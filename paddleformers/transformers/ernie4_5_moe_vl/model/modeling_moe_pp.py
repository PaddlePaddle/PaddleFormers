# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
docstring
"""


import ast
import copy
import math
from collections import OrderedDict, deque

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.fleet import get_hybrid_communicate_group as get_hcg
from paddle.distributed.fleet.layers.mpu.mp_layers import VocabParallelEmbedding
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.distributed.fleet.utils import recompute

from paddleformers.transformers.model_utils import PipelinePretrainedModel
from paddleformers.utils.log import logger

from .configuration import Ernie4_5_MoeConfig
from .distributed import ScatterOp, mark_as_sequence_parallel_parameter
from .loss.dpo import ErnieDPOCriterion
from .modeling_moe import (
    Ernie4_5_DecoderLayer,
    Ernie4_5_MoeLMHead,
    Ernie4_5_PretrainedModel,
    ErniePretrainingCriterion,
    LayerNorm,
    RMSNorm,
    _parse_moe_group,
    mtp_hidden_states_set_zero,
)

input_ids_for_mtp = deque()


def get_attr(layer, name):
    """Return attribute from layer's inner layers recursively until found."""
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


def parse_args(args, mtp_enable=False):
    """
    Parses input arguments and converts them into model-ready format.

    Processes different input argument patterns into standardized hidden states,
    attention masks and position IDs tensors. All output tensors will have
    stop_gradient=True flag set.

    Args:
        args (Union[tuple, paddle.Tensor]): Input arguments which can be either:
            - Tuple containing 3 elements: (hidden_states, attention_mask, position_ids)
            - Tuple containing 2 elements: (hidden_states, attention_mask)
            - Tuple containing 1 element: (hidden_states)
            - Single tensor: hidden_states
            If rope_embeddings are provided, they should be included in the tuple.

    Returns:
        Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[paddle.Tensor]]:
            Returns a tuple containing:
            - hidden_states (paddle.Tensor): Processed hidden states
            - attention_mask (Optional[paddle.Tensor]): Attention mask if provided
            - position_ids (Optional[paddle.Tensor]): Position IDs if provided
            All returned tensors have stop_gradient=True.
    """
    if isinstance(args, tuple):
        if not mtp_enable:
            nbatch_pack_offset = None

        if len(args) == 4:
            hidden_states, attention_mask, position_ids, nbatch_pack_offset = args
        elif len(args) == 3:
            if mtp_enable:
                hidden_states, attention_mask, nbatch_pack_offset = args
                position_ids = None
            else:
                hidden_states, attention_mask, position_ids = args
        elif len(args) == 2:
            if mtp_enable:
                hidden_states, nbatch_pack_offset = args
                attention_mask = None
            else:
                hidden_states, attention_mask = args
            position_ids = None
        elif len(args) == 1:
            (hidden_states,) = args
            attention_mask = None
            position_ids = None
            nbatch_pack_offset = None
    else:
        hidden_states = args
        attention_mask, position_ids, nbatch_pack_offset = None, None, None
    # need position_ids to compute value for PPO.
    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    if nbatch_pack_offset is not None:
        nbatch_pack_offset.stop_gradient = True

    return hidden_states, attention_mask, position_ids, nbatch_pack_offset


def return_args(hidden_states, attention_mask=None, position_ids=None):
    """
    Packages model outputs into a standardized return format.

    Returns either a single tensor or a tuple containing hidden states and
    optional attention masks/position IDs. All returned tensors are cloned
    to prevent modification of original inputs.

    Args:
        hidden_states (paddle.Tensor): Model output tensor with shape
            (batch_size, seq_len, hidden_size).
        attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            with shape (batch_size, seq_len). Defaults to None.
        position_ids (Optional[paddle.Tensor]): Position IDs tensor
            with shape (batch_size, seq_len). Defaults to None.

    Returns:
        Union[Tuple[paddle.Tensor, ...], paddle.Tensor]:
            Returns either:
            - Single tensor if only hidden_states provided
            - Tuple containing (hidden_states, attention_mask, position_ids)
              based on provided arguments
            All returned tensors are cloned copies.
    """
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def get_pp_vp_split_layers(config, skip_recompute_num=-1):
    """
    Determines the layer partitioning scheme for Pipeline Parallelism (PP) and
    Virtual Pipeline Parallelism (VP) with recomputation optimization.

    Computes the set of layers that should skip gradient recomputation based on:
    - Pipeline parallelism configuration
    - Virtual pipeline degree
    - Model architecture parameters

    Args:
        config (Config): Model configuration object containing:
            - num_hidden_layers (int): Total number of transformer layers
            - virtual_pp_degree (int): Virtual pipeline parallelism degree
            - add_tail_layers (int): Additional tail layers to append
        skip_recompute_num (int): Number of layers per virtual pipeline stage
            to exclude from recomputation. Defaults to -1 (auto-configure).

    Returns:
        Set[int]: Set of layer indices that should skip gradient recomputation.

    Raises:
        AssertionError: If invalid PP/VP configuration is detected:
            - PP size must be > 1
            - Layer count must be divisible by (PP size * VP size)
    """
    hcg = get_hcg()
    pp_size = max(hcg.get_pipe_parallel_world_size(), 1)
    vp_size = max(config.virtual_pp_degree, 1)

    assert pp_size > 1, (
        "Only support pipeline parallel, " f"pp_size must be greater than 1, but got pp_size: {pp_size}"
    )
    layer_num = config.num_hidden_layers + config.add_tail_layers

    if skip_recompute_num == -1:
        # select all layers to skip recompute
        skip_recompute_num = vp_size

    no_recompute_layer_num = []
    if skip_recompute_num == 0:
        return set(no_recompute_layer_num)

    if vp_size == 1:
        # If vp_size == 1, we can not select model chunk for pp,
        # so if skip_recompute_num > 0, we select the all layers to skip recompute.
        if skip_recompute_num > 0:
            return set(range(layer_num))
        else:
            return set()

    assert layer_num % (pp_size * vp_size) == 0, (
        "layer_num must be divisible by pp_size * vp_size,"
        f" but got layer_num: {layer_num}, pp_size: {pp_size}, vp_size: {vp_size}"
    )

    chunk_size = layer_num // (pp_size * vp_size)
    chunk_list = [list(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(pp_size * vp_size)]

    stage_chunk_list = [[] for _ in range(pp_size)]
    for i in range(pp_size * vp_size):
        stage_chunk_list[i % pp_size].append(chunk_list[i])

    for i in range(pp_size):
        no_recompute_layer_num.extend(stage_chunk_list[i][-skip_recompute_num:])

    # trick to convert to 1D list
    return set(sum(no_recompute_layer_num, []))


def create_skip_config_for_refined_recompute(layer_idx, config):
    """
    Creates a configuration for skipping recomputation based on the configuration file,
    effective only at the specified layer index.

    Args:
        layer_idx (int): The layer index used to check whether recomputation should be skipped.
        config (dict): The configuration file of the input model.

    Returns:
        dict: Returns an updated configuration file containing the following key-value pairs:
            - skip_recompute_ops (dict): A dictionary with each model layer's each operation's name and a boolean
                                         indicating whether to skip recomputation, defaults to None.
            - If the refined_recompute key does not exist or recompute is set to False,
              the original configuration file is returned.

    """
    if not config.recompute:
        return config
    skip_config = dict()

    if len(config.refined_recompute) > 0 and config.recompute_granularity not in ["full"]:
        raise ValueError(
            "Selective recompute only support full recompute now, " "please set recompute_granularity to `full`."
        )

    for op_name, skip_num in config.refined_recompute.items():
        no_recompute_layers = get_pp_vp_split_layers(config, skip_num)
        if layer_idx in no_recompute_layers:
            skip_config[op_name] = True
        else:
            skip_config[op_name] = False
    config.skip_recompute_ops[layer_idx] = skip_config
    return config


class Ernie4_5_EmbeddingPipe(nn.Layer):
    """Extends Ernie4_5_EmbeddingPipe to forward attention_mask through the pipeline."""

    def __init__(self, config):
        """
        Initializes the embedding layer with model configuration.

        Args:
            config (Config): Model configuration.
        """
        self.sequence_parallel = config.sequence_parallel
        self.config = config

        super(Ernie4_5_EmbeddingPipe, self).__init__()
        self.use_moe = config.use_moe
        if config.tensor_parallel_degree > 1:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    @property
    def embedding_weight(self):
        """
        Provides access to the underlying embedding weights.

        Returns:
            paddle.Tensor: The weight matrix of shape [vocab_size, hidden_size]
        """
        return self.embed_tokens.weight

    def forward(self, args):
        """
        Performs embedding lookup and attention mask preprocessing.

        Args:
            args (Union[Tuple, paddle.Tensor]): Input arguments which can be:
                - Tuple containing (input_ids, attention_mask, position_ids)
                - Single tensor containing input_ids

        Returns:
            Union[Tuple, paddle.Tensor]: Returns either:
                - Tuple containing (embeddings, processed_attention_mask, position_ids)
                - Single tensor of embeddings if no masks/positions provided

        Note:
            - Automatically generates position_ids if not provided
            - Supports sequence parallel redistribution of embeddings
        """
        input_ids, attention_mask, position_ids, nbatch_pack_offset = parse_args(
            args, self.config.num_nextn_predict_layers > 0
        )
        input_ids.stop_gradient = True
        emb = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        if self.config.num_nextn_predict_layers > 0:
            if self.config.enable_mtp_magic_send:
                emb = emb[:, : -self.config.num_nextn_predict_layers, :]
                if self.sequence_parallel:
                    emb = emb.reshape([-1, emb.shape[-1]])
                    emb = ScatterOp.apply(emb)
            else:
                inputs_embeds_extra = emb[:, -self.config.num_nextn_predict_layers :, :]  # [B, S, D]
                inputs_embeds = emb[:, : -self.config.num_nextn_predict_layers, :]
                inputs_embeds_ori = inputs_embeds

                if self.sequence_parallel:
                    inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                    inputs_embeds = ScatterOp.apply(inputs_embeds)
                mtp_emb_res = [inputs_embeds]
                for depth in range(self.config.num_nextn_predict_layers):
                    inputs_embeds_mtp = paddle.concat(
                        [
                            inputs_embeds_ori[:, (depth + 1) :, :],
                            inputs_embeds_extra[:, : (depth + 1), :],
                        ],
                        axis=1,
                    )
                    if self.sequence_parallel:
                        inputs_embeds_mtp = inputs_embeds_mtp.reshape([-1, inputs_embeds_mtp.shape[-1]])
                        inputs_embeds_mtp = ScatterOp.apply(inputs_embeds_mtp)

                    mtp_emb_res.append(inputs_embeds_mtp)
                res = paddle.concat(mtp_emb_res)
                ret = (res,)
        else:
            if self.sequence_parallel:
                emb = emb.reshape([-1, emb.shape[-1]])
                emb = ScatterOp.apply(emb)

            ret = (emb,)

        if attention_mask is not None:
            if attention_mask.dtype != paddle.int32:
                if len(attention_mask.shape) == 2:
                    attention_mask = attention_mask[:, None, None, :]

                attention_mask = paddle.scale(
                    x=attention_mask.astype(emb.dtype),
                    scale=1000000.0,
                    bias=-1.0,
                    bias_after_scale=False,
                )

        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if nbatch_pack_offset is not None:
            ret += (nbatch_pack_offset.clone(),)
        if len(ret) == 1:
            ret = ret[0]
        return ret


class MTPEmbeddingPipe(Ernie4_5_EmbeddingPipe):
    """Extends Ernie4_5_EmbeddingPipe to forward attention_mask through the pipeline."""

    def __init__(self, config):
        """
        Embedding Pipe 模型。
        """
        super(MTPEmbeddingPipe, self).__init__(config)

    @property
    def embedding_weight(self):
        return self.embed_tokens.weight

    def forward(self, args):
        """
        :param args: Tuple or Tensor, containing the inputs to the module.
                     If it's a tuple, then it should contain at least three elements,
                     i.e. input_ids, attention_mask and position_ids.
                     In this case, the first three elements will be used as the arguments for forward.
                     The order of these elements is defined by `forward` method.

        :return: A Tensor with shape [batch_size * beam_width, sequence_length, hidden_dim].
                  This is the output of the model when given the specified input.

        """
        assert (
            self.config.enable_mtp_magic_send
        ), "MTPEmbedding can only be added into model only support enable_mtp_magic_send=True"

        global input_ids_for_mtp
        assert len(input_ids_for_mtp) > 0, "input_ids for mtp is empty"
        hidden_states = args[0]
        input_ids = input_ids_for_mtp.popleft()
        input_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        return (hidden_states, input_embeds)


class EmptyLayer(nn.Layer):
    """
    A pass-through layer that performs no operation on its input.
    """

    def __init__(self):
        """
        Initializes the empty layer with no parameters or buffers.

        Note:
            Inherits all functionality from the base nn.Layer class
            without adding any additional components.
        """
        super().__init__()

    def forward(self, x):
        """
        Performs identity mapping of input tensor.

        Args:
            x (paddle.Tensor): Input tensor of arbitrary shape and dtype.

        Returns:
            paddle.Tensor: The exact same tensor as input (identity function).
                Preserves all input attributes including shape, dtype and gradient.

        Note:
            This implementation maintains all autograd properties of the input tensor.
        """
        return x


class Ernie4_5_DecoderLayerPipe(Ernie4_5_DecoderLayer):
    """
    Pipeline-compatible ERNIE decoder layer with enhanced recomputation support.
    """

    def __init__(self, config, layer_idx):
        """
        Initializes the pipeline decoder layer with configuration.

        Args:
            config (Config): Model configuration containing:
                - use_var_len_flash_attn (bool): Whether to use variable-length flash attention
                - recompute (bool): Whether to enable recomputation
                - recompute_granularity (str): Granularity of recomputation
            layer_idx (int): The index of this layer in the model stack

        """
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(self, args):
        """
        Processes input through the decoder layer with optional recomputation.

        Args:
            args (Union[Tuple, paddle.Tensor]): Input which can be:
                - Tuple containing (hidden_states, attention_mask, position_ids)
                - Single tensor of hidden_states

        Returns:
            Union[Tuple, paddle.Tensor]: Returns either:
                - Tuple containing (output_states, attention_mask, position_ids)
                - Single tensor of output_states if no masks/positions provided
        """
        if self.config.num_nextn_predict_layers > 0 and not self.config.enable_mtp_magic_send:
            res = args[0]
            tensor_list = paddle.split(res, self.config.num_nextn_predict_layers + 1)
            inputs_embeds = tensor_list[-self.config.num_nextn_predict_layers :]
            args = tuple(tensor_list[: -self.config.num_nextn_predict_layers]) + args[1:]
        else:
            res = None

        hidden_states, attention_mask, position_ids, nbatch_pack_offset = parse_args(
            args, self.config.num_nextn_predict_layers > 0
        )

        max_seq_len = hidden_states.shape[1]
        if self.config.sequence_parallel:
            max_seq_len = hidden_states.shape[0] * self.config.tensor_parallel_degree
        if attention_mask is None:
            tgt_mask = None
            attn_mask_start_row_indices = None
        elif attention_mask.dtype == paddle.int32:
            tgt_mask = None
            attn_mask_start_row_indices = attention_mask[:, :, :max_seq_len]
        else:
            tgt_mask = attention_mask[:, :, :max_seq_len, :max_seq_len]
            attn_mask_start_row_indices = None
            assert len(tgt_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {tgt_mask.shape}."

        position_ids_decoder = None
        if position_ids is not None:
            position_ids_decoder = position_ids[:, :max_seq_len]

        has_gradient = not hidden_states.stop_gradient
        if self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
            hidden_states = recompute(
                super().forward,
                hidden_states,
                attention_mask=tgt_mask,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
                position_ids=position_ids_decoder,
                output_gate_logits=False,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            hidden_states = super().forward(
                hidden_states=hidden_states,
                attention_mask=tgt_mask,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
                position_ids=position_ids_decoder,
                output_gate_logits=False,
            )

        if isinstance(hidden_states, paddle.Tensor):
            ret = (hidden_states,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if nbatch_pack_offset is not None:
            ret += (nbatch_pack_offset.clone(),)
        if len(ret) == 1:
            (ret,) = ret
        if self.config.num_nextn_predict_layers > 0:
            if self.config.enable_mtp_magic_send:
                ret = (ret,)
            else:
                ret = (paddle.concat([ret[0], *inputs_embeds]),) + ret[1:]
        return ret


class RMSNormPipe(RMSNorm):
    """
    Pipeline-compatible RMSNorm layer with sequence parallelism support.
    """

    def __init__(self, config):
        """
        Initializes the RMSNorm layer with pipeline-specific configurations.

        Args:
            config (Config): Model configuration.

        Note:
            Automatically marks weight parameter for sequence parallel processing
            when sequence_parallel is enabled in config
        """
        super().__init__(config)
        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, args):
        """
        Applies RMS normalization to input features.

        Args:
            args (Union[Tuple, paddle.Tensor]): Input which can be:
                - Tuple containing (hidden_states, attention_mask, position_ids)
                - Single tensor of hidden_states

        Returns:
            paddle.Tensor: Normalized output tensor with same shape as input.
        """
        if self.config.num_nextn_predict_layers > 0:
            if self.config.enable_mtp_magic_send:
                assert len(args) == self.config.num_nextn_predict_layers + 1, "the length is not valid in mtp"
                mtp_outputs = []
                for hidden_states in args:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
            else:
                tensor_list = paddle.split(args[0], self.config.num_nextn_predict_layers + 1)
                mtp_outputs = []
                for hidden_states in tensor_list:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
        else:
            hidden_states, _, _, _ = parse_args(args)
            hidden_states = super().forward(hidden_states)
            return hidden_states


class LayerNormPipe(LayerNorm):
    """
    Pipeline-compatible Layer Normalization with sequence parallelism support.
    """

    def __init__(self, config):
        """
        Initializes the LayerNorm module with pipeline-specific configurations.

        Args:
            config (Config): Model configuration.

        Note:
            Automatically marks both weight and bias parameters for sequence
            parallel processing when enabled in configuration.
        """
        super().__init__(config)
        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)
            mark_as_sequence_parallel_parameter(self.bias)

    def forward(self, args):
        """
        Applies layer normalization to input features in pipeline-compatible manner.

        Args:
            args (Union[Tuple, paddle.Tensor]): Input which can be:
                - Tuple containing (hidden_states, attention_mask, position_ids)
                - Single tensor of hidden_states

        Returns:
            paddle.Tensor: Normalized output tensor with same shape as input,
            maintaining the following properties:
                - Mean of 0
                - Standard deviation of 1
                - Original feature dimensionality
        """
        if self.config.num_nextn_predict_layers > 0:
            if self.config.enable_mtp_magic_send:
                assert len(args) == self.config.num_nextn_predict_layers + 1, "the length is not valid in mtp"
                mtp_outputs = []
                for hidden_states in args:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
            else:
                tensor_list = paddle.split(args[0], self.config.num_nextn_predict_layers + 1)
                mtp_outputs = []
                for hidden_states in tensor_list:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
        else:
            hidden_states, _, _, _ = parse_args(args)
            hidden_states = super().forward(hidden_states)
            return hidden_states


class MTPLayer(nn.Layer):
    """_summary_

    Args:
        MTPLayer (_type_): _description_
    """

    def __init__(self, config):
        """初始化模块。

        Args:
            config (Config): 模块配置参数。

        Returns:
            None。

        """
        super().__init__()
        config = copy.deepcopy(config)
        self.config = config
        if self.config.use_recompute_mtp:
            self.config.use_recompute = False
        assert (
            self.config.num_nextn_predict_layers > 0
        ), "Adding MTPLayer must assign value to num_nextn_predict_layers"

        self.mtp_block = paddle.nn.LayerList(
            [Ernie4_5_DecoderLayer(config, layer_idx) for layer_idx in range(self.config.num_nextn_predict_layers)]
        )
        Norm = RMSNorm if config.use_rmsnorm else LayerNorm
        self.mtp_hidden_norm = paddle.nn.LayerList([Norm(config) for _ in range(self.config.num_nextn_predict_layers)])
        self.mtp_emb_norm = paddle.nn.LayerList([Norm(config) for _ in range(self.config.num_nextn_predict_layers)])

        LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else paddle.nn.Linear
        self.mtp_linear_proj = paddle.nn.LayerList(
            [
                LinearFN(
                    self.config.hidden_size * 2,
                    self.config.hidden_size,
                    bias_attr=config.use_bias,
                )
                for _ in range(self.config.num_nextn_predict_layers)
            ]
        )
        if config.sequence_parallel:
            for mtp_linear in self.mtp_linear_proj:
                mark_as_sequence_parallel_parameter(mtp_linear.weight)
                if config.use_bias:
                    mark_as_sequence_parallel_parameter(mtp_linear.bias)

    def forward(self, args):
        """forward"""

        def custom_forward(*inputs):
            """custom_forward function"""
            return self.forward_impl(*inputs)

        if self.config.use_recompute_mtp:
            return recompute(custom_forward, *args)
        else:
            return custom_forward(*args)

    def forward_impl(self, *args):
        """forward_impl"""
        _, attention_mask, position_ids, nbatch_pack_offset = parse_args(
            args, self.config.num_nextn_predict_layers > 0
        )

        if self.config.enable_mtp_magic_send:
            assert isinstance(args, tuple), "Input for MTPLayer must be tuple"
            hidden_states, inputs_embeds = args
            inputs_embeds_extra = inputs_embeds[:, -self.config.num_nextn_predict_layers :, :]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.num_nextn_predict_layers, :]
            inputs_embeds_ori = inputs_embeds
        else:
            res = args[0]
            tensor_list = paddle.split(res, self.config.num_nextn_predict_layers + 1)
            hidden_states = tensor_list[0]
            inputs_embeds_cur_depth_list = tensor_list[1:]

        max_seq_len = hidden_states.shape[1]
        if self.config.sequence_parallel:
            max_seq_len = hidden_states.shape[0] * self.config.tensor_parallel_degree

        if attention_mask is None:
            tgt_mask = None
            attn_mask_start_row_indices = None
        elif attention_mask.dtype == paddle.int32:
            tgt_mask = None
            attn_mask_start_row_indices = attention_mask
        else:
            tgt_mask = attention_mask
            attn_mask_start_row_indices = None
            assert len(tgt_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {tgt_mask.shape}."

        output_list = [hidden_states]
        for depth in range(self.config.num_nextn_predict_layers):
            if position_ids is not None:
                position_ids = position_ids[:, depth + 1 : max_seq_len + depth + 1]
            if attention_mask is not None:
                if attention_mask.dtype == paddle.int32:
                    tgt_mask = None
                    attn_mask_start_row_indices = attention_mask[:, :, depth + 1 : max_seq_len + depth + 1]
                else:
                    tgt_mask = attention_mask[
                        :,
                        :,
                        depth + 1 : max_seq_len + depth + 1,
                        depth + 1 : max_seq_len + depth + 1,
                    ]
                    attn_mask_start_row_indices = None
                    assert len(tgt_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {tgt_mask.shape}."
            if self.config.enable_mtp_magic_send:
                inputs_embeds_cur_depth = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )

                if self.config.sequence_parallel or self.config.submatrix_parallel:
                    inputs_embeds_cur_depth = inputs_embeds_cur_depth.reshape([-1, inputs_embeds_cur_depth.shape[-1]])
                    inputs_embeds_cur_depth = ScatterOp.apply(inputs_embeds_cur_depth)
            else:
                inputs_embeds_cur_depth = inputs_embeds_cur_depth_list[depth]

            nbatch_pack_offset_cur_depth = nbatch_pack_offset[:, depth + 1 : max_seq_len + depth + 1]
            if self.config.sequence_parallel:
                nbatch_pack_offset_cur_depth = nbatch_pack_offset_cur_depth.reshape((-1,))
                nbatch_pack_offset_cur_depth = ScatterOp.apply(nbatch_pack_offset_cur_depth)

            hidden_states = mtp_hidden_states_set_zero(hidden_states, nbatch_pack_offset_cur_depth)
            # Norm&Concat
            inputs_embeds_cur_depth_norm = self.mtp_emb_norm[depth](inputs_embeds_cur_depth)
            hidden_states_norm = self.mtp_hidden_norm[depth](hidden_states)

            inputs_embeds_cur_depth = self.mtp_linear_proj[depth](
                paddle.concat([inputs_embeds_cur_depth_norm, hidden_states_norm], axis=-1)
            )

            decoder_layer = self.mtp_block[depth]
            layer_outputs = decoder_layer(
                inputs_embeds_cur_depth,
                tgt_mask,  # attention_mask
                attn_mask_start_row_indices,  # attn_mask_start_row_indices
                position_ids,  # position_ids
                None,  # token-type
                False,  # output-attention
                None,  # past-kv-cache
                False,  # use-cache
                False,  # output_gate_logits
            )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            output_list.append(hidden_states)

        if self.config.enable_mtp_magic_send:
            return tuple(output_list)
        else:
            res = paddle.concat(output_list)
            return (res,)


class Ernie4_5_MoeLMHeadPipe(Ernie4_5_MoeLMHead):
    """
    Pipeline-compatible Language Model Head for ERNIE MoE models.
    """

    def forward(self, args):
        """
        Computes language model logits from hidden states in pipeline-compatible manner.

        Args:
            args (Union[Tuple, paddle.Tensor]): Input which can be:
                - Tuple containing (hidden_states, attention_mask, position_ids)
                - Single tensor of hidden_states
                Note: Attention mask and position IDs are ignored in processing

        Returns:
            paddle.Tensor: Output logits tensor with shape:
                [batch_size, sequence_length, vocab_size]
                representing unnormalized log probabilities for each token
        """
        if self.config.num_nextn_predict_layers > 0:
            logits = list()
            for _hidden_states in args:
                logits.append(super().forward(_hidden_states))
            return logits
        else:
            hidden_states, _, _, _ = parse_args(args)
            logits = super().forward(hidden_states)  # 返回tensor
            return logits

    @property
    def embedding_weight(self):
        """Return the LM head embedding weights"""
        return get_attr(self, "weight")


class ErniePretrainingCriterionPipe(ErniePretrainingCriterion):
    """
    Pipeline-compatible pretraining criterion for ERNIE models.
    """

    def __init__(self, config):
        """
        Initializes the pretraining criterion with model configuration.

        Args:
            config (Config): Model configuration.
        """
        super().__init__(config)

    def forward(self, logits, labels):
        """
        Computes pretraining loss with optional loss masking.

        Args:
            logits (Union[paddle.Tensor, Tuple[paddle.Tensor]]): Model predictions which can be:
                - Single tensor of shape [batch_size, seq_len, vocab_size]
                - Tuple of tensors for multiple prediction heads
            labels (Union[paddle.Tensor, Tuple[paddle.Tensor]]): Ground truth which can be:
                - Single tensor of shape [batch_size, seq_len]
                - Tuple containing (labels_tensor, loss_mask_tensor)

        Returns:
            Union[paddle.Tensor, Tuple]:
                During training:
                    - Single loss tensor for backpropagation
                During evaluation:
                    - Tuple containing (summed_loss, loss_components) for detailed monitoring
        """
        if isinstance(labels, tuple):
            labels, loss_mask = labels
        else:
            labels, loss_mask = labels, None
        if self.config.num_nextn_predict_layers > 0:
            mtp_logits = logits[1:]
            logits = logits[0]
            loss, loss_sum = super().forward(logits, labels, loss_mask, mtp_logits=mtp_logits)
            if not self.training:
                return loss_sum
            return loss
        else:
            loss, loss_sum = super().forward(logits, labels, loss_mask)
            if not self.training:
                return loss_sum
            return loss


class Ernie4_5_MoeForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """
    Pipeline-parallel implementation of ERNIE MoE for causal language modeling.
    """

    config_class = Ernie4_5_MoeConfig
    _get_tensor_parallel_mappings = Ernie4_5_PretrainedModel._get_tensor_parallel_mappings
    _init_weights = Ernie4_5_PretrainedModel._init_weights
    _keep_in_fp32_modules = Ernie4_5_PretrainedModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]

    @classmethod
    def _prepare_pipeline_inputs_func(cls, inputs):
        first_stage_keys = [
            "input_ids",
            "attn_mask_start_row_indices",
            "position_ids",
            "nbatch_pack_offset",
        ]
        if type(inputs) is dict or type(inputs) is OrderedDict:
            if "attention_mask" in inputs:
                first_stage_keys = [
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "nbatch_pack_offset",
                ]
        else:  # inputs is list
            if "attention_mask" in inputs[0]:
                first_stage_keys = [
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "nbatch_pack_offset",
                ]
        last_stage_keys = ["labels", "loss_mask"]

        def get_expected_keys(inputs, keys):
            ret = tuple([inputs.pop(k) for k in keys if k in inputs])
            if len(ret) == 1:
                ret = ret[0]
            return ret

        if type(inputs) is dict or type(inputs) is OrderedDict:
            return [
                get_expected_keys(inputs, first_stage_keys),
                get_expected_keys(inputs, last_stage_keys),
            ]

        keys = list(inputs[0].keys())
        inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
        return [
            get_expected_keys(inputs_batch, first_stage_keys),
            get_expected_keys(inputs_batch, last_stage_keys),
        ]

    def __init__(
        self,
        config,
    ):
        """
        Initializes the pipeline-parallel ERNIE MoE model.

        Args:
            config (Ernie4_5_MoeConfig): Model configuration.
        """
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"change initializer-range from {config.initializer_range} to {new_initializer_range}")
        config.initializer_range = new_initializer_range

        if config.moe_group == "mp":
            assert config.sequence_parallel

        if config.moe_group in {"mp", "model", "tp", "mpdp"}:
            assert config.sequence_parallel
            logger.info(f"disable FFN tensor model parallel, moe-group={config.moe_group}")
            config.disable_ffn_model_parallel = True

        config.moe_group_origin = config.moe_group
        config.moe_group = _parse_moe_group(config.moe_group)
        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)

        self.config = config

        hcg = get_hcg()
        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)

        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        no_recompute_layers = get_pp_vp_split_layers(config)
        logger.info(f"use no_recompute_layers: {no_recompute_layers}")

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    "model_shared_weight",
                    Ernie4_5_EmbeddingPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                ),
                "model",
            )
        else:
            self.add_sequential_layer(LayerDesc(Ernie4_5_EmbeddingPipe, config=config), "model")

        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(
                    Ernie4_5_DecoderLayerPipe,
                    config=create_skip_config_for_refined_recompute(i, config),
                    layer_idx=i,
                ),
                f"model.layers.{i}",
            )
        for i in range(config.add_tail_layers):
            self.add_sequential_layer(
                LayerDesc(
                    EmptyLayer,
                ),
                f"model.empty.layers.{i+config.num_hidden_layers}",
            )

        if config.num_nextn_predict_layers > 0:
            if config.enable_mtp_magic_send:
                self.add_sequential_layer(
                    SharedLayerDesc(
                        key="embed_weight_share",
                        layer_func=MTPEmbeddingPipe,
                        shared_weight_attr="embedding_weight",
                        config=config,
                    ),
                    "model.embed_share",
                )
            self.add_sequential_layer(LayerDesc(MTPLayer, config=config), "model")

        self.add_sequential_layer(
            LayerDesc(RMSNormPipe if config.use_rmsnorm else LayerNormPipe, config=config),
            "model.norm",
        )

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    "model_shared_weight",
                    Ernie4_5_MoeLMHeadPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                ),
                "lm_head",
            )
        else:
            self.add_sequential_layer(LayerDesc(Ernie4_5_MoeLMHeadPipe, config=config), "lm_head")

        # NOTE(shenliang03): recompute_interval is nouse for pipeline parallel
        recompute_interval = 0

        seg_method = (
            config.pp_seg_method
            if hasattr(config, "pp_seg_method")
            else "layer:Ernie4_5_DecoderLayer|EmptyLayer|MTPLayer"
        )
        try:
            result = ast.literal_eval(seg_method)
            if isinstance(result, list):
                seg_method = result
        except Exception:
            pass

        if (
            seg_method == "layer:Ernie4_5_DecoderLayer|EmptyLayer"
            and (config.num_hidden_layers + config.add_tail_layers) % get_hcg().topology().get_dim_size("pipe") != 0
        ):
            seg_method = "uniform"
        logger.info(f"using recompute_interval={recompute_interval}, seg_method={seg_method}")

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=self.get_loss_fn(config),
            topology=get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
            num_virtual_pipeline_stages=config.virtual_pp_degree,
        )

    def get_loss_fn(self, config):
        """
        Creates the pretraining loss function for pipeline parallelism.

        Args:
            config (Ernie4_5_MoeConfig): Model configuration

        Returns:
            ErniePretrainingCriterionPipe: Configured loss function.
        """
        if config.dpo_config is not None:
            loss_fn = ErnieDPOCriterion(config, use_infohub=True)
        else:
            loss_fn = ErniePretrainingCriterionPipe(config)

        return loss_fn
