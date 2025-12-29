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


import paddle
from paddle import nn
from paddle.distributed.fleet import get_hybrid_communicate_group as get_hcg
from paddle.distributed.fleet.layers.mpu.mp_layers import VocabParallelEmbedding

from .distributed import ScatterOp


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
            - virtual_pipeline_model_parallel_size (int): Virtual pipeline parallelism degree
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
    vp_size = max(config.virtual_pipeline_model_parallel_size, 1)

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
    if config.recompute_granularity is not None or not isinstance(config.recompute_modules, dict):
        config.skip_recompute_ops[layer_idx] = {}
        return config
    skip_config = dict()

    if len(config.recompute_modules) > 0 and config.recompute_granularity != "full":
        raise ValueError(
            "Selective recompute only support full recompute now, " "please set recompute_granularity to `full`."
        )

    layer_num = config.num_layers if hasattr(config, "num_layers") else config.num_hidden_layers
    if hasattr(config, "add_tail_layer") and config.add_tail_layer:
        layer_num += 1

    for op_name, recompute_num in config.recompute_modules.items():
        skip_num = -1
        if recompute_num < 0:
            skip_num = 0
        elif recompute_num == 0:
            skip_num = -1
        else:
            skip_num = max(layer_num - recompute_num, 0)
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
        if config.tensor_model_parallel_size > 1:
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
