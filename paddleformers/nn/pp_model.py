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
import ast
import math
from typing import OrderedDict

import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed.fleet import get_hybrid_communicate_group as get_hcg
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ..transformers.configuration_utils import PretrainedConfig
from ..transformers.model_utils import PipelinePretrainedModel
from ..utils.log import logger
from .criterion import CriterionLayer
from .embedding import Embedding
from .lm_head import LMHead
from .moe.utils import _parse_moe_group
from .norm import LayerNorm, RMSNorm


def parse_args(args, mtp_enable=False, is_embed=False):
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
        mtp_enable (bool): Flag for Multi-Token Prediction.
        is_embed (bool): Flag to indicate if processing is for EmbeddingPipe,
                                  affects 3-argument tuple parsing.
    Returns:
        Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[paddle.Tensor]]:
            Returns a tuple containing:
            - hidden_states (paddle.Tensor): Processed hidden states
            - attention_mask (Optional[paddle.Tensor]): Attention mask if provided
            - position_ids (Optional[paddle.Tensor]): Position IDs if provided
            All returned tensors have stop_gradient=True.
    """
    if isinstance(args, tuple):
        position_embeddings = None
        nbatch_pack_offset = None

        if len(args) == 5:
            hidden_states, attention_mask, position_ids, position_embeddings, nbatch_pack_offset = args
        elif len(args) == 4:
            hidden_states, attention_mask, position_ids, position_embeddings = args
        elif len(args) == 3:
            if mtp_enable:
                hidden_states, attention_mask, nbatch_pack_offset = args
                position_ids = None
            elif is_embed:
                hidden_states, attention_mask, position_ids = args
            else:
                hidden_states, position_ids, position_embeddings = args
                attention_mask = None
                nbatch_pack_offset = None
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
        attention_mask, position_ids, position_embeddings, nbatch_pack_offset = None, None, None, None
    # need position_ids to compute value for PPO.
    if position_ids is not None:
        position_ids.stop_gradient = True

    if position_embeddings is not None:
        position_embeddings.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    if nbatch_pack_offset is not None:
        nbatch_pack_offset.stop_gradient = True

    return hidden_states, attention_mask, position_ids, position_embeddings, nbatch_pack_offset


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


def get_attr(layer, name):
    """Return attribute from layer's inner layers recursively until found."""
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


class RotaryEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.base = config.rope_theta

    def forward(self, x, position_ids):
        """
        Compute rotary position embeddings for given sequence length.

        Args:
            seq_length (int): Maximum sequence length
            position_ids (Tensor): Position ids of shape [batch_size, seq_length]

        Returns:
            Tensor: Rotary position embeddings of shape [1, 1, seq_length, head_dim]
        """
        indices = paddle.arange(0, self.head_dim, 2, dtype="float32")
        indices = 1 / self.base ** (indices / self.head_dim)

        sinusoid_inp = position_ids.unsqueeze(-1).astype("float32") * indices.unsqueeze(
            0
        )  # [b, s, 1] * [1, d/2] -> [b, s, d/2]
        emb = paddle.cat((sinusoid_inp, sinusoid_inp), axis=-1)
        cos = emb.cos()
        sin = emb.sin()

        # keeping it in full precision
        return cos, sin


class EmbeddingPipe(nn.Layer):
    def __init__(self, config, embed_cls=None, rotary_emb_cls=None):
        """
        Initializes the embedding layer with model configuration.
        Args:
            config (Config): Model configuration.
        """
        super(EmbeddingPipe, self).__init__()
        self.sequence_parallel = config.sequence_parallel
        self.config = config
        if rotary_emb_cls is None:
            self.rotary_emb = RotaryEmbedding(config)
        else:
            self.rotary_emb = rotary_emb_cls(config)
        if embed_cls is None:
            self.embed_tokens = Embedding.create(config)
        else:
            self.embed_tokens = embed_cls(config)

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
        num_nextn_predict_layers = self.config.get("num_nextn_predict_layers", 0)
        enable_mtp_magic_send = self.config.get("enable_mtp_magic_send", False)

        input_ids, attention_mask, position_ids, _, nbatch_pack_offset = parse_args(
            args, num_nextn_predict_layers > 0, is_embed=True
        )
        input_ids.stop_gradient = True
        emb = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        if position_ids is None and not self.config.fuse_rope:
            position_ids = (
                paddle.arange(
                    0,
                    input_ids.shape[1],
                    dtype="int64",
                )
                .unsqueeze(0)
                .tile([input_ids.shape[0], 1])
            )
        if self.config.fuse_rope:
            position_embeddings = None
        else:
            position_embeddings = paddle.stack(self.rotary_emb(emb, position_ids))  # cos and sin

        if num_nextn_predict_layers > 0:
            if enable_mtp_magic_send:
                emb = emb[:, :-num_nextn_predict_layers, :]
                if self.sequence_parallel:
                    emb = emb.reshape([-1, emb.shape[-1]])
                    emb = ScatterOp.apply(emb)
            else:
                inputs_embeds_extra = emb[:, -num_nextn_predict_layers:, :]  # [B, S, D]
                inputs_embeds = emb[:, :-num_nextn_predict_layers, :]
                inputs_embeds_ori = inputs_embeds

                if self.sequence_parallel:
                    inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                    inputs_embeds = ScatterOp.apply(inputs_embeds)
                mtp_emb_res = [inputs_embeds]
                for depth in range(num_nextn_predict_layers):
                    inputs_embeds_mtp = paddle.cat(
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
                res = paddle.cat(mtp_emb_res)
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
        if position_embeddings is not None:
            ret += (position_embeddings.clone(),)
        if nbatch_pack_offset is not None:
            ret += (nbatch_pack_offset.clone(),)
        if len(ret) == 1:
            ret = ret[0]
        return ret


class RMSNormPipe(RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.sequence_parallel:
            self.enable_sequence_parallel()

    def forward(self, args):
        hidden_states, _, _, _, _ = parse_args(args)
        hidden_states = super().forward(hidden_states)
        return hidden_states


class LayerNormPipe(LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.sequence_parallel:
            self.enable_sequence_parallel()

    def forward(self, args):
        hidden_states, _, _, _, _ = parse_args(args)
        hidden_states = super().forward(hidden_states)
        return hidden_states


class EmptyLayer(nn.Layer):
    """
    A pass-through layer that performs no operation on its input.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class LMHeadPipe(LMHead):
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
        hidden_states, _, _, _, _ = parse_args(args)
        logits = super().forward(hidden_states)
        return logits

    @property
    def embedding_weight(self):
        """Return the LM head embedding weights"""
        return get_attr(self, "weight")


def make_decoder_layer_pipe(decoder_layer):
    def forward(self, args):
        num_nextn_predict_layers = self.config.get("num_nextn_predict_layers", 0)
        enable_mtp_magic_send = self.config.get("enable_mtp_magic_send", False)
        if num_nextn_predict_layers > 0 and not enable_mtp_magic_send:
            res = args[0]
            tensor_list = paddle.split(res, num_nextn_predict_layers + 1)
            inputs_embeds = tensor_list[-num_nextn_predict_layers:]
            args = tuple(tensor_list[:-num_nextn_predict_layers]) + args[1:]
        else:
            res = None
        hidden_states, attention_mask, position_ids, position_embeddings, nbatch_pack_offset = parse_args(args)
        max_seq_len = hidden_states.shape[1]
        if self.config.sequence_parallel:
            max_seq_len = hidden_states.shape[0] * self.config.tensor_parallel_degree
        if attention_mask is None:
            tgt_mask = None
            attn_mask_startend_row_indices = None
        elif attention_mask.dtype == paddle.int32:
            tgt_mask = None
            attn_mask_startend_row_indices = attention_mask[:, :, :max_seq_len]
        else:
            tgt_mask = attention_mask[:, :, :max_seq_len, :max_seq_len]
            attn_mask_startend_row_indices = None
            assert len(tgt_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {tgt_mask.shape}."

        if position_embeddings is not None:
            position_embeddings = position_embeddings[..., :max_seq_len, :]
            tuple_position_embeddings = (position_embeddings[0], position_embeddings[1])
        else:
            tuple_position_embeddings = None

        has_gradient = not hidden_states.stop_gradient
        if self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
            hidden_states = recompute(
                decoder_layer.forward,
                self,
                hidden_states,
                attention_mask=tgt_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=tuple_position_embeddings,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            hidden_states = decoder_layer.forward(
                self,
                hidden_states=hidden_states,
                attention_mask=tgt_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_embeddings=tuple_position_embeddings,
            )

        if isinstance(hidden_states, paddle.Tensor):
            ret = (hidden_states,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if position_embeddings is not None:
            ret += (position_embeddings.clone(),)
        if nbatch_pack_offset is not None:
            ret += (nbatch_pack_offset.clone(),)
        if len(ret) == 1:
            (ret,) = ret
        if num_nextn_predict_layers > 0:
            if enable_mtp_magic_send:
                ret = (ret,)
            else:
                ret = (paddle.cat([ret[0], *inputs_embeds]),) + ret[1:]

        return ret

    return type(
        "DecoderLayerPipe",
        (decoder_layer,),
        {"forward": forward},
    )


class CriterionLayerPipe(CriterionLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_tuple = False  # loss_func only return loss, no loss_sum

    def forward(self, logits, labels):
        if isinstance(labels, tuple) and "sft" in self.loss_type:
            labels, loss_mask = labels
        loss = super().forward(logits, labels)
        return loss


class GeneralModelForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    _decoder_layer_cls = None
    _decoder_layer_pipe_cls = None
    _get_tensor_parallel_mappings = None
    _init_weights = None
    _keep_in_fp32_modules = None
    _tied_weights_keys = ["lm_head.weight"]
    config_class = PretrainedConfig
    transpose_weight_keys = None
    _embed_cls = None
    _rotary_emb_cls = None
    _norm_cls = "rms_norm"
    _mtp_layer_pipe_cls = None
    _embedding_pipe_cls = None
    _decoder_layer_pipe_cls = None
    _criterion_pipe_cls = None
    _lmhead_pipe_cls = None
    _rms_norm_pipe_cls = None

    def __init__(self, config: PretrainedConfig, **kwargs):
        if getattr(config, "sliding_window", None) is not None and "sliding_attention" in getattr(
            config, "layer_types", []
        ):
            logger.error(
                "Pipeline Parallelism (PP) does not support sliding window attention. "
                "To prevent issues during training, please set use_sliding_window=False."
            )

        # dynamic inherit DecoderLayer
        if self._decoder_layer_cls is None:
            raise ValueError("_decoder_layer_cls must be set before init.")

        EmbeddingPipeCls = self._embedding_pipe_cls if self._embedding_pipe_cls is not None else EmbeddingPipe

        if self._decoder_layer_pipe_cls is None:
            DecoderLayerPipe = make_decoder_layer_pipe(self._decoder_layer_cls)
        else:
            DecoderLayerPipe = self._decoder_layer_pipe_cls

        LMHeadPipeCls = self._lmhead_pipe_cls if self._lmhead_pipe_cls is not None else LMHeadPipe
        MTPLayerPipeCls = self._mtp_layer_pipe_cls if self._mtp_layer_pipe_cls is not None else None
        RMSNormPipeCls = self._rms_norm_pipe_cls if self._rms_norm_pipe_cls is not None else RMSNormPipe

        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"change initializer-range from {config.initializer_range} to {new_initializer_range}")
        config.initializer_range = new_initializer_range

        moe_group = config.get("moe_group", "dummy")
        if moe_group == "mp":
            assert config.sequence_parallel

        if moe_group in {"mp", "model", "tp", "mpdp"}:
            assert config.sequence_parallel
            logger.info(f"disable FFN tensor model parallel, moe-group={moe_group}")
            config.disable_ffn_model_parallel = True

        config.moe_group_origin = moe_group
        config.moe_group = _parse_moe_group(moe_group)
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
                    EmbeddingPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                    embed_cls=self._embed_cls,
                    rotary_emb_cls=self._rotary_emb_cls,
                ),
                "model",
            )
        else:
            self.add_sequential_layer(
                LayerDesc(
                    EmbeddingPipeCls, config=config, embed_cls=self._embed_cls, rotary_emb_cls=self._rotary_emb_cls
                ),
                "model",
            )

        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(
                    DecoderLayerPipe,
                    config=config,
                    layer_idx=i,
                ),
                f"model.layers.{i}",
            )
        for i in range(config.num_nextn_predict_layers):
            if MTPLayerPipeCls is not None:
                self.add_sequential_layer(
                    LayerDesc(MTPLayerPipeCls, config=config, layer_idx=config.num_hidden_layers + i),
                    f"model.layers.{config.num_hidden_layers + i}",
                )
        for i in range(config.add_tail_layers):
            self.add_sequential_layer(
                LayerDesc(
                    EmptyLayer,
                ),
                f"empty.layers.{i+config.num_hidden_layers}",
            )

        self.add_sequential_layer(
            LayerDesc(RMSNormPipeCls if self._norm_cls == "rms_norm" else LayerNormPipe, config=config),
            "model.norm",
        )

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    "model_shared_weight",
                    LMHeadPipeCls,
                    shared_weight_attr="embedding_weight",
                    config=config,
                ),
                "lm_head",
            )
        else:
            self.add_sequential_layer(LayerDesc(LMHeadPipeCls, config=config), "lm_head")
        recompute_interval = 0

        seg_method = config.pp_seg_method if hasattr(config, "pp_seg_method") else "layer:DecoderLayer|EmptyLayer"
        try:
            result = ast.literal_eval(seg_method)
            if isinstance(result, list):
                seg_method = result
        except Exception:
            pass

        if (
            seg_method == "layer:DecoderLayer|EmptyLayer"
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
        CriterionPipeCls = self._criterion_pipe_cls if self._criterion_pipe_cls is not None else CriterionLayerPipe

        if config.get("dpo_config", None) is not None:
            loss_fn = CriterionPipeCls(config, use_infohub=True)
        else:
            loss_fn = CriterionPipeCls(config)

        return loss_fn

    @classmethod
    def register_cls_attr(cls, config_class=None, pretrained_model_class=None):
        if config_class is not None:
            cls.config_class = config_class
        if pretrained_model_class is not None:
            if hasattr(pretrained_model_class, "_get_tensor_parallel_mappings"):
                cls._get_tensor_parallel_mappings = pretrained_model_class._get_tensor_parallel_mappings
            if hasattr(pretrained_model_class, "_get_fuse_or_split_param_mappings"):
                cls._get_fuse_or_split_param_mappings = pretrained_model_class._get_fuse_or_split_param_mappings
            if hasattr(pretrained_model_class, "_init_weights"):
                cls._init_weights = pretrained_model_class._init_weights
            if hasattr(pretrained_model_class, "_keep_in_fp32_modules"):
                cls._keep_in_fp32_modules = pretrained_model_class._keep_in_fp32_modules
            if hasattr(pretrained_model_class, "transpose_weight_keys"):
                cls.transpose_weight_keys = pretrained_model_class.transpose_weight_keys
        return cls

    @classmethod
    def _prepare_pipeline_inputs_func(cls, inputs):
        first_stage_keys = [
            "input_ids",
            "attn_mask_startend_row_indices",
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
            # (NOTE) attn_mask_start_row_indices is special for erniekit
            elif "attn_mask_start_row_indices" in inputs:
                first_stage_keys = [
                    "input_ids",
                    "attn_mask_start_row_indices",
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
            elif "attn_mask_start_row_indices" in inputs[0]:
                first_stage_keys = [
                    "input_ids",
                    "attn_mask_start_row_indices",
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
