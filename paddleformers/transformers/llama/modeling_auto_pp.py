# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Llama model"""
from __future__ import annotations

import os
from typing import Optional

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.pipelining.schedules import (
    Schedule1F1B,
    ScheduleFThenB,
    ScheduleVPP,
)
from paddle.distributed.auto_parallel.pipelining.stage import PipelineStage
from paddle.distributed.fleet.utils import recompute

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


from ...utils.tools import get_env_device
from . import fusion_ops
from .configuration import LlamaConfig
from .modeling import _expand_2d_mask, _make_causal_mask, build_alibi_tensor
from .modeling_auto import LlamaDecoderLayerAuto, LlamaPretrainedModelAuto

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

__all__ = [
    "get_llama_pp_schedule",
    "LlamaForCausalLM3DAutoPP",
]


def enable_fuse_ffn_qkv_pass():
    if os.getenv("FLAGS_enable_fused_ffn_qkv_pass") in [
        "True",
        "true",
        "1",
    ]:
        return True
    else:
        return False


def is_pp_enable():
    mesh = fleet.auto.get_mesh()
    return "pp" in mesh.dim_names


def get_mesh(pp_idx=0):
    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp", pp_idx)
    return mesh


def global_mesh_starts_with_pp():
    mesh = fleet.auto.get_mesh()
    if is_pp_enable():
        return mesh.get_mesh_with_dim("pp")
    else:
        return mesh


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


def parse_args(args):
    attention_mask, position_ids, alibi = None, None, None
    if isinstance(args, tuple):
        if len(args) == 4:
            hidden_states, attention_mask, position_ids, alibi = args
        if len(args) == 3:
            hidden_states, attention_mask, position_ids = args

        elif len(args) == 2:
            hidden_states, attention_mask = args

        if len(args) == 1:
            hidden_states = args[0]
    else:
        hidden_states = args

    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    if alibi is not None:
        alibi.stop_gradient = True

    return hidden_states, attention_mask, position_ids, alibi


def return_args(hidden_states, attention_mask=None, position_ids=None, alibi=None):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if alibi is not None:
        ret += (alibi.clone(),)
    if len(ret) == 1:
        ret = ret[0]

    return ret


class LlamaChunk(nn.Layer):
    def __init__(self, layers=None, is_first=False, is_last=False):
        super(LlamaChunk, self).__init__()
        assert not (is_first and is_last)
        self.layers = layers
        self.is_first = is_first
        self.is_last = is_last

    def forward(self, *args, **kwargs):
        if self.is_first:
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            outputs = tuple([input_ids, attention_mask, position_ids])
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
            return outputs
        elif self.is_last:
            outputs = args
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        else:
            outputs = args
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
        return outputs


def manual_model_split(model, stage_idx, group, mode, pp_degree, need_shared_params=False):

    num_hidden_layers = model.config.num_hidden_layers
    virtual_pp_degree = model.config.virtual_pp_degree if mode == "VPP" else 1
    chunk_size = num_hidden_layers // virtual_pp_degree // pp_degree
    chunk_num = virtual_pp_degree * pp_degree
    layer_lists = None

    layer_lists = model.layers

    if need_shared_params:
        shared_params_names = [["embedding_0.w_0.dist", "ernie_lm_head_0.w_0.dist"]]
    else:
        shared_params_names = []
    shared_mp = build_shared_param_map(model, shared_params_names)

    def _build_stage(model, stage_idx, group):
        new_model = None
        if stage_idx == 0:
            new_model = LlamaChunk(layer_lists[:chunk_size], is_first=True, is_last=False)
        elif stage_idx == chunk_num - 1:
            new_model = LlamaChunk(
                layer_lists[stage_idx * chunk_size : (stage_idx + 1) * chunk_size], is_first=False, is_last=True
            )
        else:
            new_model = LlamaChunk(
                layer_lists[stage_idx * chunk_size : (stage_idx + 1) * chunk_size], is_first=False, is_last=False
            )
        stage = PipelineStage(new_model, stage_idx, chunk_num, group=group, shared_parameters=shared_mp)
        return stage

    stages = []
    for i in range(virtual_pp_degree):
        stage = _build_stage(model, stage_idx + i * pp_degree, group)
        stages.append(stage)
    return stages


def build_shared_param_map(model, shared_params_names):
    shared_mp = []
    for pair in shared_params_names:
        assert len(pair) == 2, "Only exactly two parameters are supported for sharing."
        ori_name = pair[0]
        sync_name = pair[1]
        ori_param = get_param_from_name(ori_name, model)
        sync_param = get_param_from_name(sync_name, model)
        shared_mp.append({"params": [ori_param, sync_param]})
    return shared_mp


def get_param_from_name(param_name, model):
    for param in model.parameters():
        if param.name == param_name:
            return param
    raise ValueError(f"{param_name} not found in model parameters")


def get_llama_pp_schedule(model, n_microbatches, loss_fn, mode, pp_degree, group, need_shared_params=False):
    assert mode in ["VPP", "1F1B", "FThenB"]
    stages = manual_model_split(model, group.rank, group, mode, pp_degree, need_shared_params)
    if mode == "VPP":
        schedule = ScheduleVPP(stages, n_microbatches=n_microbatches, loss_fn=loss_fn)
    elif mode == "1F1B":
        schedule = Schedule1F1B(stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn)
    else:
        schedule = ScheduleFThenB(stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn)
    return schedule


colwise_placements = [dist.Replicate(), dist.Shard(1)]
rowise_placement = [dist.Replicate(), dist.Shard(0)]


class LlamaRMSNormAutoPP(nn.Layer):
    def __init__(self, config, ipp):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.ipp = ipp
        self.weight = dist.shard_tensor(
            self.weight,
            get_mesh(self.ipp),
            [dist.Replicate(), dist.Replicate()],
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, args):
        hidden_states, attention_mask, position_ids, alibi = parse_args(args)
        if self.config.use_fused_rms_norm:
            hidden_states = fusion_ops.fusion_rms_norm(
                hidden_states, self.weight, self.variance_epsilon, self.config.use_fast_layer_norm
            )
            return return_args(hidden_states, attention_mask, position_ids, alibi)

        with paddle.amp.auto_cast(False):
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)

        return return_args(hidden_states * self.weight, attention_mask, position_ids, alibi)


class LlamaEmbeddingAutoPP(nn.Layer):
    """Extends LlamaEmbeddings to forward attention_mask through the pipeline."""

    def __init__(self, config):
        super(LlamaEmbeddingAutoPP, self).__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        embedding_placements = (
            [dist.Replicate(), dist.Shard(1)]
            if self.config.tensor_parallel_degree > 1
            else [dist.Replicate(), dist.Replicate()]
        )

        self.embed_tokens.weight = dist.shard_tensor(
            self.embed_tokens.weight,
            get_mesh(),
            embedding_placements,
        )
        self.placements = (
            [dist.Shard(1), dist.Shard(0)] if self.config.sequence_parallel else [dist.Shard(0), dist.Replicate()]
        )

    @property
    def embedding_weight(self):
        return get_attr(self.embed_tokens, "weight")

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape, past_key_values_length=past_key_values_length
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        if get_env_device() in ["npu", "mlu", "intel_hpu"]:
            x = paddle.to_tensor(0.0, dtype="float32")
            y = paddle.to_tensor(paddle.finfo(dtype).min, dtype="float32")
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), x, y).astype(dtype)
        elif get_env_device() == "xpu":
            x = paddle.to_tensor(0.0, dtype="float32")
            y = paddle.to_tensor(-1.7005809656952787e38, dtype="float32")
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), x, y)
        elif get_env_device() == "gcu":
            min_val = paddle.finfo(dtype).min
            x = paddle.to_tensor(0.0, dtype=dtype)
            y = paddle.to_tensor(min_val, dtype=dtype)
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), x, y).astype(dtype)
        else:
            expanded_attn_mask = paddle.where(expanded_attn_mask, 0.0, paddle.finfo(dtype).min)
            expanded_attn_mask = expanded_attn_mask.astype(dtype)
        return expanded_attn_mask

    def forward(self, args):
        input_ids, attention_mask, position_ids, alibi = parse_args(args)

        input_ids.stop_gradient = True

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids")

        seq_length_with_past = seq_length
        cache_length = 0

        with paddle.amp.auto_cast(False):
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.sequence_parallel:
            # [B, S, H] -> [S, B, H]
            inputs_embeds = paddle.transpose(inputs_embeds, [1, 0, 2])

        global_mesh = global_mesh_starts_with_pp()
        if position_ids is None and self.config.sep_parallel_degree > 1:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))
        if position_ids is not None:
            position_ids = dist.shard_tensor(
                position_ids,
                global_mesh,
                [dist.Replicate() for _ in range(len(global_mesh._shape))],
            )
        # embed positions
        if not self.config.use_flash_attention and attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)

        if self.config.alibi:
            if attention_mask is None:
                attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
            alibi_place = [dist.Replicate() for _ in range(len(global_mesh._shape))]
            alibi = build_alibi_tensor(attention_mask, self.config.num_attention_heads, dtype=inputs_embeds.dtype)
            alibi = dist.shard_tensor(alibi, global_mesh, alibi_place)
        else:
            alibi = None
        if self.config.use_flash_attention and not self.config.alibi:
            # attention_mask in flash_attn is always None for pretrain
            # atttenton_mask is used in scaled_dot_product_attention with alibi_tensor
            attention_mask = None
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
            )  # [bs, 1, seq_len, seq_len]
            attention_mask = dist.shard_tensor(
                attention_mask,
                global_mesh,
                [dist.Replicate() for _ in range(len(global_mesh._shape))],
            )
        hidden_states = inputs_embeds

        hidden_states = dist.reshard(hidden_states, get_mesh(), self.placements)

        return return_args(hidden_states, attention_mask, position_ids, alibi)


class LlamaDecoderLayerAutoPP(nn.Layer):
    def __init__(self, config, idx, layerwise_recompute: bool = False, ipp: Optional[int] = None):
        super(LlamaDecoderLayerAutoPP, self).__init__()
        self.config = config
        self.layer_id = idx
        self.embed_tokens = None
        self.norm = None
        self.lm_head = None
        if self.layer_id == 0:
            self.embed_tokens = LlamaEmbeddingAutoPP(config)

        self.layer = LlamaDecoderLayerAuto(config, layerwise_recompute, ipp)
        self.ipp = ipp
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        if self.layer_id == self.config.num_hidden_layers - 1:
            self.norm = LlamaRMSNormAutoPP(config, ipp)
            self.lm_head = LlamaLMHeadAutoPP(config)

    def forward(self, args):
        if self.embed_tokens is not None:
            args = self.embed_tokens(args)
        hidden_states, attention_mask, position_ids, alibi = parse_args(args)
        output_attentions = self.config.output_attentions
        use_cache = self.config.use_cache

        past_key_value = None

        has_gradient = not hidden_states.stop_gradient

        if position_ids is not None:
            position_ids_input = dist.reshard(
                position_ids,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Replicate()],
            )
        else:
            position_ids_input = position_ids
        attention_mask_input = (
            dist.reshard(
                attention_mask,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Replicate()],
            )
            if attention_mask is not None
            else None
        )
        alibi_input = (
            dist.reshard(
                alibi,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Replicate()],
            )
            if alibi is not None
            else None
        )
        if (
            self.enable_recompute
            and self.layer_id not in self.no_recompute_layers
            and has_gradient
            and self.recompute_granularity == "full"
        ):
            layer_outputs = recompute(
                self.layer,
                hidden_states,
                position_ids_input,
                attention_mask_input,
                output_attentions,
                past_key_value,
                use_cache,
                alibi_input,
            )
        else:
            layer_outputs = self.layer(
                hidden_states,
                position_ids_input,
                attention_mask_input,
                output_attentions,
                past_key_value,
                use_cache,
                alibi_input,
            )

        if type(layer_outputs) is tuple:
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        ret_args = return_args(
            hidden_states,
            attention_mask,
            position_ids,
            alibi,
        )
        if self.norm is not None:
            ret_args = self.norm(ret_args)

        if self.lm_head is not None:
            ret_args = self.lm_head(ret_args)
        return ret_args


class LlamaLMHeadAutoPP(nn.Layer):
    def __init__(self, config: LlamaConfig):
        super(LlamaLMHeadAutoPP, self).__init__()
        self.config = config

        vocab_size = config.vocab_size
        self.weight = self.create_parameter(
            shape=[config.hidden_size, vocab_size],
            dtype=paddle.get_default_dtype(),
        )
        self.weight = dist.shard_tensor(
            self.weight,
            get_mesh(-1),
            colwise_placements,
        )

    def forward(self, args):
        hidden_states, attention_mask, position_ids, alibi = parse_args(args)

        if self.config.sequence_parallel:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(-1),
                [dist.Shard(1), dist.Replicate()],
            )
            hidden_states = paddle.transpose(hidden_states, [1, 0, 2])
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=False)
        return return_args(logits, attention_mask, position_ids, alibi)


class LlamaForCausalLM3DAutoPP(LlamaPretrainedModelAuto):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        decoder_layers = []

        def get_pp_stage_id(layer_id):
            pp_degree = global_mesh_starts_with_pp().shape[0]
            chunk_size = self.config.num_hidden_layers // (pp_degree * self.config.virtual_pp_degree)
            chunk_id = layer_id // chunk_size
            pp_stage_id = chunk_id % pp_degree
            return pp_stage_id

        for i in range(config.num_hidden_layers):
            pp_stage_id = get_pp_stage_id(i)
            decoder_layers.append(LlamaDecoderLayerAutoPP(config, i, i not in self.no_recompute_layers, pp_stage_id))
        self.layers = nn.LayerList(decoder_layers)

    def forward(
        self,
        input_ids=None,
        labels=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = return_args(input_ids, attention_mask, position_ids)

        # decoder layers
        for idx, (decoder_layer) in enumerate(self.layers):
            outputs = decoder_layer(outputs)

        return outputs[0]
