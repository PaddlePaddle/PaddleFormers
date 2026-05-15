# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The LLaVA-OneVision-1.5 Authors and The HuggingFace Inc. team. All rights reserved.
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
"""Paddle LLaVA-OneVision-1.5 model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.activation import ACT2FN
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP
from ...nn.norm import Norm as GeneralNorm
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast, ModelOutput
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS
from .configuration import Llavaonevision1_5Config, LLaVAOneVision1_5TextConfig, RiceConfig


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: paddle.Tensor, k: paddle.Tensor, cos: paddle.Tensor, sin: paddle.Tensor
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    with paddle.amp.auto_cast(False):
        q = q.astype("float32")
        k = k.astype("float32")
        cos = cos.unsqueeze(-2).astype("float32")
        sin = sin.unsqueeze(-2).astype("float32")
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_q_dtype), k_embed.astype(orig_k_dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = paddle.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        axis=unsqueeze_dim
    )
    sin = paddle.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        axis=unsqueeze_dim
    )
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(axis=unsqueeze_dim)
    sin = sin.unsqueeze(axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RiceRotaryEmbedding(nn.Layer):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs


class RicePatchEmbed(nn.Layer):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2D(
            in_channels,
            embed_dim,
            kernel_size=[patch_size, patch_size],
            stride=[patch_size, patch_size],
            bias_attr=False,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.reshape([-1, self.in_channels, self.patch_size, self.patch_size])
        hidden_states = self.proj(hidden_states.astype(target_dtype)).reshape([-1, self.embed_dim])
        return hidden_states


class RicePatchMerger(nn.Layer):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-05,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, epsilon=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.mlp(self.ln_q(x).reshape([-1, self.hidden_size]))


class RiceMlp(nn.Layer):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RiceAttention(nn.Layer):
    def __init__(self, config: RiceConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self.qkv = GeneralLinear.create(
            config.hidden_size,
            config.hidden_size * 3,
            has_bias=True,
            linear_type="default",
        )
        self.proj = GeneralLinear.create(
            config.hidden_size,
            config.hidden_size,
            linear_type="default",
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        **kwargs,
    ) -> paddle.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape([seq_length, 3, self.num_heads, -1]).transpose([1, 0, 2, 3]).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        if self.config._attn_implementation == "eager":
            attention_mask = paddle.full(
                [1, seq_length, seq_length],
                paddle.finfo(query_states.dtype).min,
                dtype=query_states.dtype,
            )
            cu_list = cu_seqlens.tolist()
            for i in range(1, len(cu_list)):
                attention_mask[
                    ...,
                    cu_list[i - 1] : cu_list[i],
                    cu_list[i - 1] : cu_list[i],
                ] = 0

            query_states = query_states.transpose([1, 0, 2])
            key_states = key_states.transpose([1, 0, 2])
            value_states = value_states.transpose([1, 0, 2])
            attn_weights = paddle.matmul(query_states, key_states.transpose([0, 2, 1])) * self.scaling
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, axis=-1, dtype=paddle.float32).astype(query_states.dtype)
            attn_output = paddle.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose([1, 0, 2]).reshape([seq_length, -1])
            return self.proj(attn_output)

        query_states = query_states.transpose([1, 0, 2]).unsqueeze(0)
        key_states = key_states.transpose([1, 0, 2]).unsqueeze(0)
        value_states = value_states.transpose([1, 0, 2]).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            paddle.split(tensor, lengths.tolist(), axis=2) for tensor in (query_states, key_states, value_states)
        ]
        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                attn_mask_startend_row_indices=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = paddle.cat(attn_outputs, axis=-2)
        attn_output = attn_output.reshape([seq_length, -1])
        return self.proj(attn_output)


class RiceBlock(nn.Layer):
    def __init__(self, config: RiceConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attn = RiceAttention(config)
        self.mlp = RiceMlp(config.hidden_size, int(config.intermediate_size), config.hidden_act)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        **kwargs,
    ) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class LLaVAOneVision1_5RotaryEmbedding(nn.Layer):
    def __init__(self, config: LLaVAOneVision1_5TextConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_parameters = config.rope_parameters
        self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[LLaVAOneVision1_5TextConfig] = None,
        seq_len: Optional[int] = None,
        device: str = "cpu",
    ) -> tuple[paddle.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (paddle.arange(0, dim, 2, dtype="int64").astype("float32").to(device) / dim)
        )
        return inv_freq, 1.0

    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            if position_ids.ndim == 2:
                inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])
                position_ids_expanded = position_ids[:, None, :].float()
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose([0, 2, 1])
            else:
                inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
                    [position_ids.shape[0], position_ids.shape[1], -1, 1]
                )
                position_ids_expanded = position_ids[:, :, None, :].float()
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose([0, 1, 3, 2])
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLaVAOneVision1_5MLP(MLP):
    pass


class LLaVAOneVision1_5Attention(nn.Layer):
    def __init__(self, config: LLaVAOneVision1_5TextConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.tensor_parallel = config.tensor_model_parallel_size > 1
        self.sequence_parallel = config.sequence_parallel
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads

        if config.tensor_model_parallel_size > 1:
            assert self.num_heads % config.tensor_model_parallel_size == 0
            assert self.num_key_value_heads % config.tensor_model_parallel_size == 0
            self.num_heads = self.num_heads // config.tensor_model_parallel_size
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        kv_hidden_size = config.num_key_value_heads * self.head_dim
        q_hidden_size = config.num_attention_heads * self.head_dim
        self.qkv_proj = GeneralLinear.create(
            config.hidden_size,
            q_hidden_size + 2 * kv_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            config.hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="rowwise",
        )
        self.q_norm = GeneralNorm.create(
            config,
            norm_type="rms_norm",
            hidden_size=self.head_dim,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=self.tensor_parallel,
        )
        self.k_norm = GeneralNorm.create(
            config,
            norm_type="rms_norm",
            hidden_size=self.head_dim,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=self.tensor_parallel,
        )
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        mix_layer = self.qkv_proj(hidden_states)
        if self.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
            q_len = max_sequence_length
            target_shape = [
                bsz,
                q_len,
                self.num_key_value_heads,
                (self.num_key_value_groups + 2) * self.head_dim,
            ]
        else:
            target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
        mix_layer = paddle.reshape_(mix_layer, target_shape)
        query_states, key_states, value_states = paddle.split(
            mix_layer,
            num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
            axis=-1,
        )
        if self.gqa_or_mqa:
            query_states = paddle.reshape_(query_states, [0, 0, self.num_heads, self.head_dim])
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        cos, sin = position_embeddings
        if cos.ndim == 3:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            mrope_section = self.config.rope_parameters.get("mrope_section")
            if mrope_section is None:
                half_dim = self.head_dim // 2
                base_section = half_dim // 3
                mrope_section = [base_section, base_section, half_dim - 2 * base_section]
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                mrope_section,
            )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class LLaVAOneVision1_5DecoderLayer(nn.Layer):
    def __init__(self, config: LLaVAOneVision1_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLaVAOneVision1_5Attention(config, layer_idx)
        self.mlp = LLaVAOneVision1_5MLP(config, fuse_up_gate=True)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.attention_type = config.layer_types[layer_idx]

        if config.sequence_parallel:
            self.post_attention_layernorm.enable_sequence_parallel()
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class LLaVAOneVision1_5PretrainedModel(PretrainedModel):
    config_class = Llavaonevision1_5Config
    base_model_prefix = "model"
    input_modalities = ["image", "video", "text"]
    _no_split_modules = ["LLaVAOneVision1_5DecoderLayer", "RiceBlock"]
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "proj",
        "merger.mlp\.\d+",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Llavaonevision1_5Config):
        mapping = getattr(cls, "_checkpoint_conversion_mapping", {})
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        aoa_statements = [
            f"model.embed_tokens.weight -> {llm_prefix}embed_tokens.weight",
            f"model.norm.weight -> {llm_prefix}norm.weight",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"model.layers.$LAYER_ID.self_attn.q_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
            f"model.layers.$LAYER_ID.self_attn.k_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
            f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
            f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
            f"visual.patch_embed.proj.weight -> {visual_prefix}patch_embed.proj.weight",
            f"visual.class_embedding -> {visual_prefix}class_embedding",
            f"visual.class_pos_emb -> {visual_prefix}class_pos_emb",
            f"visual.pre_layernorm.weight -> {visual_prefix}pre_layernorm.weight",
            f"visual.pre_layernorm.bias -> {visual_prefix}pre_layernorm.bias",
            f"visual.merger.ln_q.weight -> {visual_prefix}merger.ln_q.weight",
            f"visual.merger.ln_q.bias -> {visual_prefix}merger.ln_q.bias",
            f"visual.blocks.$LAYER_ID.norm1.weight -> {visual_prefix}blocks.$LAYER_ID.norm1.weight",
            f"visual.blocks.$LAYER_ID.norm1.bias -> {visual_prefix}blocks.$LAYER_ID.norm1.bias",
            f"visual.blocks.$LAYER_ID.norm2.weight -> {visual_prefix}blocks.$LAYER_ID.norm2.weight",
            f"visual.blocks.$LAYER_ID.norm2.bias -> {visual_prefix}blocks.$LAYER_ID.norm2.bias",
            f"visual.blocks.$LAYER_ID.attn.qkv.weight^T -> {visual_prefix}blocks.$LAYER_ID.attn.qkv.weight",
            f"visual.blocks.$LAYER_ID.attn.qkv.bias -> {visual_prefix}blocks.$LAYER_ID.attn.qkv.bias",
            f"visual.blocks.$LAYER_ID.attn.proj.weight^T -> {visual_prefix}blocks.$LAYER_ID.attn.proj.weight",
            f"visual.blocks.$LAYER_ID.attn.proj.bias -> {visual_prefix}blocks.$LAYER_ID.attn.proj.bias",
            f"visual.blocks.$LAYER_ID.mlp.fc1.weight^T -> {visual_prefix}blocks.$LAYER_ID.mlp.fc1.weight",
            f"visual.blocks.$LAYER_ID.mlp.fc1.bias -> {visual_prefix}blocks.$LAYER_ID.mlp.fc1.bias",
            f"visual.blocks.$LAYER_ID.mlp.fc2.weight^T -> {visual_prefix}blocks.$LAYER_ID.mlp.fc2.weight",
            f"visual.blocks.$LAYER_ID.mlp.fc2.bias -> {visual_prefix}blocks.$LAYER_ID.mlp.fc2.bias",
            f"visual.merger.mlp.0.weight^T -> {visual_prefix}merger.mlp.0.weight",
            f"visual.merger.mlp.0.bias -> {visual_prefix}merger.mlp.0.bias",
            f"visual.merger.mlp.2.weight^T -> {visual_prefix}merger.mlp.2.weight",
            f"visual.merger.mlp.2.bias -> {visual_prefix}merger.mlp.2.bias",
            f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}",
            f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
        ]
        if config.text_config.attention_bias:
            aoa_statements += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}, axis=0",
                f"model.layers.$LAYER_ID.self_attn.o_proj.bias -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.bias",
            ]
        if cls.base_model_prefix:
            aoa_statements += ["lm_head.weight -> lm_head.weight"]
        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: Llavaonevision1_5Config):
        mapping = getattr(cls, "_checkpoint_conversion_mapping", {})
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        aoa_statements = [
            f"{llm_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{llm_prefix}norm.weight -> model.norm.weight",
            f"{llm_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{llm_prefix}layers.$LAYER_ID.self_attn.q_norm.weight -> model.layers.$LAYER_ID.self_attn.q_norm.weight",
            f"{llm_prefix}layers.$LAYER_ID.self_attn.k_norm.weight -> model.layers.$LAYER_ID.self_attn.k_norm.weight",
            f"{llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{visual_prefix}patch_embed.proj.weight -> visual.patch_embed.proj.weight",
            f"{visual_prefix}class_embedding -> visual.class_embedding",
            f"{visual_prefix}class_pos_emb -> visual.class_pos_emb",
            f"{visual_prefix}pre_layernorm.weight -> visual.pre_layernorm.weight",
            f"{visual_prefix}pre_layernorm.bias -> visual.pre_layernorm.bias",
            f"{visual_prefix}merger.ln_q.weight -> visual.merger.ln_q.weight",
            f"{visual_prefix}merger.ln_q.bias -> visual.merger.ln_q.bias",
            f"{visual_prefix}blocks.$LAYER_ID.norm1.weight -> visual.blocks.$LAYER_ID.norm1.weight",
            f"{visual_prefix}blocks.$LAYER_ID.norm1.bias -> visual.blocks.$LAYER_ID.norm1.bias",
            f"{visual_prefix}blocks.$LAYER_ID.norm2.weight -> visual.blocks.$LAYER_ID.norm2.weight",
            f"{visual_prefix}blocks.$LAYER_ID.norm2.bias -> visual.blocks.$LAYER_ID.norm2.bias",
            f"{visual_prefix}blocks.$LAYER_ID.attn.qkv.weight^T -> visual.blocks.$LAYER_ID.attn.qkv.weight",
            f"{visual_prefix}blocks.$LAYER_ID.attn.qkv.bias -> visual.blocks.$LAYER_ID.attn.qkv.bias",
            f"{visual_prefix}blocks.$LAYER_ID.attn.proj.weight^T -> visual.blocks.$LAYER_ID.attn.proj.weight",
            f"{visual_prefix}blocks.$LAYER_ID.attn.proj.bias -> visual.blocks.$LAYER_ID.attn.proj.bias",
            f"{visual_prefix}blocks.$LAYER_ID.mlp.fc1.weight^T -> visual.blocks.$LAYER_ID.mlp.fc1.weight",
            f"{visual_prefix}blocks.$LAYER_ID.mlp.fc1.bias -> visual.blocks.$LAYER_ID.mlp.fc1.bias",
            f"{visual_prefix}blocks.$LAYER_ID.mlp.fc2.weight^T -> visual.blocks.$LAYER_ID.mlp.fc2.weight",
            f"{visual_prefix}blocks.$LAYER_ID.mlp.fc2.bias -> visual.blocks.$LAYER_ID.mlp.fc2.bias",
            f"{visual_prefix}merger.mlp.0.weight^T -> visual.merger.mlp.0.weight",
            f"{visual_prefix}merger.mlp.0.bias -> visual.merger.mlp.0.bias",
            f"{visual_prefix}merger.mlp.2.weight^T -> visual.merger.mlp.2.weight",
            f"{visual_prefix}merger.mlp.2.bias -> visual.merger.mlp.2.bias",
            f"{llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}",
            f"{llm_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.gate_proj.weight, model.layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
        ]
        for layer_id in range(config.text_config.num_hidden_layers):
            for name in ("q", "k", "v"):
                aoa_statements.append(
                    f"model.layers.{layer_id}.self_attn.{name}_proj.weight^T -> model.layers.{layer_id}.self_attn.{name}_proj.weight"
                )
            aoa_statements += [
                f"model.layers.{layer_id}.mlp.gate_proj.weight^T -> model.layers.{layer_id}.mlp.gate_proj.weight",
                f"model.layers.{layer_id}.mlp.up_proj.weight^T -> model.layers.{layer_id}.mlp.up_proj.weight",
            ]

        if config.text_config.attention_bias:
            aoa_statements += [
                f"{llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}, axis=0",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.o_proj.bias -> model.layers.$LAYER_ID.self_attn.o_proj.bias",
            ]
        if cls.base_model_prefix:
            aoa_statements += ["lm_head.weight -> lm_head.weight"]
        return {"aoa_statements": aoa_statements}


class LLaVAOneVision1_5TextModel(LLaVAOneVision1_5PretrainedModel):
    config: LLaVAOneVision1_5TextConfig
    input_modalities = "text"

    def __init__(self, config: LLaVAOneVision1_5TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.LayerList(
            [LLaVAOneVision1_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = LLaVAOneVision1_5RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = getattr(
            self.config, "sliding_window", None
        ) is not None and "sliding_attention" in getattr(self.config, "layer_types", [])

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]],
        past_key_values: Optional[Cache],
        output_attentions: bool,
        use_cache: bool,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        return recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            past_key_values,
            output_attentions,
            use_cache,
            position_embeddings,
            attn_mask_startend_row_indices,
        )

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if self.config.sequence_parallel:
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").reshape([1, -1]).expand([batch_size, -1])

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": cache_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        full_mask, full_indices = create_causal_mask_and_row_indices(**mask_kwargs)
        causal_mask_mapping = {"full_attention": full_mask}
        attn_mask_startend_row_indices_mapping = {"full_attention": full_indices}

        if self.has_sliding_layers:
            (
                causal_mask_mapping["sliding_attention"],
                attn_mask_startend_row_indices_mapping["sliding_attention"],
            ) = create_sliding_window_causal_mask_and_row_indices(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    position_embeddings,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    attn_mask_startend_row_indices_mapping[decoder_layer.attention_type],
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    **kwargs,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@dataclass
class LLaVAOneVision1_5ModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


@dataclass
class LLaVAOneVision1_5CausalLMOutputWithPast(ModelOutput):
    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


@register_base_model
class LLaVAOneVision1_5Model(LLaVAOneVision1_5PretrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config: Llavaonevision1_5Config
    _no_split_modules = ["LLaVAOneVision1_5DecoderLayer", "RiceBlock"]

    def __init__(self, config: Llavaonevision1_5Config):
        super().__init__(config)
        self.visual = RiceTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = LLaVAOneVision1_5TextModel._from_config(config.text_config)
        self.rope_deltas = None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = paddle.ones([3, input_ids.shape[0], input_ids.shape[1]], dtype=input_ids.dtype)
            image_index, video_index = 0, 0
            for i, sample_input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    sample_input_ids = sample_input_ids[attention_mask[i]]
                vision_start_indices = paddle.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = sample_input_ids[vision_start_indices + 1]
                image_nums = int((vision_tokens == image_token_id).sum().item())
                video_nums = int((vision_tokens == video_token_id).sum().item())
                input_tokens = sample_input_ids.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    ed_image = (
                        input_tokens.index(image_token_id, st)
                        if image_token_id in input_tokens and remain_images > 0
                        else len(input_tokens) + 1
                    )
                    ed_video = (
                        input_tokens.index(video_token_id, st)
                        if video_token_id in input_tokens and remain_videos > 0
                        else len(input_tokens) + 1
                    )
                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(paddle.arange(text_len).reshape([1, -1]).expand([3, -1]) + st_idx)

                    t_index = paddle.arange(llm_grid_t).reshape([-1, 1]).expand([-1, llm_grid_h * llm_grid_w]).flatten()
                    h_index = paddle.arange(llm_grid_h).reshape([1, -1, 1]).expand([llm_grid_t, -1, llm_grid_w]).flatten()
                    w_index = paddle.arange(llm_grid_w).reshape([1, 1, -1]).expand([llm_grid_t, llm_grid_h, -1]).flatten()
                    llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(paddle.arange(text_len).reshape([1, -1]).expand([3, -1]) + st_idx)

                llm_positions = paddle.cat(llm_pos_ids_list, axis=1).reshape([3, -1])
                if attention_mask is not None:
                    position_ids[..., i, attention_mask[i]] = llm_positions
                else:
                    position_ids[..., i, :] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = paddle.to_tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas

        if attention_mask is not None:
            position_ids = attention_mask.astype("int64").cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand([3, -1, -1])
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = paddle.arange(input_ids.shape[1]).reshape([1, 1, -1]).expand([3, input_ids.shape[0], -1])
            mrope_position_deltas = paddle.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas

    def get_image_features(self, pixel_values: paddle.Tensor, image_grid_thw: Optional[paddle.Tensor] = None):
        pixel_values = pixel_values.astype(self.visual.patch_embed.proj.weight.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return paddle.split(image_embeds, split_sizes)

    def get_video_features(self, pixel_values_videos: paddle.Tensor, video_grid_thw: Optional[paddle.Tensor] = None):
        pixel_values_videos = pixel_values_videos.astype(self.visual.patch_embed.proj.weight.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return paddle.split(video_embeds, split_sizes)

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        if input_ids is None:
            image_token = self.get_input_embeddings()(paddle.to_tensor(self.config.image_token_id, dtype="int64"))
            video_token = self.get_input_embeddings()(paddle.to_tensor(self.config.video_token_id, dtype="int64"))
            special_image_mask = (inputs_embeds == image_token).all(-1)
            special_video_mask = (inputs_embeds == video_token).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )
        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, LLaVAOneVision1_5ModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = paddle.cat(self.get_image_features(pixel_values, image_grid_thw), axis=0).astype(inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = paddle.cat(self.get_video_features(pixel_values_videos, video_grid_thw), axis=0).astype(inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = paddle.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                dtype="int64",
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )
        output = LLaVAOneVision1_5ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class LLaVAOneVision1_5ForConditionalGeneration(LLaVAOneVision1_5PretrainedModel):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    config_class = Llavaonevision1_5Config

    def __init__(self, config: Llavaonevision1_5Config):
        super().__init__(config)
        self.model = LLaVAOneVision1_5Model(config)
        self.lm_head = GeneralLMHead(config.text_config)
        self.criterion = CriterionLayer(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple, LLaVAOneVision1_5CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[..., slice_indices, :])

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        output = LLaVAOneVision1_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        if past_key_values is None:
            cache_position = paddle.arange(seq_length)
        else:
            cache_position = paddle.to_tensor([seq_length - 1])

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


LLaVAOneVision1_5 = LLaVAOneVision1_5Model


class RiceTransformerPretrainedModel(LLaVAOneVision1_5PretrainedModel):
    config_class = RiceConfig
    _no_split_modules = ["RiceBlock"]

    def __init__(self, config: RiceConfig) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = RicePatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = RiceRotaryEmbedding(head_dim // 2)
        scale = config.hidden_size**-0.5
        self.class_embedding = self.create_parameter(
            shape=[config.hidden_size],
            default_initializer=nn.initializer.Normal(mean=0.0, std=scale),
        )
        self.class_pos_emb = self.create_parameter(
            shape=[1, head_dim // 2],
            default_initializer=nn.initializer.Normal(mean=0.0, std=1.0),
        )
        self.window_size = None
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.blocks = nn.LayerList([RiceBlock(config) for _ in range(config.depth)])
        self.merger = RicePatchMerger(
            dim=config.text_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.gradient_checkpointing = False

    def get_dtype(self):
        return self.blocks[0].mlp.fc2.weight.dtype

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = paddle.arange(h).unsqueeze(1).expand([-1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose([0, 2, 1, 3]).flatten()

            wpos_ids = paddle.arange(w).unsqueeze(0).expand([h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3]).flatten()
            pos_ids.append(paddle.stack([hpos_ids, wpos_ids], axis=-1).tile([t, 1]))
        pos_ids = paddle.cat(pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(start_axis=1)

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        return recompute(create_custom_forward(layer_module), hidden_states, cu_seqlens, position_embeddings)

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, is_verifying: bool = False) -> paddle.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        img_feats = hidden_states.shape[0]

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu = cu_seqlens.astype("int64")

        hidden_segments = []
        rotary_segments = []
        new_cu = [0]
        for i in range(1, cu.shape[0]):
            seg_start = int(cu[i - 1].item())
            seg_end = int(cu[i].item())
            segment = hidden_states[seg_start:seg_end]
            rotary_segment = rotary_pos_emb[seg_start:seg_end]
            hidden_segments.append(paddle.cat([self.class_embedding.astype(segment.dtype).unsqueeze(0), segment], axis=0))
            rotary_segments.append(paddle.cat([self.class_pos_emb.astype(rotary_segment.dtype), rotary_segment], axis=0))
            new_cu.append(new_cu[-1] + seg_end - seg_start + 1)

        hidden_states = paddle.cat(hidden_segments, axis=0)
        rotary_pos_emb = paddle.cat(rotary_segments, axis=0)
        cu_seqlens = paddle.to_tensor(new_cu, dtype="int32", place=hidden_states.place)

        hidden_states = self.pre_layernorm(hidden_states)
        emb = paddle.cat((rotary_pos_emb, rotary_pos_emb), axis=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for blk in self.blocks:
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                hidden_states = self.recompute_training_full(
                    blk,
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )

        if is_verifying:
            segments = []
            for i in range(1, len(new_cu)):
                segments.append(hidden_states[new_cu[i - 1] + 1 : new_cu[i]])
            return paddle.cat(segments, axis=0)

        output_segments = []
        for i in range(1, len(new_cu)):
            output_segments.append(hidden_states[new_cu[i - 1] + 1 : new_cu[i]])
        hidden_states = paddle.cat(output_segments, axis=0)
        if hidden_states.shape[0] != img_feats:
            raise ValueError(f"Unexpected Rice vision sequence length: {hidden_states.shape[0]} != {img_feats}")
        return self.merger(hidden_states)


LLaVAOneVision1_5ForCausalLM = LLaVAOneVision1_5ForConditionalGeneration


__all__ = [
    "LLaVAOneVision1_5",
    "LLaVAOneVision1_5ForCausalLM",
    "LLaVAOneVision1_5ForConditionalGeneration",
    "LLaVAOneVision1_5Model",
    "LLaVAOneVision1_5PretrainedModel",
    "LLaVAOneVision1_5TextModel",
    "RiceTransformerPretrainedModel",
]
