# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import nn

from ...generation import GenerationMixin
from ...nn.activation import ACT2FN
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.norm import Norm as GeneralNorm
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    ModelOutput,
)
from ..model_utils import PretrainedModel
from .configuration import GlmOcrConfig, GlmOcrTextConfig, GlmOcrVisionConfig


class GlmOcrVisionMlp(nn.Layer):
    def __init__(self, config, bias: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half_llm(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return paddle.stack((-x2, x1), axis=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, axis=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, axis=-1)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos.astype(q.dtype)) + (rotate_half_llm(q_rot) * sin.astype(q.dtype))
    k_embed = (k_rot * cos.astype(k.dtype)) + (rotate_half_llm(k_rot) * sin.astype(k.dtype))

    q_embed = paddle.concat([q_embed, q_pass], axis=-1)
    k_embed = paddle.concat([k_embed, k_pass], axis=-1)
    return q_embed, k_embed


class GlmOcrTextAttention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: GlmOcrTextConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_parameters = config.rope_parameters
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=False)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        attention_mask: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: paddle.LongTensor | None = None,
        attn_mask_startend_row_indices: paddle.LongTensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None, tuple[paddle.Tensor] | None]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape([bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GlmOcrVisionRotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        self.inv_freq = inv_freq

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs


class GlmOcrTextMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = paddle.chunk(up_states, chunks=2, axis=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


class GlmOcrTextDecoderLayer(nn.Layer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = GlmOcrTextAttention(config, layer_idx)
        self.mlp = GlmOcrTextMLP(config)
        self.input_layernorm = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=1e-6
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=1e-6
        )
        self.post_self_attn_layernorm = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=1e-6
        )
        self.post_mlp_layernorm = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=1e-6
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class GlmOcrPreTrainedModel(PretrainedModel):
    config_class = GlmOcrConfig
    base_model_prefix = "model"

    input_modalities = ("image", "video", "text")
    _no_split_modules = ["GlmOcrTextDecoderLayer", "GlmOcrVisionBlock"]
    _keys_to_ignore_on_load_unexpected = [r"model\.language_model\.layers\.16.*"]

    transpose_weight_keys = [
        r"attn\.qkv",
        r"attn\.proj",
        r"merger\.proj",
        r"merger\.gate_proj",
        r"merger\.up_proj",
        r"merger\.down_proj",
        r"mlp\.gate_proj",
        r"mlp\.up_proj",
        r"mlp\.down_proj",
        r"self_attn\.q_proj",
        r"self_attn\.k_proj",
        r"self_attn\.v_proj",
        r"self_attn\.o_proj",
        r"mlp\.gate_up_proj",
        r"lm_head",
    ]

    def _init_weights(self, layer):
        super()._init_weights(layer)
        if isinstance(layer, GlmOcrVisionRotaryEmbedding):
            inv_freq = 1.0 / (layer.theta ** (paddle.arange(0, layer.dim, 2, dtype="float32") / layer.dim))
            layer.inv_freq = inv_freq

    @classmethod
    def _gen_aoa_config(cls, config: GlmOcrConfig):
        model_prefix = "model"
        aoa_statements = []
        aoa_statements += [
            "lm_head.weight^T -> lm_head.weight",
        ]

        # ========== Text Embedding / Final Norm ==========
        aoa_statements += [
            f"model.language_model.embed_tokens.weight -> {model_prefix}.language_model.embed_tokens.weight",
            f"model.language_model.norm.weight        -> {model_prefix}.language_model.norm.weight",
        ]

        # ========== Vision patch embed ==========
        aoa_statements += [
            f"model.visual.patch_embed.proj.weight -> {model_prefix}.visual.patch_embed.proj.weight",
            f"model.visual.patch_embed.proj.bias   -> {model_prefix}.visual.patch_embed.proj.bias",
        ]

        # ========== Vision blocks ==========
        for i in range(config.vision_config.depth):
            src = f"model.visual.blocks.{i}"
            dst = f"{model_prefix}.visual.blocks.{i}"

            aoa_statements += [
                # Norm
                f"{src}.norm1.weight -> {dst}.norm1.weight",
                f"{src}.norm2.weight -> {dst}.norm2.weight",
                # ===== Attention =====
                f"{src}.attn.qkv.weight^T  -> {dst}.attn.qkv.weight",
                f"{src}.attn.qkv.bias      -> {dst}.attn.qkv.bias",
                f"{src}.attn.proj.weight^T -> {dst}.attn.proj.weight",
                f"{src}.attn.proj.bias     -> {dst}.attn.proj.bias",
                # Q/K norm
                f"{src}.attn.q_norm.weight -> {dst}.attn.q_norm.weight",
                f"{src}.attn.k_norm.weight -> {dst}.attn.k_norm.weight",
                # ===== MLP (SwiGLU) =====
                f"{src}.mlp.gate_proj.weight^T -> {dst}.mlp.gate_proj.weight",
                f"{src}.mlp.gate_proj.bias     -> {dst}.mlp.gate_proj.bias",
                f"{src}.mlp.up_proj.weight^T   -> {dst}.mlp.up_proj.weight",
                f"{src}.mlp.up_proj.bias       -> {dst}.mlp.up_proj.bias",
                f"{src}.mlp.down_proj.weight^T -> {dst}.mlp.down_proj.weight",
                f"{src}.mlp.down_proj.bias     -> {dst}.mlp.down_proj.bias",
            ]

        # ========== Vision merger ==========
        aoa_statements += [
            f"model.visual.merger.proj.weight^T                -> {model_prefix}.visual.merger.proj.weight",
            f"model.visual.merger.post_projection_norm.weight -> {model_prefix}.visual.merger.post_projection_norm.weight",
            f"model.visual.merger.post_projection_norm.bias   -> {model_prefix}.visual.merger.post_projection_norm.bias",
            f"model.visual.merger.gate_proj.weight^T          -> {model_prefix}.visual.merger.gate_proj.weight",
            f"model.visual.merger.up_proj.weight^T            -> {model_prefix}.visual.merger.up_proj.weight",
            f"model.visual.merger.down_proj.weight^T          -> {model_prefix}.visual.merger.down_proj.weight",
        ]

        # ========== Vision downsample ==========
        aoa_statements += [
            f"model.visual.downsample.weight -> {model_prefix}.visual.downsample.weight",
            f"model.visual.downsample.bias   -> {model_prefix}.visual.downsample.bias",
            f"model.visual.post_layernorm.weight -> {model_prefix}.visual.post_layernorm.weight",
        ]

        # ========== Text blocks ==========
        for i in range(config.text_config.num_hidden_layers):
            src = f"model.language_model.layers.{i}"
            dst = f"{model_prefix}.language_model.layers.{i}"

            aoa_statements += [
                # ===== Attention =====
                f"{src}.self_attn.q_proj.weight^T -> {dst}.self_attn.q_proj.weight",
                f"{src}.self_attn.k_proj.weight^T -> {dst}.self_attn.k_proj.weight",
                f"{src}.self_attn.v_proj.weight^T -> {dst}.self_attn.v_proj.weight",
                f"{src}.self_attn.o_proj.weight^T -> {dst}.self_attn.o_proj.weight",
                # ===== MLP =====
                f"{src}.mlp.gate_up_proj.weight^T -> {dst}.mlp.gate_up_proj.weight",
                f"{src}.mlp.down_proj.weight^T    -> {dst}.mlp.down_proj.weight",
                # ===== Norm =====
                f"{src}.input_layernorm.weight          -> {dst}.input_layernorm.weight",
                f"{src}.post_attention_layernorm.weight -> {dst}.post_attention_layernorm.weight",
                f"{src}.post_self_attn_layernorm.weight -> {dst}.post_self_attn_layernorm.weight",
                f"{src}.post_mlp_layernorm.weight       -> {dst}.post_mlp_layernorm.weight",
            ]

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: GlmOcrConfig):
        """
        Inverse AOA config: convert current Paddle model state_dict keys/layout
        back to the original checkpoint keys/layout.

        Inversion rule:
        - "A -> B"   becomes "B -> A"
        - "A^T -> B" becomes "B^T -> A"
        """
        model_prefix = "model"

        aoa_statements = []

        # ========== lm_head ==========
        # forward:  lm_head.weight^T -> lm_head.weight
        # inverse:  lm_head.weight^T -> lm_head.weight  (same key, but transpose direction is reversed)
        aoa_statements += ["lm_head.weight^T -> lm_head.weight"]

        # ========== Text Embedding / Final Norm ==========
        # forward: model.language_model.xxx -> model.language_model.xxx (with prefix)
        # inverse: right -> left
        aoa_statements += [
            f"{model_prefix}.language_model.embed_tokens.weight -> model.language_model.embed_tokens.weight",
            f"{model_prefix}.language_model.norm.weight        -> model.language_model.norm.weight",
        ]

        # ========== Vision patch embed ==========
        aoa_statements += [
            f"{model_prefix}.visual.patch_embed.proj.weight -> model.visual.patch_embed.proj.weight",
            f"{model_prefix}.visual.patch_embed.proj.bias   -> model.visual.patch_embed.proj.bias",
        ]

        # ========== Vision blocks ==========
        for i in range(config.vision_config.depth):
            # forward:
            #   src = model.visual.blocks.i
            #   dst = model.visual.blocks.i (with prefix)
            # inverse:
            #   dst[...] -> src[...]
            src = f"model.visual.blocks.{i}"
            dst = f"{model_prefix}.visual.blocks.{i}"

            aoa_statements += [
                # Norm
                f"{dst}.norm1.weight -> {src}.norm1.weight",
                f"{dst}.norm2.weight -> {src}.norm2.weight",
                # ===== Attention =====
                f"{dst}.attn.qkv.weight^T  -> {src}.attn.qkv.weight",
                f"{dst}.attn.qkv.bias      -> {src}.attn.qkv.bias",
                f"{dst}.attn.proj.weight^T -> {src}.attn.proj.weight",
                f"{dst}.attn.proj.bias     -> {src}.attn.proj.bias",
                # Q/K norm
                f"{dst}.attn.q_norm.weight -> {src}.attn.q_norm.weight",
                f"{dst}.attn.k_norm.weight -> {src}.attn.k_norm.weight",
                # ===== MLP (SwiGLU) =====
                f"{dst}.mlp.gate_proj.weight^T -> {src}.mlp.gate_proj.weight",
                f"{dst}.mlp.gate_proj.bias     -> {src}.mlp.gate_proj.bias",
                f"{dst}.mlp.up_proj.weight^T   -> {src}.mlp.up_proj.weight",
                f"{dst}.mlp.up_proj.bias       -> {src}.mlp.up_proj.bias",
                f"{dst}.mlp.down_proj.weight^T -> {src}.mlp.down_proj.weight",
                f"{dst}.mlp.down_proj.bias     -> {src}.mlp.down_proj.bias",
            ]

        # ========== Vision merger ==========
        aoa_statements += [
            f"{model_prefix}.visual.merger.proj.weight^T                -> model.visual.merger.proj.weight",
            f"{model_prefix}.visual.merger.post_projection_norm.weight -> model.visual.merger.post_projection_norm.weight",
            f"{model_prefix}.visual.merger.post_projection_norm.bias   -> model.visual.merger.post_projection_norm.bias",
            f"{model_prefix}.visual.merger.gate_proj.weight^T          -> model.visual.merger.gate_proj.weight",
            f"{model_prefix}.visual.merger.up_proj.weight^T            -> model.visual.merger.up_proj.weight",
            f"{model_prefix}.visual.merger.down_proj.weight^T          -> model.visual.merger.down_proj.weight",
        ]

        # ========== Vision downsample ==========
        aoa_statements += [
            f"{model_prefix}.visual.downsample.weight -> model.visual.downsample.weight",
            f"{model_prefix}.visual.downsample.bias   -> model.visual.downsample.bias",
            f"{model_prefix}.visual.post_layernorm.weight -> model.visual.post_layernorm.weight",
        ]

        # ========== Text blocks ==========
        for i in range(config.text_config.num_hidden_layers):
            src = f"model.language_model.layers.{i}"
            dst = f"{model_prefix}.language_model.layers.{i}"

            aoa_statements += [
                # ===== Attention =====
                f"{dst}.self_attn.q_proj.weight^T -> {src}.self_attn.q_proj.weight",
                f"{dst}.self_attn.k_proj.weight^T -> {src}.self_attn.k_proj.weight",
                f"{dst}.self_attn.v_proj.weight^T -> {src}.self_attn.v_proj.weight",
                f"{dst}.self_attn.o_proj.weight^T -> {src}.self_attn.o_proj.weight",
                # ===== MLP =====
                f"{dst}.mlp.gate_up_proj.weight^T -> {src}.mlp.gate_up_proj.weight",
                f"{dst}.mlp.down_proj.weight^T    -> {src}.mlp.down_proj.weight",
                # ===== Norm =====
                f"{dst}.input_layernorm.weight          -> {src}.input_layernorm.weight",
                f"{dst}.post_attention_layernorm.weight -> {src}.post_attention_layernorm.weight",
                f"{dst}.post_self_attn_layernorm.weight -> {src}.post_self_attn_layernorm.weight",
                f"{dst}.post_mlp_layernorm.weight       -> {src}.post_mlp_layernorm.weight",
            ]

        return {"aoa_statements": aoa_statements}


@dataclass
class GlmOcrModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[paddle.Tensor] = None
    past_key_values: Cache = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype

    q = q.astype("float32")
    k = k.astype("float32")
    cos = cos.unsqueeze(-2).astype("float32")
    sin = sin.unsqueeze(-2).astype("float32")

    q_embed = ((q * cos) + (rotate_half(q) * sin)).astype(orig_q_dtype)
    k_embed = ((k * cos) + (rotate_half(k) * sin)).astype(orig_k_dtype)

    return q_embed, k_embed


class GlmOcrVisionAttention(nn.Layer):
    def __init__(self, config: GlmOcrVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias_attr=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=config.attention_bias)

        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_norm = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=self.head_dim, norm_eps=config.rms_norm_eps
        )
        self.k_norm = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=self.head_dim, norm_eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        **kwargs,
    ) -> paddle.Tensor:

        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape([seq_length, 3, self.num_heads, -1])
        qkv = qkv.transpose([1, 0, 2, 3])
        query_states, key_states, value_states = qkv[0], qkv[1], qkv[2]

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # [seq, heads, dim] -> [1, heads, seq, dim]
        query_states = query_states.transpose([1, 0, 2]).unsqueeze(0)
        key_states = key_states.transpose([1, 0, 2]).unsqueeze(0)
        value_states = value_states.transpose([1, 0, 2]).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "eager":
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            query_splits = paddle.split(query_states, lengths.numpy().tolist(), axis=2)
            key_splits = paddle.split(key_states, lengths.numpy().tolist(), axis=2)
            value_splits = paddle.split(value_states, lengths.numpy().tolist(), axis=2)

            attn_outputs = []
            for q, k, v in zip(query_splits, key_splits, value_splits):
                attn_out, _ = attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                )
                attn_outputs.append(attn_out)

            attn_output = paddle.concat(attn_outputs, axis=1)
            attn_output = attn_output.reshape([seq_length, -1])

        else:
            cu_seqlens_rm_first = cu_seqlens[1:]
            cu_seqlens_rm_last = cu_seqlens[:-1]
            repeats = (cu_seqlens_rm_first - cu_seqlens_rm_last).astype("int32")

            end_indices = paddle.repeat_interleave(cu_seqlens_rm_first, repeats).reshape([1, 1, -1, 1])
            start_indices = paddle.repeat_interleave(cu_seqlens_rm_last, repeats).reshape([1, 1, -1, 1])
            attn_mask_startend_row_indices = paddle.concat([end_indices, start_indices], axis=-1)

            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
            )
            attn_output = attn_output.reshape([seq_length, -1])

        attn_output = self.proj(attn_output)
        return attn_output


class GlmOcrVisionBlock(nn.Layer):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=config.rms_norm_eps
        )
        self.norm2 = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=config.rms_norm_eps
        )
        self.attn = GlmOcrVisionAttention(config)
        self.mlp = GlmOcrVisionMlp(config, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        **kwargs,
    ) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class GlmOcrVisionPatchMerger(nn.Layer):
    def __init__(self, dim: int, context_dim: int, hidden_act: str, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias_attr=bias)
        self.post_projection_norm = nn.LayerNorm(dim, epsilon=1e-5)
        self.gate_proj = nn.Linear(dim, context_dim, bias_attr=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias_attr=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias_attr=bias)
        self.act1 = ACT2FN["gelu"]
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state: paddle.Tensor) -> paddle.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class GlmOcrVisionPatchEmbed(nn.Layer):
    def __init__(self, config: GlmOcrVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        self.proj = nn.Conv3D(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            bias_attr=True,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.reshape(
            [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size]
        ).astype(target_dtype)
        x = self.proj(hidden_states)
        x = x.reshape([-1, self.embed_dim])
        return x


class GlmOcrVisionModel(GlmOcrPreTrainedModel):
    config: GlmOcrVisionConfig
    input_modalities = ("image", "video")
    _no_split_modules = ["GlmOcrVisionBlock"]
    _can_record_outputs = {
        "hidden_states": GlmOcrVisionBlock,
        "attentions": GlmOcrVisionAttention,
    }

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = GlmOcrVisionPatchEmbed(config)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = GlmOcrVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.LayerList([GlmOcrVisionBlock(config) for _ in range(config.depth)])

        self.merger = GlmOcrVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.out_hidden_size * config.in_channels,
            hidden_act=config.hidden_act,
        )

        self.downsample = nn.Conv2D(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=config.rms_norm_eps
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: paddle.Tensor):
        pos_ids = []

        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            hpos_ids = paddle.arange(h, dtype="int64").unsqueeze(1).expand([-1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose([0, 2, 1, 3])
            hpos_ids = hpos_ids.flatten()

            wpos_ids = paddle.arange(w, dtype="int64").unsqueeze(0).expand([h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3])
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(paddle.stack([hpos_ids, wpos_ids], axis=-1).tile([t, 1]))

        pos_ids = paddle.concat(pos_ids, axis=0)

        max_grid_size = int(paddle.max(grid_thw[:, 1:]).item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        h_idx = pos_ids[:, 0]
        w_idx = pos_ids[:, 1]
        freq_h = paddle.gather(rotary_pos_emb_full, h_idx, axis=0)
        freq_w = paddle.gather(rotary_pos_emb_full, w_idx, axis=0)

        rotary_pos_emb = paddle.concat([freq_h, freq_w], axis=-1)

        return rotary_pos_emb, pos_ids

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, **kwargs) -> BaseModelOutputWithPooling:
        hidden_states = self.patch_embed(hidden_states)
        seq_len = int(hidden_states.shape[0])
        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw.astype("int64"))
        emb = paddle.concat([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (paddle.cos(emb), paddle.sin(emb))

        lengths = (grid_thw[:, 1] * grid_thw[:, 2]).astype("int64")  # per-frame patch count = h*w
        reps = grid_thw[:, 0].astype("int64")  # repeat by t
        cu = paddle.repeat_interleave(lengths, reps, axis=0).cumsum(axis=0).astype("int32")
        cu_seqlens = paddle.nn.functional.pad(cu, pad=[1, 0], value=0)

        if int(cu_seqlens[-1].item()) != seq_len:
            raise ValueError(
                f"[VisionModel] seq mismatch: seq_len={seq_len} but cu_seqlens[-1]={int(cu_seqlens[-1].item())}, "
                f"grid_thw={grid_thw.numpy().tolist()} (need N == sum(t*h*w))"
            )

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
            )
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.reshape(
            [-1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]]
        )
        hidden_states = hidden_states.transpose([0, 3, 1, 2])
        hidden_states = self.downsample(hidden_states)
        hidden_states = hidden_states.reshape([-1, self.config.out_hidden_size])

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class GlmOcrTextRotaryEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.rope_type = self.config.rope_parameters["rope_type"]
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config)
        self.inv_freq = paddle.to_tensor(inv_freq, stop_gradient=True)
        self.original_inv_freq = self.inv_freq.clone()
        self.mrope_section = self.config.rope_parameters.get("mrope_section", [8, 12, 12])

    @staticmethod
    def compute_default_rope_parameters(config):
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        idx = paddle.arange(0, dim, 2, dtype="float32")
        dim_f = paddle.to_tensor(float(dim), dtype="float32")

        base_f = paddle.to_tensor(float(base), dtype="float32")
        inv_freq = 1.0 / (base_f ** (idx / dim_f))

        attention_factor = 1.0
        return inv_freq, attention_factor

    def apply_mrope(self, freqs, mrope_section):
        dim = freqs.shape[-1]
        total = sum(mrope_section)
        scaled = [s * dim // total for s in mrope_section]
        scaled[-1] = dim - sum(scaled[:-1])
        chunks = paddle.split(freqs, num_or_sections=scaled, axis=-1)
        picked = []
        for i, chunk in enumerate(chunks):
            picked.append(chunk[i % 3])
        return paddle.concat(picked, axis=-1)

    @paddle.no_grad()
    def forward(self, hidden_states: paddle.Tensor, position_ids: paddle.Tensor):
        inv_freq = self.inv_freq.astype("float32")
        pos = position_ids.astype("float32")

        bs = position_ids.shape[1]
        inv_freq_expanded = inv_freq.reshape([1, 1, -1, 1]).expand([3, bs, -1, 1])
        position_ids_expanded = pos.unsqueeze(2)

        freqs = paddle.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = freqs.transpose([0, 1, 3, 2])

        freqs = self.apply_mrope(freqs, self.mrope_section)
        emb = paddle.concat([freqs, freqs], axis=-1)

        cos = paddle.cos(emb) * float(self.attention_scaling)
        sin = paddle.sin(emb) * float(self.attention_scaling)

        cos = cos.astype(hidden_states.dtype)
        sin = sin.astype(hidden_states.dtype)
        return cos, sin


class GlmOcrTextModel(GlmOcrPreTrainedModel):
    config: GlmOcrTextConfig
    input_modalities = ("text",)

    def __init__(self, config: GlmOcrTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.LayerList(
            [GlmOcrTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = GeneralNorm.create(
            config, norm_type="rms_norm", hidden_size=config.hidden_size, norm_eps=config.rms_norm_eps
        )
        self.rotary_emb = GlmOcrTextRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # self._post_init()

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Any:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else True
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            if input_ids.dtype != paddle.int64:
                input_ids = input_ids.astype("int64")
            inputs_embeds = self.embed_tokens(input_ids)
        seq_len = inputs_embeds.shape[1]
        bsz = inputs_embeds.shape[0]
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if cache_position is None:
            cache_position = paddle.arange(
                start=past_seen_tokens,
                end=past_seen_tokens + seq_len,
                dtype="int64",
            )

        if position_ids is None:
            position_ids = cache_position.reshape([1, 1, -1]).expand([3, inputs_embeds.shape[0], -1])
        elif len(position_ids.shape) == 2:
            position_ids = position_ids.unsqueeze(0).expand([3, position_ids.shape[0], -1])

        if len(position_ids.shape) == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": bsz,
            "seq_length": seq_len,
            "cache_length": past_seen_tokens,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for key in [
            "attn_mask_startend_row_indices",
            "attention_mask",
            "cache_position",
            "past_key_values",
            "position_embeddings",
        ]:
            kwargs.pop(key, None)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def masked_scatter(inputs: paddle.Tensor, mask: paddle.Tensor, updates: paddle.Tensor) -> paddle.Tensor:
    x = inputs.reshape([-1])
    m = mask.astype("bool").reshape([-1])

    idx_np = np.where(m.cpu().numpy())[0]

    upd = updates.reshape([-1]).astype(inputs.dtype)

    if len(idx_np) != len(upd):
        raise ValueError(f"[masked_scatter] size mismatch: idx={len(idx_np)} vs updates={len(upd)}")

    idx = paddle.to_tensor(idx_np, dtype="int64", place=inputs.place)

    x_new = paddle.scatter(x, idx, upd, overwrite=True)
    return x_new.reshape(inputs.shape)


class GlmOcrModel(GlmOcrPreTrainedModel):
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: "GlmOcrConfig"
    _no_split_modules = ["GlmOcrTextDecoderLayer", "GlmOcrVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = GlmOcrVisionModel._from_config(config.vision_config)
        self.language_model = GlmOcrTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        spatial_merge_size = int(self.config.vision_config.spatial_merge_size)
        image_token_id = int(self.config.image_token_id)
        video_start_token_id = int(self.config.video_start_token_id)
        video_end_token_id = int(self.config.video_end_token_id)

        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            if attention_mask is None:
                attention_mask = paddle.ones_like(input_ids)

            position_ids = paddle.ones([3, input_ids.shape[0], input_ids.shape[1]], dtype="int64")

            image_index, video_index, video_group_index = 0, 0, 0

            for i in range(input_ids.shape[0]):
                mask_i = attention_mask[i] == 1
                ids_valid = paddle.masked_select(input_ids[i], mask_i)
                input_tokens = ids_valid.tolist()

                input_token_type = []
                video_check_flg = False
                for token in input_tokens:
                    if token == video_start_token_id:
                        video_check_flg = True
                    elif token == video_end_token_id:
                        video_check_flg = False

                    if token == image_token_id and not video_check_flg:
                        input_token_type.append("image")
                    elif token == image_token_id and video_check_flg:
                        input_token_type.append("video")
                    else:
                        input_token_type.append("text")

                input_type_group = []
                for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
                    group = list(group)
                    input_type_group.append((key, group[0][0], group[-1][0] + 1))

                llm_pos_ids_list = []
                video_frame_num = 1

                for modality_type, start_idx, end_idx in input_type_group:
                    st_idx = int(paddle.max(llm_pos_ids_list[-1]).item()) + 1 if llm_pos_ids_list else 0

                    if modality_type == "image":
                        t = int(image_grid_thw[image_index, 0].item())
                        h = int(image_grid_thw[image_index, 1].item())
                        w = int(image_grid_thw[image_index, 2].item())
                        llm_grid_t = t
                        llm_grid_h = h // spatial_merge_size
                        llm_grid_w = w // spatial_merge_size

                        t_index = (
                            paddle.arange(llm_grid_t).reshape([-1, 1]).expand([-1, llm_grid_h * llm_grid_w]).flatten()
                        )
                        h_index = (
                            paddle.arange(llm_grid_h)
                            .reshape([1, -1, 1])
                            .expand([llm_grid_t, -1, llm_grid_w])
                            .flatten()
                        )
                        w_index = (
                            paddle.arange(llm_grid_w)
                            .reshape([1, 1, -1])
                            .expand([llm_grid_t, llm_grid_h, -1])
                            .flatten()
                        )
                        llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + st_idx)

                        image_index += 1
                        video_frame_num = 1

                    elif modality_type == "video":
                        t = video_frame_num
                        h = int(video_grid_thw[video_index, 1].item())
                        w = int(video_grid_thw[video_index, 2].item())
                        llm_grid_h = h // spatial_merge_size
                        llm_grid_w = w // spatial_merge_size

                        for t_idx in range(t):
                            t_index = paddle.full([llm_grid_h * llm_grid_w], t_idx, dtype="int64")
                            h_index = (
                                paddle.arange(llm_grid_h).reshape([1, -1, 1]).expand([1, -1, llm_grid_w]).flatten()
                            )
                            w_index = (
                                paddle.arange(llm_grid_w).reshape([1, 1, -1]).expand([1, llm_grid_h, -1]).flatten()
                            )
                            llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + st_idx)

                        video_group_index += 1
                        if video_group_index >= int(video_grid_thw[video_index, 0].item()):
                            video_index += 1
                            video_group_index = 0
                        video_frame_num += 1

                    else:  # text
                        text_len = end_idx - start_idx
                        text_pos = paddle.arange(text_len).reshape([1, -1]).expand([3, -1]) + st_idx
                        llm_pos_ids_list.append(text_pos)
                        video_frame_num = 1

                llm_positions = paddle.concat(llm_pos_ids_list, axis=1).reshape([3, -1])

                mask_np = mask_i.numpy()
                idx_np = np.where(mask_np)[0]
                idx_tensor = paddle.to_tensor(idx_np, dtype="int64")
                for r in range(3):
                    position_ids[r, i] = paddle.scatter(position_ids[r, i], idx_tensor, llm_positions[r])

                mrope_position_deltas.append(int(paddle.max(llm_positions).item()) + 1 - input_ids.shape[1])

            mrope_position_deltas = paddle.to_tensor(mrope_position_deltas, dtype="int64").unsqueeze(1)
            return position_ids, mrope_position_deltas

        else:
            if attention_mask is not None:
                position_ids = paddle.cumsum(attention_mask.astype("int64"), axis=-1) - 1
                position_ids = paddle.where(attention_mask == 1, position_ids, paddle.ones_like(position_ids))
                position_ids = position_ids.unsqueeze(0).expand([3, -1, -1])
                max_position_ids = position_ids.max(axis=0).max(axis=-1, keepdim=True)
                mrope_position_deltas = (max_position_ids + 1 - input_ids.shape[1]).astype("int64")
            else:
                position_ids = (
                    paddle.arange(input_ids.shape[1], dtype="int64")
                    .reshape([1, 1, -1])
                    .expand([3, input_ids.shape[0], -1])
                )
                mrope_position_deltas = paddle.zeros([input_ids.shape[0], 1], dtype="int64")
            return position_ids, mrope_position_deltas

    def get_video_features(
        self,
        pixel_values_videos: paddle.Tensor,
        video_grid_thw: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        pixel_values_videos = pixel_values_videos.astype(self.visual.dtype)

        temp_frames_hw = []
        for t, h, w in video_grid_thw.numpy().tolist():
            repeated_row = paddle.to_tensor([[1, int(h), int(w)]], dtype="int64").tile([int(t), 1])
            temp_frames_hw.append(repeated_row)
        flattened_video_grid_thw = paddle.concat(temp_frames_hw, axis=0)

        vision_outputs = self.visual(
            pixel_values_videos, grid_thw=flattened_video_grid_thw, return_dict=True, **kwargs
        )

        patch_size = self.config.vision_config.patch_size
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        split_sizes = []
        for t, h, w in video_grid_thw.numpy().tolist():
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            num_tokens = int(t * (num_patches_h // spatial_merge_size) * (num_patches_w // spatial_merge_size))
            split_sizes.append(num_tokens)

        video_embeds = paddle.split(vision_outputs.pooler_output, num_or_sections=split_sizes, axis=0)
        vision_outputs.pooler_output = video_embeds
        return vision_outputs

    def get_image_features(
        self,
        pixel_values: paddle.Tensor,
        image_grid_thw: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        visual_dtype = self.visual.parameters()[0].dtype
        pixel_values = pixel_values.astype(visual_dtype)
        kwargs.pop("return_dict", None)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        split_sizes = (
            (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2] // (spatial_merge_size**2))
            .numpy()
            .astype("int64")
            .tolist()
        )

        image_embeds = paddle.split(vision_outputs.pooler_output, num_or_sections=split_sizes, axis=0)
        vision_outputs.pooler_output = image_embeds

        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: Optional[paddle.Tensor],
        inputs_embeds: paddle.Tensor,
        image_features: Optional[paddle.Tensor] = None,
        video_features: Optional[paddle.Tensor] = None,
    ):

        if input_ids is None:
            img_tok = paddle.to_tensor([self.config.image_token_id], dtype="int64")
            vid_tok = paddle.to_tensor([self.config.video_token_id], dtype="int64")

            img_emb = self.get_input_embeddings()(img_tok)  # [1, D]
            vid_emb = self.get_input_embeddings()(vid_tok)  # [1, D]

            special_image_mask_2d = (inputs_embeds == img_emb).all(axis=-1)  # [B,S]
            special_video_mask_2d = (inputs_embeds == vid_emb).all(axis=-1)  # [B,S]
        else:
            special_image_mask_2d = input_ids == self.config.image_token_id
            special_video_mask_2d = input_ids == self.config.image_token_id

        n_image_tokens = int(special_image_mask_2d.astype("int64").sum().item())

        # [B,S] -> [B,S,D]
        special_image_mask = special_image_mask_2d.unsqueeze(-1).expand_as(inputs_embeds).astype("bool")

        if image_features is not None:
            sel = paddle.masked_select(inputs_embeds, special_image_mask)
            expected_numel = image_features.numel().item()
            actual_numel = sel.numel().item()

            if actual_numel != expected_numel:
                raise ValueError(
                    f"Image features and image tokens do not match, tokens: {n_image_tokens}, "
                    f"features shape: {image_features.shape}, features numel: {expected_numel}, "
                    f"masked tokens numel: {actual_numel}"
                )

        n_video_tokens = int(special_video_mask_2d.astype("int64").sum().item())
        special_video_mask = special_video_mask_2d.unsqueeze(-1).expand_as(inputs_embeds).astype("bool")
        if video_features is not None:
            sel = paddle.masked_select(inputs_embeds, special_video_mask)
            if sel.numel().item() != video_features.numel().item():
                raise ValueError(
                    f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}"
                )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[Any] = None,
        attn_mask_startend_row_indices: Optional[Any] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids.astype("int64"))

        if pixel_values is not None:
            image_embeds_list = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = paddle.concat(image_embeds_list, axis=0).astype(inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = masked_scatter(inputs_embeds, image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds_list = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            ).pooler_output
            video_embeds = paddle.concat(video_embeds_list, axis=0).astype(inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
            inputs_embeds = masked_scatter(inputs_embeds, video_mask, video_embeds)
        if position_ids is None:
            prefill_stage = (cache_position is not None and int(cache_position[0].item()) == 0) or (
                past_key_values is None or past_key_values.get_seq_length() == 0
            )

            if prefill_stage or self.rope_deltas is None:
                position_ids, rope_deltas_calc = self.get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas_calc
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0].astype("int64") + self.rope_deltas.astype("int64")
                delta = delta.tile([batch_size // delta.shape[0], 1])
                position_ids = paddle.arange(seq_length, dtype="int64")
                position_ids = position_ids.reshape([1, -1]).expand([batch_size, -1])
                position_ids = position_ids + delta
                position_ids = position_ids.unsqueeze(0).expand([3, -1, -1])

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        return GlmOcrModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=self.rope_deltas,
        )


@dataclass
class GlmOcrCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Cache object holding past KV for fast decoding.
    rope_deltas (`paddle.Tensor` of shape `(batch_size, )` or `(batch_size, 1)`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[paddle.Tensor]] = None
    attentions: Optional[tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


class GlmOcrForConditionalGeneration(GlmOcrPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = GlmOcrModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias_attr=False)
        self.criterion = CriterionLayer(config)
        attn_impl = getattr(config, "_attn_implementation", "eager")
        config.vision_config._attn_implementation = attn_impl
        config.text_config._attn_implementation = attn_impl

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        return self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep=0,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple, GlmOcrCausalLMOutputWithPast]:
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
            cache_position=cache_position,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None, None)
        else:
            slice_indices = logits_to_keep

        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)
        if not return_dict:
            if loss is None:
                return (logits, outputs.past_key_values)
            else:
                return (loss, logits, outputs.past_key_values)
        return GlmOcrCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

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

        kwargs.pop("position_ids", None)
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=None,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["position_ids"] = None
        model_inputs["cache_position"] = cache_position

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_grid_thw"] = None
            model_inputs["video_grid_thw"] = None

        return model_inputs

    def update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
        # ---- 1) update past_key_values ----
        if isinstance(outputs, tuple):
            if len(outputs) >= 2 and not isinstance(outputs[1], paddle.Tensor):
                model_kwargs["past_key_values"] = outputs[1]
            elif len(outputs) >= 3 and not isinstance(outputs[2], paddle.Tensor):
                model_kwargs["past_key_values"] = outputs[2]
        elif hasattr(outputs, "past_key_values"):
            model_kwargs["past_key_values"] = outputs.past_key_values

        # ---- 2) update attention_mask (关键) ----
        attn = model_kwargs.get("attention_mask", None)
        if (not is_encoder_decoder) and (attn is not None):
            if len(attn.shape) == 2:
                # [B, S] -> [B, S+1]
                model_kwargs["attention_mask"] = paddle.concat(
                    [attn, paddle.ones([attn.shape[0], 1], dtype=attn.dtype)],
                    axis=-1,
                )

        return model_kwargs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[paddle.Tensor],
        inputs_embeds: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:

        if inputs_embeds is not None:
            image_id = paddle.to_tensor([self.config.image_start_token_id], dtype="int64")
            vstart_id = paddle.to_tensor([self.config.video_start_token_id], dtype="int64")
            vend_id = paddle.to_tensor([self.config.video_end_token_id], dtype="int64")

            image_emb = self.get_input_embeddings()(image_id)
            vstart_emb = self.get_input_embeddings()(vstart_id)
            vend_emb = self.get_input_embeddings()(vend_id)

            is_image = (inputs_embeds == image_emb).all(axis=-1)
            is_video_start = (inputs_embeds == vstart_emb).all(axis=-1)
            is_video_end = (inputs_embeds == vend_emb).all(axis=-1)
        else:
            is_image = input_ids == self.config.image_start_token_id
            is_video_start = input_ids == self.config.video_start_token_id
            is_video_end = input_ids == self.config.video_end_token_id

        video_level = paddle.cumsum(is_video_start.astype("int32") - is_video_end.astype("int32"), axis=1)
        inside_video = video_level > 0
        standalone_images = is_image & (~inside_video)

        image_counts = standalone_images.astype("int64").sum(axis=1)
        video_counts = is_video_start.astype("int64").sum(axis=1)

        return image_counts, video_counts

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[paddle.Tensor] = None,
        **model_kwargs,
    ) -> Tuple[Optional[paddle.Tensor], Dict[str, Any]]:

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _repeat_interleave_samples(x: paddle.Tensor, lengths, repeat_times: int):
            samples = paddle.split(x, lengths, axis=0)
            out = []
            for s in samples:
                reps = [repeat_times] + [1] * (len(s.shape) - 1)
                out.append(paddle.tile(s, reps))
            return paddle.concat(out, axis=0)

        def _expand_dict_for_generation_visual(d: Dict[str, Any]):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)

            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            image_nums_list = [int(x) for x in image_nums.tolist()]
            video_nums_list = [int(x) for x in video_nums.tolist()]

            for key in list(d.keys()):
                if key == "pixel_values":
                    samples = paddle.split(image_grid_thw, image_nums_list, axis=0)
                    lengths = [int(paddle.prod(s, axis=1).sum().item()) for s in samples]
                    d[key] = _repeat_interleave_samples(d[key], lengths=lengths, repeat_times=expand_size)
                elif key == "image_grid_thw":
                    d[key] = _repeat_interleave_samples(d[key], lengths=image_nums_list, repeat_times=expand_size)
                elif key == "pixel_values_videos":
                    samples = paddle.split(video_grid_thw, video_nums_list, axis=0)
                    lengths = [int(paddle.prod(s, axis=1).sum().item()) for s in samples]
                    d[key] = _repeat_interleave_samples(d[key], lengths=lengths, repeat_times=expand_size)
                elif key == "video_grid_thw":
                    d[key] = _repeat_interleave_samples(d[key], lengths=video_nums_list, repeat_times=expand_size)
                elif key == "second_per_grid_ts":
                    d[key] = _repeat_interleave_samples(d[key], lengths=video_nums_list, repeat_times=expand_size)

            return d

        def _expand_dict_for_generation(d: Dict[str, Any]):
            for key, val in list(d.items()):
                if (
                    key != "cache_position"
                    and val is not None
                    and isinstance(val, paddle.Tensor)
                    and key not in visual_keys
                ):
                    d[key] = paddle.repeat_interleave(val, repeats=expand_size, axis=0)
            return d

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = paddle.repeat_interleave(input_ids, repeats=expand_size, axis=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "GlmOcrTextModel",
    "GlmOcrVisionModel",
    "GlmOcrModel",
    "GlmOcrPreTrainedModel",
    "GlmOcrForConditionalGeneration",
]
