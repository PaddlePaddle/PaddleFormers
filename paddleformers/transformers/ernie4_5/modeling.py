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

"""Paddle Ernie model."""

import math
from typing import Optional, Tuple

import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)
from paddle.incubate.nn.functional import fused_rotary_position_embedding as fused_rope

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Ernie4_5MLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ..tensor_parallel_utils import model_parallel_dropout
from .configuration import Ernie4_5Config


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return paddle.stack((-x2, x1), axis=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        position_ids (`paddle.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(paddle.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # shape of q: batch_size, num_heads, seq_len, head_dim
    # glm rope style (with full dim) and full precision
    original_dtype = q.dtype

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Interleave them instead of usual shape
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, axis=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, axis=-1)

    q_embed = (q.astype("float32") * cos) + (rotate_half(q).astype("float32") * sin)
    k_embed = (k.astype("float32") * cos) + (rotate_half(k).astype("float32") * sin)

    return q_embed.astype(original_dtype), k_embed.astype(original_dtype)


def apply_fused_rope(query_states, key_states, rope_theta):
    # b h l d -> b l h d
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    _, _, num_heads, _ = query_states.shape
    _, kv_seq_len, num_key_value_heads, _ = key_states.shape
    if num_heads != num_key_value_heads:
        query_states, _, _ = fused_rope(query_states, None, None, rotary_emb_base=rope_theta)
        key_states, _, _ = fused_rope(key_states, None, None, rotary_emb_base=rope_theta)
    else:
        query_states, key_states, _ = fused_rope(
            query_states,
            key_states,
            None,
            rotary_emb_base=rope_theta,
        )
    return query_states.transpose(1, 2), key_states.transpose(1, 2)


class Ernie4_5RotaryEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.head_dim = config.head_dim
        self.base = config.rope_theta
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
        config: Optional[Ernie4_5Config] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["paddle.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`PreTrainedConfig`]):
                The model configuration.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`paddle.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x, position_ids):
        """
        Compute rotary position embeddings for given sequence length.

        Args:
            seq_length (int): Maximum sequence length
            position_ids (Tensor): Position ids of shape [batch_size, seq_length]

        Returns:
            Tensor: Rotary position embeddings of shape [1, 1, seq_length, head_dim]
        """
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])

            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Ernie4_5Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=0):
        """Initialize the attention layer.

        Args:
            config (Ernie4_5Config): Model configuration.
            layer_idx (int, optional): Index in transformer stack. Defaults to 0.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.fuse_attention_qkv = config.fuse_attention_qkv
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads

        if config.tensor_model_parallel_size > 1:
            assert (
                self.num_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_heads = self.num_heads // config.tensor_model_parallel_size

            assert (
                self.num_key_value_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_key_value_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        logger.warning_once(f"use GQA - num_heads: {self.num_heads}- num_key_value_heads: {self.num_key_value_heads}")
        assert (
            self.num_heads % self.num_key_value_heads == 0
        ), f"num_heads: {self.num_heads}, num_key_value_heads: {self.num_key_value_heads}"
        kv_hidden_size = self.head_dim * config.num_key_value_heads
        q_hidden_size = self.head_dim * config.num_attention_heads

        if not self.fuse_attention_qkv:
            self.q_proj = GeneralLinear.create(
                self.hidden_size,
                q_hidden_size,
                has_bias=config.use_bias,
                config=config,
                tp_plan="colwise",
            )
            self.k_proj = GeneralLinear.create(
                self.hidden_size,
                kv_hidden_size,
                has_bias=config.use_bias,
                config=config,
                tp_plan="colwise",
            )
            self.v_proj = GeneralLinear.create(
                self.hidden_size,
                kv_hidden_size,
                has_bias=config.use_bias,
                config=config,
                tp_plan="colwise",
            )
        else:
            self.qkv_proj = GeneralLinear.create(
                self.hidden_size,
                q_hidden_size + 2 * kv_hidden_size,
                has_bias=config.use_bias,
                config=config,
                tp_plan="colwise",
            )

        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            self.hidden_size,
            has_bias=config.use_bias,
            config=config,
            tp_plan="rowwise",
        )

        self.config = config
        self.scaling = self.head_dim**-0.5
        self.attn_implementation = config._attn_implementation

    def forward(
        self,
        hidden_states,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Compute attention outputs.

        Args:
            hidden_states (paddle.Tensor): Input tensor [bsz, seq_len, hidden_size]
            past_key_values (Optional[Cache]): Cached key/value states
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_startend_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            position_ids (Optional[paddle.Tensor]): Position indices for RoPE
            output_attentions (bool): Return attention weights if True
            use_cache (bool): Cache key/value states if True

        Returns:
            Tuple containing:
                - attention_output: [bsz, seq_len, hidden_size]
                - attention_weights: Optional attention probabilities
                - updated_key_value_cache: Optional updated cache
        """
        if not self.fuse_attention_qkv:
            if self.config.sequence_parallel:
                max_sequence_length = self.config.max_sequence_length
                bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
                q_len = max_sequence_length
            else:
                bsz, q_len, _ = hidden_states.shape

            query_states = self.q_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])
            key_states = self.k_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])
            value_states = self.v_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])
        else:
            mix_layer = self.qkv_proj(hidden_states)
            if self.config.sequence_parallel:
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

        # b l h d -> b h l d
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]

        if self.config.apply_rope_fusion:
            query_states, key_states = apply_fused_rope(query_states, key_states, self.config.rope_theta)
        else:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.config.get("attention_dropout_prob", 0.0) if self.training else 0.0,
            scaling=self.scaling,
        )

        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class Ernie4_5DecoderLayer(nn.Layer):
    """A single transformer decoder layer in ERNIE model.

    Contains self-attention and feed-forward components,
    support, residual connections, and layer normalization.
    """

    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config (Ernie4_5Config): Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        self.self_attn = Ernie4_5Attention(config, layer_idx)
        self.mlp = Ernie4_5MLP(config, fuse_up_gate=config.fuse_attention_ffn)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        self.hidden_dropout = nn.Dropout(p=config.hidden_dropout_prob, mode="upscale_in_train")

        if config.sequence_parallel:
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()
                if config.use_bias:
                    mark_as_sequence_parallel_parameter(self.self_attn.o_proj.bias)
                    mark_as_sequence_parallel_parameter(self.mlp.down_proj.bias)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (paddle.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_startend_row_indices (Optional[paddle.Tensor]): Indices for variable length attention
            position_ids (Optional[paddle.Tensor]): Position indices for rotary embeddings
            position_embeddings (Optional[paddle.Tensor]): Position embeddings tensor
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_values (Optional[Cache]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


class Ernie4_5PretrainedModel(PretrainedModel):
    """Base class for ERNIE pretrained models."""

    config_class = Ernie4_5Config
    base_model_prefix = "model"
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _gen_aoa_config(cls, config: Ernie4_5Config):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_config = {
            "aoa_statements": [
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }

        # attention qkv
        if not config.fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
            ]
            if config.use_bias:
                aoa_config["aoa_statements"] += [
                    f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                ]

        # FFN
        if not config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                for p in ("gate", "up")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
            ]

        # lm_head
        if config.tie_word_embeddings:
            aoa_config["aoa_statements"] += ["model.embed_tokens.weight -> lm_head.weight"]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Ernie4_5Config):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_statements = [
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        if not config.fuse_attention_qkv:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
            ]
            for layer_id in range(config.num_hidden_layers):
                for x in ("q", "k", "v"):
                    aoa_statements += [
                        f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                    ]
            if config.use_bias:
                aoa_statements += [
                    f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
                ]

        if not config.fuse_attention_ffn:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{y}_proj.weight"
                for y in ("gate", "up")
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.gate_proj.weight, model.layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
            ]
            for layer_id in range(config.num_hidden_layers):
                aoa_statements += [
                    f"model.layers.{layer_id}.mlp.gate_proj.weight^T -> model.layers.{layer_id}.mlp.gate_proj.weight",
                    f"model.layers.{layer_id}.mlp.up_proj.weight^T -> model.layers.{layer_id}.mlp.up_proj.weight",
                ]

        if config.tie_word_embeddings:
            aoa_statements += ["lm_head.weight -> _"]

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


@register_base_model
class Ernie4_5Model(Ernie4_5PretrainedModel):
    """The core ERNIE transformer model"""

    def __init__(self, config: Ernie4_5Config):
        """Initialize the ERNIE model architecture.

        Args:
            config (Ernie4_5Config): Model configuration.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config
        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList([Ernie4_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        self.rotary_emb = Ernie4_5RotaryEmbedding(config)

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices,
        position_ids,
        position_embeddings,
        output_attentions,
        past_key_values,
        use_cache,
    ):
        """Perform gradient checkpointing for memory-efficient training.

        Args:
            layer_module (nn.Layer): Transformer layer to recompute
            hidden_states (paddle.Tensor): Input hidden states
            attention_mask (paddle.Tensor): Attention mask
            attn_mask_startend_row_indices (paddle.Tensor): Variable length indices
            position_ids (paddle.Tensor): Position indices
            position_embeddings (paddle.Tensor): Position embeddings
            output_attentions (bool): Whether to output attention weights
            past_key_values (Optional[Cache]): Cached key/value states
            use_cache (bool): Whether to cache key/value states

        Returns:
            paddle.Tensor: Output hidden states after recomputation
        """

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            attn_mask_startend_row_indices,
            position_ids,
            position_embeddings,
            output_attentions,
            past_key_values,
            use_cache,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
    ):
        """Forward pass through the ERNIE model.

        Args:
            input_ids (Optional[paddle.Tensor]): Input token IDs
            position_ids (Optional[paddle.Tensor]): Position indices
            attention_mask (Optional[paddle.Tensor]): Attention mask
            attn_mask_startend_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            inputs_embeds (Optional[paddle.Tensor]): Precomputed embeddings
            use_cache (Optional[bool]): Whether to cache key/value states
            past_key_values (Optional[Cache]): Cached key/value states
            output_attentions (Optional[bool]): Whether to output attention weights
            output_hidden_states (Optional[bool]): Whether to output all hidden states
            return_dict (Optional[bool]): Whether to return dict or tuple

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
                Various outputs depending on configuration, including:
                - last_hidden_state: Final layer hidden states
                - past_key_values: Cached key/value states if use_cache=True
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            bsz, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            bsz, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        kv_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        hidden_states = inputs_embeds

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": bsz,
            "seq_length": seq_length,
            "cache_length": kv_seq_len,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }

        causal_attention_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        if position_ids is None:
            position_ids = paddle.arange(kv_seq_len, seq_length).unsqueeze(0).tile((bsz, 1))

        if not self.config.apply_rope_fusion:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)  # cos and sin
        else:
            position_embeddings = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    causal_attention_mask,
                    attn_mask_startend_row_indices,
                    position_ids,
                    position_embeddings,
                    output_attentions,
                    past_key_values,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    causal_attention_mask,
                    attn_mask_startend_row_indices,
                    position_ids,
                    position_embeddings,
                    output_attentions,
                    past_key_values,
                    use_cache,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )


class Ernie4_5ForCausalLM(Ernie4_5PretrainedModel):
    """ERNIE model for causal language modeling."""

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        Initializes the ERNIE model for causal language modeling.

        Args:
            config (Ernie4_5Config): Model configuration.
        """
        super().__init__(config)

        # initialize-trick for big model,
        # see https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"change initializer-range from {config.initializer_range} to {new_initializer_range}")
        config.initializer_range = new_initializer_range
        self.config = config
        self.model = Ernie4_5Model(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
        """Avoid using attention_mask with flash_attn on generation."""
        if self.config._attn_implementation == "sdpa":
            return None
        return super().prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        inputs_embeds=None,
        labels=None,
        loss_mask=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,  # true when decode, false when pretrain & eval
        **kwargs,
    ):
        """
        Forward pass for causal language modeling.

        Args:
            input_ids (paddle.Tensor): Input token IDs.
            position_ids (paddle.Tensor): Position IDs.
            attention_mask (paddle.Tensor): Attention mask.
            attn_mask_startend_row_indices (paddle.Tensor): Attention mask start indices.
            inputs_embeds (paddle.Tensor): Optional embedded inputs.
            labels (paddle.Tensor): Target labels.
            loss_mask (paddle.Tensor): Loss mask.
            use_cache (bool): Whether to use cached hidden states.
            past_key_values (Cache): Pre-computed hidden states.
            output_attentions (bool): Whether to output attentions.
            output_hidden_states (bool): Whether to output hidden states.
            return_dict (bool): Whether to return a dictionary.

        Returns:
            Union[tuple, CausalLMOutputWithCrossAttentions]: Model outputs.
        """
        if kwargs.get("attn_mask_start_row_indices", None) is not None and attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs.pop("attn_mask_start_row_indices")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        if self.criterion.loss_type == "dpo":
            logits = self.lm_head(hidden_states)
            chosen_labels = kwargs.get("chosen_labels", None)
            rejected_labels = kwargs.get("rejected_labels", None)
            response_indexs = kwargs.get("response_indexs", None)
            score_deltas = kwargs.get("score_deltas", None)
            reference_chosen_logps = kwargs.get("reference_chosen_logps", None)
            reference_rejected_logps = kwargs.get("reference_rejected_logps", None)
            labels = (
                chosen_labels,
                rejected_labels,
                response_indexs,
                score_deltas,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            return self.criterion(
                logits,
                labels,
            )

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        logits = self.lm_head(hidden_states)

        if return_dict:  # aka Generate Decoding
            if labels is not None:
                loss, _ = self.criterion(logits, labels, loss_mask)
            else:
                loss = None
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # Pretrain & Eval must have labels
        assert labels is not None
        return self.criterion(logits, labels, loss_mask)


class Ernie4_5ForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = Ernie4_5Config
    _decoder_layer_cls = Ernie4_5DecoderLayer
    _get_tensor_parallel_mappings = Ernie4_5Model._get_tensor_parallel_mappings
    _init_weights = Ernie4_5Model._init_weights
    _keep_in_fp32_modules = Ernie4_5Model._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Ernie4_5Model.transpose_weight_keys
    _gen_aoa_config = Ernie4_5ForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = Ernie4_5ForCausalLM._gen_inv_aoa_config


__all__ = ["Ernie4_5Model", "Ernie4_5ForCausalLM", "Ernie4_5ForCausalLMPipe"]
