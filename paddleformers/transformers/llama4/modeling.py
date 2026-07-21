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

from typing import Callable, cast

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.norm import Norm as GeneralNorm
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import Llama4TextConfig


class Llama4TextRotaryEmbedding(nn.Layer):
    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.rope_type = "default"
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            self.rope_type = config.rope_parameters.get("rope_type", "default")

        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(config, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype("float32") / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            if self.inv_freq.dtype != paddle.float32:
                rope_init_fn = self.compute_default_rope_parameters
                if self.rope_type != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
                inv_freq, self.attention_scaling = rope_init_fn(self.config)
                inv_freq = inv_freq.to(x.place)
                self.register_buffer("inv_freq", inv_freq, persistable=False)
                self.original_inv_freq = inv_freq

            inv_freq_expanded = self.inv_freq[None, :, None].astype("float32").expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].astype("float32")
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose([0, 2, 1])
        return freqs


def apply_rotary_emb(xq, xk, freqs):
    """Apply rotary embedding using complex multiplication.

    Args:
        xq: query states [batch, seq, heads, head_dim]
        xk: key states [batch, seq, heads, head_dim]
        freqs: frequencies [batch, seq, head_dim//2]
    """
    xq_float = xq.astype("float32")
    xk_float = xk.astype("float32")

    xq_r = xq_float.reshape(list(xq_float.shape[:-1]) + [-1, 2])
    xk_r = xk_float.reshape(list(xk_float.shape[:-1]) + [-1, 2])

    freqs = freqs[:, :, None, :]
    cos_f = paddle.cos(freqs)
    sin_f = paddle.sin(freqs)

    xq_out_r = xq_r[..., 0] * cos_f - xq_r[..., 1] * sin_f
    xq_out_i = xq_r[..., 0] * sin_f + xq_r[..., 1] * cos_f
    xq_out = paddle.stack([xq_out_r, xq_out_i], axis=-1).flatten(-2)

    xk_out_r = xk_r[..., 0] * cos_f - xk_r[..., 1] * sin_f
    xk_out_i = xk_r[..., 0] * sin_f + xk_r[..., 1] * cos_f
    xk_out = paddle.stack([xk_out_r, xk_out_i], axis=-1).flatten(-2)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


class Llama4TextL2Norm(nn.Layer):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.astype("float32")
        return (x_fp32 * paddle.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)).astype(x.dtype)


class Llama4TextExperts(nn.Layer):
    """Batched expert FFN using bmm."""

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.gate_up_proj = paddle.create_parameter(
            shape=[self.num_experts, self.hidden_size, 2 * self.expert_dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range),
        )
        self.down_proj = paddle.create_parameter(
            shape=[self.num_experts, self.expert_dim, self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range),
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [num_experts * num_tokens_per_expert, hidden_size] (pre-sorted)
        """
        hidden_states = hidden_states.reshape([self.num_experts, -1, self.hidden_size])
        gate_up = paddle.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, axis=-1)
        next_states = paddle.bmm(up * F.silu(gate), self.down_proj)
        next_states = next_states.reshape([-1, self.hidden_size])
        return next_states


class Llama4TextMLP(nn.Layer):
    """Dense MLP (shared expert or non-MoE layer)."""

    def __init__(self, config: Llama4TextConfig, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        self.config = config

        self.gate_proj = GeneralLinear.create(
            config.hidden_size,
            intermediate_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.up_proj = GeneralLinear.create(
            config.hidden_size,
            intermediate_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.down_proj = GeneralLinear.create(
            intermediate_size,
            config.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Llama4Router(nn.Layer):
    """Top-k router with sigmoid scoring."""

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.linear = nn.Linear(config.hidden_size, config.num_local_experts, bias_attr=False)

    def forward(self, hidden_states):
        router_logits = self.linear(hidden_states)
        router_top_value, router_indices = paddle.topk(router_logits, self.top_k, axis=1)
        router_scores = paddle.full_like(router_logits, float("-inf"))
        router_scores = paddle.put_along_axis(router_scores, router_indices, router_top_value, axis=1)
        router_scores = F.sigmoid(router_scores.astype("float32")).astype(router_scores.dtype)
        return router_scores, router_logits


class Llama4TextMoe(nn.Layer):
    """MoE block: router + batched experts + shared expert."""

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)
        self.router = Llama4Router(config)
        self.shared_expert = Llama4TextMLP(config)

    def forward(self, hidden_states):
        batch_seq_shape = hidden_states.shape[:-1]
        hidden_states_flat = hidden_states.reshape([-1, self.hidden_dim])

        router_scores, router_logits = self.router(hidden_states_flat)

        routed_in = hidden_states_flat.tile([self.num_experts, 1])
        routed_in = routed_in * router_scores.transpose([1, 0]).reshape([-1, 1])

        routed_out = self.experts(routed_in)

        out = self.shared_expert(hidden_states_flat)
        out = out + routed_out.reshape([self.num_experts, -1, self.hidden_dim]).sum(axis=0)
        out = out.reshape(list(batch_seq_shape) + [self.hidden_dim])
        return out, router_logits


class Llama4TextMoeFused(nn.Layer):
    """Fused MoE implementation for optimized inference."""

    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.moe = Llama4TextMoe(config)

    def forward(self, hidden_states):
        return self.moe(hidden_states)


class Llama4TextAttention(nn.Layer):
    """Multi-headed attention with RoPE/NoPE, QK-norm, and temperature tuning."""

    def __init__(self, config: Llama4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.use_rope = config.no_rope_layers[layer_idx]

        q_hidden_size = self.head_dim * config.num_attention_heads
        kv_hidden_size = self.head_dim * config.num_key_value_heads

        self.q_proj = GeneralLinear.create(
            config.hidden_size,
            q_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
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

        if config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_embeddings=None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        q_shape = [batch_size, seq_len, -1, self.head_dim]
        kv_shape = [batch_size, seq_len, -1, self.head_dim]

        query_states = self.q_proj(hidden_states).reshape(q_shape)
        key_states = self.k_proj(hidden_states).reshape(kv_shape)
        value_states = self.v_proj(hidden_states).reshape(q_shape).transpose([0, 2, 1, 3])

        if self.use_rope:
            query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)

        if hasattr(self, "qk_norm"):
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        if self.attn_temperature_tuning and not self.use_rope:
            past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
            positions = paddle.arange(seq_len) + past_seen_tokens
            attn_scales = (
                paddle.log1p(paddle.floor((positions.astype("float32") + 1.0) / self.floor_scale)) * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales.reshape([1, seq_len, 1, 1])
            query_states = (query_states * attn_scales).astype(query_states.dtype)

        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Llama4TextAttentionFused(nn.Layer):
    """Fused attention for optimized inference (delegates to standard impl)."""

    def __init__(self, config: Llama4TextConfig, layer_idx: int):
        super().__init__()
        self.attn = Llama4TextAttention(config, layer_idx)

    def forward(self, *args, **kwargs):
        return self.attn(*args, **kwargs)


class Llama4TextDecoderLayer(nn.Layer):
    def __init__(self, config: Llama4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Llama4TextAttention(config, layer_idx)

        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:
            self.feed_forward = Llama4TextMoe(config)
        else:
            self.feed_forward = Llama4TextMLP(config, intermediate_size=config.intermediate_size_mlp)

        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        position_embeddings=None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states.reshape(residual.shape)

        outputs = (hidden_states,)
        if len(outputs) == 1 and isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs


class Llama4TextPretrainedModel(PretrainedModel):
    config_class = Llama4TextConfig
    base_model_prefix = "model"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "router.linear",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Llama4TextConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight"]
        aoa_statements.append(f"model.norm.weight -> {model_prefix}norm.weight")
        aoa_statements.append(
            "model.layers.$LAYER_ID.input_layernorm.weight"
            f" -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight"
        )
        aoa_statements.append(
            "model.layers.$LAYER_ID.post_attention_layernorm.weight"
            f" -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight"
        )

        for layer_id in range(config.num_hidden_layers):
            hf_prefix = f"model.layers.{layer_id}"
            pd_prefix = f"{model_prefix}layers.{layer_id}"
            aoa_statements.extend(
                [
                    f"{hf_prefix}.self_attn.{proj_name}.weight^T -> {pd_prefix}.self_attn.{proj_name}.weight"
                    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
                ]
            )

            if layer_id in config.moe_layers:
                aoa_statements.append(
                    f"{hf_prefix}.feed_forward.router.weight^T -> {pd_prefix}.feed_forward.router.linear.weight"
                )
                aoa_statements.extend(
                    [
                        f"{hf_prefix}.feed_forward.experts.gate_up_proj -> "
                        f"{pd_prefix}.feed_forward.experts.gate_up_proj",
                        f"{hf_prefix}.feed_forward.experts.down_proj -> "
                        f"{pd_prefix}.feed_forward.experts.down_proj",
                    ]
                )
                aoa_statements.extend(
                    [
                        f"{hf_prefix}.feed_forward.shared_expert.{proj_name}.weight^T -> "
                        f"{pd_prefix}.feed_forward.shared_expert.{proj_name}.weight"
                        for proj_name in ["gate_proj", "up_proj", "down_proj"]
                    ]
                )
            else:
                aoa_statements.extend(
                    [
                        f"{hf_prefix}.feed_forward.{proj_name}.weight^T -> "
                        f"{pd_prefix}.feed_forward.{proj_name}.weight"
                        for proj_name in ["gate_proj", "up_proj", "down_proj"]
                    ]
                )

        if cls != cls.base_model_class:
            if config.tie_word_embeddings:
                aoa_statements.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: Llama4TextConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight"]
        aoa_statements.append(f"{model_prefix}norm.weight -> model.norm.weight")
        aoa_statements.append(
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight"
            " -> model.layers.$LAYER_ID.input_layernorm.weight"
        )
        aoa_statements.append(
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight"
            " -> model.layers.$LAYER_ID.post_attention_layernorm.weight"
        )

        for layer_id in range(config.num_hidden_layers):
            pd_prefix = f"{model_prefix}layers.{layer_id}"
            hf_prefix = f"model.layers.{layer_id}"
            aoa_statements.extend(
                [
                    f"{pd_prefix}.self_attn.{proj_name}.weight^T -> {hf_prefix}.self_attn.{proj_name}.weight"
                    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
                ]
            )

            if layer_id in config.moe_layers:
                aoa_statements.append(
                    f"{pd_prefix}.feed_forward.router.linear.weight^T -> {hf_prefix}.feed_forward.router.weight"
                )
                aoa_statements.extend(
                    [
                        f"{pd_prefix}.feed_forward.experts.gate_up_proj -> "
                        f"{hf_prefix}.feed_forward.experts.gate_up_proj",
                        f"{pd_prefix}.feed_forward.experts.down_proj -> "
                        f"{hf_prefix}.feed_forward.experts.down_proj",
                    ]
                )
                aoa_statements.extend(
                    [
                        f"{pd_prefix}.feed_forward.shared_expert.{proj_name}.weight^T -> "
                        f"{hf_prefix}.feed_forward.shared_expert.{proj_name}.weight"
                        for proj_name in ["gate_proj", "up_proj", "down_proj"]
                    ]
                )
            else:
                aoa_statements.extend(
                    [
                        f"{pd_prefix}.feed_forward.{proj_name}.weight^T -> "
                        f"{hf_prefix}.feed_forward.{proj_name}.weight"
                        for proj_name in ["gate_proj", "up_proj", "down_proj"]
                    ]
                )

        if not config.tie_word_embeddings and cls != cls.base_model_class:
            aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}


@register_base_model
class Llama4TextModel(Llama4TextPretrainedModel):
    def __init__(self, config: Llama4TextConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.LayerList(
            [Llama4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
        )
        self.rotary_emb = Llama4TextRotaryEmbedding(config=config)

    def _create_chunked_causal_mask(
        self, batch_size, seq_length, cache_length, dtype, chunk_size, attention_mask=None
    ):
        local_cache_length = min(cache_length, chunk_size - 1)
        kv_length = local_cache_length + seq_length
        kv_offset = cache_length - local_cache_length

        query_positions = paddle.arange(cache_length, cache_length + seq_length, dtype="int64").reshape(
            [1, seq_length, 1]
        )
        key_positions = paddle.arange(kv_offset, kv_offset + kv_length, dtype="int64").reshape([1, 1, kv_length])

        padding_mask = None
        if attention_mask is not None and attention_mask.ndim == 2:
            padding_mask = attention_mask.astype("bool")
            left_padding_tokens = (padding_mask.astype("int64").cumsum(axis=-1) == 0).astype("int64").sum(axis=-1)
        else:
            left_padding_tokens = paddle.zeros([batch_size], dtype="int64")
        left_padding_tokens = left_padding_tokens.reshape([batch_size, 1, 1])

        query_chunk = (query_positions - left_padding_tokens) // chunk_size
        key_chunk = (key_positions - left_padding_tokens) // chunk_size
        allowed = (key_positions <= query_positions) & (key_chunk == query_chunk)

        if padding_mask is not None:
            required_length = kv_offset + kv_length
            if padding_mask.shape[-1] < required_length:
                padding_mask = paddle.concat(
                    [
                        padding_mask,
                        paddle.zeros(
                            [batch_size, required_length - padding_mask.shape[-1]],
                            dtype="bool",
                        ),
                    ],
                    axis=-1,
                )
            local_padding_mask = padding_mask[:, kv_offset:required_length].unsqueeze(1)
            allowed = allowed & local_padding_mask

        min_value = paddle.full([], paddle.finfo(dtype).min, dtype=dtype)
        mask = paddle.where(allowed, paddle.zeros([], dtype=dtype), min_value)
        return mask.unsqueeze(1)

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = False,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", False)

        if not ((input_ids is None) ^ (inputs_embeds is None)):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        inputs_embeds = cast(paddle.Tensor, inputs_embeds)
        bsz, seq_length, _ = inputs_embeds.shape

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        kv_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (
                paddle.arange(kv_seq_len, seq_length + kv_seq_len, dtype=paddle.int64).unsqueeze(0).tile((bsz, 1))
            )

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
        full_causal_mask, full_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        if self.config._attn_implementation == "eager":
            chunked_causal_mask = self._create_chunked_causal_mask(
                bsz,
                seq_length,
                kv_seq_len,
                inputs_embeds.dtype,
                self.config.attention_chunk_size,
                attention_mask,
            )
        else:
            chunked_causal_mask = full_causal_mask

        causal_mask_mapping = {
            "full_attention": (full_causal_mask, full_row_indices),
            "chunked_attention": (chunked_causal_mask, full_row_indices),
        }

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        all_hidden_states = [] if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_type = self.config.layer_types[idx]
            layer_mask, layer_row_indices = causal_mask_mapping[layer_type]

            has_gradient = not hidden_states.stop_gradient
            if (
                getattr(self.config, "recompute_granularity", None) == "full"
                and getattr(self.config, "recompute_method", None) == "uniform"
                and getattr(self.config, "recompute_num_layers", 0) == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    layer_mask,
                    layer_row_indices,
                    position_ids,
                    position_embeddings,
                    past_key_values,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    attn_mask_startend_row_indices=layer_row_indices,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        all_hidden_states = tuple(all_hidden_states) if all_hidden_states else None

        if not return_dict:
            outputs = [hidden_states]
            if output_hidden_states:
                outputs.append(all_hidden_states)
            return tuple(outputs)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices,
        position_ids,
        position_embeddings,
        past_key_values,
        use_cache,
    ):
        return recompute(
            layer_module,
            hidden_states,
            attention_mask,
            attn_mask_startend_row_indices,
            position_ids,
            position_embeddings,
            past_key_values,
            use_cache,
            use_reentrant=getattr(self.config, "recompute_use_reentrant", True),
        )


class Llama4ForCausalLM(Llama4TextPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: Llama4TextConfig):
        super().__init__(config)
        self.config = config
        self.model = Llama4TextModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Cache | None = None,
        output_hidden_states: bool | None = False,
        return_dict: bool = False,
        **kwargs,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", False)

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used for full-attention layers, "
                "while attention_mask is still required by chunked-attention layers."
            )

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
