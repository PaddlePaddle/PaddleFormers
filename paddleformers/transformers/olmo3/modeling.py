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

"""OLMo3 model implementation for PaddlePaddle.

Migrated from transformers.models.olmo3.modeling_olmo3
"""

from contextlib import nullcontext
from typing import Callable, Optional, cast

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import (
    EmbeddingPipe,
    GeneralModelForCausalLMPipe,
    LMHeadPipe,
    RMSNormPipe,
    parse_args,
)
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import Olmo3Config


def _build_attention_masks(
    config,
    inputs_embeds,
    batch_size,
    seq_length,
    attention_mask,
    attn_mask_startend_row_indices,
    prepare_decoder_attention_mask,
    has_sliding_layers,
    past_key_values=None,
):
    """Prepare per-attention-type masks while preserving cache and caller-owned mask semantics."""
    cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    if isinstance(attention_mask, dict):
        causal_mask_mapping = {}
        attn_mask_startend_row_indices_mapping = {}
        for attention_type, mask_result in attention_mask.items():
            if isinstance(mask_result, tuple):
                (
                    causal_mask_mapping[attention_type],
                    attn_mask_startend_row_indices_mapping[attention_type],
                ) = mask_result
            else:
                causal_mask_mapping[attention_type] = mask_result
                attn_mask_startend_row_indices_mapping[attention_type] = attn_mask_startend_row_indices
    else:
        mask_kwargs = {
            "config": config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": cache_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": prepare_decoder_attention_mask,
        }
        full_mask, full_indices = create_causal_mask_and_row_indices(**mask_kwargs)
        causal_mask_mapping = {"full_attention": full_mask}
        attn_mask_startend_row_indices_mapping = {"full_attention": full_indices}
        if has_sliding_layers:
            (
                causal_mask_mapping["sliding_attention"],
                attn_mask_startend_row_indices_mapping["sliding_attention"],
            ) = create_sliding_window_causal_mask_and_row_indices(**mask_kwargs)
        if attn_mask_startend_row_indices is None and cache_length and seq_length > 1:
            # Cached multi-token chunks need a bottom-right aligned causal mask.
            full_mask = prepare_decoder_attention_mask(
                attention_mask=(
                    paddle.ones([batch_size, cache_length + seq_length], dtype="bool")
                    if attention_mask is None
                    else attention_mask
                ),
                input_shape=(batch_size, seq_length),
                past_key_values_length=cache_length,
                dtype=inputs_embeds.dtype,
            )
            causal_mask_mapping["full_attention"] = full_mask
            attn_mask_startend_row_indices_mapping["full_attention"] = None
        if past_key_values is not None:
            cache_position = paddle.arange(cache_length, cache_length + seq_length, dtype="int64")
            for attention_type in set(config.layer_types):
                layer_idx = config.layer_types.index(attention_type)
                kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
                mask = causal_mask_mapping[attention_type]
                row_indices = attn_mask_startend_row_indices_mapping[attention_type]
                if kv_offset:
                    if row_indices is not None:
                        raise NotImplementedError(
                            "Cached sliding-window decoding with sparse row indices is not supported. "
                            "Pass a dense attention_mask instead."
                        )
                    if mask is not None:
                        mask = mask[..., kv_offset : kv_offset + kv_length]
                    causal_mask_mapping[attention_type] = mask
    return causal_mask_mapping, attn_mask_startend_row_indices_mapping


class _SplitProjection(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, hidden_states, group):
        ctx.group = group
        return paddle.split(hidden_states, group.nranks, axis=-1)[group.rank]

    @staticmethod
    def backward(ctx, gradient):
        return mp_ops._c_concat(gradient, group=ctx.group)


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies rotary positional embedding to query and key tensors.

    NOTE: Keep computation in the original dtype (bf16) to match PyTorch's
    behavior exactly, avoiding float32-induced precision differences that
    accumulate across layers.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Olmo3Attention(nn.Layer):
    """Multi-headed attention with RMS normalization over each full Q/K projection."""

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.tensor_parallel_size = max(config.tensor_model_parallel_size, 1)
        if config.sequence_parallel and self.tensor_parallel_size == 1:
            raise ValueError("OLMo3 sequence parallelism requires tensor_model_parallel_size > 1.")
        if (
            self.num_heads % self.tensor_parallel_size != 0
            or self.num_key_value_heads % self.tensor_parallel_size != 0
        ):
            raise ValueError("Q and KV head counts must be divisible by tensor_model_parallel_size.")
        self.num_heads //= self.tensor_parallel_size
        self.num_key_value_heads //= self.tensor_parallel_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

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

        # QK-norm spans the full projection, not each head independently.
        self.q_norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=q_hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
        )
        self.k_norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=kv_hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
        )

    def _normalize_projection(self, hidden_states, norm):
        if self.tensor_parallel_size == 1:
            return norm(hidden_states)
        group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        # Splitting with a collective backward gives each replicated norm the full gradient.
        hidden_states = mp_ops._c_concat(hidden_states, group=group)
        return _SplitProjection.apply(norm(hidden_states), group)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[paddle.Tensor, list[paddle.Tensor] | None]:
        if self.config.sequence_parallel:
            seq_len = position_embeddings[0].shape[1]
            batch_size = hidden_states.shape[0] * self.config.tensor_model_parallel_size // seq_len
        else:
            batch_size, seq_len = hidden_states.shape[:2]

        q_shape = (batch_size, seq_len, -1, self.head_dim)
        kv_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self._normalize_projection(self.q_proj(hidden_states), self.q_norm)
        key_states = self._normalize_projection(self.k_proj(hidden_states), self.k_norm)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(q_shape).transpose(1, 2)
        key_states = key_states.reshape(kv_shape).transpose(1, 2)
        value_states = value_states.reshape(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if use_cache:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            elif self.layer_idx < len(past_key_values.layers):
                layer_cache = past_key_values.layers[self.layer_idx]
                if layer_cache.is_initialized:
                    key_states = paddle.concat([layer_cache.keys, key_states], axis=-2)
                    value_states = paddle.concat([layer_cache.values, value_states], axis=-2)

        backend = self.config._attn_implementation
        # Flashmask only consumes sparse row indices; preserve dense mask semantics.
        if (output_attentions and backend != "eager") or (
            backend == "flashmask" and (attention_mask is not None or (self.training and self.attention_dropout > 0))
        ):
            logger.warning_once(
                "OLMo3 is falling back to eager attention to preserve requested masks, dropout or attention "
                "weights. This can increase memory usage and runtime for long sequences."
            )
            backend = "eager"
            if attention_mask is None and attn_mask_startend_row_indices is None:
                query_positions = paddle.arange(key_states.shape[-2] - seq_len, key_states.shape[-2]).unsqueeze(-1)
                key_positions = paddle.arange(key_states.shape[-2]).unsqueeze(0)
                allowed = key_positions <= query_positions
                if self.sliding_window is not None:
                    allowed = allowed & (key_positions > query_positions - self.sliding_window)
                attention_mask = paddle.where(allowed, 0.0, paddle.finfo(query_states.dtype).min).astype(
                    query_states.dtype
                )
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[backend]

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
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Olmo3DecoderLayer(nn.Layer):
    """OLMo3 decoder layer with post-norm structure.

    Post-norm structure (vs Llama's pre-norm):
    - attn -> post_attention_layernorm -> residual -> mlp -> post_feedforward_layernorm -> residual
    """

    _attention_cls = Olmo3Attention

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = self._attention_cls(config=config, layer_idx=layer_idx)

        self.mlp = MLP(config)

        # Post-norm: applied AFTER attention and MLP
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_feedforward_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[paddle.Tensor] | tuple[paddle.Tensor, paddle.Tensor]:
        residual = hidden_states

        hidden_states, attention_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, attention_weights) if output_attentions else hidden_states


class Olmo3RotaryEmbedding(nn.Layer):
    """Rotary position embedding."""

    def __init__(self, config: Olmo3Config, rope_type: Optional[str] = None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        if rope_type is not None:
            self.rope_type = rope_type
        elif hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            self.rope_type = config.rope_parameters.get("rope_type", config.rope_parameters.get("type", "default"))
        else:
            self.rope_type = "default"

        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Olmo3Config] = None,
        seq_len: Optional[int] = None,
    ) -> tuple[paddle.Tensor, float]:
        """Computes the inverse frequencies for RoPE."""
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0

        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Olmo3PretrainedModel(PretrainedModel):
    """Base class for OLMo3 models."""

    config_class = Olmo3Config
    base_model_prefix = "model"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    def _init_weights(self, layer):
        linear_types = tuple(GeneralLinear._global_mapping.values()) + (GeneralLMHead,)
        embedding_types = tuple(GeneralEmbedding._global_mapping.values())
        if isinstance(layer, linear_types + embedding_types):
            weight = layer.weight
            if not getattr(weight, "_olmo3_initialized", False):
                context = (
                    get_rng_state_tracker().rng_state() if getattr(weight, "is_distributed", False) else nullcontext()
                )
                with context:
                    nn.initializer.Normal(std=self.config.initializer_range)(weight)
                weight._olmo3_initialized = True
            if getattr(layer, "bias", None) is not None:
                nn.initializer.Constant(0.0)(layer.bias)
            if isinstance(layer, nn.Embedding) and layer._padding_idx is not None:
                with paddle.no_grad():
                    weight[layer._padding_idx] = 0

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None,
        attn_mask_startend_row_indices: paddle.Tensor | None,
        position_ids: paddle.Tensor,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor],
        past_key_values: Cache | None,
        use_cache: bool,
        output_attentions: bool = False,
    ):
        """Recompute one decoder layer without mutating cached state."""
        cos, sin = position_embeddings
        cos = cos.clone()
        sin = sin.clone()

        position_embeddings_safe = (cos, sin)

        hidden_states = recompute(
            layer_module,
            hidden_states,
            attention_mask,
            attn_mask_startend_row_indices,
            position_ids,
            position_embeddings_safe,
            past_key_values,
            use_cache,
            output_attentions,
            use_reentrant=self.config.recompute_use_reentrant and not hidden_states.stop_gradient,
        )
        return hidden_states

    @classmethod
    def _gen_aoa_config(cls, config: Olmo3Config):
        """Generate AOA config for weight conversion.

        Note: OLMo3 uses post-norm structure:
        - post_attention_layernorm (not input_layernorm)
        - post_feedforward_layernorm (not post_attention_layernorm)

        OLMo3 follows PT-oriented checkpoint layout for projection/MLP matrices,
        so both PT and PD checkpoints use ^T mapping for these weights.
        """
        model_prefix = "" if cls == cls.base_model_class else "model."
        is_fleet = getattr(cls, "is_fleet", False)

        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"model.layers.$LAYER_ID.post_feedforward_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_feedforward_layernorm.weight",
        ]

        # Projection weights are loaded with transpose for unified PT-oriented layout.
        proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if config.attention_bias:
            aoa_statements.extend(
                f"model.layers.$LAYER_ID.self_attn.{name}.bias -> {model_prefix}layers.$LAYER_ID.self_attn.{name}.bias"
                for name in proj_names
            )
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in proj_names
            ]
        )
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.mlp.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )
        if config.get("mlp_bias", False):
            aoa_statements.extend(
                f"model.layers.$LAYER_ID.mlp.{name}.bias -> {model_prefix}layers.$LAYER_ID.mlp.{name}.bias"
                for name in ("gate_proj", "up_proj", "down_proj")
            )

        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.self_attn.q_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
                f"model.layers.$LAYER_ID.self_attn.k_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
            ]
        )

        if cls != cls.base_model_class:
            if config.get("lm_head_bias", False):
                aoa_statements.append("lm_head.bias -> lm_head.bias")
            if config.tie_word_embeddings:
                if is_fleet:
                    aoa_statements.append(f"model.embed_tokens.weight -> {model_prefix}lm_head.weight")
                else:
                    aoa_statements.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                if is_fleet:
                    aoa_statements.append(f"lm_head.weight -> {model_prefix}lm_head.weight")
                else:
                    aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: Olmo3Config):
        """Generate inverse AOA config for weight export."""
        model_prefix = "" if cls == cls.base_model_class else "model."
        is_fleet = getattr(cls, "is_fleet", False)

        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_feedforward_layernorm.weight -> model.layers.$LAYER_ID.post_feedforward_layernorm.weight",
        ]

        if config.attention_bias:
            aoa_statements.extend(
                f"{model_prefix}layers.$LAYER_ID.self_attn.{name}.bias -> model.layers.$LAYER_ID.self_attn.{name}.bias"
                for name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            )

        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> model.layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )

        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight -> model.layers.$LAYER_ID.self_attn.q_norm.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight -> model.layers.$LAYER_ID.self_attn.k_norm.weight",
            ]
        )

        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight^T -> model.layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )
        if config.get("mlp_bias", False):
            aoa_statements.extend(
                f"{model_prefix}layers.$LAYER_ID.mlp.{name}.bias -> model.layers.$LAYER_ID.mlp.{name}.bias"
                for name in ("gate_proj", "up_proj", "down_proj")
            )
        if cls != cls.base_model_class and config.get("lm_head_bias", False):
            aoa_statements.append("lm_head.bias -> lm_head.bias")

        if not config.tie_word_embeddings and cls != cls.base_model_class:
            if is_fleet:
                aoa_statements.append(f"lm_head.weight -> {model_prefix}lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}


@register_base_model
class Olmo3Model(Olmo3PretrainedModel):
    """OLMo3 model (base model without LM head).

    Key differences from OLMo2:
    - Sliding window attention is used for 3 out of 4 layers
    - RoPE scaling is not applied to sliding window attention layers
    - Uses ModuleDict with two RotaryEmbeddings instead of a single one
    """

    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.has_sliding_layers = "sliding_attention" in config.layer_types

        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=None,
        )
        self.layers = nn.LayerList(
            [Olmo3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        # Two separate RoPE embeddings: sliding_attention uses default (no scaling), full_attention uses configured rope
        self.rotary_embs = nn.LayerDict(
            {
                "sliding_attention": Olmo3RotaryEmbedding(config=config, rope_type="default"),
                "full_attention": Olmo3RotaryEmbedding(config=config),
            }
        )

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
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        do_recompute = self.training and paddle.is_grad_enabled() and self.config.recompute_granularity == "full"
        if do_recompute:
            if self.config.recompute_method != "uniform" or self.config.recompute_num_layers != 1:
                raise ValueError("OLMo3 full recompute requires uniform recompute with one layer per group.")
            if past_key_values is not None:
                raise ValueError("OLMo3 recompute training cannot consume a mutable KV cache.")
            if use_cache:
                logger.warning_once("Disabling use_cache during OLMo3 recompute training.")
                use_cache = False

        if not ((input_ids is None) ^ (inputs_embeds is None)):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        inputs_embeds = cast(paddle.Tensor, inputs_embeds)
        bsz, seq_length, _ = inputs_embeds.shape

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        kv_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (
                paddle.arange(kv_seq_len, seq_length + kv_seq_len, dtype=paddle.int64).unsqueeze(0).tile((bsz, 1))
            )

        causal_mask_mapping, attn_mask_startend_row_indices_mapping = _build_attention_masks(
            config=self.config,
            inputs_embeds=inputs_embeds,
            batch_size=bsz,
            seq_length=seq_length,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            prepare_decoder_attention_mask=self._prepare_decoder_attention_mask,
            has_sliding_layers=self.has_sliding_layers,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds

        position_embeddings_mapping = {
            "sliding_attention": self.rotary_embs["sliding_attention"](hidden_states, position_ids),
            "full_attention": self.rotary_embs["full_attention"](hidden_states, position_ids),
        }

        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            attention_type = decoder_layer.self_attn.attention_type

            layer_causal_mask = causal_mask_mapping[attention_type]
            layer_attn_mask_startend_row_indices = attn_mask_startend_row_indices_mapping[attention_type]

            layer_position_embeddings = position_embeddings_mapping[attention_type]

            if do_recompute:
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    layer_causal_mask,
                    layer_attn_mask_startend_row_indices,
                    position_ids,
                    layer_position_embeddings,
                    past_key_values,
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_causal_mask,
                    attn_mask_startend_row_indices=layer_attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    position_embeddings=layer_position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                all_attentions.append(layer_outputs[1])
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple | list) else layer_outputs

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        all_hidden_states = tuple(all_hidden_states) if all_hidden_states else None

        if not return_dict:
            outputs = []
            outputs.append(hidden_states)
            if use_cache and past_key_values is not None:
                outputs.append(past_key_values)
            if output_hidden_states:
                outputs.append(all_hidden_states)
            if output_attentions:
                outputs.append(tuple(all_attentions))
            return tuple(outputs)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=tuple(all_attentions) if output_attentions else None,
        )


class Olmo3ForCausalLM(Olmo3PretrainedModel):
    """OLMo3 model for causal language modeling."""

    _tied_weights_keys = ["lm_head.weight"]
    _model_cls = Olmo3Model

    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.config = config
        self.model = self._model_cls(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, use_cache=None, **kwargs
    ):
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None and attention_mask is not None and attention_mask.ndim == 2:
            position_ids = attention_mask.astype("int64").cumsum(-1) - 1
            position_ids = paddle.where(attention_mask.astype("bool"), position_ids, 1)
        if cache_length:
            if attention_mask is not None and attention_mask.shape[-1] > input_ids.shape[1]:
                new_length = attention_mask.shape[-1] - cache_length
                if new_length <= 0:
                    raise ValueError("Generation attention_mask must include at least one uncached token.")
                input_ids = input_ids[:, -new_length:]
            elif input_ids.shape[1] > cache_length:
                input_ids = input_ids[:, cache_length:]
        current_length = (
            inputs_embeds.shape[1] if inputs_embeds is not None and not cache_length else input_ids.shape[1]
        )
        if position_ids is None:
            position_ids = paddle.arange(cache_length, cache_length + current_length, dtype="int64").unsqueeze(0)
        else:
            position_ids = position_ids[:, -current_length:]
        model_inputs = {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": self.config.use_cache if use_cache is None else use_cache,
            "return_dict": True,
        }
        if inputs_embeds is not None and not cache_length:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        model_inputs.update(kwargs)
        return model_inputs

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        past_key_values: Cache | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        if kwargs.get("attn_mask_start_row_indices", None) is not None and attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs.pop("attn_mask_start_row_indices")
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (
            isinstance(attention_mask, paddle.Tensor)
            and attention_mask.ndim == 2
            and attention_mask.dtype != paddle.bool
        ):
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels, loss_mask=loss_mask)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Olmo3EmbeddingPipe(nn.Layer):
    def __init__(self, config, embed_cls=None, rotary_emb_cls=None):
        super().__init__()
        self.config = config
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self.embed_tokens = GeneralEmbedding.create(config=config, padding_idx=None)
        self.rotary_embs = nn.LayerDict(
            {
                "full_attention": Olmo3RotaryEmbedding(config),
                "sliding_attention": Olmo3RotaryEmbedding(config, rope_type="default"),
            }
        )

    @property
    def embedding_weight(self):
        return self.embed_tokens.weight

    def forward(self, args):
        input_ids, mask, position_ids, _, _ = parse_args(args, is_embed=True)
        hidden_states = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        batch_size, seq_length = input_ids.shape
        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").unsqueeze(0).expand([batch_size, seq_length])
        sparse = mask is not None and mask.ndim >= 3 and mask.dtype == paddle.int32
        causal_masks, row_indices = _build_attention_masks(
            config=self.config,
            inputs_embeds=hidden_states,
            batch_size=batch_size,
            seq_length=seq_length,
            attention_mask=None if sparse else mask,
            attn_mask_startend_row_indices=mask if sparse else None,
            prepare_decoder_attention_mask=Olmo3Model._prepare_decoder_attention_mask,
            has_sliding_layers=self.has_sliding_layers,
        )
        full_mask, full_indices = causal_masks["full_attention"], row_indices["full_attention"]
        sliding_mask, sliding_indices = causal_masks.get("sliding_attention"), row_indices.get("sliding_attention")
        ropes = paddle.stack(
            [
                paddle.stack(self.rotary_embs["full_attention"](hidden_states, position_ids)),
                paddle.stack(self.rotary_embs["sliding_attention"](hidden_states, position_ids)),
            ]
        )
        if self.config.sequence_parallel:
            hidden_states = ScatterOp.apply(hidden_states.reshape([-1, hidden_states.shape[-1]]))
        # Pipeline transport requires tensors; a rank-one sentinel represents an absent mask.
        metadata = tuple(
            value if value is not None else paddle.zeros([1], dtype=hidden_states.dtype)
            for value in (full_mask, full_indices, sliding_mask, sliding_indices, ropes)
        )
        for value in metadata:
            value.stop_gradient = True
        return (hidden_states,) + metadata


class Olmo3DecoderLayerPipe(Olmo3DecoderLayer):
    def forward(self, args):
        hidden_states, full_mask, full_indices, sliding_mask, sliding_indices, ropes = args
        if self.config.recompute_granularity == "full" and (
            self.config.recompute_method != "uniform" or self.config.recompute_num_layers != 1
        ):
            raise ValueError("OLMo3 full recompute requires uniform recompute with one layer per group.")
        is_sliding = self.self_attn.attention_type == "sliding_attention"
        mask, indices = (sliding_mask, sliding_indices) if is_sliding else (full_mask, full_indices)
        rope = ropes[1 if is_sliding else 0]
        kwargs = {
            "attention_mask": mask if mask.ndim > 1 else None,
            "attn_mask_startend_row_indices": indices if indices.ndim > 1 else None,
            "position_embeddings": (rope[0], rope[1]),
            "use_cache": False,
        }
        if self.training and paddle.is_grad_enabled() and self.config.recompute_granularity == "full":
            hidden_states = recompute(
                super().forward,
                hidden_states,
                use_reentrant=self.config.recompute_use_reentrant and not hidden_states.stop_gradient,
                **kwargs,
            )
        else:
            hidden_states = super().forward(hidden_states, **kwargs)
        return (hidden_states,) + tuple(value.clone() for value in args[1:])


class Olmo3RMSNormPipe(RMSNormPipe):
    def forward(self, args):
        return super().forward(args[0])


class Olmo3LMHeadPipe(LMHeadPipe):
    def forward(self, args):
        return super().forward(args[0] if isinstance(args, tuple) else args)


class Olmo3ForCausalLMPipe(GeneralModelForCausalLMPipe):
    """OLMo3 pipeline model with per-layer masks and rotary embeddings."""

    config_class = Olmo3Config
    _decoder_layer_cls = Olmo3DecoderLayer
    _decoder_layer_pipe_cls = Olmo3DecoderLayerPipe
    _embedding_pipe_cls = Olmo3EmbeddingPipe
    _rms_norm_pipe_cls = Olmo3RMSNormPipe
    _lmhead_pipe_cls = Olmo3LMHeadPipe
    _init_weights = Olmo3Model._init_weights
    _keep_in_fp32_modules = Olmo3Model._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Olmo3Model.transpose_weight_keys
    _gen_aoa_config = Olmo3ForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = Olmo3ForCausalLM._gen_inv_aoa_config

    @classmethod
    def _prepare_pipeline_inputs_func(cls, inputs):
        def normalize(record):
            record = dict(record)
            indices = record.get("attn_mask_startend_row_indices")
            if indices is None:
                indices = record.get("attn_mask_start_row_indices")
            record.pop("attn_mask_start_row_indices", None)
            record.pop("attn_mask_startend_row_indices", None)
            if indices is not None:
                if record.get("attention_mask") is not None:
                    logger.warning("Both dense mask and row indices supplied; using row indices for OLMo3 PP.")
                record.pop("attention_mask", None)
                record["attn_mask_startend_row_indices"] = indices
            elif record.get("attention_mask") is None:
                record["attention_mask"] = paddle.ones_like(record["input_ids"], dtype="bool")
            if record.get("position_ids") is None:
                record.pop("position_ids", None)
            return record

        normalized = normalize(inputs) if isinstance(inputs, dict) else [normalize(record) for record in inputs]
        return super()._prepare_pipeline_inputs_func(normalized)

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        # The generic tied-embedding branch uses EmbeddingPipe directly.
        if getattr(layer_desc, "layer_func", None) is EmbeddingPipe:
            layer_desc.layer_func = Olmo3EmbeddingPipe
        return super().add_sequential_layer(layer_desc, name_prefix)


__all__ = [
    "Olmo3Model",
    "Olmo3PretrainedModel",
    "Olmo3ForCausalLM",
    "Olmo3ForCausalLMPipe",
]
