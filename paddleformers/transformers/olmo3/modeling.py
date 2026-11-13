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
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, cast

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    build_spec_layer,
    get_rng_state_tracker,
)
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp
from paddlefleet.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_spec,
)
from paddlefleet.transformer.attention import SelfAttention
from paddlefleet.transformer.paddle_norm import RMSNorm

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..gpt_provider import GPTModel, GPTModelProvider
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import Olmo3Config


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

        key_states, value_states = _update_attention_cache(
            past_key_values, key_states, value_states, self.layer_idx, use_cache
        )
        backend, attention_mask = _select_attention_backend(
            self.config,
            query_states,
            key_states,
            attention_mask,
            attn_mask_startend_row_indices,
            self.training,
            self.sliding_window,
            output_attentions,
        )
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

        self.rope_type = _resolve_rope_type(config, rope_type)

        inv_freq, self.attention_scaling = _compute_rope_parameters(self.config, self.rope_type)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

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
                aoa_statements.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: Olmo3Config):
        """Generate inverse AOA config for weight export."""
        model_prefix = "" if cls == cls.base_model_class else "model."

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


class Olmo3FleetRotaryEmbedding(RotaryEmbedding):
    def __init__(self, source, rope_type=None):
        super().__init__(head_dim=source.head_dim, rotary_percent=1.0, rotary_base=source.rope_theta)
        self.inv_freq, _ = _compute_rope_parameters(source, rope_type)

    def forward(self, max_seq_len, offset=0, packed_seq=False, position_ids=None):
        if position_ids is None:
            position_ids = paddle.arange(offset, offset + max_seq_len).unsqueeze(0)
        frequencies = position_ids.astype("float32").unsqueeze(-1) * self.inv_freq
        return paddle.concat([frequencies, frequencies], axis=-1).unsqueeze(-2)


class Olmo3FleetEmbedding(GPTEmbedding):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.rotary_pos_emb = Olmo3FleetRotaryEmbedding(config.olmo3_config)
        if self.swa_rotary_pos_emb is not None:
            self.swa_rotary_pos_emb = Olmo3FleetRotaryEmbedding(config.olmo3_config, rope_type="default")

    def forward(self, dict_args, *args, **kwargs):
        dict_args = dict(dict_args)
        legacy_indices = dict_args.pop("attn_mask_start_row_indices", None)
        indices = dict_args.get("attn_mask_startend_row_indices")
        if indices is None:
            indices = legacy_indices
        if indices is not None:
            dict_args["attn_mask_startend_row_indices"] = indices
            if dict_args.get("attention_mask") is not None:
                logger.warning_once("Both dense mask and row indices supplied; using row indices for OLMo3 PP.")
                dict_args.pop("attention_mask")
        cache = dict_args.get("past_key_values")
        if dict_args.get("use_cache") or cache is not None:
            dict_args = dict(dict_args)
            if cache is None:
                cache = DynamicCache(config=self.config.olmo3_config)
                dict_args["past_key_values"] = cache
            length = dict_args["input_ids"].shape[1]
            if dict_args.get("position_ids") is None:
                start = cache.get_seq_length()
                dict_args["position_ids"] = paddle.arange(start, start + length).unsqueeze(0)
        output = super().forward(dict_args, *args, **kwargs)
        position_ids = dict_args.get("position_ids")
        if self.training and position_ids is not None:
            frequencies = self.rotary_pos_emb(position_ids.shape[-1], position_ids=position_ids)
            if self.config.sequence_parallel:
                frequencies = frequencies.transpose([1, 0, 2, 3]).contiguous()
            output["rotary_pos_emb"] = frequencies
        if self.config.apply_rope_fusion:
            length = dict_args["input_ids"].shape[1] if position_ids is None else position_ids.max() + 1
            dtype = self.config.params_dtype or paddle.get_default_dtype()
            frequencies = self.rotary_pos_emb(length)
            output["rotary_pos_cos"] = frequencies.cos().astype(dtype)
            output["rotary_pos_sin"] = frequencies.sin().astype(dtype)
            if self.swa_rotary_pos_emb is not None:
                frequencies = self.swa_rotary_pos_emb(length)
                output["swa_rotary_pos_cos"] = frequencies.cos().astype(dtype)
                output["swa_rotary_pos_sin"] = frequencies.sin().astype(dtype)
        mask_inputs = output["hidden_states"].astype(self.config.params_dtype or paddle.get_default_dtype())
        masks, indices = _build_attention_masks(
            config=self.config.olmo3_config,
            inputs_embeds=mask_inputs,
            batch_size=dict_args["input_ids"].shape[0],
            seq_length=dict_args["input_ids"].shape[1],
            attention_mask=dict_args.get("attention_mask"),
            attn_mask_startend_row_indices=dict_args.get("attn_mask_startend_row_indices"),
            prepare_decoder_attention_mask=Olmo3Model._prepare_decoder_attention_mask,
            has_sliding_layers="sliding_attention" in self.config.olmo3_config.layer_types,
            past_key_values=cache,
        )
        output["attention_mask"] = _pack_attention_masks(masks, indices, self.config.olmo3_config.layer_types)
        output.pop("attn_mask_startend_row_indices", None)
        return output


class Olmo3FleetSelfAttention(SelfAttention):
    def __init__(self, config, **kwargs):
        config = copy(config)
        config.use_bias = config.olmo3_config.attention_bias
        super().__init__(config=config, **kwargs)
        self.config = copy(config)
        self.attention_scaling = 1.0 if self.is_swa else config.olmo3_attention_scaling
        self.config.yarn_rotary_scaling_factor = 1.0

    def forward(self, hidden_states, attention_mask=None, attn_mask_startend_row_indices=None, **kwargs):
        if isinstance(attention_mask, (tuple, list)):
            kind = "sliding_attention" if self.is_swa else "full_attention"
            attention_mask, attn_mask_startend_row_indices = _unpack_attention_masks(attention_mask, kind)
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

    def get_query_key_value_tensors(self, *args, **kwargs):
        query, key, value = super().get_query_key_value_tensors(*args, **kwargs)
        if self.attention_scaling != 1.0:
            query = query * self.attention_scaling
            key = key * self.attention_scaling
        return query, key, value


class Olmo3FleetCoreAttention(paddle.nn.Layer):
    def __init__(self, config, is_swa=False, **kwargs):
        super().__init__()
        self.config = config.olmo3_config
        self.context_parallel_size = config.context_parallel_size
        self.attention_type = "sliding_attention" if is_swa else "full_attention"
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        use_rr_flash_attention=False,
        past_key_values=None,
        layer_idx=None,
        use_cache=False,
        **kwargs,
    ):
        selected_mask, selected_indices = attention_mask, attn_mask_startend_row_indices
        key, value = key.transpose([0, 2, 1, 3]), value.transpose([0, 2, 1, 3])
        key, value = _update_attention_cache(past_key_values, key, value, layer_idx, use_cache)
        backend, selected_mask = _select_attention_backend(
            self.config,
            query.transpose([0, 2, 1, 3]),
            key,
            selected_mask,
            selected_indices,
            self.training,
            self.config.sliding_window if self.attention_type == "sliding_attention" else None,
        )
        return ALL_ATTENTION_FUNCTIONS[backend](
            self,
            query.transpose([0, 2, 1, 3]),
            key,
            value,
            attention_mask=selected_mask,
            attn_mask_startend_row_indices=selected_indices,
            scaling=query.shape[-1] ** -0.5,
            dropout=self.config.attention_dropout if self.training else 0.0,
        )[0]


class Olmo3ProjectionInput(paddle.nn.Layer):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.dtype = config.params_dtype if config.params_dtype is not None else paddle.get_default_dtype()

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states.astype(self.dtype)


class Olmo3PostNormResidual(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config, input_is_parallel=config.sequence_parallel)

    def forward(self, training, fused):
        def apply(value_and_bias, residual, dropout):
            value, bias = value_and_bias
            if bias is not None:
                value = value + bias
            return residual + self.norm(value)

        return apply


@dataclass
class Olmo3ModelProvider(GPTModelProvider):
    olmo3_config: Optional[Olmo3Config] = None
    olmo3_attention_scaling: float = 1.0

    transform_rules = {"dtype": "params_dtype"}

    @classmethod
    def from_config(cls, config):
        source = deepcopy(config)
        normalized = deepcopy(config)
        for field in (
            "tensor_model_parallel_size",
            "context_parallel_size",
            "pipeline_model_parallel_size",
            "virtual_pipeline_model_parallel_size",
            "expert_model_parallel_size",
        ):
            setattr(normalized, field, max(getattr(normalized, field, 1), 1))
        if normalized.context_parallel_size > 1:
            raise ValueError("OLMo3 Fleet does not support context_parallel_size > 1.")
        tp_size = normalized.tensor_model_parallel_size
        if source.num_attention_heads % tp_size or source.num_key_value_heads % tp_size:
            raise ValueError("Q and KV head counts must be divisible by tensor_model_parallel_size.")
        if source.sequence_parallel and tp_size == 1:
            raise ValueError("OLMo3 sequence parallelism requires tensor_model_parallel_size > 1.")
        has_sliding = "sliding_attention" in source.layer_types
        normalized.sliding_window = (source.sliding_window - 1, 0) if has_sliding else None
        normalized.window_attn_skip_freq = [int(kind == "sliding_attention") for kind in source.layer_types]
        normalized.add_swa_attention_sink_bias = False
        normalized.use_qk_norm = True
        normalized.qk_norm_type = "per_layer"
        normalized.use_bias = source.get("mlp_bias", False)
        normalized.apply_query_key_layer_scaling = False
        normalized.hidden_dropout_prob = 0.0
        normalized.init_method_std = source.initializer_range
        normalized.embedding_init_method_std = source.initializer_range
        provider = super().from_config(normalized)
        provider.output_layer_init_method = provider.init_method
        provider.olmo3_config = source
        _, provider.olmo3_attention_scaling = _compute_rope_parameters(source)
        return provider

    def provide(self, pre_process=None, post_process=None, vp_stage=None, loss_fn=None):
        if self.olmo3_config is None:
            raise ValueError("OLMo3 requires olmo3_config; construct the provider with from_config().")
        if pre_process is not None or post_process is not None or vp_stage is not None:
            raise ValueError("OLMo3 builds all pipeline stages together; explicit stage selection is unsupported.")
        layers = []
        for index in range(self.num_hidden_layers):
            spec = get_gpt_layer_local_spec(config=self, normalization="RMSNorm", layer_number=index)
            spec.sublayers_spec.input_layernorm = Olmo3ProjectionInput
            spec.sublayers_spec.post_attention_layernorm = Olmo3ProjectionInput
            spec.sublayers_spec.self_attn_bda = LayerSpec(layer=Olmo3PostNormResidual, extra_kwargs={"config": self})
            spec.sublayers_spec.mlp_bda = LayerSpec(layer=Olmo3PostNormResidual, extra_kwargs={"config": self})
            spec.sublayers_spec.sharded_state_dict_keys_map = {}
            spec.sublayers_spec.self_attn.sublayers_spec.core_attention = LayerSpec(layer=Olmo3FleetCoreAttention)
            spec.sublayers_spec.self_attn.layer = Olmo3FleetSelfAttention
            layers.append(spec)
        spec = get_gpt_spec(
            config=self,
            transformer_layers_spec=layers,
            mtp_layers_spec=None,
            head_empty_layers_spec=[],
            tail_empty_layers_spec=[],
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type="rope",
            rotary_base=self.rope_theta,
            swa_rotary_base=self.rope_theta,
            rope_scaling=False,
            parallel_output=self.parallel_output,
            tie_word_embeddings=self.tie_word_embeddings,
        )
        spec.sublayers_spec.embedding.layer = Olmo3FleetEmbedding
        model = build_spec_layer(
            spec,
            num_stages=self.pipeline_model_parallel_size,
            loss_fn=LanguageLoss(self) if loss_fn is None else loss_fn,
        )
        wrapped = GPTModel.__new__(GPTModel)
        wrapped.__dict__.update(model.__dict__)
        return wrapped


class Olmo3ForCausalLMPipe(GeneralModelForCausalLMPipe):
    """OLMo3 pipeline model with per-layer masks and rotary embeddings."""

    config_class = Olmo3Config
    is_fleet = True

    def __new__(cls, config):
        provider = Olmo3ModelProvider.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            from ...nn.pp_model import CriterionLayerPipe

            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        model = provider.provide(loss_fn=loss_fn)
        model._gen_aoa_config = cls._gen_aoa_config
        model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        model.config_to_save = deepcopy(config)
        if not model.config_to_save.architectures:
            model.config_to_save.architectures = ["Olmo3ForCausalLM"]
        model.is_fleet = True
        return model

    @classmethod
    def _gen_aoa_config(cls, config):
        return _get_fleet_aoa_config(config)

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        statements = [
            "model.embedding.embed_tokens.weight -> model.embed_tokens.weight",
            "model.norm.weight -> model.norm.weight",
            "model.layers.$LAYER_ID.self_attn_bda.norm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            "model.layers.$LAYER_ID.mlp_bda.norm.weight -> model.layers.$LAYER_ID.post_feedforward_layernorm.weight",
            "model.layers.$LAYER_ID.self_attn.q_norm.weight -> model.layers.$LAYER_ID.self_attn.q_norm.weight",
            "model.layers.$LAYER_ID.self_attn.k_norm.weight -> model.layers.$LAYER_ID.self_attn.k_norm.weight",
            "model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            "model.layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            "model.layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, "
            "model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight, "
            f"fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
            "model.layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.gate_proj.weight, "
            "model.layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
        ]
        # Split fused parameters in their distributed layout before transposing each output.
        for layer_id in range(config.num_hidden_layers):
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"):
                key = f"model.layers.{layer_id}.{name}.weight"
                statements.append(f"{key}^T -> {key}")
        if config.attention_bias:
            statements += [
                "model.layers.$LAYER_ID.self_attn.o_proj.bias -> model.layers.$LAYER_ID.self_attn.o_proj.bias",
                "model.layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, "
                "model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, "
                f"fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]
        if config.get("mlp_bias", False):
            statements += [
                "model.layers.$LAYER_ID.mlp.down_proj.bias -> model.layers.$LAYER_ID.mlp.down_proj.bias",
                "model.layers.$LAYER_ID.mlp.up_gate_proj.bias -> model.layers.$LAYER_ID.mlp.gate_proj.bias, "
                "model.layers.$LAYER_ID.mlp.up_proj.bias, fused_ffn, axis=0",
            ]
        statements.append(
            "model.lm_head.weight -> _" if config.tie_word_embeddings else "model.lm_head.weight -> lm_head.weight"
        )
        return {"aoa_statements": statements}


_MASK_SLOTS = ("full_attention", "sliding_attention")


def _pack_attention_masks(masks, indices, layer_types):
    """Pack (dense mask, row indices) per type; bool [1] denotes an absent tensor."""
    return tuple(
        paddle.zeros([1], dtype="bool") if value is None else value.detach().clone()
        for kind in _MASK_SLOTS
        for value in ((masks.get(kind), indices.get(kind)) if kind in layer_types else (None, None))
    )


def _unpack_attention_masks(packed, kind):
    offset = 2 * _MASK_SLOTS.index(kind)
    return tuple(None if value.ndim == 1 else value for value in packed[offset : offset + 2])


def _update_attention_cache(cache, key, value, layer_idx, use_cache):
    if cache is not None:
        if use_cache:
            return cache.update(key, value, layer_idx)
        if layer_idx < len(cache.layers) and cache.layers[layer_idx].is_initialized:
            key = paddle.concat([cache.layers[layer_idx].keys, key], axis=-2)
            value = paddle.concat([cache.layers[layer_idx].values, value], axis=-2)
    return key, value


def _select_attention_backend(config, query, key, mask, indices, training, sliding_window, output_attentions=False):
    backend = config._attn_implementation
    if (output_attentions and backend != "eager") or (
        backend == "flashmask" and (mask is not None or (training and config.attention_dropout > 0))
    ):
        logger.warning_once(
            "OLMo3 is falling back to eager attention to preserve requested masks, dropout or attention "
            "weights. This can increase memory usage and runtime for long sequences."
        )
        backend = "eager"
        if mask is None and indices is None:
            query_positions = paddle.arange(key.shape[-2] - query.shape[-2], key.shape[-2]).unsqueeze(-1)
            key_positions = paddle.arange(key.shape[-2]).unsqueeze(0)
            allowed = key_positions <= query_positions
            if sliding_window is not None:
                allowed = allowed & (key_positions > query_positions - sliding_window)
            mask = paddle.where(allowed, 0.0, paddle.finfo(query.dtype).min).astype(query.dtype)
            mask = mask.unsqueeze(0).unsqueeze(0)
    return backend, mask


def _resolve_rope_type(config, rope_type=None):
    if rope_type is not None:
        return rope_type
    return config.rope_parameters.get("rope_type", config.rope_parameters.get("type", "default"))


def _compute_rope_parameters(config, rope_type=None):
    rope_type = _resolve_rope_type(config, rope_type)
    if rope_type == "default":
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, 1.0
    return ROPE_INIT_FUNCTIONS[rope_type](config)


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


def _get_fleet_aoa_config(config, include_dtype=True):
    statements = [
        "model.embed_tokens.weight -> model.embedding.embed_tokens.weight",
        "model.norm.weight -> model.norm.weight",
        "lm_head.weight -> model.lm_head.weight",
        "model.layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.self_attn_bda.norm.weight",
        "model.layers.$LAYER_ID.post_feedforward_layernorm.weight -> model.layers.$LAYER_ID.mlp_bda.norm.weight",
        "model.layers.$LAYER_ID.self_attn.q_norm.weight -> model.layers.$LAYER_ID.self_attn.q_norm.weight",
        "model.layers.$LAYER_ID.self_attn.k_norm.weight -> model.layers.$LAYER_ID.self_attn.k_norm.weight",
        "model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
        "model.layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
        "model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, "
        "model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> model.layers.$LAYER_ID.self_attn.qkv_proj.weight, "
        f"fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
        "model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> "
        "model.layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
    ]
    if config.tie_word_embeddings:
        statements.remove("lm_head.weight -> model.lm_head.weight")
        statements.append("model.embed_tokens.weight -> model.lm_head.weight")
    if config.attention_bias:
        statements += [
            "model.layers.$LAYER_ID.self_attn.o_proj.bias -> model.layers.$LAYER_ID.self_attn.o_proj.bias",
            "model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, "
            "model.layers.$LAYER_ID.self_attn.v_proj.bias -> model.layers.$LAYER_ID.self_attn.qkv_proj.bias, "
            f"fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
        ]
    if config.get("mlp_bias", False):
        statements += [
            "model.layers.$LAYER_ID.mlp.down_proj.bias -> model.layers.$LAYER_ID.mlp.down_proj.bias",
            "model.layers.$LAYER_ID.mlp.gate_proj.bias, model.layers.$LAYER_ID.mlp.up_proj.bias -> "
            "model.layers.$LAYER_ID.mlp.up_gate_proj.bias, fused_ffn, axis=0",
        ]
    dtype = getattr(config, "params_dtype", None) or getattr(config, "dtype", None)
    if include_dtype and dtype is not None:
        dtype = str(dtype).removeprefix("paddle.")
        targets = []
        for statement in statements:
            target = statement.split("->", 1)[1].split(",", 1)[0].strip()
            if "$LAYER_ID" in target:
                targets.extend(target.replace("$LAYER_ID", str(index)) for index in range(config.num_hidden_layers))
            else:
                targets.append(target)
        statements += [f"{target} -> {target}, dtype='{dtype}'" for target in targets]
    return {"aoa_statements": statements}


__all__ = [
    "Olmo3ModelProvider",
    "Olmo3Model",
    "Olmo3PretrainedModel",
    "Olmo3ForCausalLM",
    "Olmo3ForCausalLMPipe",
]
