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

from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn

from ...generation import GenerationMixin
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ..activations import ACT2FN
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import T5GemmaConfig, T5GemmaModuleConfig


class T5GemmaRMSNorm(nn.Layer):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(0.0),
        )

    def _norm(self, x):
        with paddle.amp.auto_cast(enable=False):
            return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.astype(x.dtype)


class T5GemmaMLP(nn.Layer):
    def __init__(self, config: T5GemmaModuleConfig):
        super().__init__()
        self.gate_proj = GeneralLinear.create(
            config.hidden_size,
            config.intermediate_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.up_proj = GeneralLinear.create(
            config.hidden_size,
            config.intermediate_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.down_proj = GeneralLinear.create(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )
        self.act_fn = ACT2FN[config.hidden_activation]
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        return self.down_proj(hidden_states)


class T5GemmaRotaryEmbedding(nn.Layer):
    def __init__(self, config: T5GemmaModuleConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        rope_parameters = self.config.rope_parameters
        self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        rope_init_fn = (
            self.compute_default_rope_parameters
            if self.rope_type == "default"
            else ROPE_INIT_FUNCTIONS[self.rope_type]
        )
        inv_freq, self.attention_scaling = rope_init_fn(self.config)
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[T5GemmaModuleConfig] = None,
        seq_len: Optional[int] = None,
    ) -> tuple[paddle.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(paddle.float32) / dim))
        return inv_freq, 1.0

    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


def t5gemma_eager_attention_forward(
    module: nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
):
    if scaling is None:
        scaling = module.head_dim**-0.5

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = paddle.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, axis=-1, dtype=paddle.float32).astype(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = paddle.matmul(attn_weights, value)
    attn_output = attn_output.transpose([0, 2, 1, 3]).contiguous()
    attn_output = paddle.reshape(x=attn_output, shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]])
    return attn_output, attn_weights


class T5GemmaSelfAttention(nn.Layer):
    def __init__(self, config: T5GemmaModuleConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = config.is_decoder
        self.attn_implementation = config._attn_implementation

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        q_hidden_size = config.num_attention_heads * self.head_dim
        kv_hidden_size = config.num_key_value_heads * self.head_dim
        self.q_proj = GeneralLinear.create(
            config.hidden_size, q_hidden_size, has_bias=config.attention_bias, config=config, tp_plan="colwise"
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size, kv_hidden_size, has_bias=config.attention_bias, config=config, tp_plan="colwise"
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size, kv_hidden_size, has_bias=config.attention_bias, config=config, tp_plan="colwise"
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size, config.hidden_size, has_bias=config.attention_bias, config=config, tp_plan="rowwise"
        )
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states).reshape([*input_shape, -1, self.head_dim]).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape([*input_shape, -1, self.head_dim]).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape([*input_shape, -1, self.head_dim]).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = (
            t5gemma_eager_attention_forward
            if self.attn_implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        )
        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
        )

        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class T5GemmaCrossAttention(nn.Layer):
    def __init__(self, config: T5GemmaModuleConfig, layer_idx: int):
        super().__init__()
        if config.cross_attention_hidden_size is None:
            raise ValueError("Cross-attention needs cross_attention_hidden_size to be specified.")

        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.attn_implementation = config._attn_implementation

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        q_hidden_size = config.num_attention_heads * self.head_dim
        kv_hidden_size = config.num_key_value_heads * self.head_dim
        self.q_proj = GeneralLinear.create(
            config.hidden_size, q_hidden_size, has_bias=config.attention_bias, config=config, tp_plan="colwise"
        )
        self.k_proj = GeneralLinear.create(
            config.cross_attention_hidden_size,
            kv_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            config.cross_attention_hidden_size,
            kv_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size, config.hidden_size, has_bias=config.attention_bias, config=config, tp_plan="rowwise"
        )
        self.attn_logit_softcapping = config.attn_logit_softcapping

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor],
        encoder_hidden_states: Optional[paddle.Tensor],
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
    ) -> tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        if encoder_hidden_states is None:
            raise ValueError("Encoder hidden state is required for cross attention.")

        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states).reshape([*input_shape, -1, self.head_dim]).transpose(1, 2)

        if past_key_values is not None and len(past_key_values.layers) > self.layer_idx:
            layer_cache = past_key_values.layers[self.layer_idx]
            if layer_cache.is_initialized:
                key_states, value_states = layer_cache.keys, layer_cache.values
            else:
                key_states, value_states = self._project_encoder_states(encoder_hidden_states)
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        else:
            key_states, value_states = self._project_encoder_states(encoder_hidden_states)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = (
            t5gemma_eager_attention_forward
            if self.attn_implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        )
        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=None,
            softcap=self.attn_logit_softcapping,
        )

        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

    def _project_encoder_states(self, encoder_hidden_states):
        encoder_input_shape = encoder_hidden_states.shape[:-1]
        key_states = (
            self.k_proj(encoder_hidden_states).reshape([*encoder_input_shape, -1, self.head_dim]).transpose(1, 2)
        )
        value_states = (
            self.v_proj(encoder_hidden_states).reshape([*encoder_input_shape, -1, self.head_dim]).transpose(1, 2)
        )
        return key_states, value_states


class T5GemmaEncoderLayer(nn.Layer):
    def __init__(self, config: T5GemmaModuleConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = T5GemmaSelfAttention(config=config, layer_idx=layer_idx)
        self.pre_self_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = T5GemmaMLP(config)
        self.pre_feedforward_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return (hidden_states, self_attn_weights) if output_attentions else hidden_states


class T5GemmaDecoderLayer(T5GemmaEncoderLayer):
    def __init__(self, config: T5GemmaModuleConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.cross_attn = T5GemmaCrossAttention(config=config, layer_idx=layer_idx)
        self.pre_cross_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cross_past_key_values: Optional[Cache] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_cross_attn_layernorm(hidden_states)
        hidden_states, cross_attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            past_key_values=cross_past_key_values if use_cache else None,
            output_attentions=output_attentions,
        )
        hidden_states = self.post_cross_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        if output_attentions:
            return hidden_states, self_attn_weights, cross_attn_weights
        return hidden_states


class T5GemmaClassificationHead(nn.Layer):
    def __init__(self, hidden_size: int, num_labels: int, classifier_dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=classifier_dropout_rate)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        return self.out_proj(self.dropout(hidden_states))


class T5GemmaPreTrainedModel(PretrainedModel):
    config_class = T5GemmaConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"rotary_emb.inv_freq"]
    _no_split_modules = ["T5GemmaEncoderLayer", "T5GemmaDecoderLayer"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "out_proj",
    ]

    def _init_weights(self, layer):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(paddle.normal(mean=0.0, std=std, shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(paddle.normal(mean=0.0, std=std, shape=layer.weight.shape))
        elif isinstance(layer, T5GemmaRMSNorm):
            layer.weight.set_value(paddle.zeros_like(layer.weight))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder.bos_token_id
        pad_token_id = self.config.decoder.pad_token_id
        if decoder_start_token_id is None:
            raise ValueError("self.model.config.decoder.bos_token_id has to be defined.")
        if pad_token_id is None:
            raise ValueError("self.model.config.decoder.pad_token_id has to be defined.")

        shifted_input_ids = paddle.zeros_like(input_ids)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        shifted_input_ids = paddle.where(
            shifted_input_ids == -100,
            paddle.full_like(shifted_input_ids, pad_token_id),
            shifted_input_ids,
        )
        return shifted_input_ids

    @staticmethod
    def _prepare_decoder_attention_mask(
        attention_mask,
        input_shape,
        past_key_values_length,
        dtype,
        sliding_window_size=None,
        or_mask_function=None,
    ):
        batch_size, seq_length = input_shape
        target_length = seq_length + past_key_values_length
        min_dtype = paddle.finfo(dtype).min

        query_pos = paddle.arange(past_key_values_length, target_length, dtype="int64").reshape([1, seq_length, 1])
        key_pos = paddle.arange(0, target_length, dtype="int64").reshape([1, 1, target_length])
        mask = key_pos > query_pos

        if sliding_window_size is not None:
            mask = paddle.logical_or(mask, key_pos <= (query_pos - sliding_window_size))

        if or_mask_function is not None:
            batch_idx = paddle.arange(batch_size, dtype="int64").reshape([batch_size, 1, 1])
            q_idx = paddle.arange(seq_length, dtype="int64").reshape([1, seq_length, 1])
            kv_idx = paddle.arange(target_length, dtype="int64").reshape([1, 1, target_length])
            extra_keep = or_mask_function(batch_idx, None, q_idx, kv_idx)
            mask = paddle.logical_and(mask, paddle.logical_not(extra_keep))

        if attention_mask is not None:
            padding_mask = attention_mask[:, None, :] == 0
            mask = paddle.logical_or(mask, padding_mask)

        mask = mask.astype(dtype) * min_dtype
        return mask[:, None, :, :]

    @classmethod
    def _gen_aoa_config(cls, config: T5GemmaConfig):
        statements = []
        for module_name in ["encoder", "decoder"]:
            module_config = getattr(config, module_name)
            statements.extend(
                [
                    f"model.{module_name}.embed_tokens.weight -> model.{module_name}.embed_tokens.weight",
                    f"model.{module_name}.norm.weight -> model.{module_name}.norm.weight",
                    f"model.{module_name}.layers.$LAYER_ID.pre_self_attn_layernorm.weight -> model.{module_name}.layers.$LAYER_ID.pre_self_attn_layernorm.weight",
                    f"model.{module_name}.layers.$LAYER_ID.post_self_attn_layernorm.weight -> model.{module_name}.layers.$LAYER_ID.post_self_attn_layernorm.weight",
                    f"model.{module_name}.layers.$LAYER_ID.pre_feedforward_layernorm.weight -> model.{module_name}.layers.$LAYER_ID.pre_feedforward_layernorm.weight",
                    f"model.{module_name}.layers.$LAYER_ID.post_feedforward_layernorm.weight -> model.{module_name}.layers.$LAYER_ID.post_feedforward_layernorm.weight",
                    f"model.{module_name}.layers.$LAYER_ID.self_attn.q_proj.weight^T -> model.{module_name}.layers.$LAYER_ID.self_attn.q_proj.weight",
                    f"model.{module_name}.layers.$LAYER_ID.self_attn.k_proj.weight^T -> model.{module_name}.layers.$LAYER_ID.self_attn.k_proj.weight",
                    f"model.{module_name}.layers.$LAYER_ID.self_attn.v_proj.weight^T -> model.{module_name}.layers.$LAYER_ID.self_attn.v_proj.weight",
                    f"model.{module_name}.layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.{module_name}.layers.$LAYER_ID.self_attn.o_proj.weight",
                    f"model.{module_name}.layers.$LAYER_ID.mlp.gate_proj.weight^T -> model.{module_name}.layers.$LAYER_ID.mlp.gate_proj.weight",
                    f"model.{module_name}.layers.$LAYER_ID.mlp.up_proj.weight^T -> model.{module_name}.layers.$LAYER_ID.mlp.up_proj.weight",
                    f"model.{module_name}.layers.$LAYER_ID.mlp.down_proj.weight^T -> model.{module_name}.layers.$LAYER_ID.mlp.down_proj.weight",
                ]
            )
            if module_config.attention_bias:
                statements.extend(
                    [
                        f"model.{module_name}.layers.$LAYER_ID.self_attn.q_proj.bias -> model.{module_name}.layers.$LAYER_ID.self_attn.q_proj.bias",
                        f"model.{module_name}.layers.$LAYER_ID.self_attn.k_proj.bias -> model.{module_name}.layers.$LAYER_ID.self_attn.k_proj.bias",
                        f"model.{module_name}.layers.$LAYER_ID.self_attn.v_proj.bias -> model.{module_name}.layers.$LAYER_ID.self_attn.v_proj.bias",
                        f"model.{module_name}.layers.$LAYER_ID.self_attn.o_proj.bias -> model.{module_name}.layers.$LAYER_ID.self_attn.o_proj.bias",
                    ]
                )

        statements.extend(
            [
                "model.decoder.embed_tokens.weight -> lm_head.weight",
                "model.decoder.layers.$LAYER_ID.pre_cross_attn_layernorm.weight -> model.decoder.layers.$LAYER_ID.pre_cross_attn_layernorm.weight",
                "model.decoder.layers.$LAYER_ID.post_cross_attn_layernorm.weight -> model.decoder.layers.$LAYER_ID.post_cross_attn_layernorm.weight",
                "model.decoder.layers.$LAYER_ID.cross_attn.q_proj.weight^T -> model.decoder.layers.$LAYER_ID.cross_attn.q_proj.weight",
                "model.decoder.layers.$LAYER_ID.cross_attn.k_proj.weight^T -> model.decoder.layers.$LAYER_ID.cross_attn.k_proj.weight",
                "model.decoder.layers.$LAYER_ID.cross_attn.v_proj.weight^T -> model.decoder.layers.$LAYER_ID.cross_attn.v_proj.weight",
                "model.decoder.layers.$LAYER_ID.cross_attn.o_proj.weight^T -> model.decoder.layers.$LAYER_ID.cross_attn.o_proj.weight",
            ]
        )
        if config.decoder.attention_bias:
            statements.extend(
                [
                    "model.decoder.layers.$LAYER_ID.cross_attn.q_proj.bias -> model.decoder.layers.$LAYER_ID.cross_attn.q_proj.bias",
                    "model.decoder.layers.$LAYER_ID.cross_attn.k_proj.bias -> model.decoder.layers.$LAYER_ID.cross_attn.k_proj.bias",
                    "model.decoder.layers.$LAYER_ID.cross_attn.v_proj.bias -> model.decoder.layers.$LAYER_ID.cross_attn.v_proj.bias",
                    "model.decoder.layers.$LAYER_ID.cross_attn.o_proj.bias -> model.decoder.layers.$LAYER_ID.cross_attn.o_proj.bias",
                ]
            )
        return {"aoa_statements": statements}


def make_default_2d_attention_mask(token_ids, hidden_states, pad_token_id):
    if token_ids is not None:
        if pad_token_id is None:
            raise ValueError("`pad_token_id` is required for padding information.")
        return (token_ids != pad_token_id).astype("int64")
    return paddle.ones((hidden_states.shape[0], hidden_states.shape[1]), dtype="int64")


class T5GemmaEncoder(T5GemmaPreTrainedModel):
    config_class = T5GemmaModuleConfig

    def __init__(self, config: T5GemmaModuleConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = T5GemmaRotaryEmbedding(config=config)
        self.layers = nn.LayerList([T5GemmaEncoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(config.dropout_rate)
        self.has_sliding_layers = "sliding_attention" in getattr(config, "layer_types", [])

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify either input_ids or inputs_embeds.")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]
        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand([batch_size, seq_length])
        if attention_mask is None:
            attention_mask = make_default_2d_attention_mask(input_ids, inputs_embeds, self.config.pad_token_id)

        mask_mapping = {
            "full_attention": self._prepare_bidirectional_attention_mask(
                attention_mask, seq_length, inputs_embeds.dtype
            )
        }
        if self.has_sliding_layers:
            mask_mapping["sliding_attention"] = self._prepare_bidirectional_attention_mask(
                attention_mask,
                seq_length,
                inputs_embeds.dtype,
                sliding_window_size=self.config.sliding_window,
            )

        hidden_states = inputs_embeds * paddle.to_tensor(self.config.hidden_size**0.5, dtype=inputs_embeds.dtype)
        hidden_states = self.dropout(hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer_module in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                mask_mapping[layer_module.attention_type],
                output_attentions=output_attentions,
            )
            if output_attentions:
                hidden_states, attn_weights = layer_outputs
                all_attns += (attn_weights,)
            else:
                hidden_states = layer_outputs

        hidden_states = self.dropout(self.norm(hidden_states))
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attns] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attns)

    @staticmethod
    def _prepare_bidirectional_attention_mask(attention_mask, query_length, dtype, sliding_window_size=None):
        min_dtype = paddle.finfo(dtype).min
        key_length = attention_mask.shape[-1]
        mask = attention_mask[:, None, None, :] == 0
        if sliding_window_size is not None:
            q_idx = paddle.arange(query_length, dtype="int64").reshape([1, query_length, 1])
            kv_idx = paddle.arange(key_length, dtype="int64").reshape([1, 1, key_length])
            sliding_mask = paddle.logical_or(
                kv_idx <= q_idx - sliding_window_size, kv_idx >= q_idx + sliding_window_size
            )
            mask = paddle.logical_or(mask, sliding_mask)
        mask = mask.astype(dtype) * min_dtype
        return mask.expand([attention_mask.shape[0], 1, query_length, key_length])


class T5GemmaDecoder(T5GemmaEncoder):
    def __init__(self, config: T5GemmaModuleConfig):
        super().__init__(config)
        self.layers = nn.LayerList([T5GemmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cross_past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if encoder_hidden_states is None:
            raise ValueError("`encoder_hidden_states` must be given in decoder.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You must specify exactly one of decoder_input_ids or decoder_inputs_embeds.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify either decoder_input_ids or decoder_inputs_embeds.")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]
        if not self.training and use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if not self.training and use_cache and cross_past_key_values is None:
            cross_past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = paddle.arange(cache_length, cache_length + seq_length, dtype="int64").expand(
                [batch_size, seq_length]
            )

        if attention_mask is None and past_key_values is None:
            attention_mask = make_default_2d_attention_mask(input_ids, inputs_embeds, self.config.pad_token_id)

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": cache_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": None,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        full_mask, _ = create_causal_mask_and_row_indices(**mask_kwargs)
        mask_mapping = {"full_attention": full_mask}
        if self.has_sliding_layers:
            sliding_mask, _ = create_sliding_window_causal_mask_and_row_indices(**mask_kwargs)
            mask_mapping["sliding_attention"] = sliding_mask

        encoder_seq_length = encoder_hidden_states.shape[1]
        if encoder_attention_mask is None:
            encoder_attention_mask = paddle.ones([batch_size, encoder_seq_length], dtype="int64")
        cross_mask = self._prepare_cross_attention_mask(encoder_attention_mask, seq_length, inputs_embeds.dtype)

        hidden_states = inputs_embeds * paddle.to_tensor(self.config.hidden_size**0.5, dtype=inputs_embeds.dtype)
        hidden_states = self.dropout(hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        for layer_module in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                mask_mapping[layer_module.attention_type],
                past_key_values=past_key_values if use_cache else None,
                cross_past_key_values=cross_past_key_values if use_cache else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_mask,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            if output_attentions:
                hidden_states, self_attn, cross_attn = layer_outputs
                all_self_attns += (self_attn,)
                all_cross_attns += (cross_attn,)
            else:
                hidden_states = layer_outputs

        hidden_states = self.dropout(self.norm(hidden_states))
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns, all_cross_attns]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attns,
        )

    @staticmethod
    def _prepare_cross_attention_mask(attention_mask, query_length, dtype):
        min_dtype = paddle.finfo(dtype).min
        mask = (attention_mask[:, None, None, :] == 0).astype(dtype) * min_dtype
        return mask.expand([attention_mask.shape[0], 1, query_length, attention_mask.shape[1]])


@register_base_model
class T5GemmaModel(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig):
        if not config.is_encoder_decoder:
            raise ValueError("T5GemmaModel only supports encoder-decoder modeling.")
        super().__init__(config)
        self.encoder = T5GemmaEncoder(config.encoder)
        self.decoder = T5GemmaDecoder(config.decoder)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        decoder_position_ids: Optional[paddle.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[Cache] = None,
        cross_past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        decoder_inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            cross_past_key_values=cross_past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not return_dict:
            return (
                decoder_outputs.last_hidden_state,
                decoder_outputs.past_key_values,
                decoder_outputs.hidden_states,
                decoder_outputs.attentions,
                decoder_outputs.cross_attentions,
                encoder_outputs.last_hidden_state,
                encoder_outputs.hidden_states,
                encoder_outputs.attentions,
            )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class T5GemmaEncoderModel(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig):
        if config.is_encoder_decoder:
            raise ValueError("T5GemmaEncoderModel only supports encoder-only model.")
        super().__init__(config)
        self.encoder = T5GemmaEncoder(config.encoder)

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class T5GemmaForConditionalGeneration(T5GemmaPreTrainedModel, GenerationMixin):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: T5GemmaConfig):
        config.is_encoder_decoder = True
        super().__init__(config)
        self.model = T5GemmaModel(config)
        self.vocab_size = config.decoder.vocab_size
        lm_head_config = T5GemmaModuleConfig(**config.decoder.to_dict())
        lm_head_config.vocab_size = self.vocab_size
        self.lm_head = GeneralLMHead(lm_head_config)
        self.criterion = CriterionLayer(lm_head_config)
        self.tie_weights()

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def prepare_decoder_input_ids_from_labels(self, labels):
        return self._shift_right(labels)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        decoder_position_ids: Optional[paddle.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[Cache] = None,
        cross_past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        decoder_inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            if getattr(self.config, "is_encoder_decoder", False):
                labels = paddle.concat([paddle.full_like(labels[:, :1], -100), labels[:, :-1]], axis=1)
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            cross_past_key_values=cross_past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = decoder_outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif not isinstance(logits_to_keep, int):
            hidden_states = hidden_states[:, logits_to_keep, :]

        logits = self.lm_head(hidden_states)
        decoder_config = self.get_decoder().config
        if decoder_config.final_logit_softcapping is not None:
            logits = logits / decoder_config.final_logit_softcapping
            logits = paddle.tanh(logits)
            logits = logits * decoder_config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        if not return_dict:
            output = (
                logits,
                decoder_outputs.past_key_values,
                decoder_outputs.decoder_hidden_states,
                decoder_outputs.decoder_attentions,
                decoder_outputs.cross_attentions,
                decoder_outputs.encoder_last_hidden_state,
                decoder_outputs.encoder_hidden_states,
                decoder_outputs.encoder_attentions,
            )
            return (loss,) + output if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.decoder_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=decoder_outputs.encoder_last_hidden_state,
            encoder_hidden_states=decoder_outputs.encoder_hidden_states,
            encoder_attentions=decoder_outputs.encoder_attentions,
        )


class T5GemmaForSequenceClassification(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig, is_encoder_decoder: Optional[bool] = None):
        if is_encoder_decoder is not None:
            config.is_encoder_decoder = is_encoder_decoder
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = T5GemmaModel(config) if config.is_encoder_decoder else T5GemmaEncoderModel(config)
        hidden_size = config.decoder.hidden_size if config.is_encoder_decoder else config.encoder.hidden_size
        classifier_dropout = getattr(config, "classifier_dropout_rate", 0.1)
        self.score = T5GemmaClassificationHead(hidden_size, self.num_labels, classifier_dropout)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        decoder_position_ids: Optional[paddle.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        decoder_inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_encoder_decoder and decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` is required when decoder inputs are not provided.")
            decoder_input_ids = self._shift_right(input_ids)

        if self.config.is_encoder_decoder:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                encoder_outputs=encoder_outputs,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=False,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.decoder_hidden_states
            attentions = outputs.decoder_attentions
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions

        logits = self.score(last_hidden_state)
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None or input_ids is None:
            last_non_pad_token = paddle.full([batch_size], logits.shape[1] - 1, dtype="int64")
        else:
            non_pad_mask = (input_ids != self.config.pad_token_id).astype("int64")
            token_indices = paddle.arange(input_ids.shape[-1], dtype="int64")
            last_non_pad_token = (token_indices * non_pad_mask).argmax(axis=-1)
            if self.config.is_encoder_decoder:
                last_non_pad_token = paddle.clip(last_non_pad_token + 1, max=decoder_input_ids.shape[-1] - 1)

        pooled_logits = logits[paddle.arange(batch_size), last_non_pad_token]
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.functional.mse_loss(pooled_logits.squeeze(), labels.squeeze())
            else:
                loss = nn.functional.cross_entropy(pooled_logits.reshape([-1, self.num_labels]), labels.reshape([-1]))

        if not return_dict:
            output = (pooled_logits, hidden_states, attentions)
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss, logits=pooled_logits, hidden_states=hidden_states, attentions=attentions
        )


class T5GemmaForTokenClassification(T5GemmaForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        decoder_position_ids: Optional[paddle.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        decoder_inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_encoder_decoder and decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` is required when decoder inputs are not provided.")
            decoder_input_ids = self._shift_right(input_ids)

        if self.config.is_encoder_decoder:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                encoder_outputs=encoder_outputs,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=False,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.decoder_hidden_states
            attentions = outputs.decoder_attentions
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions

        logits = self.score(last_hidden_state)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))

        if not return_dict:
            output = (logits, hidden_states, attentions)
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)


__all__ = [
    "T5GemmaPreTrainedModel",
    "T5GemmaEncoderModel",
    "T5GemmaModel",
    "T5GemmaForConditionalGeneration",
    "T5GemmaForSequenceClassification",
    "T5GemmaForTokenClassification",
]
