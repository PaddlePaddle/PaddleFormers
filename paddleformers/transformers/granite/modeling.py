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

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple, Union

import paddle
from paddle import nn
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp
from paddle.nn.functional.flash_attention import flashmask_attention

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP
from ...nn.pp_model import EmbeddingPipe, GeneralModelForCausalLMPipe, LMHeadPipe
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import GraniteConfig


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    position_ids: Optional[paddle.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class GraniteRMSNorm(nn.Layer):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class GraniteRotaryEmbedding(nn.Layer):
    def __init__(self, config: GraniteConfig):
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
        config: Optional[GraniteConfig] = None,
        seq_len: Optional[int] = None,
        device: str = "cpu",
    ) -> tuple[paddle.Tensor, float]:
        del seq_len
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype("float32").to(device) / dim))
        return inv_freq, 1.0

    @dynamic_rope_update
    def forward(self, x: paddle.Tensor, position_ids: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].astype("float32").expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].astype("float32")
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(perm=[0, 2, 1])
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


class GraniteAttention(nn.Layer):
    def __init__(self, config: GraniteConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if config.tensor_model_parallel_size > 1:
            self.num_heads = self.num_heads // config.tensor_model_parallel_size
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.attention_multiplier
        self.attention_dropout = config.attention_dropout

        q_hidden_size = config.num_attention_heads * self.head_dim
        kv_hidden_size = config.num_key_value_heads * self.head_dim
        self.q_proj = GeneralLinear.create(
            config.hidden_size,
            q_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
            gather_output=True,
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
            gather_output=True,
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
            gather_output=True,
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            config.hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="rowwise",
            input_is_parallel=False,
        )

    def _flashmask_attention_forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        attn_mask_startend_row_indices: Optional[paddle.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
    ) -> Tuple[paddle.Tensor, None]:
        # Keep Granite's custom attention parameters on the flashmask path.
        # flashmask_attention (FA2) always applies head_dim^-0.5 internally and does not
        # expose a softmax_scale parameter. To achieve the desired effective scale we
        # pre-multiply query by (scaling / head_dim^-0.5) so that:
        #   effective_scale = (scaling / head_dim^-0.5) * head_dim^-0.5 = scaling
        if scaling is not None:
            head_dim = query.shape[-1]
            default_fa_scale = head_dim**-0.5
            query = query * (scaling / default_fa_scale)
        if self.num_key_value_groups > 1:
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.ndim == 3:
            attn_mask_startend_row_indices = attn_mask_startend_row_indices.unsqueeze(-1)

        is_causal = query.shape[1] > 1
        if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.shape[-1] == 1:
            is_causal = True
        if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.shape[-1] == 4:
            is_causal = False

        attn_output = flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=attn_mask_startend_row_indices,
            dropout=dropout,
            training=self.training,
            causal=is_causal,
        )
        attn_output = paddle.reshape(
            x=attn_output,
            shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]],
        )
        return attn_output, None

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Cache]]:
        del use_cache, kwargs
        if self.config.sequence_parallel:
            seq_len = self.config.max_sequence_length
            batch_size = hidden_states.shape[0] * self.config.tensor_model_parallel_size // seq_len
            input_shape = (batch_size, seq_len)
        else:
            input_shape = hidden_states.shape[:-1]

        query_states = (
            self.q_proj(hidden_states).reshape([*input_shape, self.num_heads, self.head_dim]).transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states).reshape([*input_shape, self.num_key_value_heads, self.head_dim]).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).reshape([*input_shape, self.num_key_value_heads, self.head_dim]).transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        dropout = 0.0 if not self.training else self.attention_dropout
        if self.config._attn_implementation == "flashmask":
            attn_output, attn_weights = self._flashmask_attention_forward(
                query=query_states,
                key=key_states,
                value=value_states,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                dropout=dropout,
                scaling=self.scaling,
            )
        else:
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attention_interface(
                self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                dropout=dropout,
                scaling=self.scaling,
            )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_values


class GraniteDecoderLayer(nn.Layer):
    def __init__(self, config: GraniteConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GraniteAttention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config, has_bias=config.mlp_bias)
        self.input_layernorm = GraniteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GraniteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> Union[Tuple[paddle.Tensor], paddle.Tensor]:
        del position_ids
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


class GranitePretrainedModel(PretrainedModel):
    config_class = GraniteConfig
    base_model_prefix = "model"
    _no_split_modules = ["GraniteDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: GraniteConfig, is_split=True):
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        actions = {
            "embed_tokens.weight": partial(fn, is_column=False),
            "lm_head.weight": partial(fn, is_column=False),
        }

        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = f"layers.{layer_idx}"
            actions[f"{layer_prefix}.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            actions[f"{layer_prefix}.self_attn.k_proj.weight"] = partial(fn, is_column=True)
            actions[f"{layer_prefix}.self_attn.v_proj.weight"] = partial(fn, is_column=True)
            actions[f"{layer_prefix}.self_attn.o_proj.weight"] = partial(fn, is_column=False)
            actions[f"{layer_prefix}.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            actions[f"{layer_prefix}.mlp.up_proj.weight"] = partial(fn, is_column=True)
            actions[f"{layer_prefix}.mlp.down_proj.weight"] = partial(fn, is_column=False)

            if config.attention_bias:
                actions[f"{layer_prefix}.self_attn.q_proj.bias"] = partial(fn, is_column=True)
                actions[f"{layer_prefix}.self_attn.k_proj.bias"] = partial(fn, is_column=True)
                actions[f"{layer_prefix}.self_attn.v_proj.bias"] = partial(fn, is_column=True)
                actions[f"{layer_prefix}.self_attn.o_proj.bias"] = partial(fn, is_column=False)

            if config.mlp_bias:
                actions[f"{layer_prefix}.mlp.gate_proj.bias"] = partial(fn, is_column=True)
                actions[f"{layer_prefix}.mlp.up_proj.bias"] = partial(fn, is_column=True)
                actions[f"{layer_prefix}.mlp.down_proj.bias"] = partial(fn, is_column=False)

        return actions

    @classmethod
    def _gen_aoa_config(cls, config: GraniteConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
        ]
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.mlp.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )
        if config.attention_bias:
            aoa_statements.extend(
                [
                    f"model.layers.$LAYER_ID.self_attn.{proj_name}.bias -> {model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.bias"
                    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
                ]
            )
        if config.mlp_bias:
            aoa_statements.extend(
                [
                    f"model.layers.$LAYER_ID.mlp.{proj_name}.bias -> {model_prefix}layers.$LAYER_ID.mlp.{proj_name}.bias"
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
    def _gen_inv_aoa_config(cls, config: GraniteConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
        ]
        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> model.layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )
        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight^T -> model.layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )
        if config.attention_bias:
            aoa_statements.extend(
                [
                    f"{model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.bias -> model.layers.$LAYER_ID.self_attn.{proj_name}.bias"
                    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
                ]
            )
        if config.mlp_bias:
            aoa_statements.extend(
                [
                    f"{model_prefix}layers.$LAYER_ID.mlp.{proj_name}.bias -> model.layers.$LAYER_ID.mlp.{proj_name}.bias"
                    for proj_name in ["gate_proj", "up_proj", "down_proj"]
                ]
            )
        if not config.tie_word_embeddings and cls != cls.base_model_class:
            aoa_statements.append("lm_head.weight -> lm_head.weight")
        return {"aoa_statements": aoa_statements}


@register_base_model
class GraniteModel(GranitePretrainedModel):
    def __init__(self, config: GraniteConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.LayerList(
            [GraniteDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GraniteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = GraniteRotaryEmbedding(config=config)
        self.embedding_multiplier = config.embedding_multiplier

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")

        if inputs_embeds is None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = paddle.arange(cache_length, cache_length + seq_length, dtype="int64").unsqueeze(0)
            position_ids = position_ids.expand([batch_size, seq_length])

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
        causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        all_hidden_states = [] if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            all_hidden_states = tuple(all_hidden_states)

        if not return_dict:
            outputs = [hidden_states]
            if past_key_values is not None:
                outputs.append(past_key_values)
            if output_hidden_states:
                outputs.append(all_hidden_states)
            return tuple(outputs)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class GraniteForCausalLM(GranitePretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GraniteConfig):
        super().__init__(config)
        self.model = GraniteModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        del kwargs
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states) / self.config.logits_scaling

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
            attentions=outputs.attentions,
        )


def _scaled_embedding(config: GraniteConfig):
    embed_tokens = GeneralEmbedding.create(
        config=config,
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        padding_idx=config.pad_token_id,
    )
    forward = embed_tokens.forward

    def scaled_forward(input_ids: paddle.Tensor):
        return forward(input_ids) * config.embedding_multiplier

    embed_tokens.forward = scaled_forward
    embed_tokens._granite_scales_embeddings = True
    return embed_tokens


class GraniteEmbeddingPipe(EmbeddingPipe):
    def forward(self, args):
        outputs = super().forward(args)
        if getattr(self.embed_tokens, "_granite_scales_embeddings", False):
            return outputs
        if isinstance(outputs, tuple):
            return (outputs[0] * self.config.embedding_multiplier,) + outputs[1:]
        return outputs * self.config.embedding_multiplier


class GraniteLMHeadPipe(LMHeadPipe):
    def forward(self, args):
        logits = super().forward(args)
        return logits / self.config.logits_scaling


class GraniteForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = GraniteConfig
    _decoder_layer_cls = GraniteDecoderLayer
    _rotary_emb_cls = GraniteRotaryEmbedding
    _embed_cls = _scaled_embedding
    _embedding_pipe_cls = GraniteEmbeddingPipe
    _lmhead_pipe_cls = GraniteLMHeadPipe
    _get_tensor_parallel_mappings = GraniteModel._get_tensor_parallel_mappings
    _init_weights = GraniteModel._init_weights
    _keep_in_fp32_modules = GraniteModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = GraniteModel.transpose_weight_keys
    _gen_aoa_config = GraniteForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = GraniteForCausalLM._gen_inv_aoa_config
