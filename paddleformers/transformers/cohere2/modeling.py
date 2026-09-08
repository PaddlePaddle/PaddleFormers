# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Optional, Tuple, Union

import paddle
from paddle import nn
from paddle.distributed.fleet.recompute.recompute import recompute

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.lm_head import LMHead as GeneralLMHead
from ..activations import ACT2FN
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Cohere2Config


class Cohere2LayerNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = self.create_parameter(shape=hidden_size, default_initializer=nn.initializer.Constant(1.0))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        mean = hidden_states.mean(axis=-1, keepdim=True)
        variance = paddle.square(hidden_states - mean).mean(axis=-1, keepdim=True)
        hidden_states = (hidden_states - mean) * paddle.rsqrt(variance + self.variance_epsilon)
        return (self.weight.astype("float32") * hidden_states).astype(input_dtype)


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return paddle.stack([-x2, x1], axis=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q.astype("float32") * cos) + (rotate_half(q.astype("float32")) * sin)
    k_embed = (k.astype("float32") * cos) + (rotate_half(k.astype("float32")) * sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


class Cohere2RotaryEmbedding(nn.Layer):
    def __init__(self, config: Cohere2Config):
        super().__init__()
        self.config = config
        self.rope_theta = config.rope_theta
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (self.rope_theta ** (paddle.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.attention_scaling = 1.0

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].astype("float32").expand([position_ids.shape[0], -1, 1])
        position_ids = position_ids[:, None, :].astype("float32")
        freqs = paddle.matmul(inv_freq, position_ids).transpose([0, 2, 1])
        emb = paddle.repeat_interleave(freqs, repeats=2, axis=-1)
        cos = paddle.cos(emb) * self.attention_scaling
        sin = paddle.sin(emb) * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


class Cohere2Attention(nn.Layer):
    def __init__(self, config: Cohere2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None
        self.is_causal = True
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=config.attention_bias)
        if config.use_qk_norm:
            self.q_norm = Cohere2LayerNorm([self.head_dim], eps=config.layer_norm_eps)
            self.k_norm = Cohere2LayerNorm([self.head_dim], eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim])
        key_states = self.k_proj(hidden_states).reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states = self.v_proj(hidden_states).reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])
        if self.config.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])
        cos, sin = position_embeddings
        if self.sliding_window is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        if self.config._attn_implementation != "sdpa":
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
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None


class Cohere2MLP(nn.Layer):
    def __init__(self, config: Cohere2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Cohere2DecoderLayer(nn.Layer):
    def __init__(self, config: Cohere2Config, layer_idx: int):
        super().__init__()
        self.self_attn = Cohere2Attention(config, layer_idx)
        self.mlp = Cohere2MLP(config)
        self.input_layernorm = Cohere2LayerNorm([config.hidden_size], eps=config.layer_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states_attention, self_attn_weights = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states_attention + self.mlp(hidden_states)
        return (hidden_states, self_attn_weights) if output_attentions else (hidden_states,)


class Cohere2PreTrainedModel(PretrainedModel):
    config_class = Cohere2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r"rotary_emb.inv_freq"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _gen_aoa_config(cls, config: Cohere2Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
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
        if config.use_qk_norm:
            aoa_statements.extend(
                [
                    f"model.layers.$LAYER_ID.self_attn.{proj_name}.weight -> {model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight"
                    for proj_name in ["q_norm", "k_norm"]
                ]
            )
        if config.attention_bias:
            aoa_statements.extend(
                [
                    f"model.layers.$LAYER_ID.self_attn.{proj_name}.bias -> {model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.bias"
                    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
                ]
            )
        if cls != cls.base_model_class:
            if config.tie_word_embeddings:
                aoa_statements.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")
        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: Cohere2Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
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
        if config.use_qk_norm:
            aoa_statements.extend(
                [
                    f"{model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight -> model.layers.$LAYER_ID.self_attn.{proj_name}.weight"
                    for proj_name in ["q_norm", "k_norm"]
                ]
            )
        if config.attention_bias:
            aoa_statements.extend(
                [
                    f"{model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.bias -> model.layers.$LAYER_ID.self_attn.{proj_name}.bias"
                    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
                ]
            )
        if not config.tie_word_embeddings and cls != cls.base_model_class:
            aoa_statements.append("lm_head.weight -> lm_head.weight")
        return {"aoa_statements": aoa_statements}


def _prepare_4d_causal_attention_mask(
    attention_mask,
    input_shape,
    past_key_values_length,
    dtype,
    sliding_window_size=None,
    **kwargs,
):
    batch_size, seq_length = input_shape
    min_value = paddle.finfo(dtype).min
    key_length = seq_length + past_key_values_length
    mask = paddle.full([seq_length, key_length], min_value, dtype=dtype)
    mask = paddle.triu(mask, diagonal=1 + past_key_values_length)
    if sliding_window_size is not None:
        context_mask = (
            paddle.arange(key_length)
            <= (paddle.arange(seq_length) + past_key_values_length - sliding_window_size)[:, None]
        )
        mask = paddle.where(context_mask, min_value, mask)
    mask = mask[None, None, :, :].expand([batch_size, 1, seq_length, key_length])
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            expanded = attention_mask[:, None, None, :].astype(dtype)
            padding_mask = (1.0 - expanded) * min_value
            mask = mask + padding_mask
        elif attention_mask.ndim == 4:
            mask = attention_mask.astype(dtype)
    return mask


@register_base_model
class Cohere2Model(Cohere2PreTrainedModel):
    def __init__(self, config: Cohere2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.LayerList([Cohere2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Cohere2LayerNorm([config.hidden_size], eps=config.layer_norm_eps)
        self.rotary_emb = Cohere2RotaryEmbedding(config)
        self.has_sliding_layers = getattr(
            self.config, "sliding_window", None
        ) is not None and "sliding_attention" in getattr(self.config, "layer_types", [])

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states,
        position_embeddings,
        attention_mask,
        attn_mask_startend_row_indices,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(
                    inputs[0],
                    position_embeddings=inputs[1],
                    attention_mask=inputs[2],
                    attn_mask_startend_row_indices=inputs[3] if inputs[3] is not None else None,
                    past_key_values=None,
                    output_attentions=False,
                    use_cache=False,
                )

            return custom_forward

        return recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_embeddings,
            attention_mask,
            attn_mask_startend_row_indices,
        )

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length, _ = inputs_embeds.shape
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = paddle.arange(cache_length, cache_length + seq_length, dtype="int64").expand(
                [batch_size, seq_length]
            )
        # Prepare mask arguments (align with GPT-OSS/Gemma3_text/Qwen3 pattern)
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": cache_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": _prepare_4d_causal_attention_mask,
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
        for idx, decoder_layer in enumerate(self.layers):
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
                    position_embeddings,
                    causal_mask_mapping[decoder_layer.attention_type],
                    attn_mask_startend_row_indices_mapping[decoder_layer.attention_type],
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Cohere2ForCausalLM(Cohere2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: Cohere2Config):
        super().__init__(config)
        self.model = Cohere2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.logit_scale = config.logit_scale
        self.tie_word_embeddings = config.tie_word_embeddings
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
        logits = self.lm_head(hidden_states)
        logits = logits * self.logit_scale
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = nn.functional.cross_entropy(
                shift_logits.reshape([-1, self.vocab_size]),
                shift_labels.reshape([-1]),
                ignore_index=-100,
            )
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


__all__ = ["Cohere2Config", "Cohere2PreTrainedModel", "Cohere2Model", "Cohere2ForCausalLM"]
