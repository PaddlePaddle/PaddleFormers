# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Paddle implementation of the LFM2 text backbone."""

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...nn.criterion.interface import CriterionLayer
from ...nn.lm_head import LMHead
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Lfm2Config


class Lfm2RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter([hidden_size], default_initializer=nn.initializer.Constant(1.0))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.square().mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(dtype)


class Lfm2MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        intermediate_size = config.intermediate_size
        if config.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
                multiple = config.block_multiple_of
                intermediate_size = multiple * ((intermediate_size + multiple - 1) // multiple)
        self.w1 = nn.Linear(config.hidden_size, intermediate_size, bias_attr=False)
        self.w3 = nn.Linear(config.hidden_size, intermediate_size, bias_attr=False)
        self.w2 = nn.Linear(intermediate_size, config.hidden_size, bias_attr=False)

    def forward(self, hidden_states):
        hidden_states = F.silu(self.w1(hidden_states)) * self.w3(hidden_states)
        if hidden_states.dtype == paddle.bfloat16:
            weight = self.w2.weight.transpose([1, 0]).contiguous()
            return paddle.matmul(hidden_states, weight, transpose_y=True)
        return self.w2(hidden_states)


def rotate_half(hidden_states):
    first, second = paddle.chunk(hidden_states, 2, axis=-1)
    return paddle.concat([-second, first], axis=-1)


class Lfm2Attention(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias_attr=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias_attr=False)
        self.q_layernorm = Lfm2RMSNorm(self.head_dim, config.norm_eps)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim, config.norm_eps)

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape([batch_size, sequence_length, self.num_heads, self.head_dim])
        key = self.k_proj(hidden_states).reshape(
            [batch_size, sequence_length, self.num_key_value_heads, self.head_dim]
        )
        value = self.v_proj(hidden_states).reshape(
            [batch_size, sequence_length, self.num_key_value_heads, self.head_dim]
        )
        query = self.q_layernorm(query).transpose([0, 2, 1, 3])
        key = self.k_layernorm(key).transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        if self.num_key_value_groups > 1:
            key = paddle.repeat_interleave(key, self.num_key_value_groups, axis=1)
            value = paddle.repeat_interleave(value, self.num_key_value_groups, axis=1)
        scores = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = F.softmax(scores.astype("float32"), axis=-1).astype(query.dtype)
        output = paddle.matmul(probabilities, value).transpose([0, 2, 1, 3])
        return self.out_proj(output.reshape([batch_size, sequence_length, -1]))


class Lfm2ShortConv(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.kernel_size = config.conv_L_cache
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias_attr=config.conv_bias)
        self.conv = nn.Conv1D(
            config.hidden_size,
            config.hidden_size,
            self.kernel_size,
            groups=config.hidden_size,
            padding=self.kernel_size - 1,
            bias_attr=config.conv_bias,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=config.conv_bias)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).astype(hidden_states.dtype)
        projected = self.in_proj(hidden_states).transpose([0, 2, 1])
        b_gate, c_gate, values = paddle.chunk(projected, 3, axis=1)
        values = b_gate * values
        values_dtype = values.dtype
        values = F.conv1d(
            values.astype("float32"),
            self.conv.weight.astype("float32"),
            None if self.conv.bias is None else self.conv.bias.astype("float32"),
            padding=self.kernel_size - 1,
            groups=values.shape[1],
        )[:, :, : hidden_states.shape[1]].astype(values_dtype)
        output = (c_gate * values).transpose([0, 2, 1]).contiguous()
        return self.out_proj(output)


class Lfm2DecoderLayer(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config)
        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, hidden_states, cos, sin, causal_mask=None, attention_mask=None):
        residual = hidden_states
        normalized = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            hidden_states = self.self_attn(normalized, cos, sin, causal_mask)
        else:
            hidden_states = self.conv(normalized, attention_mask)
        hidden_states = residual + hidden_states
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class Lfm2PreTrainedModel(PretrainedModel):
    config_class = Lfm2Config
    base_model_prefix = "model"


@register_base_model
class Lfm2Model(Lfm2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.LayerList([Lfm2DecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, config.norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (config.rope_theta ** (paddle.arange(0, head_dim, 2, dtype="float32") / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, sequence_length, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = paddle.arange(sequence_length, dtype="int64").unsqueeze(0).expand([batch_size, -1])
        frequencies = position_ids.astype("float32").unsqueeze(-1) * self.inv_freq.reshape([1, 1, -1])
        embeddings = paddle.concat([frequencies, frequencies], axis=-1)
        cos, sin = embeddings.cos().astype(inputs_embeds.dtype), embeddings.sin().astype(inputs_embeds.dtype)
        minimum = paddle.finfo(inputs_embeds.dtype).min
        causal_mask = paddle.triu(
            paddle.full([sequence_length, sequence_length], minimum, dtype=inputs_embeds.dtype), diagonal=1
        )
        causal_mask = causal_mask.reshape([1, 1, sequence_length, sequence_length])
        if attention_mask is not None:
            padding_mask = (1 - attention_mask.astype(inputs_embeds.dtype))[:, None, None, :] * minimum
            causal_mask = causal_mask + padding_mask
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, causal_mask, attention_mask)
        hidden_states = self.embedding_norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)


class Lfm2ForCausalLM(Lfm2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Lfm2Model(config)
        self.lm_head = LMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, labels=None, **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = self.criterion(logits, labels)[0] if labels is not None else None
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)


__all__ = ["Lfm2ForCausalLM", "Lfm2Model", "Lfm2PreTrainedModel"]
