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

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ..cache_utils import DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from .configuration import TelechatConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    dtype = q.dtype
    q = q.astype("float32")
    k = k.astype("float32")
    return (q * cos + rotate_half(q) * sin).astype(dtype), (k * cos + rotate_half(k) * sin).astype(dtype)


class TelechatRotaryEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.n_head
        inv_freq = 1.0 / (10000.0 ** (paddle.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, x, position_ids):
        freqs = position_ids.astype("float32").unsqueeze(-1) * self.inv_freq.reshape([1, 1, -1])
        emb = paddle.concat((freqs, freqs), axis=-1)
        return emb.cos().astype(x.dtype), emb.sin().astype(x.dtype)


class TelechatRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size], dtype=paddle.get_default_dtype(), default_initializer=nn.initializer.Constant(1.0)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return hidden_states.astype(input_dtype) * self.weight.astype(input_dtype)


class TelechatAttention(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.n_head
        if self.head_dim * config.n_head != config.hidden_size:
            raise ValueError("hidden_size must be divisible by n_head")
        if config.tensor_model_parallel_size > 1:
            if config.n_head % config.tensor_model_parallel_size != 0:
                raise ValueError("n_head must be divisible by tensor_model_parallel_size")
            self.num_heads = config.n_head // config.tensor_model_parallel_size
        else:
            self.num_heads = config.n_head
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.query = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=False, config=config, tp_plan="colwise"
        )
        self.key_value = GeneralLinear.create(
            config.hidden_size, 2 * config.hidden_size, has_bias=False, config=config, tp_plan="colwise"
        )
        self.dense = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=True, config=config, tp_plan="rowwise"
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        position_embeddings=None,
        past_key_values=None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        query_states = (
            self.query(hidden_states)
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        key_value_states = self.key_value(hidden_states).reshape(
            [batch_size, seq_len, self.num_heads, 2, self.head_dim]
        )
        key_states = key_value_states[:, :, :, 0, :].transpose([0, 2, 1, 3])
        value_states = key_value_states[:, :, :, 1, :].transpose([0, 2, 1, 3])
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        attn_output, _ = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )
        return self.dense(attn_output.reshape([batch_size, seq_len, -1]))


class TelechatMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = GeneralLinear.create(
            config.hidden_size, config.ffn_hidden_size, has_bias=False, config=config, tp_plan="colwise"
        )
        self.up_proj = GeneralLinear.create(
            config.hidden_size, config.ffn_hidden_size, has_bias=False, config=config, tp_plan="colwise"
        )
        self.down_proj = GeneralLinear.create(
            config.ffn_hidden_size, config.hidden_size, has_bias=True, config=config, tp_plan="rowwise"
        )

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TelechatDecoderLayer(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attention = TelechatAttention(config, layer_idx)
        self.input_layernorm = TelechatRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.post_attention_layernorm = TelechatRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = TelechatMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        position_embeddings=None,
        past_key_values=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attention(
            hidden_states, attention_mask, attn_mask_startend_row_indices, position_embeddings, past_key_values
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class TelechatPretrainedModel(PretrainedModel):
    config_class = TelechatConfig
    base_model_prefix = "transformer"
    transpose_weight_keys = ["query", "key_value", "dense", "gate_proj", "up_proj", "down_proj", "lm_head"]

    @classmethod
    def _gen_aoa_config(cls, config):
        prefix = "" if cls == cls.base_model_class else "transformer."
        statements = [
            f"transformer.word_embeddings.weight -> {prefix}embed_tokens.weight",
            f"transformer.ln_f.weight -> {prefix}norm.weight",
            f"transformer.h.$LAYER_ID.input_layernorm.weight -> {prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"transformer.h.$LAYER_ID.post_attention_layernorm.weight -> {prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
        ]
        for name in ("query", "key_value", "dense", "gate_proj", "up_proj", "down_proj"):
            statements.append(
                f"transformer.h.$LAYER_ID.{('self_attention' if name in ('query', 'key_value', 'dense') else 'mlp')}.{name}.weight^T -> {prefix}layers.$LAYER_ID.{('self_attention' if name in ('query', 'key_value', 'dense') else 'mlp')}.{name}.weight"
            )
        statements.append(
            f"transformer.h.$LAYER_ID.self_attention.dense.bias -> {prefix}layers.$LAYER_ID.self_attention.dense.bias"
        )
        statements.append(f"transformer.h.$LAYER_ID.mlp.down_proj.bias -> {prefix}layers.$LAYER_ID.mlp.down_proj.bias")
        if cls != cls.base_model_class:
            statements.append("lm_head.weight^T -> lm_head.weight")
        return {"aoa_statements": statements}

    def _init_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(paddle.normal(0.0, self.config.initializer_range, layer.weight.shape))
            if getattr(layer, "bias", None) is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))


@register_base_model
class TelechatModel(TelechatPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = GeneralEmbedding.create(
            config, config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.embed_layernorm = (
            TelechatRMSNorm(config.hidden_size, config.layer_norm_epsilon) if config.embed_layernorm else None
        )
        self.layers = nn.LayerList([TelechatDecoderLayer(config, i) for i in range(config.n_layer)])
        self.norm = TelechatRMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.rotary_emb = TelechatRotaryEmbedding(config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        attn_mask_startend_row_indices=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if not ((input_ids is None) ^ (inputs_embeds is None)):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length, _ = inputs_embeds.shape
        if self.embed_layernorm is not None:
            inputs_embeds = self.embed_layernorm(inputs_embeds)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = (
                paddle.arange(cache_length, cache_length + seq_length, dtype="int64")
                .unsqueeze(0)
                .tile([batch_size, 1])
            )
        causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=inputs_embeds,
            batch_size=batch_size,
            seq_length=seq_length,
            cache_length=cache_length,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            prepare_decoder_attention_mask=self._prepare_decoder_attention_mask,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = decoder_layer(
                hidden_states, causal_mask, attn_mask_startend_row_indices, position_embeddings, past_key_values
            )
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            all_hidden_states = tuple(all_hidden_states)
        if not return_dict:
            return tuple(
                v for v in (hidden_states, past_key_values if use_cache else None, all_hidden_states) if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values, hidden_states=all_hidden_states
        )


class TelechatForCausalLM(TelechatPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = TelechatModel(config)
        self.lm_head = GeneralLinear.create(
            config.hidden_size, config.vocab_size, has_bias=False, config=config, tp_plan="colwise", gather_output=True
        )

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.transformer.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        inputs_embeds=None,
        labels=None,
        loss_mask=None,
        use_cache=None,
        past_key_values=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            attn_mask_startend_row_indices,
            use_cache,
            output_hidden_states,
            True,
        )
        logits = self.lm_head(outputs[0])
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape([-1, logits.shape[-1]])
            shift_labels = labels[:, 1:].reshape([-1])
            valid = shift_labels != -100
            if loss_mask is not None:
                valid = valid & loss_mask[:, 1:].reshape([-1]).astype("bool")
            safe_labels = paddle.where(valid, shift_labels, paddle.zeros_like(shift_labels))
            log_probs = F.log_softmax(shift_logits, axis=-1)
            selected = paddle.take_along_axis(log_probs, safe_labels.unsqueeze(-1), axis=-1).squeeze(-1)
            valid_float = valid.astype(selected.dtype)
            loss = -(selected * valid_float).sum() / paddle.clip(valid_float.sum(), min=1.0)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        model_inputs = (
            {"inputs_embeds": inputs_embeds}
            if inputs_embeds is not None and past_key_values is None
            else {"input_ids": input_ids}
        )
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
