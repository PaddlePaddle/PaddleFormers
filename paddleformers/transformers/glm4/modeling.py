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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..model_outputs import CausalLMOutputWithPast
from ..model_utils import PretrainedModel
from .configuration import Glm4Config

__all__ = [
    "Glm4PretrainedModel",
    "Glm4Model",
    "Glm4ForCausalLM",
]


def _normalize_config_dtype(config: Glm4Config):
    """
    Normalize HF-style torch_dtype to Paddle runtime dtype.

    Priority:
    1. keep config.dtype if already explicitly set
    2. infer from config.torch_dtype
    3. fallback to float32
    """
    cur_dtype = getattr(config, "dtype", None)
    if cur_dtype is not None and str(cur_dtype).lower() not in ["none", "null", ""]:
        return config

    torch_dtype = getattr(config, "torch_dtype", None)
    if torch_dtype is not None:
        torch_dtype = str(torch_dtype).lower()

    if torch_dtype in ["bfloat16", "bf16"]:
        config.dtype = "bfloat16"
    elif torch_dtype in ["float16", "fp16", "half"]:
        config.dtype = "float16"
    elif torch_dtype in ["float32", "fp32"]:
        config.dtype = "float32"
    else:
        config.dtype = "float32"

    return config


class Glm4RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, hidden_states):
        variance = paddle.mean(hidden_states * hidden_states, axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.eps)
        return self.weight * hidden_states


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return paddle.stack([-x2, x1], axis=-1).flatten(start_axis=-2)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = paddle.unsqueeze(cos, axis=unsqueeze_dim)
    sin = paddle.unsqueeze(sin, axis=unsqueeze_dim)
    cos = paddle.repeat_interleave(cos, repeats=2, axis=-1)
    sin = paddle.repeat_interleave(sin, repeats=2, axis=-1)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = paddle.concat([q_embed, q_pass], axis=-1)
    k_embed = paddle.concat([k_embed, k_pass], axis=-1)
    return q_embed, k_embed


class Glm4RotaryEmbedding(nn.Layer):
    def __init__(self, config: Glm4Config):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.head_dim = config.head_dim
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)

        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1

        inv_freq = 1.0 / (self.base ** (paddle.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, position_ids):
        freqs = paddle.einsum("bs,d->bsd", position_ids.astype("float32"), self.inv_freq)
        cos = paddle.cos(freqs)
        sin = paddle.sin(freqs)
        return cos, sin


class Glm4MLP(nn.Layer):
    def __init__(self, config: Glm4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size * 2,
            bias_attr=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias_attr=False,
        )

    def forward(self, hidden_states):
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = paddle.chunk(gate_up, chunks=2, axis=-1)
        hidden_states = F.silu(gate) * up
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class Glm4Attention(nn.Layer):
    def __init__(self, config: Glm4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias_attr=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias_attr=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias_attr=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias_attr=False,
        )
        self.rotary_emb = Glm4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
        past_key_value=None,
        use_cache=False,
    ):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        if position_ids is None:
            position_ids = paddle.arange(q_len, dtype="int64").unsqueeze(0).tile([bsz, 1])

        if position_embeddings is None:
            cos, sin = self.rotary_emb(position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if past_key_value is not None and past_key_value[0] is not None:
            past_k, past_v = past_key_value
            key_states = paddle.concat([past_k, key_states], axis=2)
            value_states = paddle.concat([past_v, value_states], axis=2)

        present = None
        if use_cache:
            present = (key_states, value_states)

        if self.num_key_value_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_key_value_heads
            key_states = paddle.repeat_interleave(key_states, repeat_factor, axis=1)
            value_states = paddle.repeat_interleave(value_states, repeat_factor, axis=1)

        attn_weights = paddle.matmul(query_states, key_states, transpose_y=True) / (self.head_dim**0.5)

        kv_len = key_states.shape[2]
        cur_q_len = query_states.shape[2]
        past_kv_len = kv_len - cur_q_len

        q_pos = paddle.arange(cur_q_len, dtype="int64").unsqueeze(-1) + past_kv_len
        k_pos = paddle.arange(kv_len, dtype="int64").unsqueeze(0)
        causal = (q_pos >= k_pos).astype(attn_weights.dtype)
        causal = causal.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights + (1.0 - causal) * -1e4

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                mask = attention_mask[:, None, None, :].astype(attn_weights.dtype)
                if mask.shape[-1] != kv_len:
                    pad_len = kv_len - mask.shape[-1]
                    if pad_len > 0:
                        pad = paddle.ones([mask.shape[0], 1, 1, pad_len], dtype=mask.dtype)
                        mask = paddle.concat([pad, mask], axis=-1)
                    else:
                        mask = mask[:, :, :, -kv_len:]
                attn_weights = attn_weights + (1.0 - mask) * -1e4
            else:
                attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.astype("float32"), axis=-1).astype(query_states.dtype)
        attn_output = paddle.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([bsz, cur_q_len, self.num_heads * self.head_dim])
        attn_output = self.o_proj(attn_output)

        return attn_output, present


class Glm4DecoderLayer(nn.Layer):
    def __init__(self, config: Glm4Config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.input_layernorm = Glm4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Glm4Attention(config)

        self.post_self_attn_layernorm = Glm4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = Glm4MLP(config)

        self.post_mlp_layernorm = Glm4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        attn_output = self.post_self_attn_layernorm(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.post_mlp_layernorm(mlp_output)
        hidden_states = residual + mlp_output

        return hidden_states, present


class Glm4PretrainedModel(PretrainedModel):
    config_class = Glm4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    _hf_linear_weight_suffixes = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_up_proj.weight",
        "down_proj.weight",
        "lm_head.weight",
    )

    def _init_weights(self, layer):
        return

    def init_weights(self):
        return

    @classmethod
    def _convert_hf_state_dict(cls, state_dict):
        """
        HF/PyTorch Linear weight: [out_features, in_features]
        Paddle nn.Linear weight:  [in_features, out_features]

        这里必须按名字强制转置，不能依赖 shape mismatch。
        因为 q_proj/o_proj 这类 4096x4096 方阵 shape 一样，但语义仍然相反。
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            if (
                isinstance(v, paddle.Tensor)
                and v.ndim == 2
                and any(k.endswith(suffix) for suffix in cls._hf_linear_weight_suffixes)
            ):
                v = v.transpose([1, 0])
            new_state_dict[k] = v
        return new_state_dict

    def set_state_dict(self, state_dict, *args, **kwargs):
        already_converted = kwargs.pop("already_converted", False)
        if not already_converted:
            state_dict = self._convert_hf_state_dict(state_dict)
        return super().set_state_dict(state_dict, *args, **kwargs)


class Glm4Model(Glm4PretrainedModel):
    def __init__(self, config: Glm4Config):
        config = _normalize_config_dtype(config)
        super().__init__(config)
        self.config = config

        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.LayerList([Glm4DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Glm4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Glm4RotaryEmbedding(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either input_ids or inputs_embeds.")

        if use_cache is None:
            use_cache = self.config.use_cache

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        bsz, seq_len, _ = hidden_states.shape

        past_length = 0
        if (
            past_key_values is not None
            and len(past_key_values) > 0
            and past_key_values[0] is not None
            and past_key_values[0][0] is not None
        ):
            past_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = paddle.arange(past_length, past_length + seq_len, dtype="int64").unsqueeze(0).tile([bsz, 1])

        position_embeddings = self.rotary_emb(position_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        next_past_key_values = [] if use_cache else None

        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                next_past_key_values.append(present)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            if use_cache:
                return hidden_states, next_past_key_values
            return (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_past_key_values,
        }


class Glm4ForCausalLM(Glm4PretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Glm4Config):
        config = _normalize_config_dtype(config)
        super().__init__(config)
        self.config = config
        self.model = Glm4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=True,
        **kwargs,
    ):
        if attention_mask is not None:
            position_ids = attention_mask.astype("int64").cumsum(axis=-1) - 1
            position_ids = paddle.where(
                attention_mask > 0,
                position_ids,
                paddle.zeros_like(position_ids),
            )
        else:
            position_ids = paddle.arange(input_ids.shape[1], dtype="int64").unsqueeze(0).tile([input_ids.shape[0], 1])

        if (
            past_key_values is not None
            and len(past_key_values) > 0
            and past_key_values[0] is not None
            and past_key_values[0][0] is not None
        ):
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds

        return model_inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(
                shift_logits.reshape([-1, shift_logits.shape[-1]]),
                shift_labels.reshape([-1]),
                reduction="mean",
            )

        if not return_dict:
            result = (logits, outputs.get("past_key_values", None))
            if loss is not None:
                result = (loss,) + result
            return result

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get("past_key_values", None),
            hidden_states=None,
            attentions=None,
        )
