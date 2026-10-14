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
""" Paddle InternLM25 model."""
import logging
import math
import queue
import threading
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from paddleformers.transformers import PretrainedModel, register_base_model
from paddleformers.transformers.activations import ACT2FN
from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from ..cache_utils import Cache, DynamicCache
from .configuration import InternLM25Config

logger = logging.getLogger(__name__)

try:
    from paddleformers.generation.streamers import BaseStreamer
except Exception:
    BaseStreamer = None

try:
    from paddle.nn.functional.flash_attention import flash_attention as flash_attn_func
    from paddle.nn.functional.flash_attention import (
        flash_attn_unpadded as flash_attn_varlen_func,
    )

    has_flash_attn = True
except:
    flash_attn_func, flash_attn_varlen_func = None, None
    has_flash_attn = False


def index_first_axis(tensor, index):
    return tensor[index]


def pad_input(hidden_states, indices, batch, seqlen):
    output = paddle.zeros([batch * seqlen, *hidden_states.shape[1:]], dtype=hidden_states.dtype)
    output = paddle.scatter(output, indices, hidden_states)
    return output.reshape([batch, seqlen, *hidden_states.shape[1:]])


def unpad_input(hidden_states, attention_mask):
    indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)
    hidden_states = index_first_axis(hidden_states.reshape([-1, *hidden_states.shape[2:]]), indices)
    return hidden_states, indices, cu_seqlens, max_seqlen


_CONFIG_FOR_DOC = "InternLM25Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=paddle.int32)
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0, dtype=paddle.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class InternLM25RMSNorm(nn.Layer):
    """InternLM25RMSNorm is equivalent to T5LayerNorm."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        out_2 = paddle.create_parameter(
            shape=paddle.ones(shape=hidden_size).shape,
            dtype=paddle.ones(shape=hidden_size).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=hidden_size)),
        )
        out_2.stop_gradient = not True
        self.weight = out_2
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class InternLM25RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (paddle.arange(0, self.dim, 2, dtype=paddle.int64).astype("float32") / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.max_seq_len_cached = max_position_embeddings

    @paddle.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].astype("float32").expand([position_ids.shape[0], self.inv_freq.shape[0], 1])
        )
        position_ids_expanded = position_ids[:, None, :].astype("float32")
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose([0, 2, 1])
        emb = paddle.concat((freqs, freqs), axis=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class InternLM25LinearScalingRotaryEmbedding(InternLM25RotaryEmbedding):
    def forward(self, x, position_ids):
        position_ids = position_ids.astype("float32") / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class InternLM25DynamicNTKScalingRotaryEmbedding(InternLM25RotaryEmbedding):
    def forward(self, x, position_ids):
        seq_len = paddle.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (paddle.arange(0, self.dim, 2, dtype=paddle.int64).astype("float32").to(x.place) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistable=False)

        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_axis=1):
    cos = cos.unsqueeze(unsqueeze_axis)
    sin = sin.unsqueeze(unsqueeze_axis)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InternLM25MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        return down_proj


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class InternLM25Attention(nn.Layer):
    def __init__(self, config: InternLM25Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias_attr=config.bias,
        )
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=config.bias)

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = InternLM25RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = InternLM25LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = InternLM25DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            qkv_slices = self.wqkv.weight.split(key_value_slicing, axis=0)
            qkv_states = paddle.concat([F.linear(hidden_states, qkv_slice) for qkv_slice in qkv_slices], axis=-1)
        else:
            qkv_states = self.wqkv(hidden_states)

        gs = 2 + self.num_key_value_groups
        d = self.head_dim
        h = qkv_states.shape[-1] // (gs * d)
        qkv_states = qkv_states.reshape([bsz, q_len, h, gs, d])

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = query_states.reshape([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = qkv_states[..., -2, :].transpose([0, 2, 1, 3])
        value_states = qkv_states[..., -1, :].transpose([0, 2, 1, 3])

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, axis=-1, dtype=paddle.float32).to(query_states.dtype)
        attn_output = paddle.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose([0, 2, 1, 3])

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, axis=2)
            o_proj_slices = self.wo.weight.split(self.hidden_size // self.config.pretraining_tp, axis=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class InternLM25FlashAttention2(InternLM25Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.shape

        qkv_states = self.wqkv(hidden_states)

        gs = 2 + self.num_key_value_groups
        d = self.head_dim
        h = qkv_states.shape[-1] // (gs * d)
        qkv_states = qkv_states.reshape([bsz, q_len, h, gs, d])

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = query_states.reshape([bsz, q_len, -1, self.head_dim])
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        dropout_rate = 0.0

        input_dtype = query_states.dtype
        if input_dtype == paddle.float32:
            if False:
                target_dtype = paddle.float32
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.wqkv.weight.dtype

            logger.warning(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = paddle.arange(batch_size + 1, dtype=paddle.int32)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class InternLM25SdpaAttention(InternLM25Attention):
    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if output_attentions:

            logger.warning(
                "InternLM25Model uses InternLM25SdpaAttention, but `paddle.nn.functional.scaled_dot_product_attention` "
                "does not support `output_attentions=True`. "
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.shape

        qkv_states = self.wqkv(hidden_states)

        gs = 2 + self.num_key_value_groups
        d = self.head_dim
        h = qkv_states.shape[-1] // (gs * d)
        qkv_states = qkv_states.reshape([bsz, q_len, h, gs, d])

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = query_states.reshape([bsz, q_len, -1, self.head_dim])
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.place.type == "cuda" and causal_mask is not None:
            query_states = query_states
            key_states = key_states
            value_states = value_states

        is_causal = bool(causal_mask is None and q_len > 1)

        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.wo(attn_output)

        return attn_output, None, past_key_value


INTERNLM25_ATTENTION_CLASSES = {
    "eager": InternLM25Attention,
    "flash_attention_2": InternLM25FlashAttention2,
    "sdpa": InternLM25SdpaAttention,
}


class InternLM25DecoderLayer(nn.Layer):
    def __init__(self, config: InternLM25Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.attention = INTERNLM25_ATTENTION_CLASSES[config.attn_implementation](config=config, layer_idx=layer_idx)

        self.feed_forward = InternLM25MLP(config)
        self.attention_norm = InternLM25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class InternLM25PretrainedModel(PretrainedModel):
    config_class = InternLM25Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternLM25DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    transpose_weight_keys = ["wqkv", "wo", "w1", "w2", "w3", "output"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(module.weight)
            if module.bias is not None:
                paddle.nn.initializer.Constant(0.0)(module.bias)
        elif isinstance(module, nn.Embedding):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(module.weight)
            if module._padding_idx is not None:
                module.weight[module._padding_idx].zero_()

    @classmethod
    def _gen_aoa_config(cls, config: InternLM25Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"model.tok_embeddings.weight -> {model_prefix}tok_embeddings.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.attention_norm.weight -> {model_prefix}layers.$LAYER_ID.attention_norm.weight",
            f"model.layers.$LAYER_ID.ffn_norm.weight -> {model_prefix}layers.$LAYER_ID.ffn_norm.weight",
        ]
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.attention.{w}.weight^T -> {model_prefix}layers.$LAYER_ID.attention.{w}.weight"
                for w in ["wqkv", "wo"]
            ]
        )
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.feed_forward.{w}.weight^T -> {model_prefix}layers.$LAYER_ID.feed_forward.{w}.weight"
                for w in ["w1", "w2", "w3"]
            ]
        )
        if cls != cls.base_model_class:
            if getattr(config, "tie_word_embeddings", False):
                aoa_statements.append("model.tok_embeddings.weight -> output.weight")
            else:
                aoa_statements.append("output.weight^T -> output.weight")
        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: InternLM25Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"{model_prefix}tok_embeddings.weight -> model.tok_embeddings.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.attention_norm.weight -> model.layers.$LAYER_ID.attention_norm.weight",
            f"{model_prefix}layers.$LAYER_ID.ffn_norm.weight -> model.layers.$LAYER_ID.ffn_norm.weight",
        ]
        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.attention.{w}.weight^T -> model.layers.$LAYER_ID.attention.{w}.weight"
                for w in ["wqkv", "wo"]
            ]
        )
        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.feed_forward.{w}.weight^T -> model.layers.$LAYER_ID.feed_forward.{w}.weight"
                for w in ["w1", "w2", "w3"]
            ]
        )
        if cls != cls.base_model_class:
            if getattr(config, "tie_word_embeddings", False):
                aoa_statements.append("output.weight -> model.tok_embeddings.weight")
            else:
                aoa_statements.append("output.weight^T -> output.weight")
        return {"aoa_statements": aoa_statements}


@register_base_model
class InternLM25Model(InternLM25PretrainedModel):
    _auto_class = "AutoModel"

    def __init__(self, config: InternLM25Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.LayerList(
            [InternLM25DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = InternLM25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.enable_recompute = False

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        causal_mask: Optional[paddle.Tensor],
        position_ids: Optional[paddle.Tensor],
        past_key_values: Optional[Cache],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[paddle.Tensor],
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        layer_outputs = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            causal_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            cache_position,
        )
        return layer_outputs

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        # PaddleFormers SFT trainer may pass extra kwargs like attn_mask_startend_row_indices;
        # accept and ignore them here for compatibility.
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.enable_recompute and self.training and use_cache:
            logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None or len(past_key_values) == 0:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache(ddp_cache_data=past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = paddle.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.enable_recompute and self.training:
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = tuple((layer.keys, layer.values) for layer in next_cache.layers)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: paddle.Tensor,
        input_tensor: paddle.Tensor,
        cache_position: paddle.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config.attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = False

        if self.config.attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            pass

        dtype, device = input_tensor.dtype, input_tensor.place
        min_dtype = paddle.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, paddle.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.ndim == 4:
            causal_mask = attention_mask
        else:
            causal_mask = paddle.full([sequence_length, target_length], fill_value=min_dtype, dtype=dtype)
            if device is not None:
                causal_mask = causal_mask.to(device)
            if sequence_length != 1:
                if dtype == paddle.float32:
                    causal_mask = paddle.triu(causal_mask, diagonal=1)
                else:
                    triu_mask = paddle.triu(paddle.ones(causal_mask.shape).to(device), diagonal=1).astype("bool")
                    causal_mask = paddle.where(triu_mask, causal_mask, paddle.zeros_like(causal_mask))
            causal_mask *= (paddle.arange(target_length).to(device) > cache_position.reshape(-1, 1)).astype(dtype)
            causal_mask = causal_mask[None, None, :, :].expand([input_tensor.shape[0], 1, -1, -1])
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].astype(
                    causal_mask.dtype
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config.attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.place.type == "cuda"
            and not output_attentions
        ):
            pass

        return causal_mask


class InternLM25ForCausalLM(InternLM25PretrainedModel):
    _auto_class = "AutoModelForCausalLM"
    _tied_weights_keys = ["output.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM25Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        # PaddleFormers SFT trainer may pass extra kwargs like attn_mask_startend_row_indices;
        # accept and ignore them here for compatibility.
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            output_slices = self.output.weight.split(self.vocab_size // self.config.pretraining_tp, axis=0)
            logits = [F.linear(hidden_states, output_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = paddle.concat(logits, axis=-1)
        else:
            logits = self.output(hidden_states)
        logits = logits.astype("float32")

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.reshape(-1, self.vocab_size)
            shift_labels = shift_labels.reshape(-1)
            shift_labels = shift_labels.to(shift_logits.place)
            loss = loss_fct(shift_logits, shift_labels)

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    paddle.to_tensor(
                        past_key_values.get_max_cache_shape(),
                        place=input_ids.place if hasattr(input_ids, "place") else None,
                    )
                    if past_key_values.get_max_cache_shape() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else paddle.min(max_cache_length, past_length)
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]

            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.astype("int64").cumsum(-1) - 1
            position_ids = paddle.where(attention_mask == 0, paddle.ones_like(position_ids), position_ids)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = paddle.arange(past_length, past_length + input_length)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.place)) for past_state in layer_past),
            )
        return reordered_past

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, meta_instruction=""):
        if history is None:
            history = []
        if tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = tokenizer.bos_token
        if meta_instruction:
            prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
        for record in history:
            prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        return tokenizer([prompt], return_tensors="pd")

    @paddle.no_grad()
    def chat(
        self,
        tokenizer,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        meta_instruction: str = "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory "
        "(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such "
        "as English and 中文.",
        **kwargs,
    ):
        if history is None:
            history = []
        inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
        inputs = {k: v for k, v in inputs.items() if paddle.is_tensor(v)}
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]
        outputs = self.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("<|im_end|>")[0]
        history = history + [(query, response)]
        return response, history

    @paddle.no_grad()
    def stream_chat(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        **kwargs,
    ):
        if history is None:
            history = []
        if BaseStreamer is None:
            raise ModuleNotFoundError("The version of `paddle` is too low.")

        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ""
                self.cache = []
                self.received_inputs = False
                self.queue.put((self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("ChatStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if not self.received_inputs:
                    self.received_inputs = True
                    return

                self.cache.extend(value.tolist())
                token = self.tokenizer.decode(self.cache, skip_special_tokens=True)
                if token.strip() != "<|im_end|>":
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))
                    self.cache = []
                else:
                    self.end()

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()


class InternLM25ForSequenceClassification(InternLM25PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLM25Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias_attr=False)

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = paddle.equal(input_ids, self.config.pad_token_id).astype("int32").argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.place)
            else:
                sequence_lengths = -1

        pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.place)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in (paddle.int64, paddle.int32)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.reshape(-1, self.num_labels), labels.reshape(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class InternLM25ForQuestionAnswering(InternLM25PretrainedModel):
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = InternLM25Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def get_input_embeddings(self):
        return self.transformer.tok_embeddings

    def set_input_embeddings(self, value):
        self.transformer.tok_embeddings = value

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        start_positions: Optional[paddle.Tensor] = None,
        end_positions: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = paddle.split(logits, num_or_sections=2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.place)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.place)
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class InternLM25ForTokenClassification(InternLM25PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLM25Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Aliases for intern proxy compatibility. The intern/modeling.py proxy resolves
# implementation classes by InternLM2* names via getattr(_impl_module, cls_name).
InternLM2PretrainedModel = InternLM25PretrainedModel
InternLM2DecoderLayer = InternLM25DecoderLayer
InternLM2Model = InternLM25Model
InternLM2ForCausalLM = InternLM25ForCausalLM
InternLM2ForSequenceClassification = InternLM25ForSequenceClassification
InternLM2ForQuestionAnswering = InternLM25ForQuestionAnswering
InternLM2ForTokenClassification = InternLM25ForTokenClassification
