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
""" Paddle InternLM2 model."""
import logging
import math
import warnings
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from paddleformers.transformers.activations import ACT2FN
from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from ..model_utils import PretrainedModel, register_base_model
from .configuration import InternLM2Config

logger = logging.getLogger(__name__)
try:
    from paddle.nn.functional.flash_attention import flash_attention as flash_attn_func
    from paddle.nn.functional.flash_attention import (
        flash_attn_unpadded as flash_attn_varlen_func,
    )

    has_flash_attn = True
except Exception:
    flash_attn_func, flash_attn_varlen_func = None, None
    has_flash_attn = False


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0, dtype=paddle.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


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


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def _make_causal_mask(input_ids_shape: list, dtype: paddle.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = paddle.full(shape=(tgt_len, tgt_len), fill_value=paddle.finfo(dtype).min)
    mask_cond = paddle.arange(end=mask.shape[-1])

    mask = masked_fill(mask, mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0)
    mask = mask.astype(dtype)
    if past_key_values_length > 0:
        mask = paddle.concat(x=[paddle.zeros(shape=[tgt_len, past_key_values_length], dtype=dtype), mask], axis=-1)
    return mask[(None), (None), :, :].expand(shape=[bsz, 1, tgt_len, tgt_len + past_key_values_length])


def _expand_mask(mask: paddle.Tensor, dtype: paddle.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, (None), (None), :].expand(shape=[bsz, 1, tgt_len, src_len]).astype(dtype)
    inverted_mask = 1.0 - expanded_mask

    return masked_fill(inverted_mask, inverted_mask.astype("bool"), paddle.finfo(dtype).min)


class InternLM2RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        InternLM2RMSNorm is equivalent to T5LayerNorm
        """
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
        hidden_states = hidden_states.to(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class InternLM2RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (paddle.arange(0, self.dim, 2).cast("float32") / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)  # persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=paddle.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached).to(dtype=self.inv_freq.dtype)

        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat((freqs, freqs), axis=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistable=False)  # persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistable=False)  # persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class InternLM2LinearScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached).to(dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat((freqs, freqs), axis=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistable=False)  # persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistable=False)  # persistent=False)


class InternLM2DynamicNTKScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (paddle.arange(0, self.dim, 2).cast("float32") / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistable=False)  # persistent=False)

        t = paddle.arange(self.max_seq_len_cached).to(dtype=self.inv_freq.dtype)

        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat((freqs, freqs), axis=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistable=False)  # persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistable=False)  # persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InternLM2MLP(nn.Layer):
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
    """
    This is the equivalent of paddle.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(shape=[batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


# Modified from transformers.model.llama.modeling_llama.LlamaAttention
class InternLM2Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
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
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling.get("type", "linear")
            scaling_factor = self.config.rope_scaling.get("factor", 1.0)
            if scaling_type == "dynamic":
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "linear":
                self.rotary_emb = InternLM2LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError("Currently we only support rotary embedding's type being 'dynamic' or 'linear'.")
        return self.rotary_emb

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.shape  # [1, 1847, 2048]

        try:
            qkv_states = self.wqkv(hidden_states)
        except Exception:
            qkv_states = self.wqkv(hidden_states.cast("bfloat16"))
        # [1, 1847, 4096]

        gs = 2 + self.num_key_value_groups
        d = self.head_dim
        h = qkv_states.shape[-1] // (gs * d)
        qkv_states = qkv_states.reshape([bsz, q_len, h, gs, d])
        # [1, 1847, 8, 4, 128]

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = query_states.reshape(
            [bsz, q_len, -1, self.head_dim]
        )  # [1, 1847, 8, 2, 128]->[1, 1847, 16, 128]
        key_states = qkv_states[..., -2, :]  # [1, 1847, 8, 128]
        value_states = qkv_states[..., -1, :]  # [1, 1847, 8, 128]

        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # [1, 16, 1847, 128] [1, 8, 1847, 128] [1847, 128] [1847, 128] [1, 1847]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)

        if attn_weights.shape != [bsz, self.num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of size {bsz, self.num_heads, q_len, kv_seq_len}, but is {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, axis=-1, dtype=paddle.float32).to(query_states.dtype)
        attn_output = paddle.matmul(attn_weights, value_states)

        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {attn_output.shape}"
            )

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])

        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Modified from transformers.model.llama.modeling_llama.InternLM2FlashAttention2
class InternLM2FlashAttention2(InternLM2Attention):
    """
    InternLM2 flash attention module. This module inherits from `InternLM2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        # InternLM2FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.shape  # [1, 1847, 2048]

        qkv_states = self.wqkv(hidden_states)  # [1, 1847, 4096]

        gs = 2 + self.num_key_value_groups
        d = self.head_dim
        h = qkv_states.shape[-1] // (gs * d)
        qkv_states = qkv_states.reshape([bsz, q_len, h, gs, d])
        # [1, 1847, 8, 4, 128]

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = query_states.reshape([bsz, q_len, -1, self.head_dim])
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        # new added
        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])
        #

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # real:[1, 16, 1847, 128] [1, 8, 1847, 128] [1847, 128] [1847, 128] [1, 1847]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat(x=[past_key_value[0], key_states], axis=2)
            value_states = paddle.concat(x=[past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(perm=[0, 2, 1, 3])
        key_states = key_states.transpose(perm=[0, 2, 1, 3])
        value_states = value_states.transpose(perm=[0, 2, 1, 3])

        original_dtype = query_states.dtype
        query_states = query_states.astype("bfloat16")
        key_states = key_states.astype("bfloat16")
        value_states = value_states.astype("bfloat16")
        # return tuple 0 is output, 1 is softmax return
        attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len)
        attn_output = attn_output.astype(original_dtype)
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`paddle.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`paddle.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`paddle.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`paddle.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        causal = self.is_causal and query_length != 1

        head_dim = query_states.shape[-1]
        softmax_scale = head_dim**-0.5

        if attention_mask is not None:
            batch_size = query_states.shape[0]  # [2, 3383, 16, 128]

            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._unpad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(  # TODO: flash_attn_unpadded
                query_states,  # [5998, 16, 128]
                key_states,  # [5998, 8, 128]
                value_states,  # [5998, 8, 128]
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                scale=softmax_scale,  # not softmax_scale=
                dropout=dropout,
                causal=causal,
            )[0]

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                causal=causal,  # no softmax_scale=
            )[0]

        return attn_output

    def _unpad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape([batch_size * kv_seq_len, num_key_value_heads, head_dim]), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape([batch_size * kv_seq_len, num_key_value_heads, head_dim]), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape([batch_size * kv_seq_len, self.num_heads, head_dim]), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = paddle.arange(
                batch_size + 1, dtype=paddle.int32
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q.to(paddle.int64),
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


INTERNLM2_ATTENTION_CLASSES = {
    "eager": InternLM2Attention,
    "flash_attention_2": InternLM2FlashAttention2,
    "sdpa": InternLM2Attention,
}


# Modified from transformers.model.llama.modeling_llama.LlamaDecoderLayer
class InternLM2DecoderLayer(nn.Layer):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attention = INTERNLM2_ATTENTION_CLASSES[config.attn_implementation](config=config)

        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )

        original_dtype = hidden_states.dtype
        residual = hidden_states.astype(original_dtype)  # todo

        hidden_states = self.attention_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        original_dtype = hidden_states.dtype
        residual = hidden_states.astype(original_dtype)  # todo

        hidden_states = self.ffn_norm(hidden_states)
        try:
            hidden_states = self.feed_forward(hidden_states)
        except Exception:
            hidden_states = self.feed_forward(hidden_states.cast("bfloat16"))
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class InternLM2PretrainedModel(PretrainedModel):
    config_class = InternLM2Config
    base_model_prefix = "model"
    _no_split_modules = ["InternLM2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    transpose_weight_keys = ["wqkv", "wo", "w1", "w2", "w3", "output"]
    _supports_flash_attn_2 = True

    def _init_weights(self, layer):
        std = self.config.initializer_range
        if isinstance(layer, nn.Linear):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer.bias is not None:
                paddle.nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, nn.Embedding):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer._padding_idx is not None:
                with paddle.no_grad():
                    layer.weight[layer._padding_idx] = 0.0

    @classmethod
    def _gen_aoa_config(cls, config: InternLM2Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"model.tok_embeddings.weight -> {model_prefix}tok_embeddings.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.attention_norm.weight -> {model_prefix}layers.$LAYER_ID.attention_norm.weight",
            f"model.layers.$LAYER_ID.ffn_norm.weight -> {model_prefix}layers.$LAYER_ID.ffn_norm.weight",
        ]
        for key in ["wqkv", "wo"]:
            aoa_statements.append(
                f"model.layers.$LAYER_ID.attention.{key}.weight^T -> {model_prefix}layers.$LAYER_ID.attention.{key}.weight"
            )
        for key in ["w1", "w3", "w2"]:
            aoa_statements.append(
                f"model.layers.$LAYER_ID.feed_forward.{key}.weight^T -> {model_prefix}layers.$LAYER_ID.feed_forward.{key}.weight"
            )
        if cls != cls.base_model_class:
            if config.tie_word_embeddings:
                aoa_statements.append("model.tok_embeddings.weight -> output.weight")
            else:
                aoa_statements.append("output.weight^T -> output.weight")
        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: InternLM2Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"{model_prefix}tok_embeddings.weight -> model.tok_embeddings.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.attention_norm.weight -> model.layers.$LAYER_ID.attention_norm.weight",
            f"{model_prefix}layers.$LAYER_ID.ffn_norm.weight -> model.layers.$LAYER_ID.ffn_norm.weight",
        ]
        for key in ["wqkv", "wo"]:
            aoa_statements.append(
                f"{model_prefix}layers.$LAYER_ID.attention.{key}.weight^T -> model.layers.$LAYER_ID.attention.{key}.weight"
            )
        for key in ["w1", "w3", "w2"]:
            aoa_statements.append(
                f"{model_prefix}layers.$LAYER_ID.feed_forward.{key}.weight^T -> model.layers.$LAYER_ID.feed_forward.{key}.weight"
            )
        if not config.tie_word_embeddings and cls != cls.base_model_class:
            aoa_statements.append("output.weight^T -> output.weight")
        return {"aoa_statements": aoa_statements}


@register_base_model
class InternLM2Model(InternLM2PretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`InternLM2DecoderLayer`]

    Args:
        config: InternLM2Config
    """

    _auto_class = "AutoModel"

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        if not has_flash_attn:
            self.config.attn_implementation = "eager"
            logger.warning_once("Warning: Flash attention is not available, using eager attention instead.")

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.LayerList([InternLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.enable_recompute = False
        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype="int64",
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        if self.config.attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask.numpy()) else None
        else:
            if attention_mask is None:
                attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if self.enable_recompute and has_gradient and self.training:
                layer_outputs = paddle.distributed.fleet.utils.recompute(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Modified from transformers.model.llama.modeling_llama.LlamaForCausalLM
class InternLM2ForCausalLM(InternLM2PretrainedModel):
    _auto_class = "AutoModelForCausalLM"

    _tied_weights_keys = ["output.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

        # Initialize weights and apply final processing
        # self.post_init()

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
        input_ids: paddle.Tensor = None,  # Nonr
        attention_mask: Optional[paddle.Tensor] = None,  # [2, 2336]
        position_ids: Optional[paddle.Tensor] = None,  # None
        past_key_values: Optional[List[paddle.Tensor]] = None,  # None
        inputs_embeds: Optional[paddle.Tensor] = None,  # [2, 2336, 2048]
        labels: Optional[paddle.Tensor] = None,  # None
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,  # False
        output_hidden_states: Optional[bool] = None,  # False
        return_dict: Optional[bool] = None,  # True
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, InternLM2ForCausalLM

        >>> model = InternLM2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        try:
            logits = self.output(hidden_states)
        except Exception:
            logits = self.output(hidden_states.cast("bfloat16"))

        logits = logits.cast("float32")

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            logits_flat = logits.reshape([-1, self.config.vocab_size])
            labels_flat = labels.reshape([-1])
            loss = loss_fct(logits_flat, labels_flat)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            attention_mask_int64 = attention_mask.cast("int64")
            position_ids = attention_mask_int64.cumsum(-1) - 1
            # position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = paddle.where(attention_mask_int64 == 0, paddle.ones_like(position_ids), position_ids)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(paddle.index_select(past_state, index=beam_idx, axis=0) for past_state in layer_past),
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
        query,
        history=None,
        streamer=None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        meta_instruction: str = "You are an AI assistant whose name is InternLM.\n"
        "- InternLM is a conversational language model developed by Shanghai AI Laboratory. It is designed to be helpful, honest, and harmless.\n"
        "- InternLM can understand and communicate fluently in the language chosen by the user such as English and Chinese.",
        **kwargs,
    ):
        inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
        inputs = {k: v for k, v in inputs.items() if paddle.is_tensor(v)}
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
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

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        new_num_tokens = new_embeddings.weight.shape[0]

        old_output_embeddings = self.get_output_embeddings()
        if old_output_embeddings is not None:
            new_output_embeddings = self._get_resized_embeddings(old_output_embeddings, new_num_tokens)
            self.set_output_embeddings(new_output_embeddings)
        return self.get_input_embeddings()


class InternLM2ForSequenceClassification(InternLM2PretrainedModel):
    _auto_class = "AutoModelForSequenceClassification"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLM2Model(config)
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
        past_key_values: Optional[List[paddle.Tensor]] = None,
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


class InternLM2ForQuestionAnswering(InternLM2PretrainedModel):
    _auto_class = "AutoModelForQuestionAnswering"
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = InternLM2Model(config)
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
        past_key_values: Optional[List[paddle.Tensor]] = None,
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


class InternLM2ForTokenClassification(InternLM2PretrainedModel):
    _auto_class = "AutoModelForTokenClassification"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLM2Model(config)
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
