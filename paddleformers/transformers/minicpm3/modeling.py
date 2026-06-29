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

import logging
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    mark_as_sequence_parallel_parameter,
)

import paddleformers

from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.norm import Norm as GeneralNorm
from ...trainer.utils.doc import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ..activations import ACT2FN
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from .configuration import MiniCPM3Config


def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype=paddle.get_default_dtype())  # switch to float if need + fp16 compatibility
    return head_mask


def _get_head_mask(
    self,
    head_mask: Optional[paddle.Tensor],
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
):
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


setattr(paddleformers.transformers.model_utils.PretrainedModel, "get_head_mask", _get_head_mask)

setattr(paddleformers.transformers.model_utils.PretrainedModel, "device", None)


def _post_init(self):
    if hasattr(self, "init_weights"):
        self.init_weights()
    elif hasattr(self, "_init_weights"):
        self._init_weights()


setattr(paddleformers.transformers.model_utils.PretrainedModel, "post_init", _post_init)
logger = logging.getLogger(name=__name__)
_CONFIG_FOR_DOC = "MiniCPM3Config"
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


def replace_return_docstrings(*args, **kwargs):
    def docstring_decorator(fn):
        return fn

    return docstring_decorator


def pad_input(hidden_states, indices, batch, seqlen):
    dim = hidden_states.shape[-1]
    output = paddle.zeros([batch * seqlen, dim], dtype=hidden_states.dtype)
    output = paddle.scatter(output, indices, hidden_states)
    return output.reshape([batch, seqlen, dim])


def unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = paddle.sum(attention_mask, axis=-1, dtype="int32")
    indices = paddle.nonzero(attention_mask.flatten()).flatten()
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0, dtype="int32"), (1, 0), value=0)
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    dim = hidden_states.shape[-1]
    output = paddle.gather(hidden_states.reshape([-1, dim]), indices, axis=0)
    return output, indices, cu_seqlens, max_seqlen_in_batch


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=paddle.int32)
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = paddle.max(seqlens_in_batch).item()
    cu_seqlens = paddle.compat.nn.functional.pad(
        paddle.cumsum(seqlens_in_batch, dim=0, dtype=paddle.paddle.int32), (1, 0)
    )
    return indices, cu_seqlens, max_seqlen_in_batch


def build_causal_attention_mask(
    input_ids,
    attention_mask=None,
    past_key_values_length=0,
):
    # input_ids: [bsz, seq_len]
    causal_mask, row_indices = create_causal_mask_and_row_indices(
        input_ids,
        attention_mask=attention_mask,
        past_key_values_length=past_key_values_length,
    )
    return causal_mask


def rms_layernorm(hidden: paddle.Tensor, weight: paddle.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(paddle.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * paddle.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(paddle.nn.Module):
    def __init__(self, hidden_size, eps=1e-06):
        """
        MiniCPMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.nn.Parameter(paddle.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPMRMSNorm)


class MiniCPMRotaryEmbedding(paddle.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (paddle.arange(0, self.dim, 2).float().to(device) / self.dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=paddle.float32,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(t, self.inv_freq)
        emb = paddle.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = paddle.outer(t, self.inv_freq)
        emb = paddle.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (paddle.arange(0, self.dim, 2).float().to(device) / self.dim)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = paddle.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(t, self.inv_freq)
        emb = paddle.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMLongRoPE(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        if seq_len > self.original_max_position_embeddings:
            ext_factors = paddle.tensor(self.long_factor, dtype=paddle.float32, device=device)
        else:
            ext_factors = paddle.tensor(self.short_factor, dtype=paddle.float32, device=device)
        freqs = paddle.mul(
            paddle.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype),
        )
        emb = paddle.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype) * self.scaling_factor, persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype) * self.scaling_factor, persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_fp32 = q.to(dtype=paddle.float32, device=q.device)
    k_fp32 = k.to(dtype=paddle.float32, device=k.device)
    q_embed = q_fp32 * cos + rotate_half(q_fp32) * sin
    k_embed = k_fp32 * cos + rotate_half(k_fp32) * sin
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


class MiniCPMMLP(paddle.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = GeneralLinear.create(
            self.hidden_size,
            self.intermediate_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
            gather_output=False,
        )
        self.up_proj = GeneralLinear.create(
            self.hidden_size,
            self.intermediate_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
            gather_output=False,
        )
        self.down_proj = GeneralLinear.create(
            self.intermediate_size,
            self.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
            gather_output=False,
            input_is_parallel=True,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            gate_proj_slices = paddle.split(self.gate_proj.weight, self.config.pretraining_tp, axis=1)
            up_proj_slices = paddle.split(self.up_proj.weight, self.config.pretraining_tp, axis=1)
            down_proj_slices = paddle.split(self.down_proj.weight, self.config.pretraining_tp, axis=0)
            gate_proj = paddle.cat(
                [nn.functional.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = paddle.cat(
                [nn.functional.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            intermediate_states = paddle.split(self.act_fn(gate_proj) * up_proj, self.config.pretraining_tp, axis=2)
            down_proj = [
                nn.functional.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MiniCPMAttention(paddle.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPM3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.hidden_size // config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = GeneralLinear.create(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                has_bias=False,
                config=config,
                tp_plan="colwise",
                gather_output=False,
            )
        else:
            self.q_a_proj = GeneralLinear.create(
                self.hidden_size,
                config.q_lora_rank,
                has_bias=config.attention_bias,
                config=config,
                linear_type="default",
                gather_output=False,
            )
            self.q_b_proj = GeneralLinear.create(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                has_bias=False,
                config=config,
                tp_plan="colwise",
                gather_output=False,
            )
        self.q_a_layernorm = MiniCPMRMSNorm(config.q_lora_rank)

        self.kv_a_proj_with_mqa = GeneralLinear.create(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            has_bias=config.attention_bias,
            config=config,
            linear_type="default",
            gather_output=False,
        )

        self.kv_b_proj = GeneralLinear.create(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            has_bias=False,
            config=config,
            tp_plan="colwise",
            gather_output=False,
        )

        self.o_proj = GeneralLinear.create(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="rowwise",
            gather_output=False,
            input_is_parallel=True,
        )

        self.kv_a_layernorm = MiniCPMRMSNorm(config.kv_lora_rank)

        self._init_rope()
        self.softmax_scale = self.q_head_dim**-0.5
        self.attn_implementation = config._attn_implementation

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.config.rope_scaling["factor"],
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.config.rope_scaling["factor"],
                    base=self.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPMLongRoPE(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor=self.config.rope_scaling["short_factor"],
                    long_factor=self.config.rope_scaling["long_factor"],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    # def forward(
    #     self,
    #     hidden_states: paddle.Tensor,
    #     attention_mask: Optional[paddle.Tensor] = None,
    #     position_ids: Optional[paddle.LongTensor] = None,
    #     past_key_value: Optional[Cache] = None,
    #     output_attentions: bool = False,
    #     attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    #     position_embeddings: Optional[Tuple[paddle.Tensor]] = None,
    #     use_cache: bool = False,
    #     **kwargs,
    # ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
    #     if "padding_mask" in kwargs:
    #         warnings.warn(
    #             "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
    #         )
    #     bsz, q_len, _ = hidden_states.size()
    #     q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    #     q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    #     q_nope, q_pe = paddle.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)
    #     compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    #     compressed_kv, k_pe = paddle.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)
    #     k_pe = k_pe.reshape([-1, q_len, 1, self.qk_rope_head_dim]).expand(
    #         [-1, q_len, self.num_heads, self.qk_rope_head_dim]
    #     )
    #     kv = (
    #         self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
    #         .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
    #         .transpose(1, 2)
    #     )
    #     k_nope, value_states = paddle.split(kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1)
    #     kv_seq_len = value_states.shape[-2]
    #     if past_key_value is not None:
    #         if self.layer_idx is None:
    #             raise ValueError(
    #                 f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index."
    #             )
    #         kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
    #     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    #     q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    #     # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    #     # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    #     # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    #     # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    #     # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    #     # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    #     query_states = paddle.cat([q_nope, q_pe], axis=-1)
    #     key_states = paddle.cat([k_nope, k_pe], axis=-1)
    #     if past_key_value is not None:
    #         cache_kwargs = {"sin": sin, "cos": cos}
    #         key_states, value_states = past_key_value.update(
    #             key_states, value_states, self.layer_idx, cache_kwargs
    #         )
    #     attn_weights = (
    #         paddle.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
    #     )
    #     if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #         raise ValueError(
    #             f"Attention weights should be of size {bsz, self.num_heads, q_len, kv_seq_len}, but is {attn_weights.size()}"
    #         )
    #     assert attention_mask is not None
    #     if attention_mask is not None:
    #         if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #             raise ValueError(
    #                 f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.size()}"
    #             )
    #         attn_weights = attn_weights + attention_mask
    #     attn_weights = paddle.compat.nn.functional.softmax(
    #         attn_weights, dim=-1, dtype=paddle.float32
    #     ).to(query_states.dtype)
    #     attn_weights = paddle.nn.functional.dropout(
    #         attn_weights, p=self.attention_dropout, training=self.training
    #     )
    #     attn_output = paddle.matmul(attn_weights, value_states)
    #     if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
    #         raise ValueError(
    #             f"`attn_output` should be of size {bsz, self.num_heads, q_len, self.v_head_dim}, but is {attn_output.size()}"
    #         )
    #     attn_output = attn_output.transpose(1, 2).contiguous()
    #     attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
    #     attn_output = self.o_proj(attn_output)
    #     if not output_attentions:
    #         attn_weights = None
    #     return attn_output, attn_weights, past_key_value
    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated, use `attention_mask` instead.")

        bsz, q_len, _ = hidden_states.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.reshape([bsz, q_len, self.num_heads, self.q_head_dim]).transpose([0, 2, 1, 3])
        q_nope, q_pe = paddle.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            axis=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = paddle.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
        )
        k_pe = k_pe.reshape([bsz, q_len, 1, self.qk_rope_head_dim]).transpose([0, 2, 1, 3])

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape([bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim]).transpose([0, 2, 1, 3])
        k_nope, value_states = paddle.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            axis=-1,
        )

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please initialize it with a layer index."
                )
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(
            q_pe,
            k_pe,
            cos,
            sin,
            position_ids,
        )

        k_pe = k_pe.expand([bsz, self.num_heads, q_len, self.qk_rope_head_dim])
        query_states = paddle.cat([q_nope, q_pe], axis=-1)
        key_states = paddle.cat([k_nope, k_pe], axis=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        kv_seq_len = key_states.shape[-2]

        attn_weights = (
            paddle.matmul(
                query_states,
                key_states.transpose([0, 1, 3, 2]),
            )
            * self.softmax_scale
        )

        if attention_mask is not None:
            # attention_mask: [bsz, 1, q_len, kv_seq_len]
            attn_weights = attn_weights + attention_mask

        attn_weights = paddle.nn.functional.softmax(attn_weights.astype(paddle.float32), axis=-1).astype(
            query_states.dtype
        )

        attn_weights = paddle.nn.functional.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
        )

        attn_output = paddle.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, self.num_heads * self.v_head_dim])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MiniCPMFlashAttention2(MiniCPMAttention):
    """
    MiniCPM flash attention module. This module inherits from `MiniCPMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.LongTensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
            attention_mask = kwargs.pop("padding_mask")
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = paddle.compat.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = paddle.compat.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value_states = paddle.compat.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if self.q_head_dim != self.v_head_dim:
            value_states = paddle.compat.nn.functional.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        dropout_rate = self.attention_dropout if self.training else 0.0
        input_dtype = query_states.dtype
        if input_dtype == paddle.float32:
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif paddle.is_autocast_enabled():
                target_dtype = paddle.get_autocast_gpu_dtype()
            else:
                target_dtype = self.q_a_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = paddle.nn.functional.flash_attention.flash_attn_unpadded(
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
            attn_output = paddle.nn.functional.flash_attention.flash_attention(
                query_states,
                key_states,
                value_states,
                dropout,
                causal=causal,
            )
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        key_layer = paddle.gather(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = paddle.gather(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = paddle.gather(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = paddle.arange(batch_size + 1, dtype=paddle.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            (
                query_layer,
                indices_q,
                cu_seqlens_q,
                max_seqlen_in_batch_q,
            ) = unpad_input(query_layer, attention_mask)
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MiniCPMSdpaAttention(MiniCPMAttention):
    """
    MiniCPM attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MiniCPMAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                'MiniCPM3Model is using MiniCPMSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.size()
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = paddle.compat.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = paddle.compat.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value_states = paddle.compat.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index."
                )
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.size()}"
                )
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = paddle.compat.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "flash_attention_2": MiniCPMFlashAttention2,
    "sdpa": MiniCPMSdpaAttention,
}


class MiniCPMDecoderLayer(paddle.nn.Module):
    def __init__(self, config: MiniCPM3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MINICPM_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            input_is_parallel=False,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            input_is_parallel=False,
        )
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_dropout = nn.Dropout(p=config.hidden_dropout_prob, mode="upscale_in_train")
        if config.sequence_parallel:
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()
                if config.use_bias:
                    mark_as_sequence_parallel_parameter(self.self_attn.o_proj.bias)
                    mark_as_sequence_parallel_parameter(self.mlp.down_proj.bias)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[paddle.FloatTensor, Optional[Tuple[paddle.FloatTensor, paddle.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


MINICPM_START_DOCSTRING = """
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MiniCPM3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPM3PreTrainedModel(paddleformers.transformers.PretrainedModel):
    config_class = MiniCPM3Config
    _keep_in_fp32_modules = []
    base_model_prefix = "model"
    transpose_weight_keys = [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniCPMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if "load_checkpoint_format" not in kwargs and kwargs.get("convert_from_hf", True) is not False:
            weight_path = os.path.join(str(pretrained_model_name_or_path), "pytorch_model.bin")
            if os.path.isfile(weight_path):
                kwargs["load_checkpoint_format"] = "legacy"
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (paddle.nn.Linear, paddle.compat.nn.Linear, GeneralLMHead)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, paddle.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # @classmethod
    # def _get_tensor_parallel_mappings(cls, config, is_split=True):
    #     """Generate tensor parallel mappings for model conversion."""
    #     from ..conversion_utils import split_or_merge_func

    #     fn = split_or_merge_func(
    #         is_split=is_split,
    #         tensor_model_parallel_size=config.tensor_model_parallel_size,
    #         tensor_parallel_rank=config.tensor_parallel_rank,
    #         num_attention_heads=config.num_attention_heads,
    #     )

    #     LAYER_COLWISE = [
    #         "self_attn.q_a_proj.weight",
    #         "self_attn.q_b_proj.weight",
    #         "self_attn.kv_a_proj_with_mqa.weight",
    #         "self_attn.kv_b_proj.weight",
    #         "mlp.up_proj.weight",
    #         "mlp.gate_proj.weight",
    #     ]

    #     LAYER_ROWWISE = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]

    #     BIAS_KEYS = [
    #         "self_attn.q_a_proj.bias",
    #         "self_attn.q_b_proj.bias",
    #         "self_attn.kv_a_proj_with_mqa.bias",
    #         "self_attn.kv_b_proj.bias",
    #         "mlp.up_proj.bias",
    #         "self_attn.o_proj.bias",
    #         "mlp.down_proj.bias",
    #         "mlp.gate_proj.bias",
    #         "lm_head.bias",
    #     ]

    #     def make_base_actions():
    #         actions = {
    #             "lm_head.weight": partial(fn, is_column=False),
    #             "embed_tokens.weight": partial(fn, is_column=False),
    #         }
    #         for layer_idx in range(config.num_hidden_layers):
    #             actions.update(
    #                 {
    #                     f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
    #                     for k in LAYER_COLWISE
    #                 }
    #             )
    #             actions.update(
    #                 {
    #                     f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=False)
    #                     for k in LAYER_ROWWISE
    #                 }
    #             )
    #             # bias
    #             if config.use_bias:
    #                 actions.update(
    #                     {f"{cls.base_model_prefix}.layers.0.{b}": partial(fn, is_column=True) for b in BIAS_KEYS}
    #                 )

    #         return actions

    #     mappings = make_base_actions()
    #     return mappings

    # @classmethod
    # def _gen_aoa_config(cls, config: MiniCPM3Config):
    #     model_prefix = "" if cls == cls.base_model_prefix else "model."
    #     aoa_statements = [
    #         # O 投影
    #         f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
    #         # FFN
    #         f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.gate_proj.weight",
    #         f"model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_proj.weight",
    #         # Embedding
    #         f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
    #         # LayerNorms
    #         f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
    #         f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
    #         f"model.layers.$LAYER_ID.self_attn.q_a_layernorm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_a_layernorm.weight",
    #         f"model.layers.$LAYER_ID.self_attn.kv_a_layernorm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.kv_a_layernorm.weight",
    #         f"model.norm.weight -> {model_prefix}norm.weight",
    #     ]

    #     # Attention Q/K/V 两级拆分
    #     aoa_statements += [
    #         f"model.layers.$LAYER_ID.self_attn.q_a_proj.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_a_proj.weight",
    #         f"model.layers.$LAYER_ID.self_attn.q_b_proj.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_b_proj.weight",
    #         f"model.layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.weight -> {model_prefix}layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.weight",
    #         f"model.layers.$LAYER_ID.self_attn.kv_b_proj.weight -> {model_prefix}layers.$LAYER_ID.self_attn.kv_b_proj.weight",
    #     ]

    #     # Biases（如果使用）
    #     if config.use_bias:
    #         aoa_statements += [
    #             f"model.layers.$LAYER_ID.self_attn.q_a_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.q_a_proj.bias",
    #             f"model.layers.$LAYER_ID.self_attn.q_b_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.q_b_proj.bias",
    #             f"model.layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.bias -> {model_prefix}layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.bias",
    #             # kv_b_proj 没 bias
    #             f"model.layers.$LAYER_ID.mlp.gate_proj.bias -> {model_prefix}layers.$LAYER_ID.mlp.gate_proj.bias",
    #             f"model.layers.$LAYER_ID.mlp.up_proj.bias -> {model_prefix}layers.$LAYER_ID.mlp.up_proj.bias",
    #         ]

    #     # LM head
    #     if config.tie_word_embeddings:
    #         aoa_statements += ["model.embed_tokens.weight -> lm_head.weight"]

    #     return {"aoa_statements": aoa_statements}

    # @classmethod
    # def _gen_inv_aoa_config(cls, config: MiniCPM3Config):
    #     model_prefix = "" if cls == cls.base_model_prefix else "model."
    #     # 直接反向映射
    #     aoa_statements = []
    #     for stmt in cls._gen_aoa_config(config)["aoa_statements"]:
    #         lhs, rhs = stmt.split("->")
    #         aoa_statements.append(f"{rhs.strip()} -> {lhs.strip()}")

    #     return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_aoa_config(cls, config: MiniCPM3Config):
        model_prefix = cls.base_model_prefix + "." if cls.__name__ != "MiniCPM3Model" else ""
        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            f"model.layers.$LAYER_ID.self_attn.q_a_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.q_a_proj.weight",
            f"model.layers.$LAYER_ID.self_attn.q_b_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.q_b_proj.weight",
            f"model.layers.$LAYER_ID.self_attn.q_a_layernorm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_a_layernorm.weight",
            f"model.layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.weight",
            f"model.layers.$LAYER_ID.self_attn.kv_b_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.kv_b_proj.weight",
            f"model.layers.$LAYER_ID.self_attn.kv_a_layernorm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.kv_a_layernorm.weight",
            f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
            f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.gate_proj.weight",
            f"model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_proj.weight",
            f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
        ]

        if config.attention_bias:
            aoa_statements.extend(
                [
                    f"model.layers.$LAYER_ID.self_attn.q_a_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.q_a_proj.bias",
                    f"model.layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.bias -> {model_prefix}layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.bias",
                    f"model.layers.$LAYER_ID.self_attn.o_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.bias",
                ]
            )

        if not config.tie_word_embeddings and cls.__name__ != "MiniCPM3Model":
            aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: MiniCPM3Config):
        model_prefix = cls.base_model_prefix + "." if cls.__name__ != "MiniCPM3Model" else ""
        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.q_a_proj.weight^T -> model.layers.$LAYER_ID.self_attn.q_a_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.q_b_proj.weight^T -> model.layers.$LAYER_ID.self_attn.q_b_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.q_a_layernorm.weight -> model.layers.$LAYER_ID.self_attn.q_a_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.weight^T -> model.layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.kv_b_proj.weight^T -> model.layers.$LAYER_ID.self_attn.kv_b_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.kv_a_layernorm.weight -> model.layers.$LAYER_ID.self_attn.kv_a_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.gate_proj.weight^T -> model.layers.$LAYER_ID.mlp.gate_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.up_proj.weight^T -> model.layers.$LAYER_ID.mlp.up_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
        ]

        if config.attention_bias:
            aoa_statements.extend(
                [
                    f"{model_prefix}layers.$LAYER_ID.self_attn.q_a_proj.bias -> model.layers.$LAYER_ID.self_attn.q_a_proj.bias",
                    f"{model_prefix}layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.bias -> model.layers.$LAYER_ID.self_attn.kv_a_proj_with_mqa.bias",
                    f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.bias -> model.layers.$LAYER_ID.self_attn.o_proj.bias",
                ]
            )

        if not config.tie_word_embeddings and cls.__name__ != "MiniCPM3Model":
            aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}


MINICPM_INPUTS_DOCSTRING = """
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPM3Model(MiniCPM3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMDecoderLayer`]

    Args:
        config: MiniCPM3Config
    """

    def __init__(self, config: MiniCPM3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = paddle.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = paddle.nn.ModuleList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices,
        position_ids,
        position_embeddings,
        output_attentions,
        past_key_values,
        use_cache,
    ):
        """Perform gradient checkpointing for memory-efficient training.

        Args:
            layer_module (nn.Layer): Transformer layer to recompute
            hidden_states (paddle.Tensor): Input hidden states
            attention_mask (paddle.Tensor): Attention mask
            attn_mask_startend_row_indices (paddle.Tensor): Variable length indices
            position_ids (paddle.Tensor): Position indices
            position_embeddings (paddle.Tensor): Position embeddings
            output_attentions (bool): Whether to output attention weights
            past_key_values (Optional[Cache]): Cached key/value states
            use_cache (bool): Whether to cache key/value states

        Returns:
            paddle.Tensor: Output hidden states after recomputation
        """

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            attn_mask_startend_row_indices,
            position_ids,
            position_embeddings,
            output_attentions,
            past_key_values,
            use_cache,
        )
        return hidden_states

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[List[paddle.FloatTensor]] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, paddleformers.transformers.model_outputs.BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_key_values_length = past_key_values.get_seq_length(seq_length)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = paddle.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=paddle.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        attn_mask_startend_row_indices = None
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": past_key_values_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        causal_attention_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        # if self._use_flash_attention_2:
        #     attention_mask = (
        #         attention_mask
        #         if attention_mask is not None and 0 in attention_mask
        #         else None
        #     )
        # if attention_mask is not None:
        #    attention_mask = attention_mask[:, -seq_length:]

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    causal_attention_mask,
                    attn_mask_startend_row_indices,
                    # attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return paddleformers.transformers.model_outputs.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiniCPM3ForCausalLM(MiniCPM3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPM3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[List[paddle.FloatTensor]] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        labels: Optional[paddle.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, paddleformers.transformers.model_outputs.CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniCPMForCausalLM

        >>> model = MiniCPMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pd")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\\nI'm not conscious, but I can talk to you."
        ```"""
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
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
        logits = logits.float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = paddle.nn.CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            valid_mask = shift_labels != -100
            loss = paddle.where(valid_mask, loss, paddle.zeros_like(loss))
            loss = loss.sum() / valid_mask.astype(loss.dtype).sum().clip(min=1)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return paddleformers.transformers.model_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: paddle.Tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            if not isinstance(past_key_values, Cache):
                raise ValueError("Only Cache class is supported for past_key_values.")

            cache_length = past_key_values.get_seq_length()
            past_length = cache_length
            max_cache_length = past_key_values.get_max_cache_shape()

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                max_cache_length is not None
                and max_cache_length > 0
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids = position_ids.astype("int64")
            position_ids = paddle.where(
                attention_mask == 0,
                paddle.ones_like(position_ids),
                position_ids,
            )
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {}
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", False),
            }
        )

        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    @paddle.no_grad()
    def chat(
        self,
        tokenizer,
        query: str,
        history: List[Dict] = None,
        role: str = "user",
        max_length: int = 4096,
        num_beams=1,
        do_sample=True,
        top_p=0.8,
        temperature=0.3,
        logits_processor=None,
        **kwargs,
    ):
        if history is None:
            history = []
        if logits_processor:
            gen_kwargs = {
                "max_length": max_length,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "top_p": top_p,
                "temperature": temperature,
                "logits_processor": logits_processor,
                **kwargs,
            }
        else:
            gen_kwargs = {
                "max_length": max_length,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "top_p": top_p,
                "temperature": temperature,
                "logits_processor": logits_processor,
                **kwargs,
            }
        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(history_str, return_tensors="pd")
        device = paddle.get_device()
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) : -1]
        response = tokenizer.decode(outputs)
        history.append({"role": "assistant", "content": response})
        return response, history


@add_start_docstrings(
    """
    The MiniCPM Model transformer with a sequence classification head on top (linear layer).

    [`MiniCPMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MINICPM_START_DOCSTRING,
)
class MiniCPM3ForSequenceClassification(MiniCPM3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MiniCPM3Model(config)
        self.score = paddle.compat.nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[List[paddle.FloatTensor]] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        labels: Optional[paddle.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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
        elif input_ids is not None:
            sequence_lengths = (paddle.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(logits.device)
        else:
            sequence_lengths = -1
        pooled_logits = logits[paddle.arange(batch_size, device=logits.device), sequence_lengths]
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.long or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = paddle.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
