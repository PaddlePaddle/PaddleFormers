# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2023 DeepSeek. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Paddle DeepSeek model."""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from paddle.nn.functional.flash_attention import flash_attention

from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as DeepseekV2MLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.norm import RMSNorm
from ...nn.pp_model import EmbeddingPipe, GeneralModelForCausalLMPipe, parse_args
from ...utils.initializer import kaiming_uniform_
from ...utils.log import logger
from ...utils.tools import get_env_device
from .. import linear_utils
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..llama import fusion_ops
from ..llama.modeling import get_use_casual_mask
from ..model_outputs import (
    BaseModelOutputWithPastAndMTP,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ..model_utils import PretrainedModel, register_base_model
from ..moe_gate import PretrainedMoEGate
from ..moe_layer import MoEFlexTokenLayer
from ..utils import device_guard
from .configuration import DeepseekV2Config

__all__ = [
    "DeepseekV2ForCausalLM",
    "DeepseekV2ForSequenceClassification",
    "DeepseekV2Model",
    "DeepseekV2PretrainedModel",
    "DeepseekV2ForCausalLMPipe",
]


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def assign_kv_heads(num_kv_heads: int, num_gpus: int):
    # Initialize the assignment list
    """
    Assign kv heads to different GPUs in the Tensor Parallel Setup

    Examples:
        assign_kv_heads(num_kv_heads=1, num_gpus=2): [[0], [0]]
        assign_kv_heads(num_kv_heads=2, num_gpus=2): [[0], [1]]
        assign_kv_heads(num_kv_heads=4, num_gpus=2): [[0,1], [2,3]]
        assign_kv_heads(num_kv_heads=1, num_gpus=4): [[0],[0],[0],[0]]
        assign_kv_heads(num_kv_heads=2, num_gpus=4): [[0],[0],[1],[1]]
        assign_kv_heads(num_kv_heads=4, num_gpus=4): [[0],[1],[2],[3]]
    """
    assignment_list = [[] for _ in range(num_gpus)]
    # Case 1: more heads than cards
    if num_kv_heads > num_gpus:
        num_heads_per_card = num_kv_heads // num_gpus
        for i in range(num_gpus):
            for j in range(num_heads_per_card):
                assignment_list[i].append(i * num_heads_per_card + j)
    # Case 2: more cards than heads. each card get only 1 head.
    else:
        num_card_per_heads = num_gpus // num_kv_heads
        for i in range(num_kv_heads):
            for j in range(num_card_per_heads):
                assignment_list[i * num_card_per_heads + j].append(i)
    return assignment_list


def parallel_matmul(x: Tensor, y: Tensor, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except AttributeError:
        is_fleet_init = False

    if paddle.in_dynamic_mode():
        y_is_distributed = y.is_distributed
    else:
        y_is_distributed = tensor_parallel_degree > 1

    if is_fleet_init and tensor_parallel_degree > 1 and y_is_distributed:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=False)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=False)
        return logits


def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    attn_mask_startend_row_indices=None,
    softmax_scale=1.0,
    training=True,
    sequence_parallel=False,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, v_num_heads, v_head_dim = value_states.shape

    if config.use_flash_attention and flash_attention:
        # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
        # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]

        # Note: Flash Attention does not support softmax_scale, so we need to scale the query_states
        q_head_dim = query_states.shape[-1]
        softmax_scale = softmax_scale * (q_head_dim**0.5)
        query_states = query_states * softmax_scale
        value_padding = paddle.zeros(
            [bsz, kv_seq_len, v_num_heads, head_dim - v_head_dim],
            dtype=value_states.dtype,
        )
        value_states = paddle.cat([value_states, value_padding], axis=-1)

        outputs = fusion_ops.fusion_flash_attention(
            query_states,
            config,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            sequence_parallel=sequence_parallel,
        )

        if isinstance(outputs, tuple):
            outputs[0] = outputs[0].reshape([bsz, q_len, v_num_heads, head_dim])
            outputs[0] = outputs[0][..., :v_head_dim]
            outputs[0] = outputs[0].reshape([bsz, q_len, -1])
        else:
            outputs = outputs.reshape([bsz, q_len, v_num_heads, head_dim])
            outputs = outputs[..., :v_head_dim]
            outputs = outputs.reshape([bsz, q_len, -1])

        if sequence_parallel:
            attn_output = outputs.reshape([bsz * q_len, v_head_dim * num_heads])
        else:
            attn_output = outputs.reshape([bsz, q_len, v_head_dim * num_heads])
        return attn_output

    else:
        #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        # merge with the next transpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        # matmul and divide by sqrt(head_dim)
        attn_weights = paddle.matmul(query_states * softmax_scale, key_states.transpose([0, 1, 3, 2]))

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)
        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attn_weights + attention_mask
        with paddle.amp.auto_cast(False):
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)

        attn_weights = F.dropout(attn_weights, p=config.attention_dropout, training=training)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, v_head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, v_head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask.to("bool"), y, x)


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make casual mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    if get_env_device() == "npu":
        mask = paddle.tril(paddle.ones((target_length, target_length))).astype("int32")
    else:
        mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.cat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    if get_env_device() == "npu":
        mask = mask[:, None, None, :].astype(dtype)
    else:
        mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


class DeepseekV2RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        with device_guard("cpu"):
            self.inv_freq = 1.0 / (
                self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim)
            )
            self._set_cos_sin_cache(seq_len=max_position_embeddings)

        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, axis/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, axis]
        emb = paddle.cat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, axis]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings * scaling_factor, base)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        t = t / self.scaling_factor
        # [seq_len, axis/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, axis]
        emb = paddle.cat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, axis]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]
        self.cos_sin_table = None if get_env_device() != "gcu" else paddle.cat([freqs.cos(), freqs.sin()], axis=-1)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _scale_cos_sin(self, seq_len):
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, axis/2]
        alpha = (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        base = self.base * alpha ** (self.axis / (self.axis - 2))
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, self.axis, 2), dtype="float32") / self.axis))
        freqs = paddle.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, axis]
        emb = paddle.cat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, axis]
        scale_cos = emb.cos()[None, :, None, :]
        scale_sin = emb.sin()[None, :, None, :]
        scale_cos_sin = None if get_env_device() != "gcu" else paddle.cat([freqs.cos(), freqs.sin()], axis=-1)
        return scale_cos, scale_sin, scale_cos_sin

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_position_embeddings:
            scale_cos, scale_sin, _ = self._scale_cos_sin(seq_len=seq_len)
        else:
            scale_cos, scale_sin = self.cos_cached, self.sin_cached
        cos = scale_cos[:, :seq_len, :, ...]
        sin = scale_sin[:, :seq_len, :, ...]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )

    def get_fused_cos_sin(self, x, seq_len=None):
        if seq_len > self.max_position_embeddings:
            _, _, scale_cos_sin = self._scale_cos_sin(seq_len=seq_len)
        else:
            scale_cos_sin = self.cos_sin_table
        if scale_cos_sin is not None and scale_cos_sin.dtype != x.dtype:
            return scale_cos_sin.cast(x.dtype)
        else:
            return scale_cos_sin


# Inverse axis formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find axis range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (paddle.arange(dim, dtype=paddle.float32) - min) / (max - min)
    ramp_func = paddle.clip(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.base ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        self.inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        t = paddle.arange(seq_len, dtype=paddle.float32)

        freqs = paddle.outer(t, paddle.cast(self.inv_freq, dtype="float32"))

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = paddle.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos() * _mscale
        self.sin_cached = emb.sin() * _mscale


def rotate_half(x):
    """Rotates half the hidden axiss of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, fuse_rope=False):
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
    b, s, h, d = q.shape
    q = q.reshape([b, s, h, d // 2, 2]).transpose([0, 1, 2, 4, 3]).reshape([b, s, h, d])

    b, s, h, d = k.shape
    k = k.reshape([b, s, h, d // 2, 2]).transpose([0, 1, 2, 4, 3]).reshape([b, s, h, d])

    if position_ids is None:
        # Note: Only for MixtralForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, axis]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, axis]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, axis]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, axis]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, axis]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, axis]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FakeGate(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, hidden_states, weight):
        expert_num = weight.shape[1]
        bsz, seq, _ = hidden_states.shape

        ctx.x_shape = hidden_states.shape
        ctx.x_dtype = hidden_states.dtype
        ctx.y_shape = weight.shape
        ctx.y_dtype = weight.dtype

        return paddle.randn([bsz, seq, expert_num]).cast(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return paddle.zeros(ctx.x_shape, dtype=ctx.x_dtype), paddle.zeros(ctx.y_shape, dtype=ctx.y_dtype)


class MoEGate(PretrainedMoEGate):
    def __init__(self, config, num_experts, expert_hidden_size, **kwargs):
        super().__init__(config, num_experts, expert_hidden_size, **kwargs)
        # [hidden_size, n_expert]

        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method

        self.weight = paddle.create_parameter(
            shape=[expert_hidden_size, num_experts],
            dtype=paddle.float32,
            is_bias=False,
            # default_initializer=nn.initializer.Constant(1.0),
        )

        self.config = config
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = paddle.create_parameter(
                shape=[num_experts],
                dtype=paddle.float32,
                default_initializer=nn.initializer.Constant(0.0),
            )
            self.e_score_correction_bias.is_distributed = True
            self.e_score_correction_bias.stop_gradient = True
            self.expert_usage = paddle.zeros(
                shape=[num_experts],
                dtype=paddle.int64,
            )
            self.expert_usage.stop_gradient = True

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (_type_): [batch_size * seq_len, hidden_size]
        """
        _, _, h_dim = hidden_states.shape

        # compute gating score
        with paddle.amp.auto_cast(False):
            hidden_states = hidden_states.cast(self.weight.dtype)

            if hasattr(self.config, "using_fake_gate") and self.config.using_fake_gate:
                logits = FakeGate.apply(hidden_states, self.weight)
            else:
                logits = F.linear(hidden_states, self.weight, None)

            scores = self.gate_score_func(logits=logits)
            scores = scores.cast(paddle.float32)

        scores, routing_map, exp_counts, l_aux, l_zloss = self.topkgating_nodrop(scores)
        with paddle.no_grad():
            self.expert_usage += exp_counts
        return scores, routing_map, l_aux, l_zloss


class AddAuxiliaryLoss(paddle.autograd.PyLayer):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert paddle.numel(loss) == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = not loss.stop_gradient
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = paddle.ones(1, dtype=ctx.dtype)
        return grad_output, grad_loss


class DeepseekV2MoEFlexToken(MoEFlexTokenLayer):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV2Config):
        gate = MoEGate(
            config=config,
            num_experts=config.n_routed_experts,
            expert_hidden_size=config.hidden_size,
            top_k=config.num_experts_per_tok,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
            drop_tokens=False,
        )

        hcg = fleet.get_hybrid_communicate_group()
        moe_group = hcg.get_expert_parallel_group()
        moe_grad_group = hcg.get_moe_sharding_parallel_group()
        config = deepcopy(config)
        config.tensor_parallel_degree = 1

        super().__init__(
            config=config,
            moe_num_experts=config.n_routed_experts,
            expert_class=DeepseekV2MLP,
            expert_kwargs={"config": config, "intermediate_size": config.moe_intermediate_size},
            gate=gate,
            moe_group=moe_group,
        )

        self.is_mp_moe = False
        self.is_ep_moe = True
        for p in self.experts.parameters():
            setattr(p, "is_moe_param", True)
            setattr(p, "color", {"color": "moe_expert", "group": moe_grad_group})
            p.no_sync = not self.is_mp_moe
            p.expert = not self.is_mp_moe
            logger.info(f"expert no-sync={p.no_sync}-{p.name}")
            if self.is_mp_moe or self.is_ep_moe:
                p.is_distributed = True

        self.alpha = config.aux_loss_alpha
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        final_hidden_states, l_aux, l_zloss = super().forward(hidden_states)
        if self.training and self.alpha > 0.0:
            l_aux = l_aux * self.alpha
            final_hidden_states = AddAuxiliaryLoss.apply(final_hidden_states, l_aux)

        if self.config.n_shared_experts is not None:
            shared_expert_output = self.shared_experts(hidden_states)
            final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states


class DeepseekV2Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layerwise_recompute: bool = False):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads
        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"Attention head num ({self.num_heads}) is not divisible by tensor_parallel_degree ({config.tensor_parallel_degree})."
            self.num_local_heads = self.num_heads // config.tensor_parallel_degree

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True
        self.fuse_rope = config.use_fused_rope

        self.seq_length = config.seq_length
        self.sequence_parallel = config.sequence_parallel

        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

        # Note (@DrownFish19): For tensor parallel we consider that q_a_proj and kv_a_proj_with_mqa
        # are the small weight and cannot achieve performance gain. So we use the original
        # linear layers. We use the tensor parallel linear layers for q_proj，q_b_proj and kv_b_proj
        # for which are the large weight and can achieve performance gain.

        # fmt: off

        if self.q_lora_rank is None:
            self.q_proj = GeneralLinear.create(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                has_bias=False,
                config=config,
                fuse_matmul_bias=config.fuse_linear,
                tp_plan="colwise",
                gather_output=False,
            )
        else:
            self.q_a_proj = GeneralLinear.create(
                self.hidden_size,
                config.q_lora_rank,
                has_bias=config.attention_bias,
                config=config,
                fuse_matmul_bias=config.fuse_linear,
                linear_type="default",
                gather_output=False,
            )
            self.q_b_proj = GeneralLinear.create(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                has_bias=False,
                config=config,
                fuse_matmul_bias=config.fuse_linear,
                tp_plan="colwise",
                gather_output=False,
            )
        self.q_a_layernorm = GeneralNorm.create(
            config=config,
            hidden_size=config.q_lora_rank,
            norm_type="rms_norm",
        )

        self.kv_a_proj_with_mqa = GeneralLinear.create(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            has_bias=config.attention_bias,
            config=config,
            fuse_matmul_bias=config.fuse_linear,
            linear_type="default",
            gather_output=False,
        )

        self.kv_b_proj = GeneralLinear.create(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            has_bias=False,
            config=config,
            fuse_matmul_bias=config.fuse_linear,
            tp_plan="colwise",
            gather_output=False,
        )

        self.o_proj = GeneralLinear.create(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            has_bias=config.attention_bias,
            config=config,
            fuse_matmul_bias=config.fuse_linear,
            tp_plan="rowwise",
            gather_output=False,
            input_is_parallel=True
        )

        self.kv_a_layernorm = GeneralNorm.create(
            config=config,
            hidden_size=config.kv_lora_rank,
            norm_type="rms_norm",
        )

        # fmt: on
        if self.config.tensor_parallel_degree > 1 and self.config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.kv_a_proj_with_mqa.weight)
            mark_as_sequence_parallel_parameter(self.q_a_proj.weight)
            if config.attention_bias:
                mark_as_sequence_parallel_parameter(self.kv_a_proj_with_mqa.bias)
                mark_as_sequence_parallel_parameter(self.q_a_proj.bias)

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_func = scaled_dot_product_attention

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.v_head_dim]).transpose([1, 0, 2, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        ori_shape = hidden_states.shape
        if self.config.sequence_parallel:
            seq_len, bsz, _ = hidden_states.shape
        else:
            bsz, seq_len, _ = hidden_states.shape

        # DeepSeekV2 q_lora_rank=1536
        # DeepSeekV2-lite q_lora_rank=None
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        if self.sequence_parallel:
            target_query_shape = [bsz, self.seq_length, self.num_local_heads, self.q_head_dim]
            target_key_value_shape = [
                bsz,
                self.seq_length,
                self.num_local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ]
        else:
            target_query_shape = [0, 0, self.num_heads, self.q_head_dim]
            target_key_value_shape = [0, 0, self.num_heads, self.qk_nope_head_dim + self.v_head_dim]

        q = q.reshape(shape=target_query_shape)
        q_nope, q_pe = paddle.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

        # DeepSeekV2 kv_lora_rank+qk_rope_head_dim=512+64
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = paddle.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)
        if self.sequence_parallel:
            k_pe = GatherOp.apply(k_pe)
            k_pe = paddle.transpose(k_pe, [1, 0, 2])
        k_pe = k_pe.reshape([-1, self.seq_length, 1, self.qk_rope_head_dim]).expand(
            [-1, self.seq_length, self.num_local_heads, self.qk_rope_head_dim]
        )

        # self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim = 128+64
        # self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim) = config.qk_nope_head_dim + self.v_head_dim = 128+128
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).reshape(shape=target_key_value_shape)

        k_nope, value_states = paddle.split(kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1)
        kv_seq_len = value_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, self.fuse_rope)

        query_states = paddle.cat([q_nope, q_pe], axis=-1)
        key_states = paddle.cat([k_nope, k_pe], axis=-1)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.cat([past_key_value[0], key_states], axis=1)
            value_states = paddle.cat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "core_attn"
        ):
            outputs = recompute(
                self.attn_func,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                softmax_scale=self.softmax_scale,
                training=self.training,
                sequence_parallel=self.sequence_parallel,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = self.attn_func(
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                softmax_scale=self.softmax_scale,
                training=self.training,
                sequence_parallel=self.sequence_parallel,
            )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        attn_output = self.o_proj(attn_output)
        if attn_output.shape != ori_shape:
            attn_output = attn_output.reshape(ori_shape)

        if not output_attentions:
            attn_weights = None

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class DeepseekV2DecoderLayer(nn.Layer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int, layerwise_recompute: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV2Attention(config=config, layerwise_recompute=layerwise_recompute)

        MoELayerClass = DeepseekV2MoEFlexToken

        self.mlp = (
            MoELayerClass(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )

        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
        )

    def subbatch_recompute_forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        offload_kwargs = {}
        offload_kwargs["offload_indices"] = [0]
        assert self.recompute_granularity != "full_attn"
        attn_outputs = recompute(
            self.attn,
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            **offload_kwargs,
        )

        hidden_states = attn_outputs[0]
        residual = attn_outputs[1]
        self_attn_weights = attn_outputs[2] if output_attentions else None
        present_key_value = attn_outputs[3] if use_cache else None

        assert len(hidden_states.shape) == 3
        sub_seq_len = self.config.moe_subbatch_token_num
        seq_axis = 0 if self.config.sequence_parallel else 1
        seq_len = hidden_states.shape[seq_axis]
        assert seq_len % sub_seq_len == 0
        num_chunks = seq_len // sub_seq_len
        split_list = [sub_seq_len] * num_chunks
        input_list = paddle.split(hidden_states, split_list, axis=seq_axis)
        output_list = []

        for chunk in input_list:
            out = recompute(
                self.mlp.forward,
                chunk,
                **offload_kwargs,
            )
            output_list.append(out)
        hidden_states = paddle.concat(output_list, axis=seq_axis)
        outputs = recompute(
            self.post_process,
            hidden_states,
            residual,
            output_attentions,
            use_cache,
            self_attn_weights,
            present_key_value,
            **offload_kwargs,
        )
        return outputs

    def attn(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "full_attn"
        ):
            outputs = recompute(
                self.self_attn,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                **kwargs,
            )
        else:
            outputs = self.self_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                **kwargs,
            )

        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        attn_outputs = (hidden_states, residual)

        if output_attentions:
            self_attn_weights = outputs[1]
            attn_outputs += (self_attn_weights,)

        if use_cache:
            present_key_value = outputs[2 if output_attentions else 1]
            attn_outputs += (present_key_value,)

        return attn_outputs

    def post_process(
        self,
        hidden_states,
        residual,
        output_attentions=False,
        use_cache=False,
        self_attn_weights=None,
        present_key_value=None,
    ):
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        attn_outputs = self.attn(
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        residual = attn_outputs[1]
        self_attn_weights = attn_outputs[2] if output_attentions else None
        present_key_value = attn_outputs[3] if use_cache else None
        hidden_states = self.mlp(hidden_states)
        outputs = self.post_process(
            hidden_states, residual, output_attentions, use_cache, self_attn_weights, present_key_value
        )
        return outputs


class DeepseekV2MTPLayer(DeepseekV2DecoderLayer):
    def __init__(
        self,
        config: DeepseekV2Config,
        layer_idx: int,
        layerwise_recompute: bool = False,
    ):
        super(DeepseekV2MTPLayer, self).__init__(config, layer_idx, layerwise_recompute)

        self.enorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
        )
        self.hnorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
        )
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)

        if config.sequence_parallel and config.tensor_parallel_degree > 1:
            mark_as_sequence_parallel_parameter(self.eh_proj.weight)
            mark_as_sequence_parallel_parameter(self.eh_proj.bias)

    def subbatch_recompute_forward(
        self,
        hidden_states: paddle.Tensor,
        nextn_hidden_state: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        hidden_states = self.hnorm(hidden_states)
        nextn_hidden_state = self.enorm(nextn_hidden_state)

        hidden_states = self.eh_proj(paddle.concat([nextn_hidden_state, hidden_states], axis=-1))

        layer_outputs = super(DeepseekV2MTPLayer, self).subbatch_recompute_forward(
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            **kwargs,
        )

        if type(layer_outputs) is tuple:
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        return hidden_states

    def forward(
        self,
        hidden_states: paddle.Tensor,
        nextn_hidden_state: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        hidden_states = self.hnorm(hidden_states)
        nextn_hidden_state = self.enorm(nextn_hidden_state)

        hidden_states = self.eh_proj(paddle.cat([hidden_states, nextn_hidden_state], axis=-1))

        layer_outputs = super(DeepseekV2MTPLayer, self).forward(
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            **kwargs,
        )

        if type(layer_outputs) is tuple:
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        return hidden_states


class DeepseekV2PretrainedModel(PretrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV2DecoderLayer"]
    transpose_weight_keys = [
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate",
        "eh_proj",
    ]

    def _get_model_flops(self, batch_size=1, seq_length=None, **kwargs):
        from .mfu_utils import DeepSeekProjection

        mfu_cal_proj = DeepSeekProjection(self.config)
        if seq_length is None:
            if hasattr(self.config, "seq_length"):
                seq_length = self.config.seq_length
            else:
                seq_length = 2048

        return mfu_cal_proj.get_num_flop_per_token()

    def _get_hardware_flops(self, *args, **kwargs):
        return self._get_model_flops(*args, **kwargs)

    @classmethod
    def _get_name_mappings(cls, config: DeepseekV2Config) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        # last one layer contains MTP (eagle) parameters for inference
        for layer_index in range(config.num_hidden_layers + config.num_nextn_predict_layers):
            layer_mappings = [
                [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.q_a_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.q_a_layernorm.weight"],
                [f"layers.{layer_index}.self_attn.q_b_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.kv_a_proj_with_mqa.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.kv_a_layernorm.weight"],
                [f"layers.{layer_index}.self_attn.kv_b_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
            ]
            model_mappings.extend(layer_mappings)

            # MoE parameters
            model_mappings.append([f"layers.{layer_index}.mlp.gate.weight", None, "transpose"])
            model_mappings.append([f"layers.{layer_index}.mlp.gate.e_score_correction_bias"])
            for expert_idx in range(config.n_routed_experts):
                expert_mappings = [
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.gate_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.up_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.down_proj.weight", None, "transpose"],
                ]
                model_mappings.extend(expert_mappings)
            model_mappings.append([f"layers.{layer_index}.mlp.shared_experts.gate_proj.weight", None, "transpose"])
            model_mappings.append([f"layers.{layer_index}.mlp.shared_experts.up_proj.weight", None, "transpose"])
            model_mappings.append([f"layers.{layer_index}.mlp.shared_experts.down_proj.weight", None, "transpose"])

            # MTP (eagle) parameters for inference
            if layer_index >= config.num_hidden_layers:
                model_mappings.append([f"layers.{layer_index}.embed_tokens.weight"])
                model_mappings.append([f"layers.{layer_index}.enorm.weight"])
                model_mappings.append([f"layers.{layer_index}.hnorm.weight"])
                model_mappings.append([f"layers.{layer_index}.eh_proj.weight", None, "transpose"])
                model_mappings.append([f"layers.{layer_index}.shared_head.norm.weight"])
                model_mappings.append([f"layers.{layer_index}.shared_head.head.weight", None, "transpose"])

        init_name_mappings(mappings=model_mappings)
        if cls.base_model_class.__name__ not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = f"{cls.base_model_prefix}." + mapping[1]
            if not config.tie_word_embeddings:
                model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: DeepseekV2Config, is_split=True):
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
            }

            if config.tie_word_embeddings:
                base_actions["lm_head.weight"] = partial(fn, is_column=False)
            else:
                base_actions["lm_head.weight"] = partial(fn, is_column=True)

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("embed_tokens.weight")

            # Column Linear
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_b_proj.weight"] = partial(fn, is_column=True)

            # if we have enough num_key_value_heads to split, then split it.
            if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                base_actions["layers.0.self_attn.kv_b_proj.weight"] = partial(fn, is_column=True)

            # dense mlp
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.down_proj.weight"] = partial(fn, is_column=False)

            # moe unit routed experts
            moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
            expert_parallel_degree = dist.get_world_size(moe_group)
            if expert_parallel_degree <= 1:
                for e_i in range(config.n_routed_experts):
                    base_actions[f"layers.0.mlp.experts.{e_i}.up_proj.weight"] = partial(fn, is_column=True)
                    base_actions[f"layers.0.mlp.experts.{e_i}.gate_proj.weight"] = partial(fn, is_column=True)
                    base_actions[f"layers.0.mlp.experts.{e_i}.down_proj.weight"] = partial(fn, is_column=False)

            # moe unit shared experts
            base_actions["layers.0.mlp.shared_experts.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.down_proj.weight"] = partial(fn, is_column=False)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            # for MTP (eagle) parameters for inference
            base_actions.pop("embed_tokens.weight")
            base_actions.pop("lm_head.weight")
            base_actions["layers.0.embed_tokens.weight"] = partial(fn, is_column=False)
            base_actions["layers.0.shared_head.head.weight"] = partial(fn, is_column=True)
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(
                        config.num_hidden_layers, config.num_hidden_layers + config.num_nextn_predict_layers
                    ):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                else:
                    final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    def _init_weights(self, layer):
        return
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state

        if isinstance(
            layer,
            (
                nn.Linear,
                nn.Embedding,
                mpu.VocabParallelEmbedding,
                mpu.RowParallelLinear,
                mpu.ColumnParallelLinear,
                linear_utils.RowSequenceParallelLinear,
                linear_utils.ColumnSequenceParallelLinear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                if layer.weight.is_distributed:
                    with rng_tracker():
                        layer.weight.set_value(
                            paddle.tensor.normal(
                                mean=0.0,
                                std=self.config.initializer_range
                                if hasattr(self.config, "initializer_range")
                                else self.config.initializer_range,
                                shape=layer.weight.shape,
                            )
                        )
                else:
                    layer.weight.set_value(
                        paddle.tensor.normal(
                            mean=0.0,
                            std=self.config.initializer_range
                            if hasattr(self.config, "initializer_range")
                            else self.config.initializer_range,
                            shape=layer.weight.shape,
                        )
                    )

                # set bias to zeros
                if getattr(layer, "bias", None) is not None:
                    layer.bias.set_value(paddle.zeros(shape=layer.bias.shape))

        if isinstance(layer, nn.Embedding):
            if layer._padding_idx is not None:
                layer.weight.data[layer._padding_idx].fill_(0)

        if isinstance(layer, MoEGate):
            kaiming_uniform_(layer.weight, a=math.sqrt(5))


@register_base_model
class DeepseekV2Model(DeepseekV2PretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV2DecoderLayer`]

    Args:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )

        self.layers = nn.LayerList(
            [
                DeepseekV2DecoderLayer(config, layer_idx, layer_idx not in self.no_recompute_layers)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        for layer_idx in range(config.num_hidden_layers, config.num_hidden_layers + config.num_nextn_predict_layers):
            self.layers.append(DeepseekV2MTPLayer(config, layer_idx, layer_idx not in self.no_recompute_layers))

        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
        )

        self.enable_recompute = False

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later

        expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        position_ids: Optional[Tensor],
        attention_mask: Tensor,
        output_attentions: bool,
        past_key_value: Tensor,
        use_cache: bool,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            use_reentrant=self.config.recompute_use_reentrant,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPastAndMTP]:
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
        if self.config.num_nextn_predict_layers > 0:
            seq_length -= self.config.num_nextn_predict_layers

            if attention_mask is not None:
                attention_mask = attention_mask[
                    :, :, : -self.config.num_nextn_predict_layers, : -self.config.num_nextn_predict_layers
                ].contiguous()

            # attn_mask_startend_row_indices: [b, num_head, seq_len] or [b, num_head, seq_len, C], C is 2 or 4
            if attn_mask_startend_row_indices is not None:
                if attn_mask_startend_row_indices.ndim == 3:
                    attn_mask_startend_row_indices = attn_mask_startend_row_indices[
                        :,
                        :,
                        : -self.config.num_nextn_predict_layers,
                    ].contiguous()
                elif attn_mask_startend_row_indices.ndim == 4:
                    attn_mask_startend_row_indices = attn_mask_startend_row_indices[
                        :, :, : -self.config.num_nextn_predict_layers, :
                    ].contiguous()
                else:
                    raise ValueError("attn_mask_startend_row_indices must be 3D or 4D tensor")

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[1]
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=paddle.int64
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attn_mask_startend_row_indices is not None or get_use_casual_mask():
            attention_mask = None
        else:
            # [bs, seq_len]
            attention_mask = (
                paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
                if attention_mask is None
                else attention_mask
            )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), past_key_values_length, inputs_embeds.dtype
            )  # [bs, 1, seq_len, seq_len]
            if self.config.use_flash_attention:
                attention_mask = None if is_casual_mask(attention_mask) else attention_mask

        if self.config.num_nextn_predict_layers > 0:
            inputs_embeds_extra = inputs_embeds[:, -self.config.num_nextn_predict_layers :, :]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.num_nextn_predict_layers, :]
            inputs_embeds_ori = inputs_embeds

        if self.config.sequence_parallel:
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.transpose(inputs_embeds, [1, 0, 2])  # [B, S, H] --> [S, B, H]
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds.contiguous()

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        mtp_outputs = []

        moelayer_use_subbatch_recompute = self.config.moe_subbatch_token_num > 0

        for idx in range(self.config.num_hidden_layers):
            decoder_layer = self.layers[idx]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if moelayer_use_subbatch_recompute:
                layer_outputs = decoder_layer.subbatch_recompute_forward(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                )
            elif (
                self.enable_recompute
                and idx not in self.no_recompute_layers
                and has_gradient
                and self.recompute_granularity == "full"
            ):
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None
            if type(layer_outputs) is tuple:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.config.num_nextn_predict_layers > 0:
            mtp_outputs.append(hidden_states)

            for nextn in range(self.config.num_nextn_predict_layers):
                decoder_layer = self.layers[nextn + self.config.num_hidden_layers]

                if self.config.sequence_parallel:
                    hidden_states = GatherOp.apply(hidden_states)
                    hidden_states = paddle.transpose(hidden_states, [1, 0, 2])  # [S, B, H] --> [B, S, H]

                inputs_embeds_cur_depth = paddle.cat(
                    [inputs_embeds_ori[:, (nextn + 1) :, :], inputs_embeds_extra[:, : (nextn + 1), :]], axis=1
                )

                past_key_value = None
                layer_outputs = decoder_layer(
                    hidden_states,
                    inputs_embeds_cur_depth,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                )

                if isinstance(layer_outputs, (tuple, list)):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                mtp_outputs.append(hidden_states)
            mtp_outputs = [self.norm(hidden_states) for hidden_states in mtp_outputs]
            hidden_states, mtp_outputs = mtp_outputs[0], mtp_outputs[1:]
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, mtp_outputs] if v is not None
            )
        return BaseModelOutputWithPastAndMTP(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mtp_outputs=mtp_outputs,
        )


class DeepseekV2PretrainingCriterion(nn.Layer):
    """
    Criterion for Mixtral.
    It calculates the final loss.
    """

    def __init__(self, config: DeepseekV2Config):
        super(DeepseekV2PretrainingCriterion, self).__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output

        if self.enable_parallel_cross_entropy:  # and False: # and lm_head is distributed
            self.loss_func = mpu.ParallelCrossEntropy(ignore_index=self.ignore_index)
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels, router_loss=None, mtp_logits=None):
        if self.enable_parallel_cross_entropy:
            if prediction_scores.shape[-1] == self.config.vocab_size:
                warnings.warn(
                    f"enable_parallel_cross_entropy, the vocab_size should be splitted: {prediction_scores.shape[-1]}, {self.config.vocab_size}"
                )
                self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

        def compute_loss(preds, labels):
            with paddle.amp.auto_cast(False):
                masked_lm_loss = self.loss_func(preds.astype("float32"), labels.unsqueeze(2))
                binary_sequence = paddle.where(
                    masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
                )
                count = paddle.sum(binary_sequence)
                if count == 0:
                    loss = paddle.sum(masked_lm_loss * binary_sequence)
                else:
                    loss = paddle.sum(masked_lm_loss * binary_sequence) / count
                return loss

        def add_loss(main_loss, loss):
            return main_loss + loss - loss.detach()

        if mtp_logits is not None and self.config.num_nextn_predict_layers > 0:
            assert len(mtp_logits) == self.config.num_nextn_predict_layers
            masked_lm_labels_ori = masked_lm_labels
            masked_lm_labels = masked_lm_labels[:, : -self.config.num_nextn_predict_layers]
            seq_length = masked_lm_labels.shape[1]
            loss = compute_loss(prediction_scores, masked_lm_labels)

            mtp_loss_res = []
            for depth in range(self.config.num_nextn_predict_layers):
                prediction_scores_cur_depth = mtp_logits[depth]
                masked_lm_labels_cur_depth = masked_lm_labels_ori[:, (depth + 1) : (depth + 1 + seq_length)]
                res_cur_depth = compute_loss(prediction_scores_cur_depth, masked_lm_labels_cur_depth)
                mtp_loss_res.append(res_cur_depth)
            loss = add_loss(loss, self.config.num_nextn_predict_lambda * sum([x for x in mtp_loss_res]) / len(mtp_loss_res))  # fmt: skip

        else:
            loss = compute_loss(prediction_scores, masked_lm_labels)

        if router_loss is not None and isinstance(router_loss, paddle.Tensor):
            loss = add_loss(loss, router_loss)

        return loss


class DeepseekV2ForCausalLM(DeepseekV2PretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.config = config
        self.deepseek_v2 = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)

    def get_input_embeddings(self):
        return self.deepseek_v2.embed_tokens

    def set_input_embeddings(self, value):
        self.deepseek_v2.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.deepseek_v2 = decoder

    def get_decoder(self):
        return self.deepseek_v2

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

        >>> model = DeepseekV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.deepseek_v2(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        if return_dict:
            hidden_states = outputs.hidden_states
            mtp_outputs = outputs.mtp_outputs
        else:
            hidden_states = outputs[0]
            mtp_outputs = outputs[-1]

        if labels is not None and self.config.use_fused_linear_cross_entropy:
            from paddlenlp_kernel.triton.cut_cross_entropy import linear_cross_entropy

            assert (
                self.config.tensor_parallel_degree <= 1
            ), "The argument `use_fused_linear_cross_entropy` is imcompatiable with tensor parallel "

            masked_lm_loss = linear_cross_entropy(hidden_states, self.lm_head.weight, targets=labels)

            binary_sequence = paddle.where(
                masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count
            logits = None
        else:
            # if labels is None，means we need full output, instead of tensor_parallel_output
            # tensor_parallel_output is together with ParallelCrossEntropy
            tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1
            logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)
            mtp_logits = (
                [
                    self.lm_head(_hidden_states, tensor_parallel_output=tensor_parallel_output)
                    for _hidden_states in mtp_outputs
                ]
                if len(mtp_outputs) > 0
                else []
            )

            loss = None
            if labels is not None:
                loss = self.criterion(logits, labels, mtp_logits=mtp_logits)

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
        self, input_ids, use_cache=False, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithPast) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.cat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            # TODO: support attention mask for other models
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 2:
                model_kwargs["attention_mask"] = paddle.cat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )
            elif len(attention_mask.shape) == 4:
                model_kwargs["attention_mask"] = paddle.cat(
                    [attention_mask, paddle.ones([*attention_mask.shape[:3], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )[:, :, -1:, :]

        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class DeepseekV2ForSequenceClassification(DeepseekV2PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekV2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias_attr=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

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
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
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
        else:
            if input_ids is not None:
                sequence_lengths = paddle.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int64):
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
                loss = loss_fct(pooled_logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
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


class DeepseekV2MTPLayerPipe(DeepseekV2MTPLayer):
    def forward(self, args):
        hidden_states, attention_mask, position_ids, position_embeddings, nbatch_pack_offset = parse_args(args)

        if attention_mask is None:
            attn_mask = None
            attn_mask_startend_row_indices = None
        elif attention_mask.dtype == paddle.int32:
            attn_mask = None
            attn_mask_startend_row_indices = attention_mask
        else:
            attn_mask = attention_mask
            attn_mask_startend_row_indices = None
            assert len(attn_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {attn_mask.shape}."

        hidden_states_list = paddle.split(hidden_states, self.config.num_nextn_predict_layers + 1, axis=-1)
        hidden_states_main_model = hidden_states_list[0]
        inputs_embeds_cur_depth_list = hidden_states_list[1:]
        has_gradient = not hidden_states_main_model.stop_gradient

        output_list = [hidden_states_main_model]
        hidden_states = hidden_states_main_model
        for depth in range(self.config.num_nextn_predict_layers):
            inputs_embeds_cur_depth = inputs_embeds_cur_depth_list[depth]

            moelayer_use_subbatch_recompute = self.config.moe_subbatch_token_num > 0
            if moelayer_use_subbatch_recompute:
                hidden_states = super().subbatch_recompute_forward(
                    hidden_states,
                    inputs_embeds_cur_depth,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            elif self.enable_recompute and self.config.recompute_granularity == "full" and has_gradient:
                if attn_mask is not None or attn_mask_startend_row_indices is not None:
                    hidden_states = recompute(
                        super().forward,
                        hidden_states,
                        inputs_embeds_cur_depth,
                        position_ids=position_ids,
                        attention_mask=attn_mask,
                        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                        use_reentrant=self.config.recompute_use_reentrant,
                    )
                else:
                    # for pretrain
                    hidden_states = recompute(
                        super().forward,
                        hidden_states,
                        inputs_embeds_cur_depth,
                        position_ids=position_ids,
                        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                        use_reentrant=self.config.recompute_use_reentrant,
                    )
            else:
                hidden_states = super().forward(
                    hidden_states,
                    inputs_embeds_cur_depth,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            output_list.append(hidden_states)

        hidden_states = paddle.concat(output_list, axis=-1)

        ret = (hidden_states,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)

        return ret


class DeepseekV2EmbeddingPipe(EmbeddingPipe):
    def forward(self, args):
        num_nextn_predict_layers = self.config.get("num_nextn_predict_layers", 0)
        input_ids, attention_mask, position_ids, _, _ = parse_args(args, num_nextn_predict_layers > 0)
        inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        batch_size, max_seq_len = input_ids.shape
        max_seq_len -= self.config.num_nextn_predict_layers
        if num_nextn_predict_layers > 0:
            if attention_mask is None:
                attn_mask = None
                attn_mask_startend_row_indices = None
            elif attention_mask.dtype == paddle.int32:
                attn_mask = None
                attn_mask_startend_row_indices = attention_mask[:, :, :max_seq_len]
            else:
                attn_mask = attention_mask[:, :, :max_seq_len, :max_seq_len]
                attn_mask_startend_row_indices = None
                assert len(attn_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {attn_mask.shape}."
        if attn_mask is not None:
            assert (
                attn_mask_startend_row_indices is None
            ), "attention_mask and attn_mask_startend_row_indices can not be set at same time"
            attn_mask = DeepseekV2Model._prepare_decoder_attention_mask(
                attn_mask, (batch_size, max_seq_len), 0, inputs_embeds.dtype
            )
        attn_mask = attn_mask_startend_row_indices if attn_mask_startend_row_indices is not None else attn_mask

        if num_nextn_predict_layers > 0:
            inputs_embeds_extra = inputs_embeds[:, -self.config.num_nextn_predict_layers :, :]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.num_nextn_predict_layers, :]
            inputs_embeds_ori = inputs_embeds
            batch_size, seq_length, _ = inputs_embeds.shape

            if self.sequence_parallel:
                inputs_embeds = paddle.transpose(inputs_embeds, [1, 0, 2])  # [B, S, H] --> [S, B, H]
                inputs_embeds = ScatterOp.apply(inputs_embeds)
            embeds_res = [inputs_embeds]
            for depth in range(num_nextn_predict_layers):
                inputs_embeds_mtp = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )
                if self.sequence_parallel:
                    inputs_embeds_mtp = paddle.transpose(inputs_embeds_mtp, [1, 0, 2])  # [B, S, H] --> [S, B, H]
                    inputs_embeds_mtp = ScatterOp.apply(inputs_embeds_mtp)
                embeds_res.append(inputs_embeds_mtp)
            res = paddle.concat(embeds_res, axis=-1)
            ret = (res,)
        else:
            if self.sequence_parallel:
                inputs_embeds = paddle.transpose(inputs_embeds, [1, 0, 2])  # [B, S, H] --> [S, B, H]
                inputs_embeds = ScatterOp.apply(inputs_embeds)
            ret = (inputs_embeds,)

        if attn_mask is not None:
            ret += (attn_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        return ret


class DeepseekV2DecoderLayerPipe(DeepseekV2DecoderLayer):
    def forward(self, args):
        hidden_states, attention_mask, position_ids, _, _ = parse_args(args)

        if self.config.num_nextn_predict_layers > 0:
            hidden_size = hidden_states.shape[-1]
            batch_size_mtp = hidden_size // (self.config.num_nextn_predict_layers + 1)
            inputs_embeds_mtp = hidden_states[..., -batch_size_mtp:].contiguous()
            hidden_states = hidden_states[..., :batch_size_mtp].contiguous()

        if attention_mask is None:
            attn_mask = None
            attn_mask_startend_row_indices = None
        elif attention_mask.dtype == paddle.int32:
            attn_mask = None
            attn_mask_startend_row_indices = attention_mask
        else:
            attn_mask = attention_mask
            attn_mask_startend_row_indices = None
            assert len(attn_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {attn_mask.shape}."

        has_gradient = not hidden_states.stop_gradient

        moelayer_use_subbatch_recompute = self.config.moe_subbatch_token_num > 0
        if moelayer_use_subbatch_recompute:
            hidden_states = super().subbatch_recompute_forward(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )
        elif self.enable_recompute and self.config.recompute_granularity == "full" and has_gradient:
            hidden_states = recompute(
                super().forward,
                hidden_states,
                position_ids=position_ids,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            hidden_states = super().forward(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attn_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )

        if self.config.num_nextn_predict_layers > 0:
            hidden_states = paddle.concat([hidden_states, inputs_embeds_mtp], axis=-1)

        if isinstance(hidden_states, paddle.Tensor):
            ret = (hidden_states,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if len(ret) == 1:
            (ret,) = ret
        return ret


class DeepseekV2LMHeadPipe(GeneralLMHead):
    def forward(self, args):
        if self.config.num_nextn_predict_layers > 0:
            logits = []
            for _hidden_states in args:
                logits.append(super().forward(_hidden_states))
            return logits

        hidden_states, _, _, _, _ = parse_args(args)
        logits = super().forward(hidden_states)
        return logits


class DeepseekV2PretrainingCriterionPipe(DeepseekV2PretrainingCriterion):
    def forward(self, logits, labels):

        # in GeneralModelForCausalLMPipe last_stage_keys = ["labels", "loss_mask"]
        labels = labels[0]
        if self.config.num_nextn_predict_layers > 0:
            mtp_logits = logits[1:]
            logits = logits[0]
            loss = super().forward(logits, labels, mtp_logits=mtp_logits)
        else:
            loss = super().forward(logits, labels)
        return loss


class DeepseekV2RMSNormLayerPipe(RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.sequence_parallel:
            self.enable_sequence_parallel()

    def forward(self, args):
        hidden_states, _, _, _, _ = parse_args(args)

        if self.config.num_nextn_predict_layers > 0:
            hidden_states_list = paddle.split(hidden_states, self.config.num_nextn_predict_layers + 1, axis=-1)
            hidden_states = hidden_states_list[0]
            hidden_states_mtp = hidden_states_list[-self.config.num_nextn_predict_layers :]

            output_list = [super().forward(hidden_states)]
            for hidden_states in hidden_states_mtp:
                output_list.append(super().forward(hidden_states))
            return output_list
        else:
            hidden_states = super().forward(hidden_states)
            return hidden_states


class DeepseekV2ForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = DeepseekV2Config
    _embedding_pipe_cls = DeepseekV2EmbeddingPipe
    _decoder_layer_cls = DeepseekV2DecoderLayer
    _criterion_pipe_cls = DeepseekV2PretrainingCriterionPipe
    _lmhead_pipe_cls = DeepseekV2LMHeadPipe
    _decoder_layer_pipe_cls = DeepseekV2DecoderLayerPipe
    _rms_norm_pipe_cls = DeepseekV2RMSNormLayerPipe
    _base_model = DeepseekV2PretrainedModel

    _get_tensor_parallel_mappings = DeepseekV2PretrainedModel._get_tensor_parallel_mappings
    _init_weights = DeepseekV2PretrainedModel._init_weights
    _keys_to_ignore_on_load_unexpected = DeepseekV2PretrainedModel._keys_to_ignore_on_load_unexpected
    _get_model_flops = DeepseekV2PretrainedModel._get_model_flops
    _get_hardware_flops = DeepseekV2PretrainedModel._get_hardware_flops
    transpose_weight_keys = DeepseekV2PretrainedModel.transpose_weight_keys

    _tied_weights_keys = ["lm_head.weight"]

    _mtp_layer_pipe_cls = DeepseekV2MTPLayerPipe
