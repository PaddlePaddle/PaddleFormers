# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Llama model"""
from __future__ import annotations

import math
import os
import warnings
from functools import partial
from typing import Optional, Tuple

import numpy as np
import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.recompute.recompute import recompute

from ..refined_recompute import (
    RRColumnParallelLinear,
    RRColumnSequenceParallelLinear,
    RRRowParallelLinear,
    RRRowSequenceParallelLinear,
    get_skip_recompute_ops,
)
from ..refined_recompute import recompute as rr_recompute

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    pass

from ...utils.log import logger
from ...utils.tools import get_env_device
from .. import linear_utils
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..linear_utils import Linear
from ..long_sequence_strategies import LongSequenceStrategies
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from ..model_utils import PretrainedModel, register_base_model
from ..segment_parallel_utils import ReshardLayer
from ..utils import caculate_llm_per_token_flops
from .configuration import (
    LLAMA_PRETRAINED_INIT_CONFIGURATION,
    LLAMA_PRETRAINED_RESOURCE_FILES_MAP,
    LlamaConfig,
)

try:
    if get_env_device() in ["npu", "mlu", "gcu"]:

        for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
            if lib.endswith(".so"):
                paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(lib)
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None
from . import fusion_ops

__all__ = [
    "LlamaModel",
    "LlamaLMHead",
    "LlamaPretrainedModel",
    "LlamaForCausalLM",
    "LlamaPretrainingCriterion",
]


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(np.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if np.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = int(2 ** np.floor(np.log2(n)))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def get_use_casual_mask():
    """Get the value of the 'USE_CASUAL_MASK' environment variable."""
    return os.getenv("USE_CASUAL_MASK", "False") == "True"


def build_alibi_tensor(
    bool_attention_mask: Tensor, num_heads: int, dtype: paddle.dtype, tensor_parallel_degree=1
) -> Tensor:
    batch_size, seq_length = bool_attention_mask.shape[0], bool_attention_mask.shape[-1]
    slopes = paddle.to_tensor(_get_interleave(num_heads), dtype="float32")
    alibi = slopes.unsqueeze(axis=[1, 2]) * paddle.arange(seq_length, dtype="float32").unsqueeze(axis=[0, 1]).expand(
        [num_heads, -1, -1]
    )
    alibi = alibi.reshape(shape=(1, num_heads, 1, seq_length)).expand([batch_size, -1, -1, -1])
    return paddle.cast(alibi, dtype)


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


def parallel_matmul(x: Tensor, y: Tensor, transpose_y=False, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    if paddle.in_dynamic_mode():
        y_is_distributed = y.is_distributed
    else:
        y_is_distributed = tensor_parallel_degree > 1

    if is_fleet_init and tensor_parallel_degree > 1 and y_is_distributed:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=transpose_y)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=transpose_y)
        return logits


def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    attn_mask_startend_row_indices=None,
    sequence_parallel=False,
    reshard_layer=None,
    npu_is_casual=False,
    skip_recompute=False,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    if config.use_flash_attention and flash_attention:
        return fusion_ops.fusion_flash_attention(
            query_states,
            config,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
            alibi,
            attn_mask_startend_row_indices,
            sequence_parallel,
            reshard_layer,
            npu_is_casual,
            skip_recompute=skip_recompute,
        )

        # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
        # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]

    else:
        if config.context_parallel_degree > 1:
            raise ValueError("Context parallel requires `use_flash_attention=True`")

        #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        # merge with the next transpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        # Add pre divided factor to fix nan under float16.
        if paddle.in_dynamic_mode() and query_states.dtype == paddle.float16:
            pre_divided_factor = 32
        else:
            pre_divided_factor = 1

        attn_weights = paddle.matmul(
            query_states * (1 / (math.sqrt(head_dim) * pre_divided_factor)), key_states.transpose([0, 1, 3, 2])
        )

        # then add alibi bias
        if alibi is not None:
            alibi = alibi.reshape([bsz, num_heads, 1, -1])
            attn_weights = attn_weights + alibi / pre_divided_factor

        if paddle.in_dynamic_mode() and attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        # In sep mode, the attenion mask should be created in the runtime.
        if reshard_layer is not None:
            attention_mask = None

        # NOTE: we only call get_triangle_upper_mask under PP setup
        # FIXME ZHUI when we use pipeline parallel, the attention_mask can be None
        # we just make it triangle_upper_mask
        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)
        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if paddle.in_dynamic_mode() and attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attn_weights + attention_mask
        if not paddle.in_dynamic_mode():
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        else:
            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(attn_weights.astype("float32") * pre_divided_factor, axis=-1).astype(
                    query_states.dtype
                )

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        if reshard_layer is not None:
            attn_output = reshard_layer(
                attn_output,
                split_axis=1,
                concat_axis=2,
            )
            q_len = q_len // config.sep_parallel_degree
            num_heads = num_heads * config.sep_parallel_degree

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


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

    if get_env_device() in ["npu", "mlu", "intel_hpu"]:
        mask = paddle.tril(paddle.ones((target_length, target_length))).astype("int32")
    else:
        mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    if get_env_device() == "npu" or get_env_device() == "mlu":
        mask = mask[:, None, None, :].astype(dtype)
    else:
        mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


class LlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, hidden_states):
        if self.config.use_fused_rms_norm:
            return fusion_ops.fusion_rms_norm(hidden_states, self.weight, self.variance_epsilon)

        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                # hidden_states = hidden_states.astype("float32")
                # variance = hidden_states.pow(2).mean(-1, keepdim=True)
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        else:
            hidden_states = hidden_states.astype("float32")
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(hidden_states, n_rep, axis=1). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states.unsqueeze(-2).tile([1, 1, 1, n_rep, 1])
    return hidden_states.reshape([batch, slen, num_key_value_heads * n_rep, head_dim])


class LlamaRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        if get_env_device() == "intel_hpu":
            # fallback einsum to intel Gaudi TPC since MME doesn't support FP32
            freqs = t.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        else:
            freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]
        self.cos_sin_table = None if get_env_device() != "gcu" else paddle.concat([freqs.cos(), freqs.sin()], axis=-1)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.cos_cached.dtype != x.dtype and get_env_device() == "intel_hpu":
            self.cos_cached = self.cos_cached.cast(x.dtype)
            self.sin_cached = self.sin_cached.cast(x.dtype)
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )

    def get_fused_cos_sin(self, x, seq_len=None):
        if self.cos_sin_table is not None and self.cos_sin_table.dtype != x.dtype:
            return self.cos_sin_table.cast(x.dtype)
        else:
            return self.cos_sin_table


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings * scaling_factor, base)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        t = t / self.scaling_factor
        # [seq_len, dim/2]
        if get_env_device() == "intel_hpu":
            # fallback einsum to intel Gaudi TPC since MME doesn't support FP32
            freqs = t.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        else:
            freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]
        self.cos_sin_table = None if get_env_device() != "gcu" else paddle.concat([freqs.cos(), freqs.sin()], axis=-1)


class LlamaNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        base = base * scaling_factor ** (dim / (dim - 2))
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings * scaling_factor, base)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _scale_cos_sin(self, seq_len):
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        alpha = (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        base = self.base * alpha ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        freqs = paddle.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        scale_cos = emb.cos()[None, :, None, :]
        scale_sin = emb.sin()[None, :, None, :]
        scale_cos_sin = None if get_env_device() != "gcu" else paddle.concat([freqs.cos(), freqs.sin()], axis=-1)
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


class Llama3RotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=8192,
        base=500000,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len):
        low_freq_wavelen = self.original_max_position_embeddings / self.low_freq_factor
        high_freq_wavelen = self.original_max_position_embeddings / self.high_freq_factor
        new_freqs = []
        for freq in self.inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / self.factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (self.original_max_position_embeddings / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / self.factor + smooth * freq)
        self.inv_freq = paddle.to_tensor(new_freqs, dtype=self.inv_freq.dtype)
        super()._set_cos_sin_cache(seq_len=seq_len)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):

    if position_ids is None:
        # Note: Only for LlamaForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Layer):
    def __init__(self, config, skip_recompute_ops=None):
        super().__init__()
        if skip_recompute_ops is None:
            skip_recompute_ops = {}
        self.skip_recompute_ops = skip_recompute_ops
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tensor_parallel_degree = config.tensor_parallel_degree
        self.fuse_attention_ffn = config.fuse_attention_ffn

        if config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear

            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if config.recompute and not config.recompute_use_reentrant:
                if skip_recompute_ops.get("mlp_column_ln", False):
                    ColumnParallelLinear = RRColumnSequenceParallelLinear
                if skip_recompute_ops.get("mlp_row_ln", False):
                    RowParallelLinear = RRRowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear

            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if config.recompute and not config.recompute_use_reentrant:
                if skip_recompute_ops.get("mlp_column_ln", False):
                    ColumnParallelLinear = RRColumnParallelLinear
                if skip_recompute_ops.get("mlp_row_ln", False):
                    RowParallelLinear = RRRowParallelLinear

        if config.tensor_parallel_degree > 1:
            if config.fuse_attention_ffn:
                self.gate_up_fused_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    gather_output=False,
                    has_bias=False,
                )
            else:
                self.gate_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    gather_output=False,
                    has_bias=False,
                )
                self.up_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    gather_output=False,
                    has_bias=False,
                )

            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            if config.fuse_attention_ffn:
                self.gate_up_fused_proj = Linear(self.hidden_size, self.intermediate_size * 2, bias_attr=False)
            else:
                self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
                self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)

            self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

    def forward(self, x):
        if self.fuse_attention_ffn:
            x = swiglu(self.gate_up_fused_proj(x))
        else:
            x = swiglu(self.gate_proj(x), self.up_proj(x))
        out = self.down_proj(x)
        return out


class LlamaAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layerwise_recompute: bool = False, skip_recompute_ops=None):
        super().__init__()
        if skip_recompute_ops is None:
            skip_recompute_ops = {}
        self.skip_recompute_ops = skip_recompute_ops
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_dim = self.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        assert config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.seq_length = config.seq_length
        self.sequence_parallel = config.sequence_parallel

        self.fuse_attention_qkv = config.fuse_attention_qkv

        self.kv_indices = None
        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity
        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            if self.num_key_value_heads % config.tensor_parallel_degree == 0:
                self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree
            else:
                if self.fuse_attention_qkv:
                    # TODO(Yuang): support fusion for kv when kv heads cannot be divided by mp
                    raise ValueError(
                        f"fuse_attention_qkv can't be True when num_key_value_heads {config.num_key_value_heads} % tensor_parallel_degree {config.tensor_parallel_degree} != 0"
                    )
                logger.warning(
                    f"Get num_key_value_heads: {self.num_key_value_heads}, can't split to tensor_parallel_degree: {config.tensor_parallel_degree}, so we don't spilt key value weight."
                )
                self.kv_indices = paddle.to_tensor(
                    assign_kv_heads(self.num_key_value_heads, config.tensor_parallel_degree)[
                        config.tensor_parallel_rank
                    ]
                )

        self.use_fused_rope = config.use_fused_rope
        if self.use_fused_rope and get_env_device() not in ["npu", "mlu", "xpu", "gcu", "intel_hpu"]:
            if "gpu" not in paddle.device.get_device() or fused_rotary_position_embedding is None:
                warnings.warn(
                    "Enable fuse rope in the config, but fuse rope is not available. "
                    "Will disable fuse rope. Try using latest gpu version of Paddle."
                )
                self.use_fused_rope = False

        if config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear

            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if config.recompute and not config.recompute_use_reentrant:
                if skip_recompute_ops.get("attention_column_ln", False):
                    ColumnParallelLinear = RRColumnSequenceParallelLinear
                if skip_recompute_ops.get("attention_row_ln", False):
                    RowParallelLinear = RRRowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear
            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if config.recompute and not config.recompute_use_reentrant:
                if skip_recompute_ops.get("attention_column_ln", False):
                    ColumnParallelLinear = RRColumnParallelLinear
                if skip_recompute_ops.get("attention_row_ln", False):
                    RowParallelLinear = RRRowParallelLinear

        if config.tensor_parallel_degree > 1:
            if self.fuse_attention_qkv:
                self.qkv_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size + 2 * self.config.num_key_value_heads * self.head_dim,
                    has_bias=False,
                    gather_output=False,
                )
            else:
                self.q_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    has_bias=False,
                    gather_output=False,
                )
                if self.kv_indices is None:
                    self.k_proj = ColumnParallelLinear(
                        self.hidden_size,
                        self.config.num_key_value_heads * self.head_dim,
                        has_bias=False,
                        gather_output=False,
                    )
                    self.v_proj = ColumnParallelLinear(
                        self.hidden_size,
                        self.config.num_key_value_heads * self.head_dim,
                        has_bias=False,
                        gather_output=False,
                    )
                else:
                    self.k_proj = Linear(
                        self.hidden_size,
                        self.config.num_key_value_heads * self.head_dim,
                        bias_attr=False,
                    )
                    self.v_proj = Linear(
                        self.hidden_size,
                        self.config.num_key_value_heads * self.head_dim,
                        bias_attr=False,
                    )

        else:
            if self.fuse_attention_qkv:
                self.qkv_proj = Linear(
                    self.hidden_size,
                    self.hidden_size + 2 * self.config.num_key_value_heads * self.head_dim,
                    bias_attr=False,
                )
            else:
                self.q_proj = Linear(
                    self.hidden_size,
                    self.hidden_size,
                    bias_attr=False,
                )
                self.k_proj = Linear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    bias_attr=False,
                )
                self.v_proj = Linear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    bias_attr=False,
                )

        if config.tensor_parallel_degree > 1:
            self.o_proj = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
        else:
            self.o_proj = Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=False,
            )

        if config.rope:
            if config.use_long_sequence_strategies:
                self.rotary_emb = LongSequenceStrategies.build_long_sequence_strategy(
                    config.long_sequence_strategy_type,
                    config.long_sequence_strategy_name,
                    **config.long_sequence_init_args,
                )
            else:
                self._init_rope()

        self.reshard_layer = None
        if config.sep_parallel_degree > 1:
            assert self.num_key_value_heads % config.sep_parallel_degree == 0
            assert self.num_heads % config.sep_parallel_degree == 0
            self.reshard_layer = ReshardLayer()

        self.config = config

        self.attn_func = scaled_dot_product_attention

        # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
        if config.recompute and not config.recompute_use_reentrant and skip_recompute_ops.get("flash_attn", False):
            self.attn_func = partial(scaled_dot_product_attention, skip_recompute=True)

    def _init_rope(self):
        if (
            hasattr(self.config, "rope_scaling")
            and self.config.rope_scaling is not None
            and self.config.rope_scaling.get("rope_type", None) == "llama3"
        ):
            self.rotary_emb = Llama3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
                factor=self.config.rope_scaling["factor"],
                high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
            )
        elif self.config.rope_scaling_type is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        elif self.config.rope_scaling_type == "linear":
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=self.config.rope_scaling_factor,
                base=self.config.rope_theta,
            )
        elif self.config.rope_scaling_type == "ntk":
            self.rotary_emb = LlamaNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=self.config.rope_scaling_factor,
                base=self.config.rope_theta,
            )
        elif self.config.rope_scaling_type == "dynamic_ntk":
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=self.config.rope_scaling_factor,
                base=self.config.rope_theta,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {self.config.rope_scaling_type}")

    def forward(
        self,
        hidden_states,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        alibi: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        npu_is_casual: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)

        if self.fuse_attention_qkv:
            mix_layer = self.qkv_proj(hidden_states)
            # NOTE for GQA attention fusion (compatible with MHA and MQA):
            # The weight for qkv_proj is in shape like [hidden_size, hidden_size + 2 * num_kv_heads * head_dim].
            # After the projection, the mix_layer is in shape like [b, s, hidden_size + 2 * num_kv_heads * head_dim].
            # Reshape the mix_layer into a shape like [b, s, num_kv_heads, (num_groups + 2) * head_dim],
            # where num_groups = num_q_heads // num_kv_heads.
            # Split the mix_layer on the last axis into three sections [num_groups * head_dim, head_dim, head_dim]
            # to represent the q, k and v respectively.
            # The q is in the shape like [b, s, num_kv_heads, num_groups * head_dim].
            # The k and v are in the shape like [b, s, num_kv_heads, head_dim].
            # Under MHA, the q is ready for the following calculation since num_kv_heads == num_q_heads,
            # But for the GQA or MQA, q should be reshaped into [b, s, num_q_heads, head_dim].
            if self.reshard_layer is not None:
                if self.sequence_parallel:
                    assert self.seq_length % self.config.sep_parallel_degree == 0
                    mix_layer = paddle.reshape_(
                        mix_layer,
                        [
                            -1,
                            self.seq_length // self.config.sep_parallel_degree,
                            self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim,
                        ],
                    )
                # [bs, seq_len / sep, num_head, head_dim] -> [bs, seq_len, num_head / sep, head_dim]
                mix_layer = self.reshard_layer(
                    mix_layer,
                    split_axis=2,
                    concat_axis=1,
                )
                mix_layer = paddle.reshape_(
                    mix_layer, [0, self.seq_length, -1, (self.num_key_value_groups + 2) * self.head_dim]
                )  # [bs, seq_len, num_head/k, 3*head_dim], k is sep degree
            else:
                if self.sequence_parallel:
                    target_shape = [
                        -1,
                        self.seq_length,
                        self.num_key_value_heads,
                        (self.num_key_value_groups + 2) * self.head_dim,
                    ]
                else:
                    target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
                mix_layer = paddle.reshape_(mix_layer, target_shape)
            query_states, key_states, value_states = paddle.split(
                mix_layer,
                num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
                axis=-1,
            )
            if self.gqa_or_mqa:
                query_states = paddle.reshape_(query_states, [0, 0, self.num_heads, self.head_dim])
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if self.reshard_layer is not None:
                if self.sequence_parallel:
                    assert self.seq_length % self.config.sep_parallel_degree == 0
                    query_states = paddle.reshape(
                        query_states,
                        [-1, self.seq_length // self.config.sep_parallel_degree, self.num_heads * self.head_dim],
                    )
                    key_states = paddle.reshape(
                        key_states,
                        [
                            -1,
                            self.seq_length // self.config.sep_parallel_degree,
                            self.num_key_value_heads * self.head_dim,
                        ],
                    )
                    value_states = paddle.reshape(
                        value_states,
                        [
                            -1,
                            self.seq_length // self.config.sep_parallel_degree,
                            self.num_key_value_heads * self.head_dim,
                        ],
                    )
                query_states = self.reshard_layer(
                    query_states,
                    split_axis=2,
                    concat_axis=1,
                )
                key_states = self.reshard_layer(
                    key_states,
                    split_axis=2,
                    concat_axis=1,
                )
                value_states = self.reshard_layer(
                    value_states,
                    split_axis=2,
                    concat_axis=1,
                )
                query_states = paddle.reshape(
                    query_states, [0, self.seq_length, -1, self.head_dim]
                )  # [bs, seq_len, num_head/k, head_dim], k is sep degree
                key_states = paddle.reshape(key_states, [0, self.seq_length, -1, self.head_dim])
                value_states = paddle.reshape(value_states, [0, self.seq_length, -1, self.head_dim])
            else:
                if self.sequence_parallel:
                    target_query_shape = [-1, self.seq_length, self.num_heads, self.head_dim]
                    target_key_value_shape = [-1, self.seq_length, self.num_key_value_heads, self.head_dim]
                else:
                    target_query_shape = [0, 0, self.num_heads, self.head_dim]
                    target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
                query_states = query_states.reshape(shape=target_query_shape)
                key_states = key_states.reshape(shape=target_key_value_shape)
                value_states = value_states.reshape(shape=target_key_value_shape)

        kv_seq_len = key_states.shape[-3]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        if self.config.rope:
            if self.reshard_layer is not None:
                batch_size, seq_length, _, _ = query_states.shape
                position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))
            if self.config.context_parallel_degree > 1:
                batch_size, seq_length, _, _ = query_states.shape
                group = fleet.get_hybrid_communicate_group().get_sep_parallel_group()
                chunk_size = seq_length // 2
                chunk_num = group.nranks * 2
                rank = group.rank
                first_chunk_ids = paddle.arange(rank * chunk_size, (rank + 1) * chunk_size, dtype="int64")
                second_chunk_ids = paddle.arange(
                    (chunk_num - rank - 1) * chunk_size, (chunk_num - rank) * chunk_size, dtype="int64"
                )
                position_ids = paddle.concat([first_chunk_ids, second_chunk_ids]).expand((batch_size, seq_length))
            if self.use_fused_rope:
                query_states, key_states = fusion_ops.fusion_rope(
                    query_states,
                    key_states,
                    value_states,
                    hidden_states,
                    position_ids,
                    past_key_value,
                    self.rotary_emb,
                    self.config.context_parallel_degree,
                )

            else:
                if self.config.context_parallel_degree > 1:
                    kv_seq_len *= self.config.context_parallel_degree
                if self.config.use_long_sequence_strategies:
                    cos, sin = self.rotary_emb(seq_len=kv_seq_len)
                    cos = cos[None, :, None, :]
                    sin = sin[None, :, None, :]
                    cos, sin = (
                        cos.cast(value_states.dtype) if cos.dtype != value_states.dtype else cos,
                        sin.cast(value_states.dtype) if sin.dtype != value_states.dtype else sin,
                    )
                else:
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)
            if self.config.immediate_clear_past_key_value:
                past_key_value[0]._clear_data()
                past_key_value[1]._clear_data()

        past_key_value = (key_states, value_states) if use_cache else None
        if self.kv_indices is not None:
            key_states = paddle.index_select(key_states, self.kv_indices, axis=2)
            value_states = paddle.index_select(value_states, self.kv_indices, axis=2)

        # TODO(wj-Mcat): use broadcast strategy when n_kv_heads = 1
        # repeat k/v heads if n_kv_heads < n_heads
        # paddle version > 2.6 or develop support flash-attn with gqa/mqa
        paddle_version = float(paddle.__version__[:3])
        if not self.config.use_flash_attention or ((paddle_version != 0.0) and (paddle_version <= 2.6)):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "core_attn"
        ):
            recompute_fn = rr_recompute if any(self.skip_recompute_ops.values()) else recompute
            outputs = recompute_fn(
                self.attn_func,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                alibi,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                sequence_parallel=self.sequence_parallel,
                reshard_layer=self.reshard_layer,
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
                alibi=alibi,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                sequence_parallel=self.sequence_parallel,
                reshard_layer=self.reshard_layer,
                npu_is_casual=npu_is_casual,
            )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        attn_output = self.o_proj(attn_output)

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


class LlamaDecoderLayer(nn.Layer):
    def __init__(self, config, layerwise_recompute: bool = False, skip_recompute_ops=None):
        super().__init__()
        self.config = config
        if skip_recompute_ops is None:
            skip_recompute_ops = {}
        self.skip_recompute_ops = skip_recompute_ops
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, layerwise_recompute, skip_recompute_ops=skip_recompute_ops)
        self.mlp = LlamaMLP(config, skip_recompute_ops=skip_recompute_ops)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)
        self.sequence_parallel = config.sequence_parallel
        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        alibi: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        npu_is_casual: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
            cache (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
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
            recompute_fn = rr_recompute if any(self.skip_recompute_ops.values()) else recompute
            outputs = recompute_fn(
                self.self_attn,
                hidden_states,
                position_ids,
                past_key_value,
                attention_mask,
                output_attentions,
                use_cache,
                alibi,
                attn_mask_startend_row_indices,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = self.self_attn(
                hidden_states,
                position_ids,
                past_key_value,
                attention_mask,
                output_attentions,
                use_cache,
                alibi,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                npu_is_casual=npu_is_casual,
            )

        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if output_attentions:
            self_attn_weights = outputs[1]

        if use_cache:
            present_key_value = outputs[2 if output_attentions else 1]

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class LlamaPretrainedModel(PretrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "llama"
    pretrained_init_configuration = LLAMA_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = LLAMA_PRETRAINED_RESOURCE_FILES_MAP
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]

    def _get_model_flops(self):
        if hasattr(self.config, "seq_length"):
            seq_length = self.config.seq_length
        else:
            seq_length = 2048

        return caculate_llm_per_token_flops(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            layer_num=self.config.num_hidden_layers,
            vocab_size=self.config.vocab_size,
            seq_length=seq_length,
            recompute=False,
        )

    def _get_hardware_flops(self):
        if hasattr(self.config, "seq_length"):
            seq_length = self.config.seq_length
        else:
            seq_length = 2048

        return caculate_llm_per_token_flops(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            layer_num=self.config.num_hidden_layers,
            vocab_size=self.config.vocab_size,
            seq_length=seq_length,
            recompute=self.config.recompute,
            recompute_granularity=self.config.recompute_granularity,
        )

    @classmethod
    def _get_name_mappings(cls, config: LlamaConfig) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.k_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.v_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
            ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)
        # base-model prefix "LlamaModel"
        if "LlamaModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "llama." + mapping[1]
            if not config.tie_word_embeddings:
                model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: LlamaConfig, is_split=True):

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
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            if config.tie_word_embeddings:
                base_actions["lm_head.weight"] = partial(fn, is_column=False)
            else:
                base_actions["lm_head.weight"] = partial(fn, is_column=True)

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("embed_tokens.weight")
            # Column Linear
            if config.fuse_attention_qkv:
                base_actions["layers.0.self_attn.qkv_proj.weight"] = partial(fn, is_column=True)
            else:
                base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
                # if we have enough num_key_value_heads to split, then split it.
                if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                    base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

            if config.fuse_attention_ffn:
                base_actions["layers.0.mlp.gate_up_fused_proj.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True
                )
            else:
                base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: LlamaConfig, is_fuse=False):
        # return parameter fuse utils
        from ..conversion_utils import split_or_fuse_func

        fn = split_or_fuse_func(is_fuse=is_fuse)

        # last key is fused key, other keys are to be fused.
        fuse_qkv_keys = (
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.qkv_proj.weight",
        )

        fuse_gate_up_keys = (
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.gate_up_fused_proj.weight",
        )
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)

        final_actions = {}
        if is_fuse:
            if fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_qkv_keys])
                    final_actions[keys] = partial(
                        fn, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                    )
            if fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_gate_up_keys])
                    final_actions[keys] = fn
        else:
            if not fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_qkv_keys])
                    final_actions[keys] = partial(
                        fn, split_nums=3, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                    )
            if not fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_gate_up_keys])
                    final_actions[keys] = partial(fn, split_nums=2)
        return final_actions

    def _init_weights(self, layer):
        """Initialization hook"""
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
                LlamaLMHead,
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
                                else self.llama.config.initializer_range,
                                shape=layer.weight.shape,
                            )
                        )
                else:
                    layer.weight.set_value(
                        paddle.tensor.normal(
                            mean=0.0,
                            std=self.config.initializer_range
                            if hasattr(self.config, "initializer_range")
                            else self.llama.config.initializer_range,
                            shape=layer.weight.shape,
                        )
                    )
        # Layer.apply is DFS https://github.com/PaddlePaddle/Paddle/blob/a6f5021fcc58b21f4414bae6bf4731ef6971582c/python/paddle/nn/layer/layers.py#L527-L530
        # sublayer is init first
        # scale RowParallelLinear weight
        with paddle.no_grad():
            if isinstance(layer, LlamaMLP):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.down_proj.weight.scale_(factor)
            if isinstance(layer, LlamaAttention):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.o_proj.weight.scale_(factor)


@register_base_model
class LlamaModel(LlamaPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.sequence_parallel = config.sequence_parallel
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []
        self.config = config

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            self.embed_tokens = mpu.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList(
            [
                LlamaDecoderLayer(
                    config=config,
                    layerwise_recompute=layer_idx not in self.no_recompute_layers,
                    skip_recompute_ops=get_skip_recompute_ops(config, layer_idx),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape, past_key_values_length=past_key_values_length
                    )
                    if get_env_device() in ["npu", "mlu", "intel_hpu"]:
                        expanded_attn_mask = expanded_attn_mask.astype("bool") & combined_attention_mask.astype("bool")
                    else:
                        expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        if get_env_device() in ["npu", "mlu", "intel_hpu"]:
            x = paddle.to_tensor(0.0, dtype="float32")
            y = paddle.to_tensor(paddle.finfo(dtype).min, dtype="float32")
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), x, y).astype(dtype)
        elif get_env_device() == "xpu":
            x = paddle.to_tensor(0.0, dtype="float32")
            y = paddle.to_tensor(-1.7005809656952787e38, dtype="float32")
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), x, y)
        elif get_env_device() == "gcu":
            min_val = paddle.finfo(dtype).min
            x = paddle.to_tensor(0.0, dtype=dtype)
            y = paddle.to_tensor(min_val, dtype=dtype)
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), x, y).astype(dtype)
        else:
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min)
            expanded_attn_mask = expanded_attn_mask.astype(dtype)
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
        alibi=None,
        attn_mask_startend_row_indices=None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        recompute_fn = rr_recompute if any(layer_module.skip_recompute_ops.values()) else recompute
        hidden_states = recompute_fn(
            create_custom_forward(layer_module),
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            alibi,
            attn_mask_startend_row_indices,
            use_reentrant=self.config.recompute_use_reentrant,
        )

        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ):
        if self.sequence_parallel and use_cache:
            raise ValueError("We currently only support sequence parallel without cache.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if self.config.context_parallel_degree > 1 and (attention_mask is not None or self.config.alibi):
            raise NotImplementedError("Ring FlashAttention doesn't support attention_mask or alibi")

        # embed positions
        if self.config.use_flash_attention_for_generation:
            attention_mask = None
        elif attn_mask_startend_row_indices is None and attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
        if attn_mask_startend_row_indices is None and self.config.alibi:
            if self.config.use_long_sequence_strategies:
                alibi_layer = LongSequenceStrategies.build_long_sequence_strategy(
                    self.config.long_sequence_strategy_type,
                    self.config.long_sequence_strategy_name,
                    **self.config.long_sequence_init_args,
                )
                alibi = alibi_layer(attention_mask, self.config.num_attention_heads, dtype=inputs_embeds.dtype)
            else:
                alibi = build_alibi_tensor(attention_mask, self.config.num_attention_heads, dtype=inputs_embeds.dtype)
            if self.config.tensor_parallel_degree > 1:
                block_size = self.config.num_attention_heads // self.config.tensor_parallel_degree
                alibi = alibi[
                    :,
                    self.config.tensor_parallel_rank
                    * block_size : (self.config.tensor_parallel_rank + 1)
                    * block_size,
                ]
                alibi = alibi.reshape([batch_size * block_size, 1, seq_length_with_past])
            else:
                alibi = alibi.reshape([batch_size * self.config.num_attention_heads, 1, seq_length_with_past])
        else:
            alibi = None

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        use_casual_mask = get_use_casual_mask() and not self.config.alibi

        if self.config.use_flash_attention_for_generation or use_casual_mask:
            attention_mask = None
        elif attn_mask_startend_row_indices is None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
            )  # [bs, 1, seq_len, seq_len]

        is_casual = False

        if (
            attn_mask_startend_row_indices is None
            and self.config.use_flash_attention
            and get_env_device() not in ["gcu", "intel_hpu"]
        ):
            if self.config.use_flash_attention_for_generation or use_casual_mask:
                is_casual = True
            else:
                is_casual = is_casual_mask(attention_mask)
            if get_env_device() not in ["npu", "mlu"]:
                if is_casual and alibi is None:
                    attention_mask = None
            else:
                attention_mask = None if attention_mask is None else attention_mask.astype("bool")
        hidden_states = inputs_embeds
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if (
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
                    alibi=alibi,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    alibi=alibi,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    npu_is_casual=is_casual,
                )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None
            if type(layer_outputs) is tuple:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if self.config.use_last_token_for_generation:
            hidden_states = paddle.unsqueeze(hidden_states[:, -1, :], 1)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )


class LlamaPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for Llama.
    It calculates the final loss.
    """

    def __init__(self, config):

        super(LlamaPretrainingCriterion, self).__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = (
            config.tensor_parallel_degree > 1
            and config.vocab_size % config.tensor_parallel_degree == 0
            and config.tensor_parallel_output
        )

        if self.enable_parallel_cross_entropy:  # and False: # and lm_head is distributed
            self.loss_func = mpu.ParallelCrossEntropy(ignore_index=self.ignore_index)
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels):
        if self.enable_parallel_cross_entropy:
            if prediction_scores.shape[-1] == self.config.vocab_size:
                warnings.warn(
                    f"enable_parallel_cross_entropy, the vocab_size should be splited: {prediction_scores.shape[-1]}, {self.config.vocab_size}"
                )
                self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))

            if self.config.sep_parallel_degree > 1 or self.config.context_parallel_degree > 1:
                _hcg = fleet.get_hybrid_communicate_group()
                masked_lm_loss = ConcatMaskedLoss.apply(masked_lm_loss, axis=1, group=_hcg.get_sep_parallel_group())
            # skip ignore_index which loss == 0
            # masked_lm_loss = masked_lm_loss[masked_lm_loss > 0]
            # loss = paddle.mean(masked_lm_loss)
            binary_sequence = paddle.where(
                masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count

        return loss


class ConcatMaskedLoss(PyLayer):
    @staticmethod
    def forward(ctx, inp, axis, group):
        inputs = []
        paddle.distributed.all_gather(inputs, inp, group=group)
        with paddle.no_grad():
            cat = paddle.concat(inputs, axis=axis)
        ctx.args_axis = axis
        ctx.args_group = group
        return cat

    @staticmethod
    def backward(ctx, grad):
        axis = ctx.args_axis
        group = ctx.args_group
        with paddle.no_grad():
            grads = paddle.split(grad, paddle.distributed.get_world_size(group), axis=axis)
        grad = grads[paddle.distributed.get_rank(group)]
        return grad


class LlamaLMHead(nn.Layer):
    def __init__(self, config: LlamaConfig, embedding_weights=None, transpose_y=False):
        super(LlamaLMHead, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        self.transpose_y = transpose_y
        if transpose_y:
            if embedding_weights is not None:
                self.weight = embedding_weights
            else:
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:
            if vocab_size != config.vocab_size:
                with get_rng_state_tracker().rng_state():
                    self.weight = self.create_parameter(
                        shape=[config.hidden_size, vocab_size],
                        dtype=paddle.get_default_dtype(),
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[config.hidden_size, vocab_size],
                    dtype=paddle.get_default_dtype(),
                )
        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = True if (vocab_size != config.vocab_size) else False
        if self.weight.is_distributed:
            # for tie_word_embeddings
            self.weight.split_axis = 0 if self.transpose_y else 1
        if get_env_device() == "xpu":
            try:
                from paddle_xpu.layers.nn import (  # noqa: F401
                    parallel_matmul as xpu_parallel_matmul,
                )

                self.xpu_parallel_matmul = xpu_parallel_matmul()
            except ImportError:
                self.xpu_parallel_matmul = None

    def forward(self, hidden_states, tensor_parallel_output=None):
        if self.config.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
            seq_length = self.config.seq_length
            if self.config.sep_parallel_degree > 1:
                assert seq_length % self.config.sep_parallel_degree == 0
                seq_length = seq_length // self.config.sep_parallel_degree
            if self.config.context_parallel_degree > 1:
                assert seq_length % self.config.context_parallel_degree == 0
                seq_length = seq_length // self.config.context_parallel_degree
            hidden_states = paddle.reshape_(hidden_states, [-1, seq_length, self.config.hidden_size])

        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1

        if get_env_device() == "xpu" and self.xpu_parallel_matmul is not None:
            logits = self.xpu_parallel_matmul(
                hidden_states,
                self.weight,
                transpose_y=self.transpose_y,
                tensor_parallel_output=tensor_parallel_output,
                training=self.training,
            )
        else:
            logits = parallel_matmul(
                hidden_states, self.weight, transpose_y=self.transpose_y, tensor_parallel_output=tensor_parallel_output
            )
        return logits


class LlamaForCausalLM(LlamaPretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.llama = LlamaModel(config)
        if config.tie_word_embeddings:
            self.lm_head = LlamaLMHead(config, embedding_weights=self.llama.embed_tokens.weight, transpose_y=True)
            self.tie_weights()
        else:
            self.lm_head = LlamaLMHead(config)
        self.criterion = LlamaPretrainingCriterion(config)

    def get_input_embeddings(self):
        return self.llama.embed_tokens

    def set_input_embeddings(self, value):
        self.llama.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llama = decoder

    def get_decoder(self):
        return self.llama

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
        attention_mask = kwargs.get("attention_mask", None)
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

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = paddle.concat(
                [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)], axis=-1
            )

        return model_kwargs

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        attn_mask_startend_row_indices=None,
    ):
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

        outputs = self.llama(
            input_ids,  # [bs, seq_len]
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs[0]  # [bs, seq_len, dim]

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
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                loss = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
