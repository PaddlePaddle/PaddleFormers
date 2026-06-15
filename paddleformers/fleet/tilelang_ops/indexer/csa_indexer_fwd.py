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

# TileLang fused forward kernel for DeepSeek V4 CSA compressed indexer.
#
# This kernel produces compressed top-k indices and top-k softmax probabilities
# directly from IndexQ/IndexKComp/Weights without materializing the full
# [B, S, S_comp] logits tensor. It follows the Megatron-DSA streaming top-k
# pattern, adapted to V4 CSA compressed-key semantics and BSHD/BSD layout.
#
# topk_effective semantics (controlled by caller, not by this kernel):
#   - Phase 2 (sparse warmup, dsa_indexer_use_sparse_loss=False):
#       topk_effective = n_compressed = floor(S / ratio).
#       The selected set covers the full causal compressed range, enabling
#       full-range KL loss equivalent to DSA dense warm-up.
#   - Phase 3 (sparse, dsa_indexer_use_sparse_loss=True):
#       topk_effective = min(index_topk, n_compressed), typically 512.
#       Standard selected-topk semantics for sparse training.
#   - Phase 1 (csa_dense_mode=True): this kernel is never called.
#
# Varlen support: the kernel accepts a ValidRange [B, S, 2] tensor specifying
# per-query [BOS, EOS) valid compressed K range. This enables document-mask
# (packed multi-document) training where queries must not attend to compressed
# keys from other documents. When ValidRange is not provided, the interface
# constructs it from ratio + seq_offset (causal-only mode).
#
# Padding: topk_effective is internally padded to the next power-of-2 that is
# also divisible by block_K (for bitonic sort alignment). Padded slots in the
# output are filled with -1 (indices) and 0.0 (scores). The caller receives
# only the first topk_effective columns (padding is stripped).

import math

import paddle
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def tl_csa_indexer_topk_fwd_impl(
    heads: int,
    dim: int,
    topk: int,
    ratio: int,
    seq_offset: int = 0,
    block_K: int = 32,
    dtype: str = "bfloat16",
    num_stages: int = 0,
    num_threads: int = 128,
):
    assert topk == tilelang.math.next_power_of_2(topk)
    assert topk % block_K == 0
    assert heads <= 64 and heads % 8 == 0
    assert num_stages == 0

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_comp = T.dynamic("seq_len_comp")

    INT32 = "int32"
    FP32 = "float"

    index_q_shape = [batch, seq_len, heads, dim]
    weights_shape = [batch, seq_len, heads]
    index_k_shape = [batch, seq_len_comp, dim]
    valid_range_shape = [batch, seq_len, 2]
    topk_indices_shape = [batch, seq_len, topk]
    topk_scores_shape = [batch, seq_len, topk]

    N = 2 * topk
    num_iters = int(round(math.log2(N)))
    sm_scale = dim**-0.5

    @T.macro
    def bitonic_sort(
        topk_index_shared: T.SharedBuffer([N], dtype=INT32),
        topk_value_shared: T.SharedBuffer([N], dtype=FP32),
    ):
        T.sync_threads()
        for i1 in T.serial(num_iters):
            for i2 in T.serial(i1 + 1):
                for i in T.Parallel(N):
                    ascending = (i & (1 << (i1 + 1))) != 0
                    j = i ^ (1 << (i1 - i2))
                    if i < j and (
                        (ascending and topk_value_shared[i] > topk_value_shared[j])
                        or (not ascending and topk_value_shared[i] < topk_value_shared[j])
                    ):
                        val = topk_value_shared[i]
                        topk_value_shared[i] = topk_value_shared[j]
                        topk_value_shared[j] = val
                        idx = topk_index_shared[i]
                        topk_index_shared[i] = topk_index_shared[j]
                        topk_index_shared[j] = idx
                T.sync_threads()

    @T.prim_func
    def tl_csa_indexer_topk_fwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexKComp: T.Tensor(index_k_shape, dtype),
        Weights: T.Tensor(weights_shape, FP32),
        ValidRange: T.Tensor(valid_range_shape, INT32),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        TopkScores: T.Tensor(topk_scores_shape, FP32),
    ):
        with T.Kernel(seq_len, batch, threads=num_threads) as (bx, by):
            i_t = bx
            i_b = by

            valid_start = ValidRange[i_b, i_t, 0]
            valid_end = ValidRange[i_b, i_t, 1]
            start_block = valid_start // block_K
            num_valid_blocks = T.ceildiv(valid_end, block_K) - start_block

            topk_index_shared = T.alloc_shared([N], dtype=INT32)
            topk_value_shared = T.alloc_shared([N], dtype=FP32)
            T.fill(topk_index_shared, -1)
            T.fill(topk_value_shared, float("-inf"))
            T.sync_threads()

            index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
            T.copy(IndexQ[i_b, i_t, :, :], index_q_shared)
            T.sync_threads()

            weights_shared = T.alloc_shared([heads], dtype=FP32)
            T.copy(Weights[i_b, i_t, :], weights_shared)
            T.sync_threads()

            # Fold sm_scale into weights (fp32) to avoid bf16 truncation on Q
            for i in T.Parallel(heads):
                weights_shared[i] = weights_shared[i] * sm_scale
            T.sync_threads()

            for bk_i in T.Pipelined(num_valid_blocks, num_stages=num_stages):
                k_st = (start_block + bk_i) * block_K
                k_ed = T.min(k_st + block_K, valid_end)

                index_k_shared = T.alloc_shared([block_K, dim], dtype=dtype)
                for i, j in T.Parallel(block_K, dim):
                    index_k_shared[i, j] = T.if_then_else(
                        (k_st + i >= valid_start) & (k_st + i < k_ed),
                        IndexKComp[i_b, k_st + i, j],
                        0,
                    )
                T.sync_threads()

                logits = T.alloc_fragment((block_K, heads), FP32)
                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                T.sync_threads()

                for i, j in T.Parallel(block_K, heads):
                    logits[i, j] = T.max(logits[i, j], 0) * weights_shared[j]
                T.sync_threads()

                logits_sum = T.alloc_fragment(block_K, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)
                T.sync_threads()

                # Buffer management uses relative offset (iteration count)
                k_rel = bk_i * block_K
                offset = T.alloc_var(INT32)
                if k_rel >= topk:
                    offset = topk + (k_rel % topk)
                else:
                    offset = k_rel
                T.sync_threads()

                for i in T.Parallel(block_K):
                    valid_flag = (k_st + i >= valid_start) & (k_st + i < valid_end)
                    if not valid_flag:
                        logits_sum[i] = float("-inf")
                    j = offset + i
                    topk_index_shared[j] = T.if_then_else(valid_flag, k_st + i, -1)
                    topk_value_shared[j] = logits_sum[i]
                T.sync_threads()

                k_rel_ed = (bk_i + 1) * block_K
                if k_rel_ed > topk and k_rel_ed % topk == 0:
                    bitonic_sort(topk_index_shared, topk_value_shared)

            bitonic_sort(topk_index_shared, topk_value_shared)

            logits_max_frag = T.alloc_fragment([1], dtype=FP32)
            logits_frag = T.alloc_fragment([topk], dtype=FP32)
            scores_shared = T.alloc_shared([topk], dtype=FP32)

            T.copy(topk_value_shared[:topk], logits_frag)
            T.sync_threads()
            T.reduce_max(logits_frag, logits_max_frag, dim=-1)
            T.sync_threads()

            for i in T.Parallel(topk):
                logits_frag[i] = T.if_then_else(
                    topk_index_shared[i] >= 0,
                    T.exp(logits_frag[i] - logits_max_frag[0]),
                    0,
                )
            T.sync_threads()

            lse_frag = T.alloc_fragment([1], dtype=FP32)
            T.reduce_sum(logits_frag, lse_frag)
            T.sync_threads()

            for i in T.Parallel(topk):
                scores_shared[i] = T.if_then_else(
                    topk_index_shared[i] >= 0,
                    logits_frag[i] / lse_frag[0],
                    0,
                )
            T.sync_threads()

            T.copy(topk_index_shared[:topk], TopkIndices[i_b, i_t, :])
            T.copy(scores_shared[:topk], TopkScores[i_b, i_t, :])

    return tl_csa_indexer_topk_fwd_kernel


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _pad_topk_output(indices, scores, topk: int):
    if indices.shape[-1] == topk:
        return indices, scores
    return indices[..., :topk].contiguous(), scores[..., :topk].contiguous()


def _tilelang_dtype(tensor):
    if tensor.dtype == paddle.bfloat16:
        return "bfloat16"
    if tensor.dtype == paddle.float16:
        return "float16"
    raise TypeError(f"TileLang CSA indexer expects bf16/fp16 inputs, got {tensor.dtype}")


def _shape(tensor):
    return tuple(tensor.shape)


def _require_tensor(name, tensor):
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"{name} must be a paddle.Tensor, got {type(tensor)!r}")


def _require_contiguous(name, tensor):
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_interface_inputs(
    index_q,
    index_k_comp,
    weights,
    topk_effective,
    block_K,
    num_stages,
):
    for name, tensor in (
        ("index_q", index_q),
        ("index_k_comp", index_k_comp),
        ("weights", weights),
    ):
        _require_tensor(name, tensor)
        _require_contiguous(name, tensor)
    if index_q.ndim != 4:
        raise ValueError(f"index_q must have shape [B, S, H_i, D_i], got {_shape(index_q)}")
    if index_k_comp.ndim != 3:
        raise ValueError(f"index_k_comp must have shape [B, S_comp, D_i], got {_shape(index_k_comp)}")
    if weights.ndim != 3:
        raise ValueError(f"weights must have shape [B, S, H_i], got {_shape(weights)}")

    batch, seq_len, heads, dim = _shape(index_q)
    batch_k, _, dim_k = _shape(index_k_comp)
    batch_w, seq_len_w, heads_w = _shape(weights)
    if batch != batch_k or batch != batch_w:
        raise ValueError(
            f"batch mismatch: index_q={_shape(index_q)}, index_k_comp={_shape(index_k_comp)}, weights={_shape(weights)}"
        )
    if seq_len != seq_len_w or heads != heads_w or dim != dim_k:
        raise ValueError(
            f"shape mismatch: index_q={_shape(index_q)}, index_k_comp={_shape(index_k_comp)}, weights={_shape(weights)}"
        )
    if heads > 64 or heads % 8 != 0:
        raise ValueError(f"heads must be <= 64 and divisible by 8, got {heads}")
    if int(topk_effective) <= 0:
        raise ValueError(f"topk_effective must be positive, got {topk_effective}")
    if int(block_K) <= 0:
        raise ValueError(f"block_K must be positive, got {block_K}")
    if int(num_stages) != 0:
        raise ValueError(f"num_stages must be 0, got {num_stages}")


def _build_causal_valid_range(
    batch: int,
    seq_len: int,
    seq_len_comp: int,
    ratio: int,
    seq_offset: int,
) -> "paddle.Tensor":
    """Build ValidRange [B, S, 2] for causal-only mode (no varlen).

    Equivalent to the original kernel behavior:
        valid_start = 0
        valid_end = min((t + seq_offset + 1) // ratio, seq_len_comp)
    """
    q_pos = paddle.arange(seq_len, dtype="int32") + seq_offset
    valid_end = paddle.minimum(
        (q_pos + 1) // ratio,
        paddle.full([seq_len], seq_len_comp, dtype="int32"),
    )
    # [S, 2] -> [1, S, 2] -> [B, S, 2]
    valid_start = paddle.zeros([seq_len], dtype="int32")
    vr = paddle.stack([valid_start, valid_end], axis=-1).unsqueeze(0)
    if batch > 1:
        vr = vr.expand([batch, -1, -1])
    return vr.contiguous()


def csa_indexer_topk_fwd_interface(
    index_q,
    index_k_comp,
    weights,
    ratio: int,
    topk_effective: int,
    seq_offset: int = 0,
    valid_range: paddle.Tensor | None = None,
    block_K: int = 32,
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Run V4 CSA fused compressed indexer forward.

    Args:
        index_q: [B, S, H_i, D_i] bf16/fp16, BSHD layout.
        index_k_comp: [B, S_comp, D_i] bf16/fp16, BSD layout.
        weights: [B, S, H_i] fp32 or castable to fp32.
        ratio: compression ratio. Valid compressed range for query t is
            [0, (t + seq_offset + 1) // ratio). Used only when
            valid_range is None to build causal-only ValidRange.
        topk_effective: requested output top-k width. Phase 2 may set this to
            S_comp; Phase 3 usually sets this to dsa_indexer_topk.
        seq_offset: global position offset for the first local query token.
            Used only when valid_range is None. In CP mode, cp_rank * sq_local.
        valid_range: [B, S, 2] int32 tensor specifying per-query [BOS, EOS)
            valid compressed K range (left-closed, right-open). If None,
            automatically built from ratio + seq_offset (causal-only mode).

    Returns:
        topk_indices: [B, S, topk_effective] int32, invalid slots are -1.
        topk_scores: [B, S, topk_effective] fp32 top-k softmax probabilities.
    """
    _validate_interface_inputs(
        index_q,
        index_k_comp,
        weights,
        topk_effective,
        block_K,
        num_stages,
    )

    batch, seq_len, heads, dim = index_q.shape
    seq_len_comp = index_k_comp.shape[1]

    # Build or validate ValidRange
    if valid_range is None:
        valid_range = _build_causal_valid_range(batch, seq_len, seq_len_comp, ratio, int(seq_offset))
    else:
        _require_tensor("valid_range", valid_range)
        if valid_range.ndim != 3 or valid_range.shape[2] != 2:
            raise ValueError(f"valid_range must have shape [B, S, 2], got {_shape(valid_range)}")
        if valid_range.shape[0] != batch or valid_range.shape[1] != seq_len:
            raise ValueError(
                f"valid_range shape {_shape(valid_range)} incompatible with " f"index_q shape {_shape(index_q)}"
            )
        if valid_range.dtype != paddle.int32:
            valid_range = valid_range.cast("int32")
        valid_range = valid_range.contiguous()

    padded_topk = _next_power_of_2(topk_effective)
    if padded_topk % block_K != 0:
        padded_topk = ((padded_topk + block_K - 1) // block_K) * block_K
        padded_topk = _next_power_of_2(padded_topk)

    kernel = tl_csa_indexer_topk_fwd_impl(
        heads=heads,
        dim=dim,
        topk=padded_topk,
        ratio=ratio,
        seq_offset=int(seq_offset),
        block_K=block_K,
        dtype=_tilelang_dtype(index_q),
        num_stages=num_stages,
        num_threads=num_threads,
    )

    topk_indices = paddle.empty([batch, seq_len, padded_topk], dtype="int32")
    topk_scores = paddle.empty([batch, seq_len, padded_topk], dtype="float32")

    if weights.dtype != paddle.float32:
        weights = weights.cast("float32").contiguous()

    kernel(
        index_q,
        index_k_comp,
        weights,
        valid_range,
        topk_indices,
        topk_scores,
    )

    return _pad_topk_output(topk_indices, topk_scores, topk_effective)
