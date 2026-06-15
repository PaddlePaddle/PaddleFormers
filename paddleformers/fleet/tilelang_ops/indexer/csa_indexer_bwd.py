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

# TileLang backward kernel for DeepSeek V4 CSA compressed indexer.
#
# This kernel consumes selected compressed block indices plus OGrad and computes
# gradients for IndexQ/Weights/IndexKComp without materializing full
# [B, S, S_comp] indexer logits. OGrad is the gradient w.r.t. selected indexer
# logits/scores supplied by the loss wrapper.
#
# topk_effective semantics (controlled by caller, not by this kernel):
#   - Phase 2 (sparse warmup, dsa_indexer_use_sparse_loss=False):
#       topk_effective = n_compressed. The backward covers the full compressed
#       candidate range, equivalent to full-range KL gradient.
#   - Phase 3 (sparse, dsa_indexer_use_sparse_loss=True):
#       topk_effective = min(index_topk, n_compressed), typically 512.
#       Backward only covers the selected-topk set.
#   - Phase 1 (csa_dense_mode=True): this kernel is never called.
#
# Padding: topk_effective is internally padded to the next power-of-2 that is
# also divisible by block_I. Padded slots (index == -1) are masked out and
# contribute zero gradient. The caller pads grad_scores with 0 for invalid
# slots before calling this kernel.


import paddle
import tilelang
from tilelang import language as T


def _tilelang_dtype(tensor):
    if tensor.dtype == paddle.bfloat16:
        return "bfloat16"
    if tensor.dtype == paddle.float16:
        return "float16"
    raise TypeError(f"TileLang CSA indexer backward expects bf16/fp16 inputs, got {tensor.dtype}")


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
    weights,
    index_k_comp,
    topk_indices,
    grad_scores,
    block_I,
    num_stages,
):
    for name, tensor in (
        ("index_q", index_q),
        ("weights", weights),
        ("index_k_comp", index_k_comp),
        ("topk_indices", topk_indices),
        ("grad_scores", grad_scores),
    ):
        _require_tensor(name, tensor)
        _require_contiguous(name, tensor)
    if index_q.ndim != 4:
        raise ValueError(f"index_q must have shape [B, S, H_i, D_i], got {_shape(index_q)}")
    if weights.ndim != 3:
        raise ValueError(f"weights must have shape [B, S, H_i], got {_shape(weights)}")
    if index_k_comp.ndim != 3:
        raise ValueError(f"index_k_comp must have shape [B, S_comp, D_i], got {_shape(index_k_comp)}")
    if topk_indices.ndim != 3:
        raise ValueError(f"topk_indices must have shape [B, S, topk], got {_shape(topk_indices)}")
    if grad_scores.ndim != 3:
        raise ValueError(f"grad_scores must have shape [B, S, topk], got {_shape(grad_scores)}")

    batch, seq_len, heads, dim = _shape(index_q)
    batch_w, seq_len_w, heads_w = _shape(weights)
    batch_k, _, dim_k = _shape(index_k_comp)
    batch_i, seq_len_i, topk_effective = _shape(topk_indices)
    batch_g, seq_len_g, topk_g = _shape(grad_scores)
    if not (batch == batch_w == batch_k == batch_i == batch_g):
        raise ValueError(
            "batch mismatch: "
            f"index_q={_shape(index_q)}, weights={_shape(weights)}, index_k_comp={_shape(index_k_comp)}, "
            f"topk_indices={_shape(topk_indices)}, grad_scores={_shape(grad_scores)}"
        )
    if not (seq_len == seq_len_w == seq_len_i == seq_len_g):
        raise ValueError(
            "sequence mismatch: "
            f"index_q={_shape(index_q)}, weights={_shape(weights)}, topk_indices={_shape(topk_indices)}, grad_scores={_shape(grad_scores)}"
        )
    if heads != heads_w:
        raise ValueError(f"heads mismatch: index_q={_shape(index_q)}, weights={_shape(weights)}")
    if dim != dim_k:
        raise ValueError(f"dim mismatch: index_q={_shape(index_q)}, index_k_comp={_shape(index_k_comp)}")
    if topk_effective != topk_g:
        raise ValueError(f"topk mismatch: topk_indices={_shape(topk_indices)}, grad_scores={_shape(grad_scores)}")
    if heads > 64 or heads % 8 != 0:
        raise ValueError(f"heads must be <= 64 and divisible by 8, got {heads}")
    if topk_effective <= 0:
        raise ValueError("topk_indices last dimension must be positive")
    if int(block_I) <= 0:
        raise ValueError(f"block_I must be positive, got {block_I}")
    if int(num_stages) != 0:
        raise ValueError(f"num_stages must be 0, got {num_stages}")


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
    }
)
def tl_csa_indexer_bwd_impl(
    heads: int,
    dim: int,
    topk: int,
    block_I: int = 32,
    dtype: str = "bfloat16",
    num_stages: int = 0,
    num_threads: int = 128,
):
    assert num_stages == 0
    assert topk == tilelang.math.next_power_of_2(topk)
    assert topk % block_I == 0
    assert heads <= 64 and heads % 8 == 0

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_comp = T.dynamic("seq_len_comp")

    FP32 = "float"
    INT32 = "int32"
    sm_scale = dim**-0.5

    index_q_shape = [batch, seq_len, heads, dim]
    weights_shape = [batch, seq_len, heads]
    index_k_shape = [batch, seq_len_comp, dim]
    topk_indices_shape = [batch, seq_len, topk]
    grad_scores_shape = [batch, seq_len, topk]

    @T.prim_func
    def tl_csa_indexer_bwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexKComp: T.Tensor(index_k_shape, dtype),
        Weights: T.Tensor(weights_shape, FP32),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        OGrad: T.Tensor(grad_scores_shape, FP32),
        dIndexQ: T.Tensor(index_q_shape, dtype),
        dWeights: T.Tensor(weights_shape, FP32),
        dIndexKComp: T.Tensor(index_k_shape, FP32),
    ):
        with T.Kernel(seq_len, batch, threads=num_threads) as (bx, by):
            i_t = bx
            i_b = by
            index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
            index_q_scaled_shared = T.alloc_shared([heads, dim], dtype=dtype)
            weights_shared = T.alloc_shared([heads], dtype=FP32)
            indices_shared = T.alloc_shared([block_I], dtype=INT32)
            grad_shared = T.alloc_shared([block_I], dtype=FP32)
            index_k_shared = T.alloc_shared([block_I, dim], dtype=dtype)

            d_index_q_frag = T.alloc_fragment([heads, dim], dtype=FP32)
            d_weights_frag = T.alloc_fragment([heads], dtype=FP32)
            d_index_k_frag = T.alloc_fragment([block_I, dim], dtype=FP32)
            logits = T.alloc_fragment((block_I, heads), dtype=FP32)
            d_logits_qk = T.alloc_shared((block_I, heads), dtype=FP32)
            d_logits_qk_cast1 = T.alloc_fragment((block_I, heads), dtype=dtype)
            d_logits_qk_cast2 = T.alloc_fragment((block_I, heads), dtype=dtype)

            T.copy(IndexQ[i_b, i_t, :, :], index_q_shared)
            T.copy(Weights[i_b, i_t, :], weights_shared)
            T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                index_q_scaled_shared[i, j] = index_q_shared[i, j] * sm_scale
            T.sync_threads()

            T.fill(d_index_q_frag, 0)
            T.fill(d_weights_frag, 0)
            num_blocks = T.ceildiv(topk, block_I)

            for bi_i in T.serial(num_blocks):
                for i in T.Parallel(block_I):
                    indices_shared[i] = TopkIndices[i_b, i_t, bi_i * block_I + i]
                    grad_shared[i] = OGrad[i_b, i_t, bi_i * block_I + i]
                T.sync_threads()

                for i, j in T.Parallel(block_I, dim):
                    index_k_shared[i, j] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        IndexKComp[i_b, indices_shared[i], j],
                        0,
                    )
                T.sync_threads()

                T.gemm(
                    index_k_shared,
                    index_q_scaled_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                T.sync_threads()

                for i, j in T.Parallel(block_I, heads):
                    logits[i, j] = T.max(logits[i, j], 0)
                T.sync_threads()

                d_weights_i = T.alloc_fragment((block_I, heads), dtype=FP32)
                for i, j in T.Parallel(block_I, heads):
                    d_weights_i[i, j] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        grad_shared[i] * logits[i, j],
                        0,
                    )
                T.reduce_sum(d_weights_i, d_weights_frag, dim=0, clear=False)

                for i, j in T.Parallel(block_I, heads):
                    d_logits_qk[i, j] = T.if_then_else(
                        ((indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp)) & (logits[i, j] > 0),
                        grad_shared[i] * weights_shared[j],
                        0,
                    )
                T.sync_threads()

                T.copy(d_logits_qk, d_logits_qk_cast1)
                T.gemm(
                    d_logits_qk_cast1,
                    index_k_shared,
                    d_index_q_frag,
                    transpose_A=True,
                    transpose_B=False,
                    clear_accum=False,
                )

                T.copy(d_logits_qk, d_logits_qk_cast2)
                T.gemm(
                    d_logits_qk_cast2,
                    index_q_scaled_shared,
                    d_index_k_frag,
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=True,
                )

                for i, j in T.Parallel(block_I, dim):
                    if (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp):
                        T.atomic_add(
                            dIndexKComp[i_b, indices_shared[i], j],
                            d_index_k_frag[i, j],
                        )

                # Prevent the race condition: threads in the warp might overwrite topk in the next loop
                T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                d_index_q_frag[i, j] = d_index_q_frag[i, j] * sm_scale

            T.copy(d_index_q_frag, dIndexQ[i_b, i_t, :, :])
            T.copy(d_weights_frag, dWeights[i_b, i_t, :])

    return tl_csa_indexer_bwd_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
    }
)
def tl_csa_indexer_bwd_det_impl(
    heads: int,
    dim: int,
    topk: int,
    block_I: int = 32,
    dtype: str = "bfloat16",
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Deterministic indexer backward: writes dIndexKComp to per-token buffer (no atomics)."""
    assert num_stages == 0
    assert topk == tilelang.math.next_power_of_2(topk)
    assert topk % block_I == 0
    assert heads <= 64 and heads % 8 == 0

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_comp = T.dynamic("seq_len_comp")

    FP32 = "float"
    INT32 = "int32"
    sm_scale = dim**-0.5

    index_q_shape = [batch, seq_len, heads, dim]
    weights_shape = [batch, seq_len, heads]
    index_k_shape = [batch, seq_len_comp, dim]
    topk_indices_shape = [batch, seq_len, topk]
    grad_scores_shape = [batch, seq_len, topk]

    @T.prim_func
    def tl_csa_indexer_bwd_det_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexKComp: T.Tensor(index_k_shape, dtype),
        Weights: T.Tensor(weights_shape, FP32),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        OGrad: T.Tensor(grad_scores_shape, FP32),
        dIndexQ: T.Tensor(index_q_shape, dtype),
        dWeights: T.Tensor(weights_shape, FP32),
        dIndexKComp_buf: T.Tensor([batch, seq_len, topk, dim], FP32),
    ):
        with T.Kernel(seq_len, batch, threads=num_threads) as (bx, by):
            i_t = bx
            i_b = by
            index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
            index_q_scaled_shared = T.alloc_shared([heads, dim], dtype=dtype)
            weights_shared = T.alloc_shared([heads], dtype=FP32)
            indices_shared = T.alloc_shared([block_I], dtype=INT32)
            grad_shared = T.alloc_shared([block_I], dtype=FP32)
            index_k_shared = T.alloc_shared([block_I, dim], dtype=dtype)

            d_index_q_frag = T.alloc_fragment([heads, dim], dtype=FP32)
            d_weights_frag = T.alloc_fragment([heads], dtype=FP32)
            d_index_k_frag = T.alloc_fragment([block_I, dim], dtype=FP32)
            logits = T.alloc_fragment((block_I, heads), dtype=FP32)
            d_logits_qk = T.alloc_shared((block_I, heads), dtype=FP32)
            d_logits_qk_cast1 = T.alloc_fragment((block_I, heads), dtype=dtype)
            d_logits_qk_cast2 = T.alloc_fragment((block_I, heads), dtype=dtype)

            T.copy(IndexQ[i_b, i_t, :, :], index_q_shared)
            T.copy(Weights[i_b, i_t, :], weights_shared)
            T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                index_q_scaled_shared[i, j] = index_q_shared[i, j] * sm_scale
            T.sync_threads()

            T.fill(d_index_q_frag, 0)
            T.fill(d_weights_frag, 0)
            num_blocks = T.ceildiv(topk, block_I)

            for bi_i in T.serial(num_blocks):
                for i in T.Parallel(block_I):
                    indices_shared[i] = TopkIndices[i_b, i_t, bi_i * block_I + i]
                    grad_shared[i] = OGrad[i_b, i_t, bi_i * block_I + i]
                T.sync_threads()

                for i, j in T.Parallel(block_I, dim):
                    index_k_shared[i, j] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        IndexKComp[i_b, indices_shared[i], j],
                        0,
                    )
                T.sync_threads()

                T.gemm(
                    index_k_shared,
                    index_q_scaled_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                T.sync_threads()

                for i, j in T.Parallel(block_I, heads):
                    logits[i, j] = T.max(logits[i, j], 0)
                T.sync_threads()

                d_weights_i = T.alloc_fragment((block_I, heads), dtype=FP32)
                for i, j in T.Parallel(block_I, heads):
                    d_weights_i[i, j] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        grad_shared[i] * logits[i, j],
                        0,
                    )
                T.reduce_sum(d_weights_i, d_weights_frag, dim=0, clear=False)

                for i, j in T.Parallel(block_I, heads):
                    d_logits_qk[i, j] = T.if_then_else(
                        ((indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp)) & (logits[i, j] > 0),
                        grad_shared[i] * weights_shared[j],
                        0,
                    )
                T.sync_threads()

                T.copy(d_logits_qk, d_logits_qk_cast1)
                T.gemm(
                    d_logits_qk_cast1,
                    index_k_shared,
                    d_index_q_frag,
                    transpose_A=True,
                    transpose_B=False,
                    clear_accum=False,
                )

                T.copy(d_logits_qk, d_logits_qk_cast2)
                T.gemm(
                    d_logits_qk_cast2,
                    index_q_scaled_shared,
                    d_index_k_frag,
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=True,
                )

                # Deterministic: write to per-token buffer (no atomics)
                for i, j in T.Parallel(block_I, dim):
                    dIndexKComp_buf[i_b, i_t, bi_i * block_I + i, j] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        d_index_k_frag[i, j],
                        0,
                    )
                # Prevent the race condition: threads in the warp might overwrite topk in the next loop
                T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                d_index_q_frag[i, j] = d_index_q_frag[i, j] * sm_scale

            T.copy(d_index_q_frag, dIndexQ[i_b, i_t, :, :])
            T.copy(d_weights_frag, dWeights[i_b, i_t, :])

    return tl_csa_indexer_bwd_det_kernel


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def dindexk_reduce(
    B,
    S,
    S_comp,
    topk,
    D,
    BK=4,
    threads=128,
    indices_dtype=T.int32,
    accum_dtype=T.float32,
):
    """Deterministic reduction for dIndexKComp using CSR structure."""
    buf_shape = [B, S, topk, D]
    sort_perm_shape = [B, S * topk]
    seg_offsets_shape = [B, S_comp + 1]
    out_shape = [B, S_comp, D]

    @T.prim_func
    def dindexk_reduce_kernel(
        dIndexKComp_buf: T.Tensor(buf_shape, accum_dtype),
        sort_perm: T.Tensor(sort_perm_shape, indices_dtype),
        seg_offsets: T.Tensor(seg_offsets_shape, indices_dtype),
        dIndexKComp: T.Tensor(out_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(S_comp, BK), B, threads=threads) as (bk, by):
            acc = T.alloc_fragment([D], accum_dtype)

            for ki in range(BK):
                k = bk * BK + ki
                if k < S_comp:
                    T.clear(acc)
                    start = seg_offsets[by, k]
                    end = seg_offsets[by, k + 1]

                    for fi in T.serial(end - start):
                        flat_pos = sort_perm[by, start + fi]
                        s_idx = flat_pos // topk
                        t_idx = flat_pos % topk
                        for d_i in T.Parallel(D):
                            acc[d_i] += dIndexKComp_buf[by, s_idx, t_idx, d_i]

                    for d_i in T.Parallel(D):
                        dIndexKComp[by, k, d_i] = acc[d_i]

    return dindexk_reduce_kernel


def _build_indexer_csr_index(topk_indices, S_comp):
    """Build CSR inverse index for deterministic dIndexKComp reduction.

    Args:
        topk_indices: [B, S, topk] int32.
        S_comp: int — number of compressed KV positions.

    Returns:
        sort_perm:   [B, S*topk] int32.
        seg_offsets: [B, S_comp+1] int32.
    """
    B, S, topk = topk_indices.shape
    flat_idx = topk_indices.reshape([B, S * topk])

    flat_idx_shifted = paddle.where(
        flat_idx < 0,
        paddle.full_like(flat_idx, S_comp),
        flat_idx,
    )

    sort_perm = paddle.argsort(flat_idx_shifted, axis=1, stable=True).cast("int32")
    sorted_idx = paddle.take_along_axis(flat_idx_shifted, sort_perm.cast("int64"), axis=1)

    boundaries = paddle.arange(S_comp + 1, dtype="int64").unsqueeze(0).expand([B, -1])
    seg_offsets = paddle.searchsorted(sorted_idx, boundaries).cast("int32")

    return sort_perm, seg_offsets


def csa_indexer_bwd_interface(
    index_q,
    weights,
    index_k_comp,
    topk_indices,
    grad_scores,
    block_I: int = 32,
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Run V4 CSA compressed indexer backward.

    Args:
        index_q: [B, S, H_i, D_i] bf16/fp16, BSHD layout.
        weights: [B, S, H_i] fp32 or castable to fp32.
        index_k_comp: [B, S_comp, D_i] bf16/fp16, BSD layout.
        topk_indices: [B, S, topk_effective] int32, invalid slots are -1.
        grad_scores: [B, S, topk_effective] fp32 OGrad for selected logits.

    Returns:
        grad_q: [B, S, H_i, D_i] same dtype as index_q.
        grad_weights: [B, S, H_i] fp32.
        grad_k_comp: [B, S_comp, D_i] fp32.
    """
    _validate_interface_inputs(
        index_q,
        weights,
        index_k_comp,
        topk_indices,
        grad_scores,
        block_I,
        num_stages,
    )

    deterministic = paddle.get_flags(["FLAGS_cudnn_deterministic"])["FLAGS_cudnn_deterministic"]

    batch, seq_len, heads, dim = index_q.shape
    _, seq_len_comp, _ = index_k_comp.shape
    topk_effective = topk_indices.shape[-1]
    padded_topk = 1 << (topk_effective - 1).bit_length()
    if padded_topk % block_I != 0:
        padded_topk = ((padded_topk + block_I - 1) // block_I) * block_I
        padded_topk = 1 << (padded_topk - 1).bit_length()

    if padded_topk != topk_effective:
        pad = padded_topk - topk_effective
        topk_pad = paddle.full(
            [batch, seq_len, pad],
            -1,
            topk_indices.dtype,
        )
        grad_pad = paddle.zeros(
            [batch, seq_len, pad],
            grad_scores.dtype,
        )
        topk_indices = paddle.concat([topk_indices, topk_pad], axis=-1).contiguous()
        grad_scores = paddle.concat([grad_scores, grad_pad], axis=-1).contiguous()

    if weights.dtype != paddle.float32:
        weights = weights.cast("float32").contiguous()
    if grad_scores.dtype != paddle.float32:
        grad_scores = grad_scores.cast("float32").contiguous()

    grad_q = paddle.empty_like(index_q)
    grad_weights = paddle.empty_like(weights, dtype="float32")

    if not deterministic:
        # === Non-deterministic path (atomic_add) ===
        kernel = tl_csa_indexer_bwd_impl(
            heads=heads,
            dim=dim,
            topk=padded_topk,
            block_I=block_I,
            dtype=_tilelang_dtype(index_q),
            num_stages=num_stages,
            num_threads=num_threads,
        )
        grad_k_comp = paddle.zeros([batch, seq_len_comp, dim], dtype="float32")
        kernel(
            index_q,
            index_k_comp,
            weights,
            topk_indices,
            grad_scores,
            grad_q,
            grad_weights,
            grad_k_comp,
        )
    else:
        # === Deterministic path (per-token buffer + CSR reduction) ===
        kernel_det = tl_csa_indexer_bwd_det_impl(
            heads=heads,
            dim=dim,
            topk=padded_topk,
            block_I=block_I,
            dtype=_tilelang_dtype(index_q),
            num_stages=num_stages,
            num_threads=num_threads,
        )
        dindexk_buf = paddle.empty([batch, seq_len, padded_topk, dim], dtype="float32")
        kernel_det(
            index_q,
            index_k_comp,
            weights,
            topk_indices,
            grad_scores,
            grad_q,
            grad_weights,
            dindexk_buf,
        )

        # CSR-ordered deterministic reduction
        sort_perm, seg_offsets = _build_indexer_csr_index(topk_indices, seq_len_comp)
        reduce_kernel = dindexk_reduce(batch, seq_len, seq_len_comp, padded_topk, dim)
        grad_k_comp = reduce_kernel(dindexk_buf, sort_perm, seg_offsets)

    return grad_q, grad_weights, grad_k_comp
