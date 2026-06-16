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

# Refer to https://github.com/radixark/miles/pull/1045/

import paddle
import tilelang
from tilelang import language as T


@tilelang.jit(out_idx=[-1])
def preprocess(
    B,
    S,
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(
                    O[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    o,
                )
                T.copy(
                    dO[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    do,
                )
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    B,
    S_kv,
    D,
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    dkv_shape = [B, S_kv, D]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), B, threads=threads) as (bx, by):
            T.copy(
                dKV[by, bx * block_N : (bx + 1) * block_N, :],
                dKV_out[by, bx * block_N : (bx + 1) * block_N, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def bwd(
    B,
    S,
    S_kv,
    H,
    D,
    topk,
    sm_scale=None,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert topk % block_size == 0, (
        f"topk ({topk}) must be divisible by block_size ({block_size})"
    )
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]

    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mqa_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(kv_shape, accum_dtype),
        dAttnSink: T.Tensor(attn_sink_shape, accum_dtype),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(
                dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared
            )

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] != -1

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_p.dtype)
                    )

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[
                        by, Indices[by, s_i, i_i * BS + bi_i], d_i
                    ]

                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_p,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                # P = exp2(scores * sm_scale_log2e - LSE)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2
                        - Lse[by, s_i, bz * block_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                # dP = P * (dO @ KV^T - Delta)
                T.gemm(
                    dO_shared,
                    KV_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i]
                        * (
                            acc_dp[h_i, bi_i]
                            - Delta[by, s_i, bz * block_H + h_i]
                        )
                        * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)

                # dQ += dP @ KV
                T.gemm(
                    dP_shared_cast,
                    KV_shared,
                    acc_dq,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                # dKV += dP^T @ Q + P^T @ dO
                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(
                    P_shared_cast,
                    dO_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                # Atomic store dKV with split to reduce register pressure
                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[
                                bi_i + s * (BS // split_store), d_i
                            ]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[
                                by,
                                Indices[
                                    by,
                                    s_i,
                                    i_i * BS + bi_i + s * (BS // split_store),
                                ],
                                d_i * 4,
                            ],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

            # Store dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(
                dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D]
            )

            # dAttnSink[h] = -sum_{b,s}( Delta[b,s,h] * p_sink[b,s,h] )
            # where p_sink = exp(attn_sink[h]) / Z = exp2(attn_sink[h]*log2e - LSE)
            # attn_sink is a pre-scaled logit, so only convert to log2 base (no sm_scale)
            for h_i in T.Parallel(block_H):
                T.atomic_add(
                    dAttnSink[bz * block_H + h_i],
                    -Delta[by, s_i, bz * block_H + h_i]
                    * T.exp2(
                        AttnSink[bz * block_H + h_i] * 1.44269504
                        - Lse[by, s_i, bz * block_H + h_i]
                    ),
                )

    return sparse_mqa_bwd_kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def bwd_det(
    B,
    S,
    S_kv,
    H,
    D,
    topk,
    sm_scale=None,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Deterministic backward: writes dKV/dAttnSink to per-block buffers (no atomics)."""
    assert topk % block_size == 0
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]

    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mqa_bwd_det_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV_buf: T.Tensor([B, S, topk, D], accum_dtype),
        dAttnSink_buf: T.Tensor([S, B, H], accum_dtype),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(
                dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared
            )

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] != -1

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_p.dtype)
                    )

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[
                        by, Indices[by, s_i, i_i * BS + bi_i], d_i
                    ]

                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_p,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2
                        - Lse[by, s_i, bz * block_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                T.gemm(
                    dO_shared,
                    KV_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i]
                        * (
                            acc_dp[h_i, bi_i]
                            - Delta[by, s_i, bz * block_H + h_i]
                        )
                        * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)

                T.gemm(
                    dP_shared_cast,
                    KV_shared,
                    acc_dq,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(
                    P_shared_cast,
                    dO_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                # Deterministic: write dKV to per-block buffer (no atomics)
                for bi_i, d_i in T.Parallel(BS, D):
                    dKV_buf[by, s_i, i_i * BS + bi_i, d_i] = acc_dkv[bi_i, d_i]

            # Store dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(
                dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D]
            )

            # Deterministic: write dAttnSink to per-block buffer (no atomics)
            for h_i in T.Parallel(block_H):
                dAttnSink_buf[s_i, by, bz * block_H + h_i] = -Delta[
                    by, s_i, bz * block_H + h_i
                ] * T.exp2(
                    AttnSink[bz * block_H + h_i] * 1.44269504
                    - Lse[by, s_i, bz * block_H + h_i]
                )

    return sparse_mqa_bwd_det_kernel


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def dkv_reduce(
    B,
    S,
    S_kv,
    topk,
    D,
    BK=4,
    threads=128,
    indices_dtype=T.int32,
    accum_dtype=T.float32,
):
    """Deterministic K-major reduction: gather dKV_buf entries by target KV position.

    Uses pre-sorted CSR structure for fixed-order accumulation.
    Grid: (ceildiv(S_kv, BK), B).
    """
    dkv_buf_shape = [B, S, topk, D]
    sort_perm_shape = [B, S * topk]
    seg_offsets_shape = [B, S_kv + 1]
    dkv_shape = [B, S_kv, D]

    @T.prim_func
    def dkv_reduce_kernel(
        dKV_buf: T.Tensor(dkv_buf_shape, accum_dtype),
        sort_perm: T.Tensor(sort_perm_shape, indices_dtype),
        seg_offsets: T.Tensor(seg_offsets_shape, indices_dtype),
        dKV: T.Tensor(dkv_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, BK), B, threads=threads) as (bk, by):
            acc = T.alloc_fragment([D], accum_dtype)

            for ki in range(BK):
                k = bk * BK + ki
                if k < S_kv:
                    T.clear(acc)
                    start = seg_offsets[by, k]
                    end = seg_offsets[by, k + 1]

                    for fi in T.serial(end - start):
                        flat_pos = sort_perm[by, start + fi]
                        s_idx = flat_pos // topk
                        t_idx = flat_pos % topk
                        for d_i in T.Parallel(D):
                            acc[d_i] += dKV_buf[by, s_idx, t_idx, d_i]

                    for d_i in T.Parallel(D):
                        dKV[by, k, d_i] = acc[d_i]

    return dkv_reduce_kernel


def _build_csr_index(topk_idxs, S_kv):
    """Build CSR inverse index for deterministic dKV reduction.

    Args:
        topk_idxs: [B, S, topk] int32 — KV indices per query token.
        S_kv: int — total KV positions.

    Returns:
        sort_perm:   [B, S*topk] int32 — permutation that sorts flat indices.
        seg_offsets: [B, S_kv+1] int32 — CSR row pointers per KV position.
    """
    B, S, topk = topk_idxs.shape
    flat_idx = topk_idxs.reshape([B, S * topk])  # [B, S*topk]

    # Shift -1 to S_kv so they sort to the end (won't be visited)
    flat_idx_shifted = paddle.where(
        flat_idx < 0,
        paddle.full_like(flat_idx, S_kv),
        flat_idx,
    )

    # Deterministic stable sort per batch
    sort_perm = paddle.argsort(flat_idx_shifted, axis=1, stable=True).cast(
        "int32"
    )
    sorted_idx = paddle.take_along_axis(
        flat_idx_shifted, sort_perm.cast("int64"), axis=1
    )

    # Build CSR offsets: seg_offsets[b, k] = first position in sorted_idx where value >= k
    boundaries = (
        paddle.arange(S_kv + 1, dtype="int64").unsqueeze(0).expand([B, -1])
    )
    seg_offsets = paddle.searchsorted(sorted_idx, boundaries).cast("int32")

    return sort_perm, seg_offsets


def sparse_mqa_bwd_interface(
    q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=None
):
    """Backward interface for DSv4 sparse MQA attention.

    Args:
        q:         [B, S, H, D] bf16
        kv:        [B, S_kv, D] bf16
        attn_sink: [H] fp32
        o:         [B, S, H, D] bf16 (forward output)
        do:        [B, S, H, D] bf16 (grad of output)
        topk_idxs: [B, S, topk] int32
        lse:       [B, S, H] fp32 (log-sum-exp from forward)
        sm_scale:  float or None

    Returns:
        dq:         [B, S, H, D] bf16
        dkv:        [B, S_kv, D] bf16
        d_attn_sink: [H] fp32
    """
    assert q.is_contiguous() and kv.is_contiguous()
    assert topk_idxs.is_contiguous() and lse.is_contiguous()
    deterministic = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
        "FLAGS_cudnn_deterministic"
    ]

    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    topk = topk_idxs.shape[-1]

    # Pad topk to next multiple of block_size (kernel requires divisibility)
    block_size = 32
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        pad = paddle.full([B, S, padded_topk - topk], -1, dtype=topk_idxs.dtype)
        topk_idxs = paddle.concat([topk_idxs, pad], axis=-1).contiguous()
        topk = padded_topk

    preprocess_kernel = preprocess(B, S, H, D)
    delta = preprocess_kernel(o, do)

    if not deterministic:
        # === Non-deterministic path (original) ===
        bwd_kernel = bwd(B, S, S_kv, H, D, topk, sm_scale)
        postprocess_kernel = postprocess(B, S_kv, D)

        dkv = paddle.zeros_like(kv, dtype="float32")
        d_attn_sink = paddle.zeros_like(attn_sink)
        dq = bwd_kernel(
            q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink
        )
        dkv = postprocess_kernel(dkv)

        return dq, dkv, d_attn_sink
    else:
        # === Deterministic path ===
        bwd_det_kernel = bwd_det(B, S, S_kv, H, D, topk, sm_scale)
        postprocess_kernel = postprocess(B, S_kv, D)

        dkv_buf = paddle.empty([B, S, topk, D], dtype="float32")
        d_attn_sink_buf = paddle.empty([S, B, H], dtype="float32")
        dq = bwd_det_kernel(
            q,
            kv,
            do,
            attn_sink,
            topk_idxs,
            lse,
            delta,
            dkv_buf,
            d_attn_sink_buf,
        )

        # dAttnSink: deterministic sum over (S, B) dimensions
        d_attn_sink = d_attn_sink_buf.sum(axis=[0, 1])

        # dKV: build CSR inverse index + deterministic reduction
        sort_perm, seg_offsets = _build_csr_index(topk_idxs, S_kv)
        reduce_kernel = dkv_reduce(B, S, S_kv, topk, D)
        dkv = reduce_kernel(dkv_buf, sort_perm, seg_offsets)
        dkv = postprocess_kernel(dkv)

        return dq, dkv, d_attn_sink
