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

"""Backward for the independent MHA block-score attention
(:mod:`block_score_mha`): a standard flash-attention backward (dQ, dK, dV) for
full attention with H INDEPENDENT heads (per-head K/V) and causal + document
masking.

Each head has its own K/V ``[B, S_kv, H, D]`` / ``[B, S_kv, H, D_v]``. One
program handles a (query tile, head, batch) triple; dK/dV from the query tiles
sharing a (head, batch) are scattered into that head's K/V grad via
``atomic_add``. The block-score / lse outputs of the forward feed a
non-differentiable TopK and carry no gradient here.

Flash-attn backward math (per head):
    P     = exp(scaled_logit - lse)          (masked -> 0)
    dP    = dO . V^T
    Delta = rowsum(O * dO)
    dS    = P * (dP - Delta) * sm_scale
    dQ   += dS . K
    dK   += dS^T . Q
    dV   += P^T . dO
"""

import paddle
import tilelang
from tilelang import language as T


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def block_score_mha_bwd(
    H,
    D,
    sm_scale,
    D_v=None,
    block_M=64,
    block_N=64,
    block_B=64,
    num_stages=1,
    threads=128,
):
    """MHA flash-attention backward kernel (per-head K/V, dQ/dK/dV)."""
    if D_v is None:
        D_v = D
    assert D % 16 == 0, (
        f"D must be a multiple of 16 (tensor-core k-tile), got {D}"
    )
    assert D_v % 16 == 0, (
        f"D_v must be a multiple of 16 (tensor-core k-tile), got {D_v}"
    )
    assert block_B % block_N == 0, "block_B must be a multiple of block_N"

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_bm = T.dynamic("num_bm")

    q_shape = [batch, seq_len, H, D]
    k_shape = [batch, seq_len_kv, H, D]
    v_shape = [batch, seq_len_kv, H, D_v]
    do_shape = [batch, seq_len, H, D_v]
    lse_shape = [batch, seq_len, H]
    vr_shape = [batch, seq_len, 2]
    br_shape = [batch, num_bm, 2]

    dtype = T.bfloat16
    accum_dtype = T.float32
    idx_dtype = T.int32
    BM = block_M
    BN = block_N
    ratio = block_B // block_N  # key sub-tiles per block_B selection block

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        dO: T.Tensor(do_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(lse_shape, accum_dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
        BlockRange: T.Tensor(br_shape, idx_dtype),
        dK: T.Tensor(k_shape, accum_dtype),
        dV: T.Tensor(v_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, BM), H, batch, threads=threads) as (
            bm,
            bh,
            bb,
        ):
            Q_shared = T.alloc_shared([BM, D], dtype)
            dO_shared = T.alloc_shared([BM, D_v], dtype)
            K_shared = T.alloc_shared([BN, D], dtype)
            V_shared = T.alloc_shared([BN, D_v], dtype)
            P_shared = T.alloc_shared([BM, BN], dtype)
            dS_shared = T.alloc_shared([BM, BN], dtype)

            acc_s = T.alloc_fragment([BM, BN], accum_dtype)
            acc_p = T.alloc_fragment([BM, BN], accum_dtype)
            acc_dp = T.alloc_fragment([BM, BN], accum_dtype)
            acc_dq = T.alloc_fragment([BM, D], accum_dtype)
            acc_dk = T.alloc_fragment([BN, D], accum_dtype)
            acc_dv = T.alloc_fragment([BN, D_v], accum_dtype)
            lse_f = T.alloc_fragment([BM], accum_dtype)
            delta_f = T.alloc_fragment([BM], accum_dtype)
            bos = T.alloc_fragment([BM], idx_dtype)
            eos = T.alloc_fragment([BM], idx_dtype)

            for i in T.Parallel(BM):
                row = bm * BM + i
                in_range = row < seq_len
                bos[i] = T.if_then_else(in_range, ValidRange[bb, row, 0], 0)
                eos[i] = T.if_then_else(in_range, ValidRange[bb, row, 1], 0)
                lse_f[i] = T.if_then_else(in_range, Lse[bb, row, bh], 0)
                delta_f[i] = T.if_then_else(in_range, Delta[bb, row, bh], 0)

            # Guarded per-row load of Q / dO for this head (ragged tail zeroed).
            for i, d in T.Parallel(BM, D):
                row = bm * BM + i
                in_range = row < seq_len
                safe_row = T.if_then_else(in_range, row, 0)
                Q_shared[i, d] = T.if_then_else(
                    in_range, Q[bb, safe_row, bh, d], T.cast(0, dtype)
                )
            for i, d in T.Parallel(BM, D_v):
                row = bm * BM + i
                in_range = row < seq_len
                safe_row = T.if_then_else(in_range, row, 0)
                dO_shared[i, d] = T.if_then_else(
                    in_range, dO[bb, safe_row, bh, d], T.cast(0, dtype)
                )
            T.clear(acc_dq)

            # document-tight early-exit: [jl, jh) is the block-B key window
            # reachable by this query tile's rows (host-precomputed). Every
            # skipped block is fully masked for all rows -> no grad. Iterate the
            # window in block_N sub-tiles (ratio per block).
            jl = BlockRange[bb, bm, 0]
            jh = BlockRange[bb, bm, 1]
            for nn in T.Pipelined((jh - jl) * ratio, num_stages=num_stages):
                col0 = (jl * ratio + nn) * BN
                T.copy(K[bb, col0 : col0 + BN, bh, :], K_shared)
                T.copy(V[bb, col0 : col0 + BN, bh, :], V_shared)

                # P = softmax prob = exp(raw*sm_scale - lse); masked -> 0
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, c in T.Parallel(BM, BN):
                    col = col0 + c
                    keep = (col >= bos[i]) and (col < eos[i])
                    acc_p[i, c] = T.if_then_else(
                        keep,
                        T.exp2(
                            (acc_s[i, c] * sm_scale - lse_f[i]) * 1.44269504
                        ),
                        0.0,
                    )
                T.copy(acc_p, P_shared)

                # dP = dO @ V^T
                T.clear(acc_dp)
                T.gemm(
                    dO_shared,
                    V_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                # dS = P * (dP - Delta) * sm_scale
                for i, c in T.Parallel(BM, BN):
                    acc_dp[i, c] = (
                        acc_p[i, c] * (acc_dp[i, c] - delta_f[i]) * sm_scale
                    )
                T.copy(acc_dp, dS_shared)

                # dQ += dS @ K
                T.gemm(
                    dS_shared,
                    K_shared,
                    acc_dq,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                # dV += P^T @ dO ; dK += dS^T @ Q  (scattered to this head)
                T.clear(acc_dv)
                T.gemm(
                    P_shared,
                    dO_shared,
                    acc_dv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.clear(acc_dk)
                T.gemm(
                    dS_shared,
                    Q_shared,
                    acc_dk,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for c, d in T.Parallel(BN, D):
                    T.atomic_add(dK[bb, col0 + c, bh, d], acc_dk[c, d])
                for c, d in T.Parallel(BN, D_v):
                    T.atomic_add(dV[bb, col0 + c, bh, d], acc_dv[c, d])

            for i, d in T.Parallel(BM, D):
                if bm * BM + i < seq_len:
                    dQ[bb, bm * BM + i, bh, d] = acc_dq[i, d]

    return main


def _fit_block_mn(D, block_B, D_v=None, cap_bytes=230000):
    """Pick (block_M, block_N) maximising the query tile then the key sub-tile.

    Shared budget holds Q ``[block_M, D]`` + dO ``[block_M, D_v]`` + K
    ``[block_N, D]`` + V ``[block_N, D_v]`` + P/dS ``[block_M, block_N]`` in
    bf16 (dQ writes straight from its accumulator). Prefer the largest
    ``block_M`` (<=64), then the largest ``block_N`` dividing ``block_B``.
    """
    if D_v is None:
        D_v = D
    cands_n = [n for n in (block_B, 32, 16) if block_B % n == 0]
    cands_n = sorted(set(cands_n), reverse=True)
    for bm in (64, 48, 32, 16):
        for bn in cands_n:
            shared = 2 * (bm * (D + D_v) + bn * (D + D_v) + 2 * bm * bn)
            if shared <= cap_bytes:
                return bm, bn
    return 16, min(16, block_B)


def block_score_mha_bwd_interface(
    q,
    k,
    v,
    o,
    do,
    lse,
    valid_range,
    sm_scale=None,
    block_B=64,
    block_M=None,
    block_N=None,
):
    """Backward interface for the independent MHA block-score attention.

    Args:
        q:           [B, S, H, D] bf16 forward query.
        k:           [B, S_kv, H, D] bf16 key (per head).
        v:           [B, S_kv, H, D_v] bf16 value (per head).
        o:           [B, S, H, D_v] bf16 forward output.
        do:          [B, S, H, D_v] bf16 grad of output.
        lse:         [B, S, H] fp32 natural-log LSE from forward.
        valid_range: [B, S, 2] int32.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size.
        block_M/N:   optional tiling overrides (auto-fit otherwise); the result
                     is mathematically identical for any valid choice.

    Returns:
        dq [B,S,H,D] bf16, dk [B,S_kv,H,D] bf16, dv [B,S_kv,H,D_v] bf16.
    """
    assert q.is_contiguous() and do.is_contiguous() and lse.is_contiguous()
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    d_v = v.shape[-1]
    if sm_scale is None:
        sm_scale = d**-0.5
    if valid_range.dtype != paddle.int32:
        valid_range = valid_range.cast("int32")

    # kernel reads whole block_B blocks; pad K/V (and dK/dV) so the last block
    # is addressable.
    pad = (block_B - s_kv % block_B) % block_B
    s_kv_pad = s_kv + pad
    if pad > 0:
        k = paddle.nn.functional.pad(k, [0, 0, 0, 0, 0, pad])
        v = paddle.nn.functional.pad(v, [0, 0, 0, 0, 0, pad])
    k = k.contiguous()
    v = v.contiguous()
    valid_range = valid_range.contiguous()

    delta = (o.astype("float32") * do.astype("float32")).sum(-1).contiguous()
    dk = paddle.zeros([b, s_kv_pad, h, d], dtype="float32")
    dv = paddle.zeros([b, s_kv_pad, h, d_v], dtype="float32")

    fit_m, fit_n = _fit_block_mn(d, block_B, d_v)
    if block_M is None:
        block_M = fit_m
    if block_N is None:
        block_N = fit_n
    num_kv_blocks = s_kv_pad // block_B

    # Per query-tile key-block window [jl, jh): jl skips leading blocks before
    # the tile's document start; jh caps at the tile's causal reach. Padded rows
    # get bos=+big / eos=0 so they never widen a tile's window.
    num_bm = (s + block_M - 1) // block_M
    pad_rows = num_bm * block_M - s
    bos = valid_range[:, :, 0]
    eos = valid_range[:, :, 1]
    if pad_rows > 0:
        bos = paddle.nn.functional.pad(bos, [0, pad_rows], value=s_kv_pad)
        eos = paddle.nn.functional.pad(eos, [0, pad_rows], value=0)
    bos = bos.reshape([b, num_bm, block_M])
    eos = eos.reshape([b, num_bm, block_M])
    jl = (bos.min(-1) // block_B).clip(0, num_kv_blocks)
    jh = ((eos.max(-1) + block_B - 1) // block_B).clip(0, num_kv_blocks)
    jh = paddle.maximum(jh, jl)
    block_range = paddle.stack([jl, jh], axis=-1).astype("int32").contiguous()

    bwd = block_score_mha_bwd(
        h,
        d,
        float(sm_scale),
        D_v=d_v,
        block_M=block_M,
        block_N=block_N,
        block_B=block_B,
    )
    dq = bwd(q, k, v, do, lse, delta, valid_range, block_range, dk, dv)

    dk_bf = dk.astype(q.dtype)
    dv_bf = dv.astype(q.dtype)
    if pad > 0:
        dk_bf = dk_bf[:, :s_kv, :, :].contiguous()
        dv_bf = dv_bf[:, :s_kv, :, :].contiguous()
    return dq, dk_bf, dv_bf
