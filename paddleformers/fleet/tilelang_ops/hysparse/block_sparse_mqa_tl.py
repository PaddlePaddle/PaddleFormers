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

"""Independent TileLang block-sparse MQA (absorbed-MLA) gather attention.

A standalone correctness-verification oracle / drop-in replacement for the DSA
operator ``block_sparse_mqa_attention_dsa`` (see
:mod:`paddleformers.fleet.cudnn_ops.block_sparse_mqa_dsa`). Pure TileLang forward +
backward, so it runs on any TileLang-capable GPU (no FlashMLA / cuDNN DSA
dependency) and natively handles ``kv_lora_rank`` other than 512.

Layout (absorbed-MLA, single shared K/V latent):

* ``Q``            ``[B, S, H, Dk]`` bf16 (H query heads; Dk = kv_lora_rank +
  rope, e.g. 576).
* ``K``            ``[B, S_kv, Dk]`` bf16 -- the one shared latent. The **key**
  for the ``q·k`` logit is the full ``Dk``; the **value** for ``p·v`` is its
  leading ``Dv = kv_lora_rank`` slice (e.g. 512). So ``Dk != Dv``.
* ``Indices``      ``[B, S, nsel]`` int32 document-relative block ids
  (``-1`` = padding), shared across heads.
* ``ValidRange``   ``[B, S, 2]`` int32 per-query ``[bos, eos)``.
* ``AttnSink``     ``[H]`` fp32 per-head sink logit (virtual softmax column);
  a very-negative sentinel recovers the plain sinkless softmax.
* ``Output``       ``[B, S, H, Dv]`` bf16, ``Lse`` ``[B, S, H]`` fp32
  (natural-log sum-exp, **including** the folded sink term).

Design notes:

* **Dk/Dv split.** ``K_shared`` is loaded full ``Dk`` (for ``q·k``);
  ``V_shared`` is loaded from the *leading* ``Dv`` columns of the same ``K``
  tensor (the unified latent), so no separate V tensor / copy is needed.
* **Shared-latent gradient combination.** Because K (full Dk) and V (leading
  Dv) are two views of the same tensor, its gradient combines both paths:
  ``dK_score = dS^T·Q`` (width Dk) and ``dV = P^T·dO`` (width Dv). The backward
  scatters ``dK_score`` into all Dk columns and ``dV`` into the leading Dv
  columns of a single fp32 ``dKV [B, S_kv, Dk]`` accumulator via ``atomic_add``
  (many query tokens select the same block), yielding the combined latent grad
  directly. ``dQ = dS·K`` has width Dk.
* **attn_sink folding.** Copied from :mod:`windowed_mqa_attn`: the per-head
  pre-scaled sink logit is folded as a virtual key column into the online
  softmax denominator (base-2, log2(e) only, no ``sm_scale``). The sink
  gradient ``d_sink[h] = -sum_{b,s}(p_sink * Delta)`` is computed on the host in
  the PyLayer backward (like DSA), from the saved ``out``/``dO``/``lse``.
"""

import paddle
import tilelang
from tilelang import language as T

_NEG_SINK = -1e30  # sink so large-negative that exp(sink - m) underflows to 0.


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _block_sparse_mqa_fwd(
    H,
    Dk,
    Dv,
    nsel,
    sm_scale,
    block_B=64,
    num_stages=2,
    threads=128,
):
    """Forward kernel: one program per query token, H heads on the GEMM M dim."""
    assert Dk % 16 == 0, f"Dk must be a multiple of 16, got {Dk}"
    assert Dv % 16 == 0, f"Dv must be a multiple of 16, got {Dv}"
    assert Dv <= Dk, f"Dv ({Dv}) must be <= Dk ({Dk})"
    assert H <= 128, "this kernel supports up to 128 query heads"
    scale_log2 = sm_scale * 1.44269504  # log2(e); online softmax in base 2

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, H, Dk]
    kv_shape = [batch, seq_len_kv, Dk]
    o_shape = [batch, seq_len, H, Dv]
    idx_shape = [batch, seq_len, nsel]
    vr_shape = [batch, seq_len, 2]
    lse_shape = [batch, seq_len, H]
    sink_shape = [H]

    dtype = T.bfloat16
    accum_dtype = T.float32
    idx_dtype = T.int32
    PH = max(tilelang.math.next_power_of_2(H), 16)  # padded heads on M
    BB = block_B

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(idx_shape, idx_dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
        AttnSink: T.Tensor(sink_shape, accum_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, batch, threads=threads) as (bs, bb):
            Q_shared = T.alloc_shared([PH, Dk], dtype)
            K_shared = T.alloc_shared([BB, Dk], dtype)
            V_shared = T.alloc_shared([BB, Dv], dtype)
            P_shared = T.alloc_shared([PH, BB], dtype)

            acc_o = T.alloc_fragment([PH, Dv], accum_dtype)
            acc_s = T.alloc_fragment([PH, BB], accum_dtype)
            row_max = T.alloc_fragment([PH], accum_dtype)
            m_i = T.alloc_fragment([PH], accum_dtype)
            m_prev = T.alloc_fragment([PH], accum_dtype)
            l_i = T.alloc_fragment([PH], accum_dtype)
            l_new = T.alloc_fragment([PH], accum_dtype)
            alpha = T.alloc_fragment([PH], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(m_i, -(2**30))
            T.fill(l_i, 0)

            bos = ValidRange[bb, bs, 0]
            eos = ValidRange[bb, bs, 1]

            # load this token's H query heads onto M (pad rows >= H with 0)
            for h, d in T.Parallel(PH, Dk):
                Q_shared[h, d] = T.if_then_else(
                    h < H, Q[bb, bs, h, d], T.cast(0, dtype)
                )

            for i in T.Pipelined(nsel, num_stages=num_stages):
                blk = Indices[bb, bs, i]
                valid_blk = blk >= 0
                safe_blk = T.if_then_else(valid_blk, blk, 0)

                # document-relative gather: relative block ``blk`` spans absolute
                # columns [bos + blk*BB, bos + (blk+1)*BB). Guard the read
                # against the K/V length (cols >= eos are masked below, so a
                # clamped dummy read is harmless). K = full Dk; V = leading Dv of
                # the same shared latent.
                for c, d in T.Parallel(BB, Dk):
                    col = bos + safe_blk * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    K_shared[c, d] = T.if_then_else(
                        in_bounds, K[bb, safe_col, d], T.cast(0, dtype)
                    )
                for c, d in T.Parallel(BB, Dv):
                    col = bos + safe_blk * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    V_shared[c, d] = T.if_then_else(
                        in_bounds, K[bb, safe_col, d], T.cast(0, dtype)
                    )

                # raw q·k^T
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # mask: keep col iff block valid and in [bos, eos). The relative
                # col bos + blk*BB + c is always >= bos, so only the upper bound
                # (col < eos) needs checking.
                for h, c in T.Parallel(PH, BB):
                    col = bos + safe_blk * BB + c
                    keep = valid_blk and (col < eos)
                    acc_s[h, c] = T.if_then_else(
                        keep, acc_s[h, c], -T.infinity(accum_dtype)
                    )

                # online softmax (base 2)
                T.reduce_max(acc_s, row_max, dim=1, clear=True)
                T.copy(m_i, m_prev)
                for h in T.Parallel(PH):
                    m_i[h] = T.max(m_i[h], row_max[h] * sm_scale)
                for h in T.Parallel(PH):
                    alpha[h] = T.exp2((m_prev[h] - m_i[h]) * 1.44269504)
                for h, c in T.Parallel(PH, BB):
                    acc_s[h, c] = T.exp2(
                        acc_s[h, c] * scale_log2 - m_i[h] * 1.44269504
                    )
                T.reduce_sum(acc_s, l_new, dim=1)
                for h in T.Parallel(PH):
                    l_i[h] = l_i[h] * alpha[h] + l_new[h]
                for h, d in T.Parallel(PH, Dv):
                    acc_o[h, d] = acc_o[h, d] * alpha[h]
                T.copy(acc_s, P_shared)
                T.gemm(
                    P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                )

            # Fold the (optional) learnable attention sink as a virtual key
            # column competing in the same softmax denominator. AttnSink is a
            # pre-scaled logit (same units as m_i), converted to base-2 with
            # log2(e) only (no sm_scale). A very-negative sink -> exp(sink - m)
            # underflows to 0, recovering plain softmax bit-for-bit.
            for h in T.Parallel(PH):
                safe_h = T.if_then_else(h < H, h, 0)
                sink_h = T.if_then_else(
                    h < H, AttnSink[safe_h], -T.infinity(accum_dtype)
                )
                m_prev[h] = m_i[h]
                m_i[h] = T.max(m_i[h], sink_h)
                alpha[h] = T.exp2((m_prev[h] - m_i[h]) * 1.44269504)
                l_i[h] = l_i[h] * alpha[h] + T.exp2(
                    (sink_h - m_i[h]) * 1.44269504
                )
            for h, d in T.Parallel(PH, Dv):
                acc_o[h, d] = acc_o[h, d] * alpha[h]

            # normalize; empty rows (no valid key) -> 0 out / -inf lse
            for h, d in T.Parallel(PH, Dv):
                acc_o[h, d] = T.if_then_else(
                    l_i[h] > 0, acc_o[h, d] / l_i[h], 0.0
                )
            for h, d in T.Parallel(PH, Dv):
                if h < H:
                    Output[bb, bs, h, d] = acc_o[h, d]
            for h in T.Parallel(PH):
                if h < H:
                    Lse[bb, bs, h] = T.if_then_else(
                        l_i[h] > 0,
                        m_i[h] + T.log(l_i[h]),
                        -T.infinity(accum_dtype),
                    )

    return main


def _fwd_interface(
    query, shared_key_sq, indices, valid_range, sink, sm_scale, block_B, d_v
):
    """Run the forward kernel. Returns (out [B,S,H,Dv], lse [B,S,H]).

    ``d_v`` (= kv_lora_rank) is the value/output width; the kernel reads V from
    the leading ``d_v`` columns of the full-Dk ``shared_key_sq`` latent.
    """
    b, s, h, dk = query.shape
    nsel = indices.shape[-1]
    kernel = _block_sparse_mqa_fwd(
        h, dk, d_v, nsel, float(sm_scale), block_B=block_B
    )
    out, lse = kernel(query, shared_key_sq, indices, valid_range, sink)
    return out, lse


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _block_sparse_mqa_bwd(
    H,
    Dk,
    Dv,
    nsel,
    sm_scale,
    block_B=64,
    block_H=None,
    num_stages=1,
    threads=128,
):
    """Backward kernel: dQ local per token; dK/dV scattered into one shared
    fp32 latent buffer ``dKV [B, S_kv, Dk]`` via atomic_add (dK->all Dk cols,
    dV->leading Dv cols, so the combined shared-latent grad falls out directly).
    """
    assert Dk % 16 == 0, f"Dk must be a multiple of 16, got {Dk}"
    assert Dv % 16 == 0, f"Dv must be a multiple of 16, got {Dv}"
    assert Dv <= Dk, f"Dv ({Dv}) must be <= Dk ({Dk})"
    assert H <= 128, "this kernel supports up to 128 query heads"

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, H, Dk]
    kv_shape = [batch, seq_len_kv, Dk]
    do_shape = [batch, seq_len, H, Dv]
    idx_shape = [batch, seq_len, nsel]
    vr_shape = [batch, seq_len, 2]
    lse_shape = [batch, seq_len, H]

    dtype = T.bfloat16
    accum_dtype = T.float32
    idx_dtype = T.int32
    # Heads tiled on the GEMM M dim in groups of BH so the [·, Dk] shared
    # buffers fit on-chip at large head dims (MLA Dk=576); BH == H keeps the
    # single-program-per-token fast path.
    BH = block_H if block_H is not None else H
    PH = max(tilelang.math.next_power_of_2(BH), 16)
    num_hg = (H + BH - 1) // BH
    BB = block_B

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(do_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(lse_shape, accum_dtype),
        Indices: T.Tensor(idx_shape, idx_dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
        dKV: T.Tensor(kv_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(num_hg, seq_len, batch, threads=threads) as (hg, bs, bb):
            h0 = hg * BH
            Q_shared = T.alloc_shared([PH, Dk], dtype)
            dO_shared = T.alloc_shared([PH, Dv], dtype)
            K_shared = T.alloc_shared([BB, Dk], dtype)
            V_shared = T.alloc_shared([BB, Dv], dtype)
            P_shared = T.alloc_shared([PH, BB], dtype)
            dS_shared = T.alloc_shared([PH, BB], dtype)
            dQ_shared = T.alloc_shared([PH, Dk], dtype)

            acc_s = T.alloc_fragment([PH, BB], accum_dtype)
            acc_p = T.alloc_fragment([PH, BB], accum_dtype)
            acc_dp = T.alloc_fragment([PH, BB], accum_dtype)
            acc_dq = T.alloc_fragment([PH, Dk], accum_dtype)
            acc_dk = T.alloc_fragment([BB, Dk], accum_dtype)
            acc_dv = T.alloc_fragment([BB, Dv], accum_dtype)
            lse_f = T.alloc_fragment([PH], accum_dtype)
            delta_f = T.alloc_fragment([PH], accum_dtype)

            bos = ValidRange[bb, bs, 0]
            eos = ValidRange[bb, bs, 1]

            for h, d in T.Parallel(PH, Dk):
                gh = h0 + h
                use = (h < BH) and (gh < H)
                sh = T.if_then_else(use, gh, 0)
                Q_shared[h, d] = T.if_then_else(
                    use, Q[bb, bs, sh, d], T.cast(0, dtype)
                )
            for h, d in T.Parallel(PH, Dv):
                gh = h0 + h
                use = (h < BH) and (gh < H)
                sh = T.if_then_else(use, gh, 0)
                dO_shared[h, d] = T.if_then_else(
                    use, dO[bb, bs, sh, d], T.cast(0, dtype)
                )
            for h in T.Parallel(PH):
                gh = h0 + h
                use = (h < BH) and (gh < H)
                sh = T.if_then_else(use, gh, 0)
                lse_f[h] = T.if_then_else(use, Lse[bb, bs, sh], 0.0)
                delta_f[h] = T.if_then_else(use, Delta[bb, bs, sh], 0.0)

            T.clear(acc_dq)

            for i in T.Pipelined(nsel, num_stages=num_stages):
                blk = Indices[bb, bs, i]
                valid_blk = blk >= 0
                safe_blk = T.if_then_else(valid_blk, blk, 0)

                for c, d in T.Parallel(BB, Dk):
                    col = bos + safe_blk * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    K_shared[c, d] = T.if_then_else(
                        in_bounds, K[bb, safe_col, d], T.cast(0, dtype)
                    )
                for c, d in T.Parallel(BB, Dv):
                    col = bos + safe_blk * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    V_shared[c, d] = T.if_then_else(
                        in_bounds, K[bb, safe_col, d], T.cast(0, dtype)
                    )

                # P = softmax prob = exp(raw*sm_scale - lse); masked -> 0. The
                # forward Lse already folds the sink, so sum_c P < 1 (the sink
                # holds the remaining mass), matching the sink-aware softmax.
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for h, c in T.Parallel(PH, BB):
                    col = bos + safe_blk * BB + c
                    keep = valid_blk and (col < eos)
                    acc_p[h, c] = T.if_then_else(
                        keep,
                        T.exp2(
                            (acc_s[h, c] * sm_scale - lse_f[h]) * 1.44269504
                        ),
                        0.0,
                    )
                T.copy(acc_p, P_shared)

                # dP = dO @ V^T  (V is leading Dv of the latent)
                T.clear(acc_dp)
                T.gemm(
                    dO_shared,
                    V_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                # dS = P * (dP - Delta) * sm_scale
                for h, c in T.Parallel(PH, BB):
                    acc_dp[h, c] = (
                        acc_p[h, c] * (acc_dp[h, c] - delta_f[h]) * sm_scale
                    )
                T.copy(acc_dp, dS_shared)

                # dQ += dS @ K   (width Dk)
                T.gemm(
                    dS_shared,
                    K_shared,
                    acc_dq,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                # dV = P^T @ dO (width Dv) ; dK_score = dS^T @ Q (width Dk)
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
                # Scatter into the single shared-latent grad: dK_score fills all
                # Dk columns; dV adds into the leading Dv columns -> the combined
                # d(shared_key_sq) falls out. atomic_add: many query tokens hit
                # the same key block.
                for c, d in T.Parallel(BB, Dk):
                    col = bos + safe_blk * BB + c
                    if valid_blk and (col < seq_len_kv):
                        T.atomic_add(dKV[bb, col, d], acc_dk[c, d])
                for c, d in T.Parallel(BB, Dv):
                    col = bos + safe_blk * BB + c
                    if valid_blk and (col < seq_len_kv):
                        T.atomic_add(dKV[bb, col, d], acc_dv[c, d])

            T.copy(acc_dq, dQ_shared)
            for h, d in T.Parallel(PH, Dk):
                gh = h0 + h
                if (h < BH) and (gh < H):
                    dQ[bb, bs, gh, d] = dQ_shared[h, d]

    return main


@tilelang.jit(out_idx=[-1])
def _cast_bf16_kv(D, block_N=64, threads=128):
    """Cast a shared-latent grad accumulator [B, S_kv, D] fp32 -> bf16."""
    batch = T.dynamic("batch")
    seq_len_kv = T.dynamic("seq_len_kv")
    shape = [batch, seq_len_kv, D]

    @T.prim_func
    def main(
        X: T.Tensor(shape, T.float32),
        Out: T.Tensor(shape, T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(seq_len_kv, block_N), batch, threads=threads
        ) as (bn, bb):
            for i, d in T.Parallel(block_N, D):
                if bn * block_N + i < seq_len_kv:
                    Out[bb, bn * block_N + i, d] = X[bb, bn * block_N + i, d]

    return main


def _fit_block_h(H, Dk, Dv, block_B, cap_bytes=200000):
    """Largest head-group whose per-token backward shared buffers fit.

    bf16 buffers: Q/dQ ``[PH, Dk]`` (x2), dO ``[PH, Dv]``, K ``[BB, Dk]``,
    V ``[BB, Dv]``, P/dS ``[PH, BB]`` (x2). For large Dk (MLA 576) all H heads
    on M can overflow shared memory, so tile them.
    """
    for bh in (H, 64, 32, 16):
        if bh > H:
            continue
        ph = max(tilelang.math.next_power_of_2(bh), 16)
        shared = 2 * (
            2 * ph * Dk
            + ph * Dv
            + block_B * Dk
            + block_B * Dv
            + 2 * ph * block_B
        )
        if shared <= cap_bytes:
            return bh
    return min(H, 16)


def _bwd_interface(
    query,
    shared_key_sq,
    out,
    do,
    lse,
    indices,
    valid_range,
    sm_scale,
    block_B,
    d_v,
):
    """Run the backward kernel. Returns (dq [B,S,H,Dk], dkv [B,S_kv,Dk] bf16).

    ``d_v`` (= kv_lora_rank) is the value width (leading slice of the latent);
    the returned ``dkv`` is the *combined* shared-latent grad (dK_score over all
    Dk columns + dV over the leading Dv columns).
    """
    b, s, h, dk = query.shape
    s_kv = shared_key_sq.shape[1]
    nsel = indices.shape[-1]

    delta = (out.astype("float32") * do.astype("float32")).sum(-1).contiguous()
    dkv = paddle.zeros([b, s_kv, dk], dtype="float32")

    bwd = _block_sparse_mqa_bwd(
        h,
        dk,
        d_v,
        nsel,
        float(sm_scale),
        block_B=block_B,
        block_H=_fit_block_h(h, dk, d_v, block_B),
    )
    dq = bwd(query, shared_key_sq, do, lse, delta, indices, valid_range, dkv)
    dkv_bf = _cast_bf16_kv(dk)(dkv)
    return dq, dkv_bf


class _BlockSparseMQATL(paddle.autograd.PyLayer):
    """TileLang block-sparse MQA gather over the absorbed-MLA shared latent.

    forward inputs (tensors + non-tensors):
        query:         [B, S, H, Dk] bf16.
        shared_key_sq: [B, S_kv, Dk] bf16 shared latent (value = leading Dv).
        indices:       [B, S, nsel] int32 doc-relative block ids (-1 padding).
        valid_range:   [B, S, 2] int32 [bos, eos).
        sm_scale, block_B (non-tensor)
        attn_sink:     [H] fp32 tensor (learnable sink) or None (sinkless).
    output: out [B, S, H, Dv] bf16 (Dv = shared_key_sq.shape[-1] value slice).
    """

    @staticmethod
    def forward(
        ctx,
        query,
        shared_key_sq,
        indices,
        valid_range,
        sm_scale,
        block_B,
        d_v,
        attn_sink=None,
    ):
        b, s, h, dk = query.shape
        ctx.sm_scale = float(sm_scale)
        ctx.block_B = int(block_B)
        ctx.d_v = int(d_v)
        ctx.query_dtype = query.dtype
        ctx.kv_dtype = shared_key_sq.dtype

        ctx.learnable_sink = attn_sink is not None
        if attn_sink is None:
            sink = paddle.full([h], _NEG_SINK, dtype="float32")
        else:
            assert list(attn_sink.shape) == [h], (
                f"attn_sink must be [H={h}]; got {attn_sink.shape}"
            )
            sink = attn_sink.cast("float32")
        sink = sink.contiguous()

        out, lse = _fwd_interface(
            query,
            shared_key_sq,
            indices,
            valid_range,
            sink,
            ctx.sm_scale,
            ctx.block_B,
            ctx.d_v,
        )
        ctx.save_for_backward(
            query, shared_key_sq, out, lse, indices, valid_range, sink
        )
        ctx.needs_grad = (
            not query.stop_gradient,
            not shared_key_sq.stop_gradient,
            ctx.learnable_sink and not attn_sink.stop_gradient,
        )
        return out  # [B, S, H, Dv]

    @staticmethod
    def backward(ctx, grad_output):
        query, shared_key_sq, out, lse, indices, valid_range, sink = (
            ctx.saved_tensor()
        )
        do = grad_output.contiguous()

        gq, gk, gsink = ctx.needs_grad
        dq, dkv = _bwd_interface(
            query,
            shared_key_sq,
            out,
            do,
            lse,
            indices,
            valid_range,
            ctx.sm_scale,
            ctx.block_B,
            ctx.d_v,
        )
        dq = dq.cast(ctx.query_dtype) if gq else None
        dkv = dkv.cast(ctx.kv_dtype) if gk else None

        d_attn_sink = None
        if gsink:
            # Analytic sink grad from the saved forward tensors (like DSA). The
            # forward Lse already folds the sink, so p_sink = exp(sink - lse)
            # directly. Delta[b,s,h] = sum_dv(out * dO).
            out_f = out.astype("float32")
            do_f = do.astype("float32")
            delta = (out_f * do_f).sum(axis=-1)  # [B, S, H]
            h = query.shape[2]
            sink_h = sink.astype("float32").reshape([1, 1, h])
            lse_f = lse.astype("float32")
            # Empty rows have lse = -inf -> p_sink = 0 (no contribution).
            p_sink = paddle.where(
                paddle.isfinite(lse_f),
                paddle.exp(sink_h - lse_f),
                paddle.zeros_like(lse_f),
            )
            d_attn_sink = (-(delta * p_sink).sum(axis=[0, 1])).contiguous()
            d_attn_sink = d_attn_sink.cast("float32")

        # One grad per tensor input, in order: query, shared_key_sq, indices,
        # valid_range, [attn_sink if it was a tensor]. Non-tensor inputs
        # (sm_scale, block_B) occupy no slot.
        grads = [dq, dkv, None, None]
        if ctx.learnable_sink:
            grads.append(d_attn_sink)
        return tuple(grads)


def block_sparse_mqa_attention_tl(
    query,
    shared_key_sq,
    shared_block_indices,
    valid_range,
    sm_scale=None,
    block_B=64,
    kv_lora_rank=512,
    attn_sink=None,
):
    """HySparse block-sparse MQA gather attention (TileLang oracle).

    Drop-in replacement for
    :func:`paddleformers.fleet.cudnn_ops.block_sparse_mqa_attention_dsa` with an
    identical signature. Unlike DSA it needs no head padding to 64 and handles
    any ``kv_lora_rank`` natively (no pad to 512).

    Args:
        query:                [B, S, H, Dk] (Dk = kv_lora_rank + rope, e.g. 576).
        shared_key_sq:        [B, S, Dk] shared K/V latent; value = leading
                              ``kv_lora_rank`` slice.
        shared_block_indices: [B, S, topk] int document-relative block ids
                              (-1 padding), from ``select_topk_blocks``.
        valid_range:          [B, S, 2] int per-query ``[bos, eos)``.
        sm_scale:             softmax scale (defaults to ``Dk ** -0.5``).
        block_B:              block size in tokens (64).
        kv_lora_rank:         value dim ``Dv`` (leading slice of the latent).
        attn_sink:            [H] fp32 per-head learnable sink logit, or ``None``
                              for the default sinkless softmax.

    Returns:
        ``(out, None)`` where ``out`` is ``[B, S, H * kv_lora_rank]`` and carries
        gradient to ``query`` and ``shared_key_sq`` (and ``attn_sink`` when a
        learnable sink is supplied). The ``None`` matches the DSA call site
        ``sparse_core_attn_out, _ = block_sparse_mqa_attention_...``.
    """
    if sm_scale is None:
        sm_scale = query.shape[-1] ** -0.5

    b, s, num_heads, dk = query.shape
    assert kv_lora_rank <= dk, (
        f"kv_lora_rank ({kv_lora_rank}) must be <= Dk ({dk})"
    )
    assert shared_key_sq.shape[-1] == dk, (
        f"shared_key_sq last dim ({shared_key_sq.shape[-1]}) must equal query "
        f"Dk ({dk})"
    )

    # The full-Dk latent is the *key* (q·k score); its leading ``kv_lora_rank``
    # columns are the *value*. The kernel reads V from K[..., :Dv] internally,
    # so hand it the full latent and pass Dv = kv_lora_rank.
    if valid_range.dtype != paddle.int32:
        valid_range = valid_range.cast("int32")
    if shared_block_indices.dtype != paddle.int32:
        shared_block_indices = shared_block_indices.cast("int32")
    valid_range = valid_range.contiguous()
    shared_block_indices = shared_block_indices.contiguous()
    query = query.contiguous()
    shared_key_sq = shared_key_sq.contiguous()

    out = _BlockSparseMQATL.apply(
        query,
        shared_key_sq,
        shared_block_indices,
        valid_range,
        float(sm_scale),
        int(block_B),
        int(kv_lora_rank),
        attn_sink,
    )
    out = out.reshape([b, s, num_heads * kv_lora_rank])
    return out, None
