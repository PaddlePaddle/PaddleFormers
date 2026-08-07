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

"""Causal windowed MQA flash attention (single shared K/V head).

The fused attention kernel behind the HySparse **SWA layers'** sliding-window
main path. Q is ``[B, S, H, D]`` (H query heads); K, V are one shared head
``[B, S_kv, D]`` / ``[B, S_kv, D_v]`` (MQA/MLA). Causal + document + sliding
window masking is expressed entirely through ``valid_range`` ``[B, S, 2]`` --
each query's half-open valid key column range ``[bos, eos)`` -- and the
kernel's ``eos - bos`` early-exit bounds the per-token work to that range.

Why a bespoke kernel: absorbed-MLA MQA has Dk=576 / Dv=512 (the key carries an
extra RoPE slice, so ``D`` exceeds ``D_v``). FlashAttention (FA2/3/4) only
covers ``head_dim <= 256`` and would fall back to eager O(S^2) attention at
these dims; this fused TileLang path is far cheaper.

Returns ``(out [B,S,H,D_v], lse [B,S,H])``. ``lse`` is the natural-log sum-exp
of the scaled logits, consumed by the backward (:mod:`windowed_mqa_attn_bwd`);
it carries no gradient.
"""

import paddle
import tilelang
from tilelang import language as T


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def windowed_mqa_fwd(
    H,
    D,
    sm_scale,
    D_v=None,
    block_B=64,
    num_stages=2,
    threads=128,
):
    """Windowed MQA flash-attention kernel (single shared K/V head).

    One program per query **token**, with the ``H`` query heads placed on the
    GEMM ``M`` dimension. Keys are streamed in ``block_B``-sized tiles over the
    token's valid range ``[bos, eos)``; the ``ceildiv(eos - bos, block_B)``
    early-exit skips fully-masked tiles (pure causal halves the work, a sliding
    window bounds work to the window, packed documents scan only their own doc).

    ``D`` is the query/key head dim (for the ``q·k`` logit); ``D_v`` is the
    value/output head dim. They are equal for plain attention; for absorbed MLA
    the key carries an extra RoPE slice so ``D`` (e.g. 576) exceeds ``D_v``
    (e.g. 512). ``D_v`` defaults to ``D``.
    """
    if D_v is None:
        D_v = D
    assert D % 16 == 0, (
        f"D must be a multiple of 16 (tensor-core k-tile), got {D}"
    )
    assert D_v % 16 == 0, (
        f"D_v must be a multiple of 16 (tensor-core k-tile), got {D_v}"
    )
    assert H <= 128, "this kernel supports up to 128 query heads"
    scale_log2 = sm_scale * 1.44269504  # log2(e), online softmax in base 2

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, H, D]
    kv_shape = [batch, seq_len_kv, D]
    v_shape = [batch, seq_len_kv, D_v]
    o_shape = [batch, seq_len, H, D_v]
    lse_shape = [batch, seq_len, H]
    vr_shape = [batch, seq_len, 2]
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
        V: T.Tensor(v_shape, dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
        AttnSink: T.Tensor(sink_shape, accum_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, batch, threads=threads) as (bs, bb):
            Q_shared = T.alloc_shared([PH, D], dtype)
            K_shared = T.alloc_shared([BB, D], dtype)
            V_shared = T.alloc_shared([BB, D_v], dtype)
            P_shared = T.alloc_shared([PH, BB], dtype)

            acc_o = T.alloc_fragment([PH, D_v], accum_dtype)
            acc_s = T.alloc_fragment([PH, BB], accum_dtype)
            tile_max = T.alloc_fragment([PH], accum_dtype)
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
            for h, d in T.Parallel(PH, D):
                Q_shared[h, d] = T.if_then_else(
                    h < H, Q[bb, bs, h, d], T.cast(0, dtype)
                )

            # causal / window / document early-exit: only tiles overlapping the
            # token's valid range [bos, eos) can hold an unmasked key; later
            # tiles are fully masked (all cols >= eos) and add nothing.
            num_valid_tiles = T.ceildiv(eos - bos, BB)
            for j in T.Pipelined(num_valid_tiles, num_stages=num_stages):
                # gather tile j: cols [bos + j*BB, bos + (j+1)*BB). Guard the
                # read against the padded K/V length (cols >= eos are masked
                # below, so a clamped dummy read is harmless).
                for c, d in T.Parallel(BB, D):
                    col = bos + j * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    K_shared[c, d] = T.if_then_else(
                        in_bounds, K[bb, safe_col, d], T.cast(0, dtype)
                    )
                for c, d in T.Parallel(BB, D_v):
                    col = bos + j * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    V_shared[c, d] = T.if_then_else(
                        in_bounds, V[bb, safe_col, d], T.cast(0, dtype)
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

                # causal + document mask (col >= bos automatic for tile j)
                for h, c in T.Parallel(PH, BB):
                    col = bos + j * BB + c
                    acc_s[h, c] = T.if_then_else(
                        col < eos, acc_s[h, c], -T.infinity(accum_dtype)
                    )

                # online softmax (base 2) over scaled logits
                T.reduce_max(acc_s, tile_max, dim=1, clear=True)
                T.copy(m_i, m_prev)
                for h in T.Parallel(PH):
                    m_i[h] = T.max(m_i[h], tile_max[h] * sm_scale)
                for h in T.Parallel(PH):
                    alpha[h] = T.exp2((m_prev[h] - m_i[h]) * 1.44269504)
                for h, c in T.Parallel(PH, BB):
                    acc_s[h, c] = T.exp2(
                        acc_s[h, c] * scale_log2 - m_i[h] * 1.44269504
                    )
                T.reduce_sum(acc_s, l_new, dim=1)
                for h in T.Parallel(PH):
                    l_i[h] = l_i[h] * alpha[h] + l_new[h]
                for h, d in T.Parallel(PH, D_v):
                    acc_o[h, d] = acc_o[h, d] * alpha[h]
                T.copy(acc_s, P_shared)
                T.gemm(
                    P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                )

            # Fold the (optional) learnable attention sink as a virtual key
            # column: a per-head logit ``AttnSink[h]`` competing in the same
            # softmax denominator (reduces every weight so they sum < 1). A
            # very-negative sink (e.g. -1e30) makes ``exp(sink - m)`` underflow
            # to 0, recovering the plain sinkless softmax bit-for-bit.
            #   AttnSink is a pre-scaled logit (same units as ``m_i``), so it is
            #   converted to base-2 with log2(e) only (no ``sm_scale``).
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
            for h, d in T.Parallel(PH, D_v):
                acc_o[h, d] = acc_o[h, d] * alpha[h]

            # normalize; empty rows (no valid key) -> 0 out / -inf lse
            for h, d in T.Parallel(PH, D_v):
                acc_o[h, d] = T.if_then_else(
                    l_i[h] > 0, acc_o[h, d] / l_i[h], 0.0
                )
            for h, d in T.Parallel(PH, D_v):
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


def windowed_mqa_attn_fwd(
    q, k, v, valid_range, attn_sink=None, sm_scale=None, block_B=64
):
    """Forward interface for windowed MQA (shared K/V head) flash attention.

    Args:
        q:           [B, S, H, D] bf16 query (H heads).
        k, v:        [B, S_kv, D] / [B, S_kv, D_v] bf16 single shared K/V head.
        valid_range: [B, S, 2] int32 per-query [bos, eos) valid key columns
            (encodes causal + document + sliding-window masking).
        attn_sink:   [H] fp32 per-head learnable sink logit (attention sink /
            softmax off-by-one). ``None`` -> a very-negative sink is used,
            recovering the plain sinkless softmax bit-for-bit.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key tile size streamed by the kernel.

    Returns:
        out [B,S,H,D_v], lse [B,S,H].
    """
    assert q.is_contiguous()
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    d_v = v.shape[-1]
    # lightweight host-side shape checks (no device sync) so mismatched inputs
    # fail early and clearly instead of causing undefined kernel behaviour.
    assert len(k.shape) == 3 and k.shape[0] == b and k.shape[2] == d, (
        f"k must be [B, S_kv, D] matching q; got k {k.shape}, q {q.shape}"
    )
    assert len(v.shape) == 3 and list(v.shape[:2]) == [b, s_kv], (
        f"v must be [B, S_kv, D_v] matching k seqlen; got {v.shape}, k {k.shape}"
    )
    assert list(valid_range.shape) == [b, s, 2], (
        f"valid_range must be [B, S, 2]; got {valid_range.shape}"
    )
    if sm_scale is None:
        sm_scale = d**-0.5
    if valid_range.dtype != paddle.int32:
        valid_range = valid_range.cast("int32")

    # No learnable sink -> a very-negative per-head sink makes exp(sink - m)
    # underflow to 0, so the kernel produces the plain sinkless softmax.
    if attn_sink is None:
        attn_sink = paddle.full([h], -1e30, dtype="float32")
    else:
        assert list(attn_sink.shape) == [h], (
            f"attn_sink must be [H={h}]; got {attn_sink.shape}"
        )
        attn_sink = attn_sink.cast("float32")
    attn_sink = attn_sink.contiguous()

    # kernel streams whole block_B tiles; pad K/V so the last tile is in bounds
    pad = (block_B - s_kv % block_B) % block_B
    if pad > 0:
        k = paddle.nn.functional.pad(k, [0, 0, 0, pad])
        v = paddle.nn.functional.pad(v, [0, 0, 0, pad])
    k = k.contiguous()
    v = v.contiguous()
    valid_range = valid_range.contiguous()

    kernel = windowed_mqa_fwd(h, d, float(sm_scale), D_v=d_v, block_B=block_B)
    out, lse = kernel(q, k, v, valid_range, attn_sink)
    return out, lse
