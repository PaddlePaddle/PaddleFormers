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

"""Independent TileLang MHA block-score attention (correctness oracle).

A standalone, per-(query token, head) TileLang forward that mirrors the FA4
fused ``block_score_fa4_attn_fwd`` operator but with H INDEPENDENT heads (real
MHA, per-head K/V), written for verifiability rather than speed. It is a
drop-in replacement for ``block_score_fa4_attn_fwd`` used to cross-check the
fused kernel while debugging a training-loss anomaly.

Q is ``[B, S, H, D]``; K is ``[B, S_kv, H, D]``; V is ``[B, S_kv, H, D_v]`` --
every head has its own K/V (NOT MQA). Masking (causal + document) is expressed
through ``valid_range`` ``[B, S, 2]`` giving each query token's half-open valid
key column range ``[bos, eos)``.

The forward returns:
* ``out``         ``[B, S, H, D_v]`` attention output.
* ``lse``         ``[B, S, H]`` natural log-sum-exp of the *scaled* logits.
* ``block_logit`` ``[B, H, S, num_blocks]`` per-(query, key-block) max of the
  *scaled* logit ``sm_scale * q.k`` over the block's valid columns; fully
  masked / never-visited blocks stay ``-inf``.

Blocks are DOCUMENT-RELATIVE: block ``j`` of a query spans absolute key columns
``[bos + j*block_B, bos + (j+1)*block_B)``. num_blocks = ceil(S_kv / block_B).

Because the emitted logit is already SCALED, it matches the current
``pipeline.block_scores_from_logit`` which recovers the eq.(3) probability as
``exp(block_logit - lse)`` WITHOUT re-applying ``sm_scale`` (unlike the older
MQA scaffold that stored the raw logit).
"""

import paddle
import tilelang
from tilelang import language as T

from .block_score_mha_bwd import block_score_mha_bwd_interface


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def block_score_mha_fwd(
    H,
    D,
    sm_scale,
    D_v=None,
    block_B=64,
    num_stages=2,
    threads=128,
):
    """MHA block-score kernel: one program per (query token, head).

    The single query row ``Q[b, s, h, :]`` is placed on GEMM ``M`` row 0 (the
    tile is padded to 16 rows with zeros so the tensor-core M-tile is legal);
    only row 0 carries the real query. That head's own K/V blocks are streamed
    in ``block_B``-sized tiles over the token's valid range ``[bos, eos)`` with a
    ``ceildiv(eos - bos, block_B)`` early-exit. Online softmax runs base-2; the
    per-block max of the SCALED logit is written to ``BlockLogit``.
    """
    if D_v is None:
        D_v = D
    assert D % 16 == 0, (
        f"D must be a multiple of 16 (tensor-core k-tile), got {D}"
    )
    assert D_v % 16 == 0, (
        f"D_v must be a multiple of 16 (tensor-core k-tile), got {D_v}"
    )
    scale_log2 = sm_scale * 1.44269504  # log2(e), online softmax in base 2

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_blocks = T.dynamic("num_blocks")

    q_shape = [batch, seq_len, H, D]
    k_shape = [batch, seq_len_kv, H, D]
    v_shape = [batch, seq_len_kv, H, D_v]
    o_shape = [batch, seq_len, H, D_v]
    lse_shape = [batch, seq_len, H]
    vr_shape = [batch, seq_len, 2]
    blk_shape = [batch, H, seq_len, num_blocks]

    dtype = T.bfloat16
    accum_dtype = T.float32
    idx_dtype = T.int32
    PM = 16  # padded query rows on M (only row 0 is the real query)
    BB = block_B

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
        BlockLogit: T.Tensor(blk_shape, accum_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, H, batch, threads=threads) as (bs, bh, bb):
            Q_shared = T.alloc_shared([PM, D], dtype)
            K_shared = T.alloc_shared([BB, D], dtype)
            V_shared = T.alloc_shared([BB, D_v], dtype)
            P_shared = T.alloc_shared([PM, BB], dtype)

            acc_o = T.alloc_fragment([PM, D_v], accum_dtype)
            acc_s = T.alloc_fragment([PM, BB], accum_dtype)
            blk_max = T.alloc_fragment([PM], accum_dtype)
            m_i = T.alloc_fragment([PM], accum_dtype)
            m_prev = T.alloc_fragment([PM], accum_dtype)
            l_i = T.alloc_fragment([PM], accum_dtype)
            l_new = T.alloc_fragment([PM], accum_dtype)
            alpha = T.alloc_fragment([PM], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(m_i, -(2**30))
            T.fill(l_i, 0)

            bos = ValidRange[bb, bs, 0]
            eos = ValidRange[bb, bs, 1]

            # load this token's single query head onto M row 0 (pad the rest)
            for i, d in T.Parallel(PM, D):
                Q_shared[i, d] = T.if_then_else(
                    i == 0, Q[bb, bs, bh, d], T.cast(0, dtype)
                )

            # causal/document early-exit: only this token's own valid blocks
            # (relative block j in [0, ceil((eos-bos)/block_B))) can hold a
            # valid key; every later block is fully masked and its per-block max
            # stays -inf (the host buffer is pre-filled with -inf).
            num_valid_blocks = T.ceildiv(eos - bos, BB)
            for j in T.Pipelined(num_valid_blocks, num_stages=num_stages):
                # gather relative block j: cols [bos + j*BB, bos + (j+1)*BB).
                for c, d in T.Parallel(BB, D):
                    col = bos + j * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    K_shared[c, d] = T.if_then_else(
                        in_bounds, K[bb, safe_col, bh, d], T.cast(0, dtype)
                    )
                for c, d in T.Parallel(BB, D_v):
                    col = bos + j * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    V_shared[c, d] = T.if_then_else(
                        in_bounds, V[bb, safe_col, bh, d], T.cast(0, dtype)
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

                # causal + document mask (col >= bos automatic for block j)
                for i, c in T.Parallel(PM, BB):
                    col = bos + j * BB + c
                    acc_s[i, c] = T.if_then_else(
                        col < eos, acc_s[i, c], -T.infinity(accum_dtype)
                    )

                # per-block max of raw logit over valid cols; store the SCALED
                # max logit (sm_scale * raw) so downstream pipeline.py can do
                # exp(block_logit - lse) with no extra sm_scale multiply.
                T.reduce_max(acc_s, blk_max, dim=1, clear=True)
                for i in T.Parallel(PM):
                    if i == 0:
                        BlockLogit[bb, bh, bs, j] = blk_max[i] * sm_scale

                # online softmax (base 2) over scaled logits
                T.copy(m_i, m_prev)
                for i in T.Parallel(PM):
                    m_i[i] = T.max(m_i[i], blk_max[i] * sm_scale)
                for i in T.Parallel(PM):
                    alpha[i] = T.exp2((m_prev[i] - m_i[i]) * 1.44269504)
                for i, c in T.Parallel(PM, BB):
                    acc_s[i, c] = T.exp2(
                        acc_s[i, c] * scale_log2 - m_i[i] * 1.44269504
                    )
                T.reduce_sum(acc_s, l_new, dim=1)
                for i in T.Parallel(PM):
                    l_i[i] = l_i[i] * alpha[i] + l_new[i]
                for i, d in T.Parallel(PM, D_v):
                    acc_o[i, d] = acc_o[i, d] * alpha[i]
                T.copy(acc_s, P_shared)
                T.gemm(
                    P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                )

            # normalize; empty rows (no valid key) -> 0 out / -inf lse
            for i, d in T.Parallel(PM, D_v):
                acc_o[i, d] = T.if_then_else(
                    l_i[i] > 0, acc_o[i, d] / l_i[i], 0.0
                )
            for i, d in T.Parallel(PM, D_v):
                if i == 0:
                    Output[bb, bs, bh, d] = acc_o[i, d]
            for i in T.Parallel(PM):
                if i == 0:
                    Lse[bb, bs, bh] = T.if_then_else(
                        l_i[i] > 0,
                        m_i[i] + T.log(l_i[i]),
                        -T.infinity(accum_dtype),
                    )

    return main


def _default_causal_valid_range(b, s):
    """Single-document causal valid_range [B, S, 2]: bos=0, eos=t+1."""
    pos = paddle.arange(s, dtype="int32")
    bos = paddle.zeros([s], dtype="int32")
    eos = pos + 1
    vr = paddle.stack([bos, eos], axis=-1)  # [S, 2]
    return vr.unsqueeze(0).expand([b, s, 2]).contiguous()


class BlockScoreMHAAttn(paddle.autograd.PyLayer):
    """Differentiable MHA block-score attention (independent oracle).

    ``out`` carries gradient (TileLang fwd + bwd kernels); ``lse`` and the
    in-place ``block_logit`` buffer are non-differentiable (they feed the hard
    TopK block selection).
    """

    @staticmethod
    def forward(ctx, q, k, v, block_logit, valid_range, sm_scale, block_B):
        b, s, h, d = q.shape
        s_kv = k.shape[1]
        d_v = v.shape[-1]

        # kernel streams whole block_B tiles; pad K/V so the last tile is in
        # bounds. Padding does not change valid_range (eos <= s_kv still).
        pad = (block_B - s_kv % block_B) % block_B
        if pad > 0:
            kp = paddle.nn.functional.pad(k, [0, 0, 0, 0, 0, pad])
            vp = paddle.nn.functional.pad(v, [0, 0, 0, 0, 0, pad])
        else:
            kp = k
            vp = v
        kp = kp.contiguous()
        vp = vp.contiguous()

        kernel = block_score_mha_fwd(
            h, d, float(sm_scale), D_v=d_v, block_B=block_B
        )
        out, lse = kernel(q, kp, vp, valid_range, block_logit)

        ctx.save_for_backward(q, k, v, valid_range, out, lse)
        ctx.needs_grad = (
            not q.stop_gradient,
            not k.stop_gradient,
            not v.stop_gradient,
        )
        ctx.mark_non_differentiable(lse)
        ctx.sm_scale = float(sm_scale)
        ctx.block_B = block_B
        return out, lse

    @staticmethod
    def backward(ctx, dout, *_):
        q, k, v, valid_range, out, lse = ctx.saved_tensor()
        dq, dk, dv = block_score_mha_bwd_interface(
            q,
            k,
            v,
            out,
            dout.contiguous(),
            lse,
            valid_range,
            sm_scale=ctx.sm_scale,
            block_B=ctx.block_B,
        )
        gq, gk, gv = ctx.needs_grad
        # One grad slot per forward Tensor input, in order:
        #   q, k, v, block_logit, valid_range, sm_scale, block_B.
        # block_logit / valid_range are non-differentiable buffers; sm_scale and
        # block_B are python scalars -> all None.
        return (
            dq if gq else None,
            dk if gk else None,
            dv if gv else None,
            None,  # block_logit
            None,  # valid_range
        )


def block_score_mha_attn_fwd(
    q,
    k,
    v,
    valid_range=None,
    sm_scale=None,
    block_B=64,
    causal=True,
    startend_row_indices=None,
):
    """Independent MHA block-score forward (drop-in for FA4 oracle).

    Args:
        q:           [B, S, H, D] bf16 query.
        k:           [B, S_kv, H, D] bf16 key (per head).
        v:           [B, S_kv, H, D_v] bf16 value (per head).
        valid_range: [B, S, 2] int32 per-query [bos, eos) valid key columns
            (encodes causal + document masking). Block coordinates are anchored
            at each query's own ``bos``. When ``None`` a single-document causal
            range (bos=0, eos=t+1) is built (requires ``causal=True``).
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size (document-relative), default 64.
        causal:      only used to build the default valid_range when it is None.
        startend_row_indices: accepted for signature compatibility with the FA4
            operator; unused here because masking flows through valid_range.

    Returns:
        (out [B,S,H,D_v], lse [B,S,H], block_logit [B,H,S,num_blocks]).
        num_blocks = ceil(S_kv / block_B). block_logit stores the SCALED
        per-block max logit (-inf for masked / never-visited blocks).
    """
    assert q.is_contiguous(), "q must be contiguous"
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    assert list(k.shape) == [b, s_kv, h, d], (
        f"k must be [B, S_kv, H, D] matching q; got k {k.shape}, q {q.shape}"
    )
    d_v = v.shape[-1]
    assert list(v.shape) == [b, s_kv, h, d_v], (
        f"v must be [B, S_kv, H, D_v]; got {v.shape}"
    )
    if sm_scale is None:
        sm_scale = d**-0.5

    if valid_range is None:
        if not causal:
            raise ValueError(
                "valid_range=None only supported with causal=True "
                "(single-document causal default)"
            )
        valid_range = _default_causal_valid_range(b, s)
    else:
        assert list(valid_range.shape) == [b, s, 2], (
            f"valid_range must be [B={b}, S={s}, 2]; got {valid_range.shape}"
        )
        if valid_range.dtype != paddle.int32:
            valid_range = valid_range.cast("int32")
    valid_range = valid_range.contiguous()

    num_blocks = (s_kv + block_B - 1) // block_B
    # Pre-fill -inf: the kernel only writes the blocks its early-exit visits;
    # skipped blocks must read back as -inf so their host-side score is 0.
    block_logit = paddle.full(
        [b, h, s, num_blocks], float("-inf"), dtype="float32"
    )

    out, lse = BlockScoreMHAAttn.apply(
        q, k, v, block_logit, valid_range, float(sm_scale), block_B
    )
    return out, lse, block_logit
