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


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mqa_fwd(
    heads,
    dim,
    topk,
    sm_scale=None,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be power of 2, got {dim}"
    assert topk % block_I == 0, f"topk ({topk}) must be divisible by block_I ({block_I})"
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, topk]
    lse_shape = [batch, seq_len, heads]
    attn_sink_shape = [heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H = heads
    padded_H = max(tilelang.math.next_power_of_2(heads), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim

    if heads > 64:
        assert heads % 64 == 0, "heads should be a multiple of 64"
        REPLICATE_H = heads // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, threads=threads) as (
            bx,
            by,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_log2 = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i = by
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)

            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, i_i * BI + bi_i] != -1

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, i_i * BI + bi_i], d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Update max to include sink logit for numerical stability
            # m_i_log2: global max in log2 space (includes both KV scores and sink)
            for h_i in T.Parallel(H_per_block):
                m_i_log2[h_i] = T.max(
                    m_i[h_i] * sm_scale,
                    AttnSink[H0 + h_i] * 1.44269504,
                )
            for h_i in T.Parallel(H_per_block):
                alpha[h_i] = T.exp2(m_i[h_i] * sm_scale - m_i_log2[h_i])
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = sumexp[h_i] * alpha[h_i] + T.exp2(AttnSink[H0 + h_i] * 1.44269504 - m_i_log2[h_i])
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i] / sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i_log2[h_i]

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mqa_fwd_interface(
    q,
    kv,
    attn_sink,
    topk_idxs,
    sm_scale=None,
    block_I=64,
    num_stages=2,
    threads=256,
):
    """Forward interface for DSv4 sparse MQA attention."""
    assert q.is_contiguous() and kv.is_contiguous() and topk_idxs.is_contiguous()
    batch, seq_len, heads, dim = q.shape
    _, _, topk = topk_idxs.shape
    _, _, kv_dim = kv.shape
    assert kv_dim == dim

    padded_topk = (topk + block_I - 1) // block_I * block_I
    if padded_topk != topk:
        pad = paddle.full([batch, seq_len, padded_topk - topk], -1, dtype=topk_idxs.dtype)
        topk_idxs = paddle.concat([topk_idxs, pad], axis=-1).contiguous()
        topk = padded_topk

    kernel = sparse_mqa_fwd(
        heads,
        dim,
        topk,
        sm_scale,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    out, lse = kernel(q, kv, attn_sink, topk_idxs)
    return out, lse
