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

"""Unified CSA Sparse Attention PyLayer with backend dispatch.

Supports:
  - "tilelang": tilelang forward + tilelang backward
  - "cudnn": flash_mla forward + cuDNN backward
"""

import paddle


class CSASparseAttention(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, query, kv_full, attn_sink, topk_idxs, softmax_scale, backend):
        from paddleformers.fleet.tilelang_ops.attn.sparse_mqa import (
            _prepare_inputs,
            sparse_attn,
        )

        b, sq, np_heads, hn = query.shape
        ctx.query_shape = (b, sq, np_heads, hn)
        ctx.softmax_scale = float(softmax_scale)
        ctx.attn_sink_dtype = attn_sink.dtype
        ctx.backend = backend

        query, kv_full, attn_sink, topk_idxs = _prepare_inputs(
            query,
            kv_full,
            attn_sink,
            topk_idxs,
        )
        output, lse = sparse_attn(
            query,
            kv_full,
            attn_sink,
            topk_idxs,
            sm_scale=ctx.softmax_scale,
            backend=backend,
        )
        ctx.save_for_backward(query, kv_full, attn_sink, topk_idxs, output, lse)
        return output.reshape([b, sq, np_heads * hn])

    @staticmethod
    def backward(ctx, grad_output):
        query, kv_full, attn_sink, topk_idxs, output, lse = ctx.saved_tensor()
        b, sq, np_heads, hn = ctx.query_shape

        if ctx.backend == "cudnn":
            from paddleformers.fleet.cudnn_ops import csa_sparse_attn_bwd_cudnn
            from paddleformers.fleet.tilelang_ops.attn.sparse_mqa import (
                _local_to_global_flat,
            )

            _, s_kv, dkv_dim = kv_full.shape

            q_flat = query.reshape([b * sq, np_heads, hn])
            o_flat = output.reshape([b * sq, np_heads, hn])
            do_flat = grad_output.reshape([b * sq, np_heads, hn])
            kv_flat = kv_full.reshape([b * s_kv, dkv_dim])
            lse_flat = lse.reshape([b * sq, np_heads])
            topk_idxs_flat = _local_to_global_flat(topk_idxs, s_kv)

            dq_flat, dkv_flat, d_sink = csa_sparse_attn_bwd_cudnn(
                q_flat,
                kv_flat,
                o_flat,
                do_flat,
                lse_flat,
                attn_sink,
                topk_idxs_flat,
                softmax_scale=ctx.softmax_scale,
            )
            dq = dq_flat.reshape(query.shape)
            dkv = dkv_flat.reshape(kv_full.shape)
            d_attn_sink = d_sink.reshape(attn_sink.shape).cast(ctx.attn_sink_dtype)
        else:
            from paddleformers.fleet.tilelang_ops.attn import sparse_mqa_bwd

            grad_output = grad_output.reshape([b, sq, np_heads, hn])
            dq, dkv, d_attn_sink = sparse_mqa_bwd.sparse_mqa_bwd_interface(
                query,
                kv_full,
                attn_sink,
                output,
                grad_output,
                topk_idxs,
                lse,
                ctx.softmax_scale,
            )
            dq = dq.reshape(query.shape)
            dkv = dkv.reshape(kv_full.shape)
            d_attn_sink = d_attn_sink.reshape(attn_sink.shape).cast(ctx.attn_sink_dtype)

        return (dq, dkv, d_attn_sink, None)


def csa_sparse_attn(query, kv_full, attn_sink, topk_idxs, softmax_scale, backend="tilelang"):
    """Unified CSA sparse attention entry point.

    Args:
        backend: "tilelang" or "cudnn"
    """
    return CSASparseAttention.apply(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        backend,
    )
