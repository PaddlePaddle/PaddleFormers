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

import paddle

from .attn import sparse_mqa_bwd
from .attn.sparse_mqa import _prepare_inputs, sparse_attn


class CSASparseAttention(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, query, kv_full, attn_sink, topk_idxs, softmax_scale):
        b, sq, np_heads, hn = query.shape
        ctx.query_shape = (b, sq, np_heads, hn)
        ctx.softmax_scale = float(softmax_scale)
        ctx.attn_sink_dtype = attn_sink.dtype
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
        )
        ctx.save_for_backward(query, kv_full, attn_sink, topk_idxs, output, lse)
        return output.reshape([b, sq, np_heads * hn])

    @staticmethod
    def backward(ctx, grad_output):
        query, kv_full, attn_sink, topk_idxs, output, lse = ctx.saved_tensor()
        b, sq, np_heads, hn = ctx.query_shape
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


def csa_sparse_attn(query, kv_full, attn_sink, topk_idxs, softmax_scale):
    return CSASparseAttention.apply(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
    )
