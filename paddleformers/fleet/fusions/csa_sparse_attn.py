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

"""Unified CSA Sparse Attention entry with single-switch backend dispatch.

A single ``backend`` argument selects one of three implementations of the
final sparse MQA attention:
  - "unfused": pure-Paddle einsum forward + Paddle autograd backward
  - "tilelang": TileLang sparse MQA forward + TileLang backward
  - "cudnn": FlashMLA sparse forward + cuDNN DSA backward
"""

import paddle
from paddle import Tensor


def unfused_compressed_sparse_attn(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Sparse attention with MQA and learnable attention sink.

    Args:
        query: [b, sq, np, hn] multi-head query
        kv_full: [b, n_kv, hn] single-head KV (original + compressed concatenated)
        attn_sink: [np] per-head learnable bias (attention sink)
        topk_indices: [b, sq, topk] indices into kv_full dim=1 (-1 = invalid)
        softmax_scale: attention scale factor

    Returns:
        output: [b, sq, np * hn]
    """
    b, sq, np_heads, hn = query.shape
    topk = topk_indices.shape[-1]

    # Clamp negative indices to 0 for gathering, mask them later
    safe_indices = paddle.clip(topk_indices, min=0).cast(
        paddle.int64
    )  # [b, sq, topk]
    safe_indices_exp = safe_indices.unsqueeze(-1).expand(
        [-1, -1, -1, hn]
    )  # [b, sq, topk, hn]

    # Gather KV at selected positions: [b, n_kv, hn] -> [b, sq, topk, hn]
    kv_gathered = paddle.gather(
        kv_full.unsqueeze(1).expand([-1, sq, -1, -1]),
        dim=2,
        index=safe_indices_exp,
    )
    with paddle.amp.auto_cast(False):
        # Compute attention scores: [b, np, sq, topk]
        q = query.transpose([0, 2, 1, 3]).cast("float32")  # [b, np, sq, hn]
        kv_g = kv_gathered.cast("float32")
        scores = (
            paddle.einsum("bnsh,bskh->bnsk", q, kv_g) * softmax_scale
        )  # [b, np, sq, topk]
        # Mask invalid positions (topk_indices < 0) with -inf
        invalid_mask = (topk_indices < 0).unsqueeze(1)  # [b, 1, sq, topk]
        scores = scores.masked_fill(invalid_mask, float("-inf"))

        # Softmax with attention sink
        # sink: [np] -> [1, np, 1, 1]
        sink = attn_sink.reshape([1, np_heads, 1, 1])
        # Compute stable softmax: max over scores and sink
        scores_max = scores.max(axis=-1, keepdim=True)  # [b, np, sq, 1]
        scores_max = paddle.maximum(scores_max, sink)

        exp_scores = paddle.exp(scores - scores_max)  # [b, np, sq, topk]
        exp_sink = paddle.exp(sink - scores_max)  # [b, np, sq, 1]

        sum_exp = (
            exp_scores.sum(axis=-1, keepdim=True) + exp_sink
        )  # [b, np, sq, 1]
        attn_weights = exp_scores / sum_exp  # [b, np, sq, topk]

        # Weighted sum: [b, np, sq, topk] x [b, sq, topk, hn] -> [b, np, sq, hn]
        output = paddle.einsum("bnsk,bskh->bnsh", attn_weights, kv_g)
    output = output.cast(query.dtype)

    # Reshape: [b, np, sq, hn] -> [b, sq, np * hn]
    output = output.transpose([0, 2, 1, 3]).reshape([b, sq, np_heads * hn])
    return output


class CSASparseAttention(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx, query, kv_full, attn_sink, topk_idxs, softmax_scale, backend
    ):
        from paddleformers.fleet.fusions.csa_sparse_attn_utils import (
            prepare_inputs,
        )

        b, sq, np_heads, hn = query.shape
        ctx.query_shape = (b, sq, np_heads, hn)
        ctx.softmax_scale = float(softmax_scale)
        ctx.attn_sink_dtype = attn_sink.dtype
        ctx.backend = backend

        query, kv_full, attn_sink, topk_idxs = prepare_inputs(
            query,
            kv_full,
            attn_sink,
            topk_idxs,
        )
        if backend == "cudnn":
            from paddleformers.fleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
                flash_mla_sparse_attn,
            )

            output, lse, _ = flash_mla_sparse_attn(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                sm_scale=ctx.softmax_scale,
            )
        else:
            from paddleformers.fleet.tilelang_ops.attn.sparse_mqa import (
                sparse_attn,
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

        if ctx.backend == "cudnn":
            from paddleformers.fleet.cudnn_ops import csa_sparse_attn_bwd_cudnn
            from paddleformers.fleet.fusions.csa_sparse_attn_utils import (
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
            d_attn_sink = d_sink.reshape(attn_sink.shape).cast(
                ctx.attn_sink_dtype
            )
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
            d_attn_sink = d_attn_sink.reshape(attn_sink.shape).cast(
                ctx.attn_sink_dtype
            )

        return (dq, dkv, d_attn_sink, None)


def csa_sparse_attn(
    query, kv_full, attn_sink, topk_idxs, softmax_scale, backend="tilelang"
):
    """Unified CSA sparse attention entry point.

    Args:
        backend: one of {"unfused", "tilelang", "cudnn"}.
    """
    if backend == "unfused":
        return unfused_compressed_sparse_attn(
            query,
            kv_full,
            attn_sink,
            topk_idxs,
            softmax_scale,
        )
    if backend not in ("tilelang", "cudnn"):
        raise ValueError(
            f"csa_sparse_attn_backend={backend!r} is invalid. "
            "Must be one of {'unfused', 'tilelang', 'cudnn'}."
        )
    return CSASparseAttention.apply(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        backend,
    )
