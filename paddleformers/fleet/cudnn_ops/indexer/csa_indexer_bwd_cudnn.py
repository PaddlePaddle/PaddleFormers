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

"""Paddle wrapper around the cuDNN-frontend DSA indexer backward.

Calls ``paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api
.indexer_backward_wrapper`` directly on Paddle tensors. The wrapper performs
the score-gradient precompute kernel internally, matching Megatron's
``FusedIndexerSparseAttnFunc`` data flow, so this entry point exposes the
*raw* (target, predict) pair instead of pre-computed ``grad_scores``.

Returns d_index_q / d_weights / d_index_k_comp matching input dtypes/shapes.
"""

from __future__ import annotations

import paddle
from paddlefleet_ops import CUDNN_FRONTEND_HINT, is_cudnn_frontend_available


def _require_cudnn_frontend():
    if not is_cudnn_frontend_available():
        raise ImportError(CUDNN_FRONTEND_HINT)


def _to_bf16(t: paddle.Tensor) -> paddle.Tensor:
    if t.dtype != paddle.bfloat16:
        return t.cast(paddle.bfloat16)
    return t


def csa_indexer_bwd(
    index_q: paddle.Tensor,
    weights: paddle.Tensor,
    index_k_comp: paddle.Tensor,
    target: paddle.Tensor,
    topk_probs: paddle.Tensor,
    topk_indices: paddle.Tensor,
    loss_coeff: float,
    grad_loss: paddle.Tensor | None = None,
    block_I: int = 128,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Run cuDNN-frontend DSA indexer backward on Paddle tensors.

    Args:
        index_q:       [B, S, H, D] bf16, indexer queries.
        weights:       [B, S, H], per-head weights. bf16 expected; fp32 is
            cast to bf16 internally and the gradient cast back.
        index_k_comp:  [B, S_comp, D] bf16, compressed indexer keys.
        target:        [B, S, topk] fp32, multi-head aggregated attention
            target (probability over selected slots).
        topk_probs:    [B, S, topk] fp32, indexer post-softmax probs over the
            same selected slots.
        topk_indices:  [B, S, topk] int32, selected positions (-1 = invalid).
        loss_coeff:    scalar, KL loss coefficient (e.g. 0.01).
        grad_loss:     0-D fp32 paddle tensor, upstream gradient w.r.t. the
            scalar KL loss. ``None`` is treated as 1.0.
        block_I:       cuDNN tile size. 128 matches Megatron production.

    Returns:
        (grad_index_q, grad_weights, grad_index_k_comp) with the same shapes
        as the corresponding inputs and dtypes restored to the original
        ``weights`` dtype.
    """
    orig_weights_dtype = weights.dtype
    orig_q_dtype = index_q.dtype
    orig_k_dtype = index_k_comp.dtype

    index_q_bf = _to_bf16(index_q)
    weights_bf = _to_bf16(weights)
    index_k_bf = _to_bf16(index_k_comp)

    # cuDNN overwrites attn_score (target) and index_score (topk_probs)
    # in-place during the score-grad precompute. Clone so the saved
    # forward tensors are untouched, casting to fp32 in one step.
    target_buf = (
        target.cast(paddle.float32)
        if target.dtype != paddle.float32
        else target.clone()
    )
    predict_buf = (
        topk_probs.cast(paddle.float32)
        if topk_probs.dtype != paddle.float32
        else topk_probs.clone()
    )
    if topk_indices.dtype != paddle.int32:
        topk_indices = topk_indices.cast(paddle.int32)

    # grad_loss as a 0-D fp32 tensor.
    if grad_loss is None:
        grad_loss_paddle = paddle.ones([], dtype=paddle.float32)
    else:
        grad_loss_paddle = grad_loss
        if grad_loss_paddle.dtype != paddle.float32:
            grad_loss_paddle = grad_loss_paddle.cast(paddle.float32)

    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api import (
        indexer_backward_wrapper,
    )

    out = indexer_backward_wrapper(
        index_q_bf,
        weights_bf,
        index_k_bf,
        target_buf,
        predict_buf,
        topk_indices,
        sm_scale=float(index_q_bf.shape[-1]) ** -0.5,
        loss_coeff=float(loss_coeff),
        grad_loss=grad_loss_paddle,
        block_I=int(block_I),
    )

    grad_q = out["d_index_q"]
    grad_weights = out["d_weights"]
    grad_k = out["d_index_k"]

    if grad_q.dtype != orig_q_dtype:
        grad_q = grad_q.cast(orig_q_dtype)
    if grad_weights.dtype != orig_weights_dtype:
        grad_weights = grad_weights.cast(orig_weights_dtype)
    if grad_k.dtype != orig_k_dtype:
        grad_k = grad_k.cast(orig_k_dtype)

    return grad_q, grad_weights, grad_k
