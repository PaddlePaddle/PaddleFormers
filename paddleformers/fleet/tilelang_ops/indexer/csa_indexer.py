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

DEFAULT_INDEXER_BLOCK = 32


def _get_csa_indexer_topk_fwd_interface():
    from .csa_indexer_fwd import csa_indexer_topk_fwd_interface

    return csa_indexer_topk_fwd_interface


def _get_csa_indexer_bwd_interface():
    from .csa_indexer_bwd import csa_indexer_bwd_interface

    return csa_indexer_bwd_interface


def _get_csa_attn_target_reducesum_interface():
    from .csa_attn_target import csa_attn_target_reducesum_interface

    return csa_attn_target_reducesum_interface


def _validate_indexer_inputs(index_q, index_k_comp, weights):
    if not isinstance(index_q, paddle.Tensor):
        raise TypeError(
            f"index_q must be a paddle.Tensor, got {type(index_q)!r}"
        )
    if not isinstance(index_k_comp, paddle.Tensor):
        raise TypeError(
            f"index_k_comp must be a paddle.Tensor, got {type(index_k_comp)!r}"
        )
    if not isinstance(weights, paddle.Tensor):
        raise TypeError(
            f"weights must be a paddle.Tensor, got {type(weights)!r}"
        )
    if len(index_q.shape) != 4:
        raise ValueError(
            f"index_q must have shape [B, S, H_i, D_i], got {index_q.shape}"
        )
    if len(index_k_comp.shape) != 3:
        raise ValueError(
            f"index_k_comp must have shape [B, S_comp, D_i], got {index_k_comp.shape}"
        )
    if len(weights.shape) != 3:
        raise ValueError(
            f"weights must have shape [B, S, H_i], got {weights.shape}"
        )

    batch, seq_len, heads, dim = index_q.shape
    batch_k, _, dim_k = index_k_comp.shape
    batch_w, seq_len_w, heads_w = weights.shape
    if batch != batch_k or batch != batch_w:
        raise ValueError(
            f"batch mismatch: index_q={index_q.shape}, index_k_comp={index_k_comp.shape}, weights={weights.shape}"
        )
    if seq_len != seq_len_w or heads != heads_w or dim != dim_k:
        raise ValueError(
            f"shape mismatch: index_q={index_q.shape}, index_k_comp={index_k_comp.shape}, weights={weights.shape}"
        )


def _validate_topk_and_grad(index_q, topk_indices, grad_scores):
    if not isinstance(topk_indices, paddle.Tensor):
        raise TypeError(
            f"topk_indices must be a paddle.Tensor, got {type(topk_indices)!r}"
        )
    if not isinstance(grad_scores, paddle.Tensor):
        raise TypeError(
            f"grad_scores must be a paddle.Tensor, got {type(grad_scores)!r}"
        )
    if len(topk_indices.shape) != 3:
        raise ValueError(
            f"topk_indices must have shape [B, S, topk], got {topk_indices.shape}"
        )
    if len(grad_scores.shape) != 3:
        raise ValueError(
            f"grad_scores must have shape [B, S, topk], got {grad_scores.shape}"
        )
    batch, seq_len, _, _ = index_q.shape
    if topk_indices.shape != grad_scores.shape:
        raise ValueError(
            f"topk_indices shape {topk_indices.shape} must match grad_scores shape {grad_scores.shape}"
        )
    topk_batch, topk_seq_len, topk_last_dim = topk_indices.shape
    if topk_batch != batch or topk_seq_len != seq_len:
        raise ValueError(
            f"topk_indices shape {topk_indices.shape} is incompatible with index_q shape {index_q.shape}"
        )
    if topk_last_dim <= 0:
        raise ValueError("topk_indices last dimension must be positive")


def _prepare_forward_inputs(index_q, index_k_comp, weights, topk_effective):
    _validate_indexer_inputs(index_q, index_k_comp, weights)
    if int(topk_effective) <= 0:
        raise ValueError(
            f"topk_effective must be positive, got {topk_effective}"
        )
    if weights.dtype != paddle.float32:
        weights = weights.cast("float32")
    return (
        index_q.contiguous(),
        index_k_comp.contiguous(),
        weights.contiguous(),
        int(topk_effective),
    )


def _prepare_backward_inputs(
    index_q, weights, index_k_comp, topk_indices, grad_scores
):
    _validate_indexer_inputs(index_q, index_k_comp, weights)
    _validate_topk_and_grad(index_q, topk_indices, grad_scores)
    if topk_indices.dtype != paddle.int32:
        topk_indices = topk_indices.cast("int32")
    if grad_scores.dtype != paddle.float32:
        grad_scores = grad_scores.cast("float32")
    if weights.dtype != paddle.float32:
        weights = weights.cast("float32")

    topk_indices = topk_indices.contiguous()
    grad_scores = paddle.where(
        topk_indices >= 0, grad_scores, paddle.zeros_like(grad_scores)
    )
    return (
        index_q.contiguous(),
        weights.contiguous(),
        index_k_comp.contiguous(),
        topk_indices,
        grad_scores.contiguous(),
    )


def csa_indexer_topk_fwd(
    index_q,
    index_k_comp,
    weights,
    ratio: int,
    topk_effective: int,
    seq_offset: int = 0,
    valid_range: paddle.Tensor | None = None,
    block_K: int = DEFAULT_INDEXER_BLOCK,
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Paddle entry for V4 CSA compressed indexer forward.

    Args:
        index_q: [B, S, H_i, D_i] indexer queries.
        index_k_comp: [B, S_comp, D_i] compressed indexer keys.
        weights: [B, S, H_i] per-head weights for score aggregation.
        ratio: compression ratio (e.g. 4). Used only when valid_range is None
            to build causal-only ValidRange. Causal range:
            [0, (t + seq_offset + 1) // ratio).
        topk_effective: number of top-k entries to select per query position.
            - Phase 2 (dsa_indexer_use_sparse_loss=False): set to n_compressed
              = floor(S / ratio) for full-candidate selection.
            - Phase 3 (dsa_indexer_use_sparse_loss=True): set to
              min(index_topk, n_compressed), typically 512.
        seq_offset: global position offset for the first local query token.
            In CP mode, this is cp_rank * sq_local. Default 0.
            Used only when valid_range is None.
        valid_range: [B, S, 2] int32 tensor specifying per-query [BOS, EOS)
            valid compressed K range (left-closed, right-open). If None,
            automatically built from ratio + seq_offset (causal-only mode).
        block_K: tile size for streaming over compressed keys (default 32).

    Returns:
        topk_indices: [B, S, topk_effective] int32, invalid slots are -1.
        topk_scores: [B, S, topk_effective] fp32 top-k softmax probabilities.
    """
    index_q, index_k_comp, weights, topk_effective = _prepare_forward_inputs(
        index_q,
        index_k_comp,
        weights,
        topk_effective,
    )
    csa_indexer_topk_fwd_interface = _get_csa_indexer_topk_fwd_interface()
    topk_indices, topk_scores = csa_indexer_topk_fwd_interface(
        index_q,
        index_k_comp,
        weights,
        ratio=int(ratio),
        topk_effective=topk_effective,
        seq_offset=int(seq_offset),
        valid_range=valid_range,
        block_K=int(block_K),
        num_stages=int(num_stages),
        num_threads=int(num_threads),
    )
    batch, seq_len = index_q.shape[:2]
    expected_shape = [batch, seq_len, topk_effective]
    if (
        topk_indices.shape != expected_shape
        or topk_scores.shape != expected_shape
    ):
        raise RuntimeError(
            f"unexpected CSA indexer forward output shapes: indices={topk_indices.shape}, scores={topk_scores.shape}, expected={expected_shape}"
        )
    if not isinstance(topk_indices, paddle.Tensor) or not isinstance(
        topk_scores, paddle.Tensor
    ):
        raise RuntimeError(
            "TileLang must return Paddle tensors. "
            "Ensure paddle.enable_compat(scope={'tilelang'}) runs before import tilelang."
        )
    if topk_indices.dtype != paddle.int32:
        topk_indices = topk_indices.cast("int32")
    if topk_scores.dtype != paddle.float32:
        topk_scores = topk_scores.cast("float32")
    return topk_indices, topk_scores


def csa_attn_target_reducesum(
    query_mla,
    key_comp_mla,
    topk_indices,
    softmax_scale: float,
    block_I: int = DEFAULT_INDEXER_BLOCK,
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Paddle entry for V4 CSA indexer-loss attention target computation.

    Computes the selected-set multi-head target distribution used by the CSA
    indexer KL loss. This replaces materializing full [B, H, S, S_comp]
    attention scores in the Paddle reference path.
    """
    if not isinstance(query_mla, paddle.Tensor):
        raise TypeError(
            f"query_mla must be a paddle.Tensor, got {type(query_mla)!r}"
        )
    if not isinstance(key_comp_mla, paddle.Tensor):
        raise TypeError(
            f"key_comp_mla must be a paddle.Tensor, got {type(key_comp_mla)!r}"
        )
    if not isinstance(topk_indices, paddle.Tensor):
        raise TypeError(
            f"topk_indices must be a paddle.Tensor, got {type(topk_indices)!r}"
        )
    if len(query_mla.shape) != 4:
        raise ValueError(
            f"query_mla must have shape [B, S, H, D], got {query_mla.shape}"
        )
    if len(key_comp_mla.shape) != 3:
        raise ValueError(
            f"key_comp_mla must have shape [B, S_comp, D], got {key_comp_mla.shape}"
        )
    if len(topk_indices.shape) != 3:
        raise ValueError(
            f"topk_indices must have shape [B, S, topk], got {topk_indices.shape}"
        )
    if (
        query_mla.shape[0] != key_comp_mla.shape[0]
        or query_mla.shape[0] != topk_indices.shape[0]
    ):
        raise ValueError(
            f"batch mismatch: query_mla={query_mla.shape}, key_comp_mla={key_comp_mla.shape}, topk_indices={topk_indices.shape}"
        )
    if query_mla.shape[1] != topk_indices.shape[1]:
        raise ValueError(
            f"sequence mismatch: query_mla={query_mla.shape}, topk_indices={topk_indices.shape}"
        )
    if query_mla.shape[3] != key_comp_mla.shape[2]:
        raise ValueError(
            f"dim mismatch: query_mla={query_mla.shape}, key_comp_mla={key_comp_mla.shape}"
        )
    if topk_indices.dtype != paddle.int32:
        topk_indices = topk_indices.cast("int32")
    topk_indices = topk_indices.contiguous()
    csa_attn_target_reducesum_interface = (
        _get_csa_attn_target_reducesum_interface()
    )
    target = csa_attn_target_reducesum_interface(
        query_mla.contiguous(),
        key_comp_mla.contiguous(),
        topk_indices,
        float(softmax_scale),
        block_I=int(block_I),
        num_stages=int(num_stages),
        num_threads=int(num_threads),
    )
    expected_shape = topk_indices.shape
    if target.shape != expected_shape:
        raise RuntimeError(
            f"unexpected CSA attention target shape: target={target.shape}, expected={expected_shape}"
        )
    if not isinstance(target, paddle.Tensor):
        raise RuntimeError(
            "TileLang must return Paddle tensors. "
            "Ensure paddle.enable_compat(scope={'tilelang'}) runs before import tilelang."
        )
    return target


def csa_indexer_bwd(
    index_q,
    weights,
    index_k_comp,
    topk_indices,
    grad_scores,
    block_I: int = DEFAULT_INDEXER_BLOCK,
    num_stages: int = 0,
    num_threads: int = 128,
):
    """Paddle entry for V4 CSA compressed indexer backward.

    Computes gradients for IndexQ, Weights, and IndexKComp given the selected
    top-k indices and the gradient of the loss w.r.t. the selected scores.

    Args:
        index_q: [B, S, H_i, D_i] indexer queries (same as forward input).
        weights: [B, S, H_i] per-head weights (same as forward input).
        index_k_comp: [B, S_comp, D_i] compressed indexer keys.
        topk_indices: [B, S, topk_effective] int32, from forward output.
            Invalid slots must be -1 (they are masked to zero gradient).
        grad_scores: [B, S, topk_effective] fp32, gradient of loss w.r.t.
            the selected indexer scores (typically ``(probs - target) * coeff``).

    Returns:
        grad_q: [B, S, H_i, D_i] gradient for indexer queries.
        grad_weights: [B, S, H_i] gradient for per-head weights.
        grad_k_comp: [B, S_comp, D_i] gradient for compressed indexer keys.
    """
    index_q, weights, index_k_comp, topk_indices, grad_scores = (
        _prepare_backward_inputs(
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            grad_scores,
        )
    )
    csa_indexer_bwd_interface = _get_csa_indexer_bwd_interface()
    grad_q, grad_weights, grad_k_comp = csa_indexer_bwd_interface(
        index_q,
        weights,
        index_k_comp,
        topk_indices,
        grad_scores,
        block_I=int(block_I),
        num_stages=int(num_stages),
        num_threads=int(num_threads),
    )
    if (
        grad_q.shape != index_q.shape
        or grad_weights.shape != weights.shape
        or grad_k_comp.shape != index_k_comp.shape
    ):
        raise RuntimeError(
            "unexpected CSA indexer backward output shapes: "
            f"grad_q={grad_q.shape}, grad_weights={grad_weights.shape}, grad_k_comp={grad_k_comp.shape}"
        )
    if (
        not isinstance(grad_q, paddle.Tensor)
        or not isinstance(grad_weights, paddle.Tensor)
        or not isinstance(grad_k_comp, paddle.Tensor)
    ):
        raise RuntimeError(
            "TileLang must return Paddle tensors. "
            "Ensure paddle.enable_compat(scope={'tilelang'}) runs before import tilelang."
        )
    return grad_q, grad_weights, grad_k_comp
