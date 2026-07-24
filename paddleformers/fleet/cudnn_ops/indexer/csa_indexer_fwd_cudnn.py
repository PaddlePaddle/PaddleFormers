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

"""Paddle wrapper around the cuDNN-frontend DSA indexer forward.

Calls ``paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_forward.api
.indexer_forward_wrapper`` and ``paddlefleet_ops.cudnn.deepseek_sparse_attention
.indexer_top_k.api.indexer_top_k_wrapper`` directly on Paddle tensors.

Returns selected compressed KV indices and per-row valid counts.
"""

from __future__ import annotations

import paddle

from paddlefleet_ops import CUDNN_FRONTEND_HINT, is_cudnn_frontend_available

# Packed-global fallback materializes a dense ``[B, S_q, S_k]`` fp32 score
# matrix plus left-align gather buffers. At 128K with CP that peaks at tens of
# GiB (e.g. S_q=S_k=32768 => 4GiB scores + ~16GiB int64 gather indices). Tiling
# the query dimension bounds the peak to ``O(tile * S_k)`` without changing the
# result: the indexer forward and radix top-k are both per-query-row
# independent, so a tile's output equals the matching slice of the full run.
# Default target keeps one fp32 score tile near 256MiB (tile*S_k <= 64Mi elems).
_DEFAULT_QUERY_TILE_ELEMS = 1 << 26


def _resolve_indexer_query_tile(sq: int, sk: int) -> int:
    """Query-dim tile size bounding the dense score matrix per kernel call."""
    sq = int(sq)
    if sk <= 0:
        return sq
    tile = max(1, _DEFAULT_QUERY_TILE_ELEMS // int(sk))
    return min(tile, sq)


def _require_cudnn_frontend():
    if not is_cudnn_frontend_available():
        raise ImportError(CUDNN_FRONTEND_HINT)


def _check_cudnn_indexer_shape_support(
    index_q, index_k_comp, ratio, seq_offset=0
):
    """Guard host-side shape contracts the cuDNN indexer forward cannot honor.

    The cuDNN CSA indexer forward kernel has historically crashed for short
    compressed-KV shapes: ``S_k == 1`` triggers ``cudaErrorIllegalInstruction``
    (715) rather than failing cleanly. Keep this cheap host-side assert so an
    unsupported short-sequence case fails clearly instead of poisoning the CUDA
    context. TileLang / pure-Paddle remain the recommended backends for
    standalone short sequences.

    cuDNN-frontend v1.26 no longer needs a host-side ``S_q + seq_offset <=
    S_k * ratio`` guard: the SM100 kernel clamps the ratio-causal block count to
    ``seqlen_k`` and skipped/masked positions remain ``-inf``. Tail query rows in
    non-ratio-aligned sequences are therefore valid and should be allowed.
    """
    sk = int(index_k_comp.shape[1])
    if sk < 2:
        raise ValueError(
            "cuDNN CSA indexer currently requires compressed KV length >= 2; "
            f"got S_k={sk}. Use the TileLang/Paddle indexer for short sequences."
        )


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
            f"batch mismatch: index_q={index_q.shape}, "
            f"index_k_comp={index_k_comp.shape}, weights={weights.shape}"
        )
    if seq_len != seq_len_w or heads != heads_w or dim != dim_k:
        raise ValueError(
            f"shape mismatch: index_q={index_q.shape}, "
            f"index_k_comp={index_k_comp.shape}, weights={weights.shape}"
        )
    if heads not in (32, 64):
        raise ValueError(
            f"cuDNN IndexerForward requires H_i (qhead_per_kv_head) in {{32, 64}}, got {heads}"
        )
    if dim != 128:
        raise ValueError(f"cuDNN IndexerForward requires D_i=128, got {dim}")


def _doc_lens_from_startend(startend_row_indices, sq):
    if startend_row_indices is None:
        return None
    if int(startend_row_indices.shape[0]) != 1:
        return None

    # ``startend_row_indices`` is metadata-sized. Reading it on host lets us
    # build cu_seqlens and max_seqlen values required by the cuDNN THD API.
    ends = startend_row_indices.reshape([-1]).numpy().astype("int64").tolist()
    doc_lens = []
    pos = 0
    sq = int(sq)
    while pos < sq:
        end = int(ends[pos])
        if end <= pos:
            break
        end = min(end, sq)
        doc_lens.append(end - pos)
        pos = end
    return doc_lens


def _doc_lens_to_list(doc_lens):
    if doc_lens is None:
        return None
    if isinstance(doc_lens, paddle.Tensor):
        return doc_lens.reshape([-1]).numpy().astype("int64").tolist()
    return [int(length) for length in doc_lens]


def _make_cu_seqlens(lengths, place):
    offsets = [0]
    acc = 0
    for length in lengths:
        acc += int(length)
        offsets.append(acc)
    return paddle.to_tensor(offsets, dtype="int32", place=place)


def _cudnn_indexer_topk_fwd_docmask_thd(
    index_q,
    index_k_comp,
    weights,
    ratio,
    topk_effective,
    sm_scale,
    valid_range,
    startend_row_indices,
    doc_lens=None,
    return_topk_scores=False,
):
    """Docmask fast path: run cuDNN forward in THD/varlen local-K space."""
    batch = int(index_q.shape[0])
    sq = int(index_q.shape[1])
    if batch != 1:
        return None
    if int(startend_row_indices.reshape([-1]).shape[0]) != sq:
        return None

    doc_lens = _doc_lens_to_list(doc_lens)
    if doc_lens is None:
        doc_lens = _doc_lens_from_startend(startend_row_indices, sq)
    if not doc_lens:
        return None
    if sum(int(length) for length in doc_lens) > sq:
        return None

    # The cuDNN ratio-causal forward requires S_q <= S_k * ratio for each THD
    # segment. PaddleFleet docmask semantics still keep the tail query rows of
    # a non-ratio-aligned document valid; those rows may attend to the last real
    # compressed block. Dropping them would change top-k semantics, so use the
    # packed-global fallback unless every document is ratio-aligned.
    if any(int(length) % int(ratio) != 0 for length in doc_lens):
        return None

    q_lens = [int(length) for length in doc_lens]
    comp_lens = [int(length) // int(ratio) for length in doc_lens]
    if any(int(length) < 2 for length in comp_lens):
        return None

    total_q = sum(q_lens)
    total_k = sum(comp_lens)
    max_q = max(q_lens) if q_lens else 0
    max_k = max(comp_lens) if comp_lens else 0
    if total_q <= 0 or total_k <= 0 or max_k < 2:
        return None
    if total_k > int(index_k_comp.shape[1]):
        raise ValueError(
            f"docmask compressed length mismatch: documents need {total_k}, "
            f"but index_k_comp has {int(index_k_comp.shape[1])} rows"
        )

    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_forward.api import (
        indexer_forward_wrapper,
    )
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_top_k.api import (
        indexer_top_k_wrapper,
    )

    from .docmask_utils import topk_local_to_global, valid_range_to_counts

    q_parts = []
    w_parts = []
    vr_parts = []
    q_pos = 0
    for doc_len in doc_lens:
        q_parts.append(index_q[0, q_pos : q_pos + int(doc_len)])
        w_parts.append(weights[0, q_pos : q_pos + int(doc_len)])
        vr_parts.append(valid_range[0, q_pos : q_pos + int(doc_len)])
        q_pos += int(doc_len)
    if not q_parts:
        return None

    q_thd = paddle.concat(q_parts, axis=0).contiguous()
    k_thd = index_k_comp[0, :total_k].unsqueeze(1).contiguous()
    w_thd = paddle.concat(w_parts, axis=0).contiguous()
    cu_q = _make_cu_seqlens(q_lens, index_q.place)
    cu_k = _make_cu_seqlens(comp_lens, index_q.place)

    scores = indexer_forward_wrapper(
        q_thd,
        k_thd,
        w_thd,
        ratio=int(ratio),
        sm_scale=float(sm_scale),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=int(max_q),
        max_seqlen_k=int(max_k),
    )["scores"]

    topk = int(topk_effective)
    topk_k = min(topk, int(max_k))
    vr_thd = paddle.concat(vr_parts, axis=0)
    counts = valid_range_to_counts(vr_thd)
    result = indexer_top_k_wrapper(
        scores.reshape([total_q, int(max_k)]).contiguous(),
        counts.cast("int32"),
        top_k=topk_k,
        next_n=1,
        return_val=False,
    )
    topk_local = result["indices"].reshape([total_q, topk_k]).cast("int32")
    valid_local = (topk_local >= 0) & (
        topk_local < counts.reshape([total_q, 1])
    )
    topk_local = paddle.where(
        valid_local, topk_local, paddle.full_like(topk_local, -1)
    )

    topk_scores = None
    if return_topk_scores:
        invalid_mask = topk_local < 0
        safe_local = paddle.where(
            invalid_mask, paddle.zeros_like(topk_local), topk_local
        )
        topk_scores = paddle.take_along_axis(
            scores, safe_local.cast("int64"), axis=1
        )
        topk_scores = paddle.where(
            invalid_mask,
            paddle.full_like(topk_scores, float("-inf")),
            topk_scores,
        )

    topk_global = topk_local_to_global(topk_local, vr_thd)
    if topk_k < topk:
        pad_shape = [total_q, topk - topk_k]
        topk_global = paddle.concat(
            [topk_global, paddle.full(pad_shape, -1, dtype="int32")], axis=-1
        )
        if return_topk_scores:
            topk_scores = paddle.concat(
                [
                    topk_scores,
                    paddle.full(pad_shape, float("-inf"), dtype="float32"),
                ],
                axis=-1,
            )

    if total_q < sq:
        full_topk = paddle.full([sq, topk], -1, dtype="int32")
        full_scores = None
        if return_topk_scores:
            full_scores = paddle.full(
                [sq, topk], float("-inf"), dtype="float32"
            )
        src_pos = 0
        dst_pos = 0
        for doc_len in doc_lens:
            full_topk[dst_pos : dst_pos + int(doc_len)] = topk_global[
                src_pos : src_pos + int(doc_len)
            ]
            if return_topk_scores:
                full_scores[dst_pos : dst_pos + int(doc_len)] = topk_scores[
                    src_pos : src_pos + int(doc_len)
                ]
            src_pos += int(doc_len)
            dst_pos += int(doc_len)
        topk_global = full_topk
        if return_topk_scores:
            topk_scores = full_scores

    topk_global = topk_global.reshape([1, sq, topk])
    topk_length = (topk_global >= 0).sum(axis=-1).cast("int32")
    if return_topk_scores:
        return topk_global, topk_length, topk_scores.reshape([1, sq, topk])
    return topk_global, topk_length


def cudnn_indexer_forward(
    index_q, index_k_comp, weights, ratio=4, sm_scale=None, seq_offset=0
):
    """Compute indexer scores using cuDNN CuTe-DSL kernel (SM100).

    Args:
        index_q:       [B, S_q, H_i, D_i] bf16, indexer queries.
        index_k_comp:  [B, S_k, D_i] bf16, compressed indexer keys.
        weights:       [B, S_q, H_i] bf16, per-head weights.
        ratio:         compression ratio for the causal mask.
        sm_scale:      scale factor applied to QK scores (default: dim**-0.5).

    Returns:
        scores: [B, S_q, S_k] fp32 Paddle tensor. Masked positions are -inf.
    """
    seq_offset = int(seq_offset)
    if seq_offset < 0:
        raise ValueError(f"seq_offset must be >= 0, got {seq_offset}")
    _check_cudnn_indexer_shape_support(
        index_q, index_k_comp, ratio, seq_offset=seq_offset
    )
    if sm_scale is None:
        sm_scale = float(index_q.shape[-1]) ** -0.5
    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_forward.api import (
        indexer_forward_wrapper,
    )

    q_causal_offsets = None
    if seq_offset != 0:
        q_causal_offsets = paddle.full(
            [int(index_q.shape[0])],
            seq_offset,
            dtype="int32",
        )

    result = indexer_forward_wrapper(
        index_q.contiguous(),
        index_k_comp.unsqueeze(2).contiguous(),
        weights.contiguous(),
        ratio=int(ratio),
        sm_scale=float(sm_scale),
        q_causal_offsets=q_causal_offsets,
    )
    return result["scores"]


def cudnn_indexer_topk(scores, sq, ratio, topk, valid_range=None, seq_offset=0):
    """Select top-K indices using cuDNN TRT-LLM radix kernel (SM100).

    Args:
        scores:  [B, S_q, S_k] fp32 Paddle tensor.
        sq:      query sequence length.
        ratio:   compression ratio.
        topk:    number of entries to select per query position.
        valid_range: optional [B, S_q, 2] int32 per-query left-closed
            compressed-KV range ``[valid_start, valid_end)`` for document-mask
            (packed multi-document) training. ``None`` => causal-only mode
            (legacy single-document behavior, byte-for-byte unchanged).

    Returns:
        topk_indices: [B, S_q, topk] int32 **global** compressed-buffer ids,
            invalid slots are -1.
        topk_length:  [B, S_q] int32, per-row valid count.
    """
    batch = int(scores.shape[0])
    sk = int(scores.shape[2])
    sq = int(sq)
    topk = int(topk)
    seq_offset = int(seq_offset)
    topk_k = min(topk, sk)

    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_top_k.api import (
        indexer_top_k_wrapper,
    )

    if valid_range is None:
        # Causal-only (single-document): the radix kernel's per-row prefix
        # length is exactly the ratio-causal limit. No id remap needed —
        # local == global because there is a single compressed buffer.
        q_idx = paddle.arange(seq_offset, seq_offset + sq, dtype="int32")
        seq_lens = paddle.clip((q_idx + 1) // int(ratio), max=sk).tile([batch])
        scores_for_topk = scores
        valid_range_for_remap = None
    else:
        # Document-mask: the valid window [valid_start, valid_end) is an
        # arbitrary sub-interval, but the radix kernel only honors prefixes
        # [0, seq_lens). Left-align each query's window to [0, count), run
        # top-k in that local space, then map the selected local ids back to
        # global compressed-buffer ids by adding valid_start.
        from .docmask_utils import (
            shift_scores_to_local_window,
            topk_local_to_global,
        )

        if valid_range.shape[0] != batch or valid_range.shape[1] != sq:
            raise ValueError(
                f"valid_range must have shape [{batch}, {sq}, 2], got "
                f"{list(valid_range.shape)}"
            )
        scores_for_topk, counts = shift_scores_to_local_window(
            scores, valid_range
        )
        seq_lens = counts.reshape([batch * sq]).cast("int32")
        valid_range_for_remap = valid_range

    result = indexer_top_k_wrapper(
        scores_for_topk.reshape([batch * sq, sk]).contiguous(),
        seq_lens,
        top_k=topk_k,
        next_n=1,
        return_val=False,
    )
    topk_indices = result["indices"].reshape([batch, sq, topk_k]).cast("int32")

    if valid_range_for_remap is not None:
        # local (per-document, [0, count)) -> global; -1 slots preserved.
        topk_indices = topk_local_to_global(topk_indices, valid_range_for_remap)

    if topk_k < topk:
        padding = paddle.full([batch, sq, topk - topk_k], -1, dtype="int32")
        topk_indices = paddle.concat([topk_indices, padding], axis=-1)

    topk_length = (topk_indices >= 0).sum(axis=-1).cast("int32")
    return topk_indices, topk_length


def cudnn_indexer_topk_fwd(
    index_q,
    index_k_comp,
    weights,
    ratio=4,
    topk_effective=64,
    indexer_softmax_scale=1.0,
    valid_range=None,
    startend_row_indices=None,
    doc_lens=None,
    seq_offset=0,
    return_topk_scores=False,
):
    """Run cuDNN-frontend DSA indexer forward on Paddle tensors.

    Args:
        index_q:                [B, S, H_i, D_i] bf16, indexer queries.
        index_k_comp:           [B, S_comp, D_i] bf16, compressed indexer keys.
        weights:                [B, S, H_i] bf16, per-head weights.
        ratio:                  compression ratio (e.g. 4).
        topk_effective:         number of entries to select per query position.
        indexer_softmax_scale:  additional scale on weights.
        valid_range:            optional [B, S, 2] int32 per-query left-closed
            compressed-KV range for document-mask (packed multi-document)
            training. ``None`` => causal-only single-document mode (unchanged).

        startend_row_indices: optional [1, S, 1] doc end metadata. When present
            with ``valid_range`` and ``seq_offset == 0``, the docmask path uses
            cuDNN THD/varlen forward so score computation is document-local
            instead of packed-global. CP docmask uses the packed-global fallback
            because local query slices do not match the global docmask length.
        doc_lens: optional precomputed document lengths from reusable CSA
            docmask metadata. When supplied, the THD docmask path reuses it
            instead of reparsing ``startend_row_indices``.
        seq_offset: global query position offset for CP causal-only mode.
        return_topk_scores: return selected raw scores as a third output. This
            avoids gathering from a packed-global score tensor on the THD path.

    Returns:
        topk_indices: [B, S, topk_effective] int32 global compressed-buffer ids,
            invalid slots are -1.
        topk_length:  [B, S] int32, per-row valid count.
        topk_scores:  optional [B, S, topk_effective] fp32 selected scores.
    """
    return _cudnn_indexer_topk_fwd_impl(
        index_q,
        index_k_comp,
        weights,
        ratio=ratio,
        topk_effective=topk_effective,
        indexer_softmax_scale=indexer_softmax_scale,
        valid_range=valid_range,
        startend_row_indices=startend_row_indices,
        doc_lens=doc_lens,
        seq_offset=seq_offset,
        return_topk_scores=return_topk_scores,
    )


def _cudnn_indexer_topk_fwd_impl(
    index_q,
    index_k_comp,
    weights,
    ratio=4,
    topk_effective=64,
    indexer_softmax_scale=1.0,
    valid_range=None,
    startend_row_indices=None,
    doc_lens=None,
    seq_offset=0,
    return_topk_scores=False,
):
    _validate_indexer_inputs(index_q, index_k_comp, weights)
    if int(topk_effective) <= 0:
        raise ValueError(
            f"topk_effective must be positive, got {topk_effective}"
        )
    seq_offset = int(seq_offset)
    if seq_offset < 0:
        raise ValueError(f"seq_offset must be >= 0, got {seq_offset}")

    # sm_scale combines base dim**-0.5 with any additional indexer_softmax_scale
    _sm = float(index_q.shape[-1]) ** -0.5
    if float(indexer_softmax_scale) != 1.0:
        _sm = _sm * float(indexer_softmax_scale)

    if (
        seq_offset == 0
        and valid_range is not None
        and startend_row_indices is not None
    ):
        thd_result = _cudnn_indexer_topk_fwd_docmask_thd(
            index_q,
            index_k_comp,
            weights,
            ratio,
            topk_effective,
            _sm,
            valid_range,
            startend_row_indices,
            doc_lens=doc_lens,
            return_topk_scores=return_topk_scores,
        )
        if thd_result is not None:
            return thd_result

    sq_total = int(index_q.shape[1])
    sk = int(index_k_comp.shape[1])
    query_tile = _resolve_indexer_query_tile(sq_total, sk)

    if query_tile >= sq_total:
        return _dense_indexer_topk_single(
            index_q,
            index_k_comp,
            weights,
            ratio,
            topk_effective,
            _sm,
            valid_range,
            seq_offset,
            return_topk_scores,
        )

    # Tile the query dimension: each tile's forward + top-k is independent of
    # the others, so concatenating tile outputs reproduces the single-shot
    # result byte-for-byte while capping peak memory at ``O(query_tile * S_k)``.
    idx_parts = []
    len_parts = []
    score_parts = [] if return_topk_scores else None
    for start in range(0, sq_total, query_tile):
        end = min(start + query_tile, sq_total)
        vr_chunk = None if valid_range is None else valid_range[:, start:end]
        chunk = _dense_indexer_topk_single(
            index_q[:, start:end],
            index_k_comp,
            weights[:, start:end],
            ratio,
            topk_effective,
            _sm,
            vr_chunk,
            seq_offset + start,
            return_topk_scores,
        )
        if return_topk_scores:
            idx_chunk, len_chunk, score_chunk = chunk
            score_parts.append(score_chunk)
        else:
            idx_chunk, len_chunk = chunk
        idx_parts.append(idx_chunk)
        len_parts.append(len_chunk)

    topk_indices = paddle.concat(idx_parts, axis=1)
    topk_length = paddle.concat(len_parts, axis=1)
    if return_topk_scores:
        return topk_indices, topk_length, paddle.concat(score_parts, axis=1)
    return topk_indices, topk_length


def _dense_indexer_topk_single(
    index_q,
    index_k_comp,
    weights,
    ratio,
    topk_effective,
    sm_scale,
    valid_range,
    seq_offset,
    return_topk_scores,
):
    """Single-shot packed-global forward + top-k over the full query slice."""
    scores = cudnn_indexer_forward(
        index_q,
        index_k_comp,
        weights,
        ratio=ratio,
        sm_scale=sm_scale,
        seq_offset=seq_offset,
    )
    topk_indices, topk_length = cudnn_indexer_topk(
        scores,
        int(index_q.shape[1]),
        ratio,
        topk_effective,
        valid_range=valid_range,
        seq_offset=seq_offset,
    )
    if not return_topk_scores:
        return topk_indices, topk_length

    invalid_mask = topk_indices < 0
    safe_indices = paddle.where(
        invalid_mask, paddle.zeros_like(topk_indices), topk_indices
    )
    topk_scores = paddle.take_along_axis(
        scores, safe_indices.cast("int64"), axis=2
    )
    topk_scores = paddle.where(
        invalid_mask,
        paddle.full_like(topk_scores, float("-inf")),
        topk_scores,
    )
    return topk_indices, topk_length, topk_scores
