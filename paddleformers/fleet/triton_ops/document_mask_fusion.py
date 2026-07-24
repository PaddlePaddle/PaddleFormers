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

"""
Triton document-mask forward kernel.
"""

import paddle
from paddle import Tensor

from .utils import is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@triton.jit
def _max_combine(a, b):
    """Associative combine op for cumulative max."""
    return tl.maximum(a, b)


@triton.jit
def document_mask_fwd_kernel(
    End_ptr,  # start_end_row_indices [seq_len], exclusive end per position
    DocStart_ptr,  # output doc_start_per_pos [seq_len]
    DocLen_ptr,  # output doc_len_per_pos [seq_len]
    PosInDoc_ptr,  # output pos_in_doc [seq_len]
    seq_len,
    BLOCK_N: tl.constexpr,  # power of 2 block along seq_len
):
    """Single program, streams the 1-D row in BLOCK_N chunks."""
    # running doc-start carried across blocks; real starts are >= 0, so -1 is a
    # safe "-inf" for the cumulative max.
    running_start = tl.full((), -1, tl.int32)

    for start in range(0, seq_len, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < seq_len

        end = tl.load(End_ptr + cols, mask=mask, other=0).to(tl.int32)

        # previous position's end value; read straight from global memory so the
        # first lane of each block correctly picks up the previous block's tail.
        prev_cols = cols - 1
        prev_mask = (prev_cols >= 0) & mask
        end_prev = tl.load(End_ptr + prev_cols, mask=prev_mask, other=-1).to(
            tl.int32
        )

        # a document boundary starts wherever the end value changes; position 0
        # (end_prev == -1) is always a boundary.
        is_boundary = end != end_prev

        # candidate start = this column index at a boundary, else -1; masked
        # lanes are also -1 so they never win the cumulative max.
        cand = tl.where(is_boundary & mask, cols.to(tl.int32), -1)

        # inclusive cumulative max within the block, then fold in prior blocks.
        cm = tl.associative_scan(cand, 0, _max_combine)
        doc_start = tl.maximum(cm, running_start)

        pos_in_doc = cols.to(tl.int32) - doc_start
        doc_len = end - doc_start

        tl.store(
            DocStart_ptr + cols,
            doc_start.to(DocStart_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            DocLen_ptr + cols,
            doc_len.to(DocLen_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            PosInDoc_ptr + cols,
            pos_in_doc.to(PosInDoc_ptr.dtype.element_ty),
            mask=mask,
        )

        # carry the largest doc-start seen so far into the next block.
        running_start = tl.max(doc_start, axis=0)


def document_mask_triton(start_end_row_indices: Tensor):
    """Compute (doc_start_per_pos, doc_len_per_pos, pos_in_doc) for a 1-D input.

    Args:
        start_end_row_indices: 1-D tensor [seq_len]; each entry is the exclusive
            end index of the document that position belongs to (non-decreasing).

    Returns:
        tuple of three int64 tensors, each of shape [seq_len].
    """
    assert start_end_row_indices.ndim == 1, "expect 1-D input [seq_len]"

    x = start_end_row_indices.astype("int32")
    if not x.is_contiguous():
        x = x.contiguous()

    (seq_len,) = x.shape
    doc_start = paddle.empty([seq_len], dtype="int64")
    doc_len = paddle.empty([seq_len], dtype="int64")
    pos_in_doc = paddle.empty([seq_len], dtype="int64")

    BLOCK_N = 4096
    grid = (1,)
    document_mask_fwd_kernel[grid](
        x,
        doc_start,
        doc_len,
        pos_in_doc,
        seq_len,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return doc_start, doc_len, pos_in_doc


@triton.jit
def cutoff_compact_kernel(
    PosInDoc_ptr,  # pos_in_doc [seq_len]
    DocLen_ptr,  # doc_len_per_pos [seq_len]
    GatherIdx_ptr,  # output cutoff_gather_indices [seq_len]
    CutoffPos_ptr,  # output cutoff_pos_in_doc [seq_len]
    NCompressed_ptr,  # output n_compressed [1], number of kept tokens
    IsFirst_ptr,  # output compressed_is_first [seq_len // ratio]
    CompressedPos_ptr,  # output compressed_pos_in_doc [seq_len // ratio]
    seq_len,
    ratio,  # compression ratio (runtime scalar, need NOT be a power of 2)
    pad_value: tl.constexpr,  # value written to trailing padding slots
    BLOCK_N: tl.constexpr,  # power of 2 block along seq_len
):
    """
    KV-compression compaction.

    Each document keeps only its first ``doc_len // ratio * ratio`` tokens
    (the trailing ``doc_len % ratio`` tokens are dropped); the survivors from
    all documents are packed contiguously toward the front. For every kept
    token in order, its source column index is written to
    ``cutoff_gather_indices`` and its ``pos_in_doc`` to ``cutoff_pos_in_doc``.
    Trailing slots (>= number of kept tokens) are filled with ``pad_value``.
    The total kept count is written to ``n_compressed`` [1].

    Also emits ``compressed_is_first`` of length ``seq_len // ratio``:
    entry ``g`` is True iff compressed group ``g`` (compacted tokens
    ``[g*ratio, (g+1)*ratio)``) starts a document, i.e. its first compacted
    token has ``cutoff_pos_in_doc == 0``. Groups beyond the actual kept count
    (``g >= n_compressed // ratio``) are padding and set to False.

    And ``compressed_pos_in_doc`` of length ``seq_len // ratio``: entry ``g``
    holds the ``cutoff_pos_in_doc`` of compressed group ``g``'s first compacted
    token (i.e. ``cutoff_pos_in_doc[g*ratio]``). Padding groups are set to -1.

    One program per row, streaming in BLOCK_N chunks with a running kept-count
    carried across blocks.
    """
    # number of kept tokens before the current block (exclusive prefix base)
    running_count = tl.zeros((), tl.int32)

    for start in range(0, seq_len, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < seq_len

        pos = tl.load(PosInDoc_ptr + cols, mask=mask, other=0).to(tl.int32)
        dlen = tl.load(DocLen_ptr + cols, mask=mask, other=0).to(tl.int32)

        # cutoff length = largest multiple of ratio not exceeding doc_len.
        cutoff = dlen - (dlen % ratio)
        keep = (pos < cutoff) & mask
        m = keep.to(tl.int32)

        # exclusive prefix count within the row = destination slot in compact out
        incl = tl.cumsum(m, axis=0)
        dest = running_count + incl - m

        tl.store(
            GatherIdx_ptr + dest,
            cols.to(GatherIdx_ptr.dtype.element_ty),
            mask=keep,
        )
        tl.store(
            CutoffPos_ptr + dest,
            pos.to(CutoffPos_ptr.dtype.element_ty),
            mask=keep,
        )

        running_count += tl.sum(m, axis=0)

    # total number of kept tokens across the whole row.
    tl.store(
        NCompressed_ptr + tl.arange(0, 1),
        running_count.to(NCompressed_ptr.dtype.element_ty),
    )

    # fill the trailing padding [running_count, seq_len) in the same launch.
    # scattered slots [0, running_count) and pad slots are disjoint, so every
    # output position is written exactly once (safe on an uninitialized buffer).
    pad_g = tl.full([BLOCK_N], pad_value, GatherIdx_ptr.dtype.element_ty)
    pad_c = tl.full([BLOCK_N], pad_value, CutoffPos_ptr.dtype.element_ty)
    for start in range(running_count, seq_len, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        pad_mask = cols < seq_len
        tl.store(GatherIdx_ptr + cols, pad_g, mask=pad_mask)
        tl.store(CutoffPos_ptr + cols, pad_c, mask=pad_mask)

    # ---- compressed_is_first (reads back CutoffPos written above) ----
    # single-CTA launch (grid=(1,)): the barrier makes the CutoffPos scatter
    # visible to the group loads below.
    tl.debug_barrier()

    # n_compressed is divisible by ratio (each doc's cutoff len is), so valid
    # groups exactly tile [0, running_count).
    n_valid = running_count // ratio
    n_groups = seq_len // ratio
    for gstart in range(0, n_groups, BLOCK_N):
        gids = gstart + tl.arange(0, BLOCK_N)
        gmask = gids < n_groups
        valid = gids < n_valid

        tok = gids * ratio  # compacted index at each group start
        gpos = tl.load(CutoffPos_ptr + tok, mask=valid, other=0).to(tl.int32)
        # valid groups: first iff group-start token has pos_in_doc==0;
        # padding groups (>= n_valid): forced False.
        is_first = tl.where(valid, gpos == 0, False)
        # compressed_pos_in_doc: group-start token's pos_in_doc; padding -> -1.
        compressed_pos = tl.where(valid, gpos, -1)

        tl.store(
            IsFirst_ptr + gids,
            is_first.to(IsFirst_ptr.dtype.element_ty),
            mask=gmask,
        )
        tl.store(
            CompressedPos_ptr + gids,
            compressed_pos.to(CompressedPos_ptr.dtype.element_ty),
            mask=gmask,
        )


def cutoff_compact_triton(
    pos_in_doc: Tensor, doc_len_per_pos: Tensor, ratio: int, pad_value: int = -1
):
    """Compact KV tokens after dropping each document's non-ratio-divisible tail.

    Args:
        pos_in_doc: 1-D int tensor [seq_len], offset of each position in its doc.
        doc_len_per_pos: 1-D int tensor [seq_len], length of the doc per position.
        ratio: compression ratio (int, need NOT be a power of 2).
        pad_value: fill value for the trailing (dropped) slots, default -1.

    Both inputs are typically produced by ``document_mask_triton``.

    Returns:
        (cutoff_gather_indices, cutoff_pos_in_doc, n_compressed, compressed_is_first, compressed_pos_in_doc):
        - cutoff_gather_indices, cutoff_pos_in_doc: int64 tensors [seq_len];
          the first ``num_kept`` entries hold the compacted source column
          indices / pos_in_doc, the rest are ``pad_value``.
        - n_compressed: int64 tensor [1] holding ``num_kept`` (on device).
        - compressed_is_first: bool tensor [seq_len // ratio]; entry
          ``g`` marks whether compressed group ``g`` starts a document. Groups
          beyond ``num_kept // ratio`` are padding and set to False.
        - compressed_pos_in_doc: int64 tensor [seq_len // ratio]; entry ``g``
          holds the ``cutoff_pos_in_doc`` of group ``g``'s first compacted
          token. Padding groups are set to -1.
    """
    assert pos_in_doc.ndim == 1 and doc_len_per_pos.ndim == 1, (
        "expect 1-D inputs"
    )
    assert pos_in_doc.shape == doc_len_per_pos.shape, "inputs must share shape"
    assert ratio >= 1, "ratio must be >= 1"

    pos = pos_in_doc.astype("int32")
    dlen = doc_len_per_pos.astype("int32")
    if not pos.is_contiguous():
        pos = pos.contiguous()
    if not dlen.is_contiguous():
        dlen = dlen.contiguous()

    (seq_len,) = pos.shape
    ratio = int(ratio)
    n_groups = seq_len // ratio

    gather_idx = paddle.empty([seq_len], dtype="int64")
    cutoff_pos = paddle.empty([seq_len], dtype="int64")
    n_cutoff = paddle.empty([1], dtype="int64")
    is_first = paddle.empty([n_groups], dtype="bool")
    compressed_pos_in_doc = paddle.empty([n_groups], dtype="int64")

    BLOCK_N = 4096
    grid = (1,)
    cutoff_compact_kernel[grid](
        pos,
        dlen,
        gather_idx,
        cutoff_pos,
        n_cutoff,
        is_first,
        compressed_pos_in_doc,
        seq_len,
        ratio,
        pad_value=pad_value,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return gather_idx, cutoff_pos, n_cutoff, is_first, compressed_pos_in_doc


@triton.jit
def window_topk_idxs_kernel(
    DocStart_ptr,  # doc_start_per_pos [seqlen]
    DocLen_ptr,  # doc_len_per_pos [seqlen]
    Out_ptr,  # output [seqlen, window_size]
    seqlen,
    window_size,
    BLOCK_W: tl.constexpr,  # power of 2 block along the window axis
):
    """
    Per-position sliding-window index table (batch_size == 1).

    For position ``i`` the window starts at ``max(doc_start[i], i - window + 1)``
    and lists ``window`` consecutive source indices. Entries beyond ``i`` (future
    tokens) or before ``doc_start[i]`` are set to -1. One program per position,
    streaming the window in BLOCK_W chunks.

    Trailing padding is supported: a position ``i`` is a valid query iff
    ``pos_in_doc[i] = i - doc_start[i] < doc_len[i]`` (equivalently ``i`` is
    inside a real document). Invalid (padding) query rows are filled with -1.

    All arithmetic stays in int32: seqlen fits comfortably, so this avoids
    mixed-width issues while the output buffer is int64.
    """
    pid = tl.program_id(0)  # position index i
    pos = pid
    doc_start = tl.load(DocStart_ptr + pid).to(tl.int32)
    doc_len = tl.load(DocLen_ptr + pid).to(tl.int32)
    # padding query: its offset within the (continued) doc reaches doc_len.
    q_pad = (pos - doc_start) >= doc_len
    win_start = tl.maximum(doc_start, pos - window_size + 1)
    row_out = pid.to(tl.int64) * window_size  # big number (e.g. 1M * 4096)

    for start in range(0, window_size, BLOCK_W):
        offs = start + tl.arange(0, BLOCK_W)
        mask = offs < window_size

        indices = win_start + offs
        invalid = (indices > pos) | (indices < doc_start) | q_pad
        result = tl.where(invalid, -1, indices)

        tl.store(
            Out_ptr + row_out + offs,
            result.to(Out_ptr.dtype.element_ty),
            mask=mask,
        )


def window_topk_idxs_triton(
    doc_start_per_pos: Tensor, doc_len_per_pos: Tensor, window_size: int
) -> Tensor:
    """Fused version of ``_build_window_topk_idxs_from_doc_bounds``.

    Assumptions baked in per request: ``batch_size == 1`` and
    ``seqlen == doc_start_per_pos.shape[0]``. Trailing padding positions
    (``pos_in_doc >= doc_len``) are treated as invalid queries and emit an
    all -1 row.

    Args:
        doc_start_per_pos: 1-D int tensor [seqlen] (e.g. from document_mask_triton).
        doc_len_per_pos: 1-D int tensor [seqlen] (e.g. from document_mask_triton),
            used to detect trailing-padding query positions.
        window_size: positive window length.

    Returns:
        int64 tensor [1, seqlen, window_size]; each valid row holds the
        sliding-window source indices for a position, with out-of-window slots
        set to -1; padding-query rows are all -1.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    assert doc_start_per_pos.ndim == 1, "expect 1-D doc_start_per_pos [seqlen]"
    assert doc_start_per_pos.shape == doc_len_per_pos.shape, (
        "inputs must share shape"
    )

    ds = doc_start_per_pos.contiguous()
    dl = doc_len_per_pos.contiguous()

    (seqlen,) = ds.shape
    out = paddle.empty([seqlen, window_size], dtype="int64")

    BLOCK_W = min(triton.next_power_of_2(window_size), 2048)
    grid = (seqlen,)
    window_topk_idxs_kernel[grid](
        ds,
        dl,
        out,
        seqlen,
        window_size,
        BLOCK_W=BLOCK_W,
        num_warps=4,
    )
    return out.unsqueeze(0)


@triton.jit
def compressed_doc_start_kernel(
    End_ptr,  # start_end_row_indices [seq_len]
    DocStart_ptr,  # doc_start_per_pos [seq_len]
    Out_ptr,  # output compressed_doc_start_per_pos [seq_len]
    seq_len,
    ratio,  # compression ratio (runtime scalar, need NOT be a power of 2)
    BLOCK_N: tl.constexpr,  # power of 2 block along seq_len
):
    """
    Per-position global compressed-kv doc offset.

    For every position, emit the number of compressed kv slots contributed by
    all documents strictly before its own document, i.e. the global index of
    the first compressed slot of its document. Document ``d`` contributes
    ``doc_len_d // ratio`` slots.

    Implementation: at each document boundary ``b_k`` (start of doc ``k``) place
    a contribution ``len_{k-1} // ratio`` (the just-finished doc's slot count);
    the inclusive prefix sum of these contributions over positions yields, for
    any position in doc ``k``, ``sum_{m<k} len_m // ratio``. Streamed in BLOCK_N
    chunks with a running sum carried across blocks.
    """
    running_sum = tl.zeros((), tl.int32)

    for start in range(0, seq_len, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < seq_len

        end = tl.load(End_ptr + cols, mask=mask, other=0).to(tl.int32)

        # previous position's end / doc_start, read from global memory so lane 0
        # of each block correctly picks up the previous block's tail.
        prev_cols = cols - 1
        prev_mask = (prev_cols >= 0) & mask
        end_prev = tl.load(End_ptr + prev_cols, mask=prev_mask, other=-1).to(
            tl.int32
        )
        ds_prev = tl.load(DocStart_ptr + prev_cols, mask=prev_mask, other=0).to(
            tl.int32
        )

        # a document boundary starts wherever the end value changes; position 0
        # (prev_cols < 0) is the first doc and contributes nothing.
        is_boundary = (end != end_prev) & (prev_cols >= 0) & mask
        # length of the just-finished doc = its exclusive end minus its start.
        prev_doc_len = end_prev - ds_prev
        contrib = tl.where(is_boundary, prev_doc_len // ratio, 0)

        incl = tl.cumsum(contrib, axis=0)
        out = running_sum + incl
        tl.store(Out_ptr + cols, out.to(Out_ptr.dtype.element_ty), mask=mask)

        running_sum += tl.sum(contrib, axis=0)


def compressed_doc_start_triton(
    start_end_row_indices: Tensor, doc_start_per_pos: Tensor, ratio: int
):
    """Per-position global compressed-kv doc offset.

    Args:
        start_end_row_indices: 1-D int tensor [seq_len]; exclusive end index of
            each position's document (non-decreasing).
        doc_start_per_pos: 1-D int tensor [seq_len] from ``document_mask_triton``.
        ratio: compression ratio (int, need NOT be a power of 2).

    Returns:
        int64 tensor [seq_len]; each entry is the global index of the first
        compressed kv slot belonging to that position's document.
    """
    assert start_end_row_indices.ndim == 1 and doc_start_per_pos.ndim == 1, (
        "expect 1-D inputs"
    )
    assert start_end_row_indices.shape == doc_start_per_pos.shape, (
        "inputs must share shape"
    )
    assert ratio >= 1, "ratio must be >= 1"

    end = start_end_row_indices.contiguous()
    ds = doc_start_per_pos.contiguous()

    (seq_len,) = end.shape
    ratio = int(ratio)
    out = paddle.empty([seq_len], dtype="int64")

    BLOCK_N = 4096
    grid = (1,)
    compressed_doc_start_kernel[grid](
        end,
        ds,
        out,
        seq_len,
        ratio,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out


@triton.jit
def compressed_topk_idxs_kernel(
    CompDocStart_ptr,  # compressed_doc_start_per_pos [seqlen]
    PosInDoc_ptr,  # pos_in_doc [seqlen]
    DocLen_ptr,  # doc_len_per_pos [seqlen]
    Out_ptr,  # output [seqlen, n_groups]
    seqlen,
    n_groups,
    ratio,  # compression ratio (runtime scalar, need NOT be power of 2)
    offset,  # constant added to every accessible slot index (-1 kept as -1)
    BLOCK_G: tl.constexpr,  # power of 2 block along the compressed-kv axis
):
    """
    Per-query compressed-kv access table (batch_size == 1).

    The last axis is aligned with ``compressed_is_first`` / ``compressed_pos_in_doc``:
    entry ``g`` names global compressed slot ``g``. For query position ``i`` the
    output holds ``g + offset`` where slot ``g`` is accessible and ``-1`` otherwise.
    ``offset`` shifts every accessible id by a constant (e.g. compressed kv are
    appended after the uncompressed kv during training); ``-1`` padding is kept.

    A slot ``g`` is accessible by ``q[i]`` iff:
      * ``q[i]`` is a valid (non-padding) query: ``pos_in_doc[i] < doc_len[i]``,
      * it belongs to ``q``'s own document, and
      * it is causally complete for ``q``: with local slot index
        ``j = g - compressed_doc_start[i]``, ``0 <= j < (pos_in_doc[i] + 1) // ratio``.
    That is, slot ``j`` covers local doc tokens ``[j*ratio, (j+1)*ratio)`` and only
    becomes visible once ``q`` has reached the last of those tokens. Trailing
    padding query rows are all -1.

    For a valid query these conditions collapse to the contiguous range
    ``[compressed_doc_start[i], compressed_doc_start[i] + (pos_in_doc[i]+1)//ratio)``.
    One program per position, streaming the compressed axis in BLOCK_G chunks.
    """
    pid = tl.program_id(0)  # query position i
    cds = tl.load(CompDocStart_ptr + pid).to(tl.int32)
    pos = tl.load(PosInDoc_ptr + pid).to(tl.int32)
    dlen = tl.load(DocLen_ptr + pid).to(tl.int32)

    # padding query (pos_in_doc >= doc_len) sees nothing.
    valid_q = pos < dlen
    n_access = (pos + 1) // ratio  # causally-complete slots in the doc
    lo = cds
    hi = cds + tl.where(valid_q, n_access, 0)
    row_out = pid.to(tl.int64) * n_groups  # big number (e.g. 128K * 32K)

    for start in range(0, n_groups, BLOCK_G):
        offs = start + tl.arange(0, BLOCK_G)
        mask = offs < n_groups

        accessible = (offs >= lo) & (offs < hi)
        result = tl.where(accessible, offs + offset, -1)

        tl.store(
            Out_ptr + row_out + offs,
            result.to(Out_ptr.dtype.element_ty),
            mask=mask,
        )


def compressed_topk_idxs_triton(
    compressed_doc_start_per_pos: Tensor,
    pos_in_doc: Tensor,
    doc_len_per_pos: Tensor,
    ratio: int,
    offset: int = 0,
) -> Tensor:
    """Build ``compressed_topk_ids`` for KV-compressed sparse attention.

    Assumptions baked in per request: ``batch_size == 1``. Trailing padding
    query positions (``pos_in_doc >= doc_len``) emit an all -1 row.

    Args:
        compressed_doc_start_per_pos: 1-D int tensor [seqlen] from
            ``compressed_doc_start_triton``.
        pos_in_doc: 1-D int tensor [seqlen] from ``document_mask_triton``.
        doc_len_per_pos: 1-D int tensor [seqlen] from ``document_mask_triton``,
            used to detect trailing-padding query positions.
        ratio: compression ratio (int, need NOT be a power of 2).
        offset: int constant added to every accessible compressed-kv id; ``-1``
            padding is preserved. Use this when the compressed kv are appended
            after the uncompressed kv (offset == number of uncompressed kv).

    Returns:
        int32 tensor [1, seqlen, seqlen // ratio]; entry ``[0, i, g]`` is
        ``g + offset`` if query ``i`` may attend to compressed slot ``g`` (valid
        query, same doc, causally complete), else ``-1``. The last axis aligns
        with ``compressed_is_first``.
    """
    assert (
        compressed_doc_start_per_pos.ndim == 1
        and pos_in_doc.ndim == 1
        and doc_len_per_pos.ndim == 1
    ), "expect 1-D inputs"
    assert (
        compressed_doc_start_per_pos.shape
        == pos_in_doc.shape
        == doc_len_per_pos.shape
    ), "inputs must share shape"
    assert ratio >= 1, "ratio must be >= 1"

    cds = compressed_doc_start_per_pos.contiguous()
    pos = pos_in_doc.contiguous()
    dlen = doc_len_per_pos.contiguous()

    (seqlen,) = cds.shape
    ratio = int(ratio)
    offset = int(offset)
    n_groups = seqlen // ratio

    out = paddle.empty([seqlen, n_groups], dtype="int32")

    BLOCK_G = min(triton.next_power_of_2(max(n_groups, 1)), 2048)
    grid = (seqlen,)
    compressed_topk_idxs_kernel[grid](
        cds,
        pos,
        dlen,
        out,
        seqlen,
        n_groups,
        ratio,
        offset,
        BLOCK_G=BLOCK_G,
        num_warps=4,
    )
    return out.unsqueeze(0)
