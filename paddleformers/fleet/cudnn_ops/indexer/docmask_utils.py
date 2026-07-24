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

"""Document-mask helpers for the cuDNN sparse CSA indexer backward.

The production sparse indexer backward (``indexer_backward_sm100`` via
``csa_indexer_bwd``) assumes single-document inputs: top-k indices address one
contiguous compressed-KV buffer and tensors are dense BSHD. Document-mask
(packed multi-document) training breaks both assumptions:

* Top-k indices may be **per-document local** (each document's selected
  compressed positions numbered from 0, then concatenated -- e.g. three docs
  giving ``[0,1,2, 0,2,3, 0,4,7]``) instead of **global** (numbered
  continuously across the whole compressed buffer). The kernel indexes K/dK by
  global flat ids, so local ids must be offset by each query's document column
  start before the kernel runs, and converted back afterwards.

* Layouts differ across the pipeline: some stages use **THD** (packed, trailing
  padding removed) while others use **BSHD with b==1** (sequence padded to a
  fixed length such as 8192). Inputs must be converted / (de)padded at the
  right boundary.

These are pure-Paddle tensor helpers (no GPU-kernel dependency) so they are unit
testable on any device. The per-query document column start is read from
``valid_range[..., 0]`` (left-closed compressed-KV range), the same tensor
produced by ``csa_attention.get_valid_range``.
"""

from __future__ import annotations

import paddle

__all__ = [
    "topk_local_to_global",
    "topk_global_to_local",
    "thd_to_bshd_b1",
    "bshd_b1_to_thd",
    "valid_range_to_counts",
    "shift_scores_to_local_window",
]


def _validate_topk_valid_range(topk_indices, valid_range):
    if not isinstance(topk_indices, paddle.Tensor):
        raise TypeError(
            f"topk_indices must be a paddle.Tensor, got {type(topk_indices)!r}"
        )
    if not isinstance(valid_range, paddle.Tensor):
        raise TypeError(
            f"valid_range must be a paddle.Tensor, got {type(valid_range)!r}"
        )
    if topk_indices.ndim < 2:
        raise ValueError(
            f"topk_indices must be at least 2D [..., topk], got shape {topk_indices.shape}"
        )
    if valid_range.ndim != topk_indices.ndim or valid_range.shape[-1] != 2:
        raise ValueError(
            f"valid_range must have shape [..., 2] matching topk_indices leading "
            f"dims; got valid_range={valid_range.shape}, topk_indices={topk_indices.shape}"
        )
    if list(valid_range.shape[:-1]) != list(topk_indices.shape[:-1]):
        raise ValueError(
            "valid_range and topk_indices must share leading dims: "
            f"valid_range={valid_range.shape}, topk_indices={topk_indices.shape}"
        )


def _shift_topk(topk_indices, valid_range, sign):
    """Shift valid top-k ids by ``sign * valid_range[..., 0]``; keep -1 slots."""
    _validate_topk_valid_range(topk_indices, valid_range)
    int_dtype = topk_indices.dtype
    # doc_col_start per query; broadcast over the top-k dim.
    offsets = valid_range[..., 0:1].cast(int_dtype)
    valid = topk_indices >= 0
    shifted = topk_indices + sign * offsets
    return paddle.where(valid, shifted, paddle.full_like(topk_indices, -1))


def topk_local_to_global(topk_indices, valid_range):
    """Per-document-local top-k ids -> global compressed-buffer ids.

    ``global = local + valid_range[..., 0]`` broadcast over the top-k dim.
    Invalid slots (``local < 0``, conventionally ``-1``) are preserved as ``-1``.

    Args:
        topk_indices: int32 ``[..., topk]`` (e.g. ``[B, S, topk]`` BSHD or
            ``[T, topk]`` THD). Per-document-local ids.
        valid_range:  int ``[..., 2]`` left-closed compressed-KV range; column
            0 is the document column start (``doc_col_start``).

    Returns:
        int32 tensor, same shape as ``topk_indices``, with global ids.
    """
    return _shift_topk(topk_indices, valid_range, sign=1)


def topk_global_to_local(topk_indices, valid_range):
    """Global compressed-buffer top-k ids -> per-document-local ids.

    Inverse of :func:`topk_local_to_global`. Preserves ``-1`` invalid slots.
    """
    return _shift_topk(topk_indices, valid_range, sign=-1)


def valid_range_to_counts(valid_range):
    """Per-query valid compressed-column count: ``valid_end - valid_start``.

    Args:
        valid_range: int ``[..., 2]`` left-closed range ``[start, end)``.

    Returns:
        int32 ``[...]`` counts (clamped to ``>= 0``).
    """
    if not isinstance(valid_range, paddle.Tensor):
        raise TypeError(
            f"valid_range must be a paddle.Tensor, got {type(valid_range)!r}"
        )
    if valid_range.ndim < 1 or valid_range.shape[-1] != 2:
        raise ValueError(
            f"valid_range must have shape [..., 2], got {valid_range.shape}"
        )
    counts = valid_range[..., 1] - valid_range[..., 0]
    return counts.clip(min=0).cast("int32")


def shift_scores_to_local_window(scores, valid_range, fill_value=float("-inf")):
    """Left-align each row's ``[valid_start, valid_end)`` window to ``[0, count)``.

    The radix top-k kernel treats ``seq_lens[i]`` as a **prefix** length: it
    only considers columns ``[0, seq_lens[i])`` of row ``i``. Document-mask
    valid ranges are arbitrary sub-intervals ``[valid_start, valid_end)``, not
    prefixes. This helper gathers each query's valid window to the front so the
    kernel can run with ``seq_lens = count``; the resulting local top-k ids are
    mapped back to global ids via :func:`topk_local_to_global`.

    Because each document's compressed columns are causally ordered and the
    document token counts are cut to multiples of ``ratio``, the valid window
    end equals the global ratio-causal limit exactly, so no kernel-side ``-inf``
    intrudes into the window.

    Args:
        scores:      ``[B, S, Sk]`` fp32 dense indexer scores.
        valid_range: ``[B, S, 2]`` int left-closed compressed-KV range.
        fill_value:  value written to padding columns ``>= count`` (never read
            by the kernel; kept ``-inf`` for safety / determinism).

    Returns:
        ``(local_scores, counts)`` where ``local_scores`` is ``[B, S, Sk]`` with
        the valid window left-aligned and the tail filled with ``fill_value``,
        and ``counts`` is ``[B, S]`` int32 valid-column counts.
    """
    if not isinstance(scores, paddle.Tensor):
        raise TypeError(f"scores must be a paddle.Tensor, got {type(scores)!r}")
    if scores.ndim != 3:
        raise ValueError(f"scores must be 3D [B, S, Sk], got {scores.shape}")
    if valid_range.ndim != 3 or valid_range.shape[-1] != 2:
        raise ValueError(
            f"valid_range must be [B, S, 2], got {valid_range.shape}"
        )
    if list(valid_range.shape[:2]) != list(scores.shape[:2]):
        raise ValueError(
            "valid_range and scores must share [B, S]: "
            f"valid_range={valid_range.shape}, scores={scores.shape}"
        )

    sk = int(scores.shape[2])
    valid_start = valid_range[..., 0:1].cast("int64")  # [B, S, 1]
    counts = valid_range_to_counts(valid_range)  # [B, S] int32
    col = paddle.arange(sk, dtype="int64").reshape([1, 1, sk])
    gather_idx = valid_start + col  # [B, S, Sk]
    in_range = (col < counts.unsqueeze(-1).cast("int64")) & (gather_idx < sk)
    safe_idx = paddle.where(in_range, gather_idx, paddle.zeros_like(gather_idx))
    local = paddle.take_along_axis(scores, safe_idx, axis=2)
    local = paddle.where(in_range, local, paddle.full_like(local, fill_value))
    return local, counts


def thd_to_bshd_b1(tensor, pad_len, pad_value=0):
    """Packed THD ``[T, ...]`` -> BSHD ``b==1`` ``[1, pad_len, ...]``.

    The leading dim is the packed sequence dim. Rows ``[0, T)`` are copied;
    rows ``[T, pad_len)`` are filled with ``pad_value``. A leading batch dim of
    size 1 is added so the result matches the dense BSHD contract the kernel
    consumes.

    Args:
        tensor:    ``[T, ...]`` packed tensor (no batch dim).
        pad_len:   target padded sequence length (``>= T``).
        pad_value: fill value for padding rows (use ``-1`` for top-k indices,
            ``0`` for scores / features).
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"tensor must be a paddle.Tensor, got {type(tensor)!r}")
    if tensor.ndim < 1:
        raise ValueError("tensor must have at least 1 dim (the packed seq dim)")
    t = int(tensor.shape[0])
    pad_len = int(pad_len)
    if pad_len < t:
        raise ValueError(
            f"pad_len ({pad_len}) must be >= packed length T ({t})"
        )
    if pad_len == t:
        out = tensor
    else:
        pad_shape = [pad_len - t, *list(tensor.shape[1:])]
        pad = paddle.full(pad_shape, pad_value, dtype=tensor.dtype)
        out = paddle.concat([tensor, pad], axis=0)
    return out.unsqueeze(0)


def bshd_b1_to_thd(tensor, total_len):
    """BSHD ``b==1`` ``[1, S_pad, ...]`` -> packed THD ``[total_len, ...]``.

    Inverse of :func:`thd_to_bshd_b1` (with ``total_len == T``): drops the
    batch dim and slices off trailing padding rows ``[total_len, S_pad)``.

    Args:
        tensor:    ``[1, S_pad, ...]`` BSHD tensor with batch size 1.
        total_len: number of valid packed rows to keep (``<= S_pad``).
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"tensor must be a paddle.Tensor, got {type(tensor)!r}")
    if tensor.ndim < 2:
        raise ValueError("tensor must be at least 2D [1, S_pad, ...]")
    if int(tensor.shape[0]) != 1:
        raise ValueError(
            f"bshd_b1_to_thd expects batch size 1, got {tensor.shape[0]}"
        )
    body = tensor[0]
    total_len = int(total_len)
    s_pad = int(body.shape[0])
    if total_len > s_pad:
        raise ValueError(
            f"total_len ({total_len}) must be <= padded length ({s_pad})"
        )
    if total_len == s_pad:
        return body
    return body[:total_len]
