# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Utilities for transformer layers."""

from __future__ import annotations

import contextlib
from functools import lru_cache

import paddle

from paddleformers.fleet.training.global_vars import get_profile_timers


@lru_cache(maxsize=32)
def get_default_causal_mask(sq: int) -> paddle.Tensor:
    """Return the causal upper triangular mask for softmax input."""
    return paddle.triu(paddle.ones(sq, sq), diagonal=1).bool()


@lru_cache(maxsize=32)
def get_sliding_window_causal_mask(sq, skv, sliding_window):
    """Create the equivalent attention mask for SWA in [sq, skv] shape"""
    m = paddle.ones(sq, skv, dtype=paddle.bool)
    mu = paddle.triu(m, diagonal=skv - sq - sliding_window[0])
    ml = paddle.tril(mu, diagonal=skv - sq + sliding_window[1])
    ml = ~ml

    return ml


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


@contextlib.contextmanager
def profile(name, use_event=True):
    """Record a timer scope when profile timers are available."""
    if not name:
        yield
        return

    timers = get_profile_timers()
    timer = timers(name, use_event=use_event) if timers is not None else None
    if timer is not None:
        timer.start()
    try:
        yield
    finally:
        if timer is not None:
            timer.stop()


def is_layer_window_attention(
    sliding_window: tuple[int, int] | None,
    window_attn_skip_freq: int | list,
    layer_number: int,
) -> bool:
    # layer_number is 0-indexed
    if not sliding_window:
        return False
    if window_attn_skip_freq is None:
        return True
    if isinstance(window_attn_skip_freq, int):
        return layer_number % window_attn_skip_freq != 0
    if isinstance(window_attn_skip_freq, list):
        return bool(window_attn_skip_freq[layer_number])

    raise ValueError(f"Invalid `window_attn_skip_freq`: {type(window_attn_skip_freq)}, " f"{window_attn_skip_freq}")


def startend_row_indices_add_sliding_window(
    startend_row_indices: paddle.Tensor,
    sliding_window: tuple[int, int] | None,
    head_wise_swa_ratio: float,
    kv_num_heads: int,
) -> paddle.Tensor:
    """
    Args:
        startend_row_indices: [bsz, heads, seq, 2].
        window_size: int or None
    """
    if not sliding_window:
        return startend_row_indices
    # construct sliding window mask
    bsz, heads, seq, num_vec = startend_row_indices.shape
    if num_vec not in [1, 2]:
        raise ValueError("only support LTS and LTS & UTE now")

    swa_head_num = int(head_wise_swa_ratio * kv_num_heads)

    if heads == 1:
        heads = kv_num_heads
        startend_row_indices = startend_row_indices.repeat([1, kv_num_heads, 1, 1])  # 扩展到多头，方便后续对每个头做不同的操作

    window_size = sliding_window[0]
    LTS_SWA = (
        paddle.arange(window_size, seq + window_size, dtype=paddle.int32).unsqueeze([0, 1]).repeat([bsz, heads, 1])
    )  # (bsz, heads, seq)
    startend_row_indices_new_LTS = paddle.where(
        startend_row_indices[..., 0] < LTS_SWA,
        startend_row_indices[..., 0],
        LTS_SWA,
    )
    if 0 < swa_head_num and swa_head_num < heads:
        # 说明有部分head只swa-head，我们把剩余的non-swa-head的mask还原回去
        non_swa_head_num = heads - swa_head_num
        startend_row_indices_new_LTS[:, :non_swa_head_num, :] = startend_row_indices[:, :non_swa_head_num, :, 0]

    if num_vec == 1:
        startend_row_indices = startend_row_indices_new_LTS.unsqueeze(-1)
    else:
        startend_row_indices = paddle.stack(
            [startend_row_indices_new_LTS, startend_row_indices[..., 1]],
            axis=-1,
        )
    return startend_row_indices


# ---------------------------------------------------------------------------
# Helper functions for document mask
# ---------------------------------------------------------------------------


def get_doc_lens(startend_row_indices: paddle.Tensor) -> paddle.Tensor:
    """Derive document lengths from startend_row_indices.

    Args:
        startend_row_indices: [batch_size, h, seqlen, 1] tensor where
            each value is the end boundary (exclusive) of the document that
            position belongs to.

    Returns:
        doc_lens: [n_docs] int32 tensor of document lengths.
    """
    mask = startend_row_indices.flatten().cast("int64")
    seqlen = mask.shape[0]
    positions = paddle.arange(seqlen, dtype="int64")

    is_boundary = paddle.zeros([seqlen], dtype="bool")
    is_boundary[0] = True
    is_boundary[1:] = (positions[1:] == mask[:-1]) & (mask[1:] != mask[:-1])

    boundary_indices = paddle.nonzero(is_boundary).flatten()
    doc_ends = mask[boundary_indices]
    doc_lens = (doc_ends - boundary_indices).cast("int32")
    return doc_lens


def get_doc_starts(doc_lens: paddle.Tensor) -> paddle.Tensor:
    """Compute document start positions from document lengths.

    Args:
        doc_lens: [n_docs] tensor of document lengths.

    Returns:
        doc_starts: [n_docs] int32 tensor of cumulative start positions.
    """
    lens = doc_lens.flatten().cast("int32")
    cum = paddle.cumsum(lens, axis=0)
    starts = paddle.zeros_like(cum)
    if cum.shape[0] > 1:
        starts[1:] = cum[:-1]
    return starts
