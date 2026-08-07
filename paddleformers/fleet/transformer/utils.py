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


def get_sliding_window_left_size(
    sliding_window: int | tuple[int, int],
) -> int:
    """Return the left (past) window size for both accepted forms.

    - int W          -> W          (HF causal one-sided semantics)
    - (left, right)  -> left       (Fleet native two-sided semantics)

    The caller decides how to treat non-positive results (`0` / `-1`), which
    denote "no finite left window" (i.e. `-1` = infinite window).
    """
    if isinstance(sliding_window, int):
        return sliding_window
    return sliding_window[0]


@lru_cache(maxsize=32)
def get_sliding_window_causal_mask(sq, skv, sliding_window):
    """Create the equivalent attention mask for SWA in [sq, skv] shape.

    sliding_window: when int, use causal one-sided semantics (left=W, right=0);
                    when tuple, use native (left, right) two-sided semantics.
    A negative left window (e.g. `-1`) denotes an infinite past window, which is
    a plain causal mask (no left truncation).
    """
    m = paddle.ones(sq, skv, dtype=paddle.bool)
    if isinstance(sliding_window, int):
        left, right = sliding_window, 0
    else:
        left, right = sliding_window[0], sliding_window[1]
    # left < 0 => infinite window: skip the upper-triangular (left) truncation
    # so every query can attend to all past keys, yielding a causal mask.
    mu = m if left < 0 else paddle.triu(m, diagonal=skv - sq - left)
    ml = paddle.tril(mu, diagonal=skv - sq + right)
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
    sliding_window: int | tuple[int, int] | None,
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

    raise ValueError(
        f"Invalid `window_attn_skip_freq`: {type(window_attn_skip_freq)}, "
        f"{window_attn_skip_freq}"
    )


def startend_row_indices_add_sliding_window(
    startend_row_indices: paddle.Tensor,
    sliding_window: int | tuple[int, int] | None,
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
    window_size = get_sliding_window_left_size(sliding_window)
    if window_size <= 0:
        # -1 (infinite window) / 0 => no sliding-window truncation.
        return startend_row_indices
    # construct sliding window mask
    bsz, heads, seq, num_vec = startend_row_indices.shape
    if num_vec not in [1, 2]:
        raise ValueError("only support LTS and LTS & UTE now")

    swa_head_num = int(head_wise_swa_ratio * kv_num_heads)

    if heads == 1:
        heads = kv_num_heads
        startend_row_indices = startend_row_indices.repeat(
            [1, kv_num_heads, 1, 1]
        )  # 扩展到多头，方便后续对每个头做不同的操作

    LTS_SWA = (
        paddle.arange(window_size, seq + window_size, dtype=paddle.int32)
        .unsqueeze([0, 1])
        .repeat([bsz, heads, 1])
    )  # (bsz, heads, seq)
    startend_row_indices_new_LTS = paddle.where(
        startend_row_indices[..., 0] < LTS_SWA,
        startend_row_indices[..., 0],
        LTS_SWA,
    )
    if 0 < swa_head_num and swa_head_num < heads:
        # 说明有部分head只swa-head，我们把剩余的non-swa-head的mask还原回去
        non_swa_head_num = heads - swa_head_num
        startend_row_indices_new_LTS[:, :non_swa_head_num, :] = (
            startend_row_indices[:, :non_swa_head_num, :, 0]
        )

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


def inspect_tensor(tag, layer_idx, tensor, save=False, load=True):
    """Inspect tensor info, optionally save to .npy and/or load override from .npy.

    Controlled by environment variables:
        ABLATION_INSPECT_TENSOR: "1" to enable printing tensor info.
        ABLATION_SAVE_TENSOR_PATH: path to save .npy files (layer 0 only, first call).
        ABLATION_LOAD_TENSOR_PATH: path to directory with saved .npy files (layer 0 only).
        ABLATION_INFO_SKIP_TAGS: comma-separated tags to skip for info printing.
        ABLATION_DUMP_SKIP_TAGS: comma-separated tags to skip for tensor saving/loading.

    Args:
        tag: identifier for the tensor checkpoint.
        layer_idx: transformer layer index.
        tensor: the live paddle tensor.
        save: if True, attempt to save the tensor to .npy file. Default is False.
        load: if True, attempt to load and override the tensor from disk (layer 0 only).
              Default is True.

    Returns:
        The original tensor (if no load or no file found), or the loaded tensor.
    """
    import hashlib
    import os

    import numpy as np

    inspect_flag = os.environ.get("ABLATION_INSPECT_TENSOR", "0") == "1"
    ablation_save_path = os.environ.get("ABLATION_SAVE_TENSOR_PATH", "")
    ablation_load_path = os.environ.get("ABLATION_LOAD_TENSOR_PATH", "")
    skip_tags = set(
        filter(None, os.environ.get("ABLATION_INFO_SKIP_TAGS", "").split(","))
    )
    dump_skip_tags = set(
        filter(None, os.environ.get("ABLATION_DUMP_SKIP_TAGS", "").split(","))
    )

    if (not inspect_flag) or (tensor is None):
        return tensor

    rank = (
        paddle.distributed.get_rank()
        if paddle.distributed.is_initialized()
        else 0
    )

    # --- info (print) ---
    if tag not in skip_tags:
        try:
            t_f = tensor.astype("float32")
            abssum = t_f.abs().sum().item()
            md5 = hashlib.md5(t_f.numpy().tobytes()).hexdigest()
            print(
                f"[ABLATION_train] tag={tag} rank={rank} layer={layer_idx} "
                f"abssum={abssum} md5={md5} shape={list(tensor.shape)} dtype={tensor.dtype}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[ABLATION_train] tag={tag} layer={layer_idx} info_failed={e}",
                flush=True,
            )

    # --- save (dump .npy, rank_xxx/layer_xxx directory) ---
    if save and ablation_save_path and tag not in dump_skip_tags:
        rank_dir = os.path.join(ablation_save_path, f"rank_{rank}")
        layer_dir = os.path.join(rank_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        fpath = os.path.join(layer_dir, f"{tag}.npy")
        arr = tensor.astype("float32").numpy()
        np.save(fpath, arr)
        abssum = float(np.abs(arr).sum())
        md5 = hashlib.md5(arr.tobytes()).hexdigest()
        print(
            f"[ABLATION_dump_tensor] saved {tag} rank={rank} layer={layer_idx} shape={list(tensor.shape)} "
            f"dtype={tensor.dtype} abssum={abssum} md5={md5} -> {fpath}",
            flush=True,
        )

    # --- load (override, rank_xxx/layer_xxx directory) ---
    if load and ablation_load_path and tag not in dump_skip_tags:
        rank_dir = os.path.join(ablation_load_path, f"rank_{rank}")
        layer_dir = os.path.join(rank_dir, f"layer_{layer_idx}")
        path = os.path.join(layer_dir, f"{tag}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape != tuple(tensor.shape):
                arr = arr.reshape(tensor.shape)
            # Load as float32 first, then cast to target dtype
            # (paddle.to_tensor doesn't support float8 dtypes directly)
            loaded = paddle.to_tensor(arr, dtype="float32", place=tensor.place)
            if tensor.dtype != paddle.float32:
                loaded = loaded.astype(tensor.dtype)
            load_f32 = loaded.astype("float32")
            abssum = load_f32.abs().sum().item()
            md5 = hashlib.md5(load_f32.numpy().tobytes()).hexdigest()
            print(
                f"[ABLATION_load_tensor] loaded {tag} rank={rank} shape={list(loaded.shape)} dtype={loaded.dtype} abssum={abssum} md5={md5}",
                flush=True,
            )
            orig_f32 = tensor.astype("float32")
            diff = (orig_f32 - load_f32).abs()
            max_abs_diff = diff.max().item()
            mean_abs_diff = diff.mean().item()
            rel_diff = mean_abs_diff / (load_f32.abs().mean().item() + 1e-12)
            print(
                f"[ABLATION_load_tensor] diff {tag} max_abs_diff={max_abs_diff} mean_abs_diff={mean_abs_diff} relative_diff={rel_diff}",
                flush=True,
            )
            return loaded

    return tensor
