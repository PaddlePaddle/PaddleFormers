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

"""Collation: batch of EncodedSamples → padded numpy dict for training.

Supports:
  - Non-packed (simple padding) and packed (greedy/binpacking with
    block-diagonal attention mask) modes.
  - MTP (Multi-Token Prediction) outputs: nbatch_pack_offset,
    mtp_attn_mask / mtp_attn_mask_startend_row_indices, mtp_layer_mask.

Output format is compatible with PaddleFormers' existing Trainer.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .encode import EncodedSample
from .packing import binpack_ffd, greedy_pack


def collate_sft(
    batch: List[EncodedSample],
    pad_token_id: int,
    max_seq_len: int,
    packing: bool = False,
    packing_method: str = "greedy",
    packing_interval: int = 1000,
    use_attn_mask_startend_row_indices: bool = False,
    num_nextn_predict_layers: int = 0,
    eos_token_id: Optional[int] = None,
    use_global_causal_attn: bool = False,
) -> Dict[str, np.ndarray]:
    """Collate a batch of EncodedSamples into training-ready numpy arrays.

    Args:
        batch: List of EncodedSample from DataLoader.
        pad_token_id: Tokenizer's pad token ID for input_ids padding.
        max_seq_len: Maximum sequence length. Sequences are padded to
            min(max_seq_len, max_length_in_batch) when not packing,
            or exactly max_seq_len when packing.
        packing: If True, use packing to combine short sequences.
        packing_method: "greedy" (default) or "binpacking" (FFD).
        packing_interval: Buffer size for binpack_ffd algorithm.
        use_attn_mask_startend_row_indices: If True, output compact flashmask
            format (attn_mask_startend_row_indices) instead of 4D attention_mask.
        num_nextn_predict_layers: MTP depth D. When > 0, max_seq_len is extended
            by D and MTP outputs are generated.
        eos_token_id: Tokenizer's EOS token ID for mtp_layer_mask.
        use_global_causal_attn: If True, MTP masks use single global causal block
            instead of per-document block-causal.

    Returns:
        Dict with numpy arrays:
            - input_ids:      [B, S] int64
            - labels:         [B, S] int64, -100 for masked positions
            - position_ids:   [B, S] int64
            - attention_mask: [B, 1, S, S] float32  (when not using startend)
            - attn_mask_startend_row_indices: [B, 1, S, 1] int32  (when using startend)
        When num_nextn_predict_layers > 0, additionally:
            - nbatch_pack_offset:  [B, S] int32
            - mtp_attn_mask:       [B, D, 1, S, S] float32  (or startend variant)
            - mtp_layer_mask:      [B, D, S] int32
    """
    # Handle dict input from streaming .map() — convert to EncodedSample
    if batch and isinstance(batch[0], dict):
        batch = [
            EncodedSample(
                input_ids=item["input_ids"],
                labels=item["labels"],
                seq_len=item["seq_len"],
            )
            for item in batch
        ]

    if not batch:
        raise ValueError("collate_sft received an empty batch")

    # Effective max_seq_len (extended for MTP)
    effective_max_seq_len = max_seq_len
    if num_nextn_predict_layers > 0:
        effective_max_seq_len = max_seq_len + num_nextn_predict_layers

    if packing:
        # Select packing algorithm
        if packing_method == "binpacking":
            groups = binpack_ffd(batch, max_seq_len, packing_interval)
        else:
            groups = greedy_pack(batch, max_seq_len)
        return _collate_packed(
            groups,
            pad_token_id,
            effective_max_seq_len,
            use_attn_mask_startend_row_indices,
            num_nextn_predict_layers,
            eos_token_id,
            use_global_causal_attn,
        )
    else:
        return _collate_simple(
            batch,
            pad_token_id,
            effective_max_seq_len,
            use_attn_mask_startend_row_indices,
            num_nextn_predict_layers,
            eos_token_id,
            use_global_causal_attn,
        )


def _collate_simple(
    batch: List[EncodedSample],
    pad_token_id: int,
    max_seq_len: int,
    use_startend: bool = False,
    num_nextn_predict_layers: int = 0,
    eos_token_id: Optional[int] = None,
    use_global_causal_attn: bool = False,
) -> Dict[str, np.ndarray]:
    """Simple collation: pad each sample independently, causal mask."""
    batch_size = len(batch)
    pad_len = min(max_seq_len, max(s.seq_len for s in batch))

    input_ids = np.full((batch_size, pad_len), pad_token_id, dtype=np.int64)
    labels = np.full((batch_size, pad_len), -100, dtype=np.int64)
    position_ids = np.zeros((batch_size, pad_len), dtype=np.int64)

    for i, sample in enumerate(batch):
        seq_len = min(sample.seq_len, pad_len)
        input_ids[i, :seq_len] = sample.input_ids[:seq_len]
        labels[i, :seq_len] = sample.labels[:seq_len]
        position_ids[i, :seq_len] = np.arange(seq_len)

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
    }

    if use_startend:
        result["attn_mask_startend_row_indices"] = _build_startend_simple(batch, pad_len)
    else:
        attention_mask = np.zeros((batch_size, 1, pad_len, pad_len), dtype=np.float32)
        for i, sample in enumerate(batch):
            seq_len = min(sample.seq_len, pad_len)
            attention_mask[i, 0, :seq_len, :seq_len] = np.tril(np.ones((seq_len, seq_len)))
        result["attention_mask"] = attention_mask

    # MTP outputs (single-sequence case: each sample is one "document")
    if num_nextn_predict_layers > 0:
        all_pack_offset = []
        all_mtp_mask = []
        all_mtp_layer_mask = []
        for i, sample in enumerate(batch):
            seq_len = min(sample.seq_len, pad_len)
            seq_lens_i = [seq_len]  # single document
            all_pack_offset.append(_build_nbatch_pack_offset(seq_lens_i, pad_len))
            if use_startend:
                all_mtp_mask.append(
                    gen_mtp_attn_mask_startend_row_indices(
                        seq_lens_i, pad_len, num_nextn_predict_layers, use_global_causal_attn
                    )
                )
            else:
                all_mtp_mask.append(
                    gen_mtp_attn_mask(seq_lens_i, pad_len, num_nextn_predict_layers, use_global_causal_attn)
                )
            all_mtp_layer_mask.append(
                gen_mtp_layer_mask(input_ids[i], seq_len, pad_len, num_nextn_predict_layers, eos_token_id)
            )

        result["nbatch_pack_offset"] = np.concatenate(all_pack_offset, axis=0)
        if use_startend:
            result["mtp_attn_mask_startend_row_indices"] = np.stack(all_mtp_mask)
        else:
            result["mtp_attn_mask"] = np.stack(all_mtp_mask)
        result["mtp_layer_mask"] = np.stack(all_mtp_layer_mask)

    return result


def _collate_packed(
    groups: List[List[EncodedSample]],
    pad_token_id: int,
    max_seq_len: int,
    use_startend: bool = False,
    num_nextn_predict_layers: int = 0,
    eos_token_id: Optional[int] = None,
    use_global_causal_attn: bool = False,
) -> Dict[str, np.ndarray]:
    """Packed collation: groups already formed, build block-diagonal attention mask."""
    batch_size = len(groups)

    input_ids = np.full((batch_size, max_seq_len), pad_token_id, dtype=np.int64)
    labels = np.full((batch_size, max_seq_len), -100, dtype=np.int64)
    position_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)

    # Track sub-sequence lengths for each group
    group_seq_lens: List[List[int]] = []

    for i, group in enumerate(groups):
        offset = 0
        seq_lens = []
        for sample in group:
            seq_len = min(sample.seq_len, max_seq_len - offset)
            if seq_len <= 0:
                break

            input_ids[i, offset : offset + seq_len] = sample.input_ids[:seq_len]
            labels[i, offset : offset + seq_len] = sample.labels[:seq_len]
            position_ids[i, offset : offset + seq_len] = np.arange(seq_len)

            seq_lens.append(seq_len)
            offset += seq_len
        group_seq_lens.append(seq_lens)

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
    }

    if use_startend:
        result["attn_mask_startend_row_indices"] = _build_startend_packed(group_seq_lens, max_seq_len)
    else:
        attention_mask = np.zeros((batch_size, 1, max_seq_len, max_seq_len), dtype=np.float32)
        for i, seq_lens in enumerate(group_seq_lens):
            offset = 0
            for seq_len in seq_lens:
                causal_block = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
                attention_mask[i, 0, offset : offset + seq_len, offset : offset + seq_len] = causal_block
                offset += seq_len
        result["attention_mask"] = attention_mask

    # MTP outputs
    if num_nextn_predict_layers > 0:
        all_pack_offset = []
        all_mtp_mask = []
        all_mtp_layer_mask = []

        for i, seq_lens in enumerate(group_seq_lens):
            all_pack_offset.append(_build_nbatch_pack_offset(seq_lens, max_seq_len))

            if use_startend:
                all_mtp_mask.append(
                    gen_mtp_attn_mask_startend_row_indices(
                        seq_lens, max_seq_len, num_nextn_predict_layers, use_global_causal_attn
                    )
                )
            else:
                all_mtp_mask.append(
                    gen_mtp_attn_mask(seq_lens, max_seq_len, num_nextn_predict_layers, use_global_causal_attn)
                )

            total_len = sum(seq_lens)
            all_mtp_layer_mask.append(
                gen_mtp_layer_mask(input_ids[i], total_len, max_seq_len, num_nextn_predict_layers, eos_token_id)
            )

        result["nbatch_pack_offset"] = np.concatenate(all_pack_offset, axis=0)
        if use_startend:
            result["mtp_attn_mask_startend_row_indices"] = np.stack(all_mtp_mask)
        else:
            result["mtp_attn_mask"] = np.stack(all_mtp_mask)
        result["mtp_layer_mask"] = np.stack(all_mtp_layer_mask)

    return result


# ---------------------------------------------------------------------------
# Attention mask helpers
# ---------------------------------------------------------------------------


def _build_startend_simple(
    batch: List[EncodedSample],
    pad_len: int,
) -> np.ndarray:
    """Build attn_mask_startend_row_indices for non-packed batch.

    Each sample is a single sequence: all tokens attend to [0, seq_len).
    Padding tokens get identity (each attends only to itself).

    Returns: [B, 1, S, 1] int32
    """
    batch_size = len(batch)
    indices = np.zeros((batch_size, 1, pad_len, 1), dtype=np.int32)
    for i, sample in enumerate(batch):
        seq_len = min(sample.seq_len, pad_len)
        indices[i, 0, :seq_len, 0] = seq_len
        indices[i, 0, seq_len:, 0] = np.arange(seq_len, pad_len)
    return indices


def _build_startend_packed(
    group_seq_lens: List[List[int]],
    max_seq_len: int,
) -> np.ndarray:
    """Build attn_mask_startend_row_indices for packed batch.

    For each group, each sub-sequence of length L at offset O:
    all L positions get value O+L (meaning "can attend up to position O+L-1").
    Padding positions get identity (self-attend only).

    Returns: [B, 1, S, 1] int32
    """
    batch_size = len(group_seq_lens)
    indices = np.zeros((batch_size, 1, max_seq_len, 1), dtype=np.int32)
    for i, seq_lens in enumerate(group_seq_lens):
        offset = 0
        for seq_len in seq_lens:
            indices[i, 0, offset : offset + seq_len, 0] = offset + seq_len
            offset += seq_len
        # Padding region: each position attends only to itself
        if offset < max_seq_len:
            indices[i, 0, offset:, 0] = np.arange(offset, max_seq_len)
    return indices


# ---------------------------------------------------------------------------
# MTP (Multi-Token Prediction) helpers
# ---------------------------------------------------------------------------


def _build_nbatch_pack_offset(
    group_seq_lens: List[int],
    max_seq_len: int,
) -> np.ndarray:
    """Build document boundary markers for MTP.

    Places 1 at the last position of each sub-sequence (except the final one).
    This tells MTP layers where document boundaries are within a packed group.

    Args:
        group_seq_lens: List of sub-sequence lengths in one packed group.
        max_seq_len: Padded sequence length.

    Returns: [1, S] int32
    """
    offset_arr = np.zeros(max_seq_len, dtype=np.int32)
    pos = 0
    for seq_len in group_seq_lens[:-1]:
        pos += seq_len
        offset_arr[pos - 1] = 1
    return offset_arr[None, :]  # [1, S]


def gen_mtp_attn_mask(
    group_seq_lens: List[int],
    max_seq_len: int,
    mtp_depth: int,
    use_global_causal_attn: bool = False,
) -> np.ndarray:
    """Generate MTP per-layer attention mask (2D matrix form).

    For each MTP layer d (0-indexed), document boundaries are shifted left
    by d+1 positions. This creates the correct causal mask for each
    speculative prediction depth.

    Args:
        group_seq_lens: Sub-sequence lengths in one packed group.
        max_seq_len: Padded sequence length (already extended by mtp_depth).
        mtp_depth: Number of MTP prediction layers D.
        use_global_causal_attn: If True, single global causal block.

    Returns: [D, 1, S, S] float32
    """
    total_len = sum(group_seq_lens)

    if use_global_causal_attn:
        single = np.zeros((max_seq_len, max_seq_len), dtype=np.float32)
        single[:total_len, :total_len] = np.tril(np.ones((total_len, total_len)))
        return np.stack([single] * mtp_depth, axis=0)[:, None, :, :]

    # Compute internal boundaries (cumulative sum of seq_lens, excluding last)
    boundaries = []
    offset = 0
    for sl in group_seq_lens[:-1]:
        offset += sl
        boundaries.append(offset)

    result = []
    for mtp_idx in range(mtp_depth):
        shift = mtp_idx + 1
        shifted = [b - shift for b in boundaries if b - shift > 0] + [total_len]
        mask = np.zeros((max_seq_len, max_seq_len), dtype=np.float32)
        prev = 0
        for boundary in shifted:
            if boundary > prev:
                mask[prev:boundary, prev:boundary] = np.tril(
                    np.ones((boundary - prev, boundary - prev), dtype=np.float32)
                )
            prev = boundary
        result.append(mask)

    return np.stack(result, axis=0)[:, None, :, :]


def gen_mtp_attn_mask_startend_row_indices(
    group_seq_lens: List[int],
    max_seq_len: int,
    mtp_depth: int,
    use_global_causal_attn: bool = False,
) -> np.ndarray:
    """Generate MTP per-layer attention mask (compressed startend form).

    Args:
        group_seq_lens: Sub-sequence lengths in one packed group.
        max_seq_len: Padded sequence length (already extended by mtp_depth).
        mtp_depth: Number of MTP prediction layers D.
        use_global_causal_attn: If True, single global block.

    Returns: [D, 1, S, 1] int32
    """
    total_len = sum(group_seq_lens)
    pad_indices = list(range(total_len, max_seq_len))

    if use_global_causal_attn:
        row = [total_len] * total_len + pad_indices
        return np.array([row] * mtp_depth, dtype=np.int32)[:, None, :, None]

    boundaries = []
    offset = 0
    for sl in group_seq_lens[:-1]:
        offset += sl
        boundaries.append(offset)

    result = []
    for mtp_idx in range(mtp_depth):
        shift = mtp_idx + 1
        shifted = [b - shift for b in boundaries if b - shift > 0] + [total_len]
        indices = []
        prev = 0
        for boundary in shifted:
            indices.extend([boundary] * (boundary - prev))
            prev = boundary
        result.append(indices + pad_indices)

    return np.array(result, dtype=np.int32)[:, None, :, None]


def gen_mtp_layer_mask(
    input_ids_row: np.ndarray,
    total_len: int,
    max_seq_len: int,
    mtp_depth: int,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """Generate MTP per-layer hidden inputs mask.

    Zeroes out positions where EOS token appears at the shifted offset,
    preventing MTP layers from predicting across document boundaries.

    Args:
        input_ids_row: The input_ids for one batch row, shape [S].
        total_len: Actual total token length (sum of sub-sequences).
        max_seq_len: Padded sequence length.
        mtp_depth: Number of MTP prediction layers D.
        eos_token_id: If provided, zero out EOS positions in shifted input.

    Returns: [D, S] int32
    """
    if eos_token_id is None:
        return np.ones((mtp_depth, max_seq_len), dtype=np.int32)

    result = []
    for mtp_idx in range(mtp_depth):
        mask = np.ones(max_seq_len, dtype=np.int32)
        shifted = input_ids_row[mtp_idx + 1 : total_len]
        eos_positions = np.where(shifted == eos_token_id)[0]
        mask[eos_positions] = 0
        result.append(mask)

    return np.stack(result, axis=0)


# ---------------------------------------------------------------------------
# VL (Vision-Language) collation
# ---------------------------------------------------------------------------


def collate_vl_sft(
    batch: List[Any],
    pad_token_id: int,
    max_seq_len: int,
    get_rope_func: Optional[Callable] = None,
    use_attn_mask_startend_row_indices: bool = False,
) -> Dict[str, Any]:
    """Collate a batch of VLEncodedSamples into training-ready arrays.

    Handles pixel_values concatenation and 3D position_ids via get_rope_func.

    Args:
        batch: List of VLEncodedSample from DataLoader.
        pad_token_id: Tokenizer's pad token ID.
        max_seq_len: Maximum sequence length for padding.
        get_rope_func: Model's get_rope_index method for 3D position_ids.
            If None, falls back to simple 1D position_ids.
        use_attn_mask_startend_row_indices: If True, use compact flashmask format.

    Returns:
        Dict with numpy/tensor arrays:
            - input_ids, labels, position_ids, attention_mask (same as collate_sft)
            - pixel_values: concatenated vision features
            - image_grid_thw: stacked grid dimensions
    """
    import paddle

    batch_size = len(batch)
    pad_len = min(max_seq_len, max(s.seq_len for s in batch))

    input_ids = np.full((batch_size, pad_len), pad_token_id, dtype=np.int64)
    labels = np.full((batch_size, pad_len), -100, dtype=np.int64)

    for i, sample in enumerate(batch):
        seq_len = min(sample.seq_len, pad_len)
        input_ids[i, :seq_len] = sample.input_ids[:seq_len]
        labels[i, :seq_len] = sample.labels[:seq_len]

    # Collect vision inputs
    all_pixel_values = []
    all_image_grid_thw = []

    for sample in batch:
        mm = sample.mm_inputs
        if "pixel_values" in mm and mm["pixel_values"] is not None:
            pv = mm["pixel_values"]
            if not isinstance(pv, paddle.Tensor):
                pv = paddle.to_tensor(pv)
            all_pixel_values.append(pv)
        if "image_grid_thw" in mm and mm["image_grid_thw"] is not None:
            grid = mm["image_grid_thw"]
            if not isinstance(grid, paddle.Tensor):
                grid = paddle.to_tensor(grid, dtype="int64")
            all_image_grid_thw.append(grid)

    pixel_values = paddle.concat(all_pixel_values, axis=0) if all_pixel_values else None
    image_grid_thw = paddle.concat(all_image_grid_thw, axis=0) if all_image_grid_thw else None

    # Compute position_ids
    if get_rope_func is not None and image_grid_thw is not None:
        input_ids_tensor = paddle.to_tensor(input_ids, dtype="int64")
        attention_mask_tensor = paddle.ones_like(input_ids_tensor)
        for i, sample in enumerate(batch):
            seq_len = min(sample.seq_len, pad_len)
            attention_mask_tensor[i, seq_len:] = 0

        position_ids, rope_deltas = get_rope_func(
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask_tensor,
        )
        position_ids = position_ids.numpy()
    else:
        position_ids = np.zeros((batch_size, pad_len), dtype=np.int64)
        for i, sample in enumerate(batch):
            seq_len = min(sample.seq_len, pad_len)
            position_ids[i, :seq_len] = np.arange(seq_len)

    result: Dict[str, Any] = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
    }

    if pixel_values is not None:
        result["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        result["image_grid_thw"] = image_grid_thw

    if use_attn_mask_startend_row_indices:
        result["attn_mask_startend_row_indices"] = _build_startend_simple(batch, pad_len)
    else:
        attention_mask = np.zeros((batch_size, 1, pad_len, pad_len), dtype=np.float32)
        for i, sample in enumerate(batch):
            seq_len = min(sample.seq_len, pad_len)
            attention_mask[i, 0, :seq_len, :seq_len] = np.tril(np.ones((seq_len, seq_len)))
        result["attention_mask"] = attention_mask

    return result
