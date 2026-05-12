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

Supports both non-packed (simple padding) and packed (greedy packing with
block-diagonal attention mask) modes.

Output format is compatible with PaddleFormers' existing Trainer.
"""

from typing import Dict, List

import numpy as np

from .encode import EncodedSample
from .packing import greedy_pack


def collate_sft(
    batch: List[EncodedSample],
    pad_token_id: int,
    max_seq_len: int,
    packing: bool = False,
    use_attn_mask_startend_row_indices: bool = False,
) -> Dict[str, np.ndarray]:
    """Collate a batch of EncodedSamples into training-ready numpy arrays.

    Args:
        batch: List of EncodedSample from DataLoader.
        pad_token_id: Tokenizer's pad token ID for input_ids padding.
        max_seq_len: Maximum sequence length. Sequences are padded to
            min(max_seq_len, max_length_in_batch) when not packing,
            or exactly max_seq_len when packing.
        packing: If True, use greedy packing to combine short sequences.
        use_attn_mask_startend_row_indices: If True, output compact flashmask
            format (attn_mask_startend_row_indices) instead of 4D attention_mask.

    Returns:
        Dict with numpy arrays:
            - input_ids:      [B, S] int64
            - labels:         [B, S] int64, -100 for masked positions
            - position_ids:   [B, S] int64
            - attention_mask: [B, 1, S, S] float32  (when use_attn_mask_startend_row_indices=False)
            - attn_mask_startend_row_indices: [B, 1, S, 1] int32  (when use_attn_mask_startend_row_indices=True)
    """
    if packing:
        return _collate_packed(batch, pad_token_id, max_seq_len, use_attn_mask_startend_row_indices)
    else:
        return _collate_simple(batch, pad_token_id, max_seq_len, use_attn_mask_startend_row_indices)


def _collate_simple(
    batch: List[EncodedSample],
    pad_token_id: int,
    max_seq_len: int,
    use_startend: bool = False,
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

    return result


def _collate_packed(
    batch: List[EncodedSample],
    pad_token_id: int,
    max_seq_len: int,
    use_startend: bool = False,
) -> Dict[str, np.ndarray]:
    """Packed collation: bin samples together, block-diagonal attention mask."""
    # Pack samples into groups
    groups = greedy_pack(batch, max_seq_len)
    batch_size = len(groups)

    input_ids = np.full((batch_size, max_seq_len), pad_token_id, dtype=np.int64)
    labels = np.full((batch_size, max_seq_len), -100, dtype=np.int64)
    position_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)

    # Track sub-sequence lengths for each group (needed for startend format)
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

    return result


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
