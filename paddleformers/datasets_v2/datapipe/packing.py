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

"""Packing algorithms: bin multiple short sequences into max_seq_len slots.

Two strategies:
  - greedy_pack: largest-remaining-capacity first-fit (fast, no external deps)
  - binpack_ffd: First-Fit Decreasing via `binpacking` library (better packing efficiency)
"""

from typing import List

import numpy as np

from .encode import EncodedSample


def greedy_pack(
    samples: List[EncodedSample],
    max_seq_len: int,
) -> List[List[EncodedSample]]:
    """Pack samples into bins using greedy first-fit (largest gap first).

    For each sample, finds the open bin with the most remaining capacity.
    If the sample fits, it's placed there. Otherwise, a new bin is opened.

    Args:
        samples: List of encoded samples to pack.
        max_seq_len: Maximum total token count per bin.

    Returns:
        List of packed groups. Each group is a list of samples whose
        combined seq_len <= max_seq_len.
    """
    if not samples:
        return []

    n = len(samples)
    remaining = np.full(n, -1, dtype=np.int64)
    remaining[0] = max_seq_len
    packs: List[List[EncodedSample]] = [[]]
    next_bin = 1

    for sample in samples:
        seq_len = sample.seq_len
        if seq_len > max_seq_len:
            # Sample exceeds max_seq_len — put it alone in its own bin (will be truncated by collate)
            packs.append([sample])
            continue

        best_bin = int(np.argmax(remaining))
        if seq_len <= remaining[best_bin]:
            packs[best_bin].append(sample)
            remaining[best_bin] -= seq_len
        else:
            # Open a new bin
            if next_bin >= n:
                # Extend array if needed
                remaining = np.append(remaining, np.full(n, -1, dtype=np.int64))
            remaining[next_bin] = max_seq_len - seq_len
            packs.append([sample])
            next_bin += 1

    return [p for p in packs if p]


def binpack_ffd(
    samples: List[EncodedSample],
    max_seq_len: int,
    packing_interval: int = 1000,
) -> List[List[EncodedSample]]:
    """Bin-pack using First-Fit Decreasing (binpacking library).

    Processes samples in chunks of packing_interval for memory efficiency.
    The last incomplete bin in each chunk is carried over to the next chunk,
    allowing cross-chunk optimization.

    Args:
        samples: List of encoded samples to pack.
        max_seq_len: Maximum total token count per bin.
        packing_interval: Buffer size — how many samples to accumulate
            before running the FFD algorithm. Larger values give better
            packing efficiency but use more memory.

    Returns:
        List of packed groups. Each group is a list of samples whose
        combined seq_len <= max_seq_len.
    """
    import binpacking

    if not samples:
        return []

    # Build (sample, weight) tuples for the binpacking library
    items = [(s, s.seq_len) for s in samples]
    accumulated = []
    result: List[List[EncodedSample]] = []

    for start in range(0, len(items), packing_interval):
        end = min(start + packing_interval, len(items))
        is_finished = end == len(items)
        accumulated += items[start:end]

        bins = binpacking.to_constant_volume(accumulated, max_seq_len, weight_pos=1)

        if bins and not is_finished:
            # Emit all complete bins, keep last one for next iteration
            for b in bins[:-1]:
                result.append([item[0] for item in b])
            accumulated = bins[-1]
        else:
            for b in bins:
                result.append([item[0] for item in b])
            accumulated = []

    return [g for g in result if g]
