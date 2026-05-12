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

"""Greedy packing: bin multiple short sequences into max_seq_len slots.

Algorithm matches PaddleFormers' existing greedy_intokens approach:
largest-remaining-capacity first-fit.
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
