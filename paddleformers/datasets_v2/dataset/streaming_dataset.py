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

"""Streaming dataset wrapper: adapts HF IterableDataset to paddle.io.IterableDataset."""

import numpy as np
import paddle


class StreamingDataset(paddle.io.IterableDataset):
    """Wraps a HuggingFace IterableDataset as a paddle.io.IterableDataset.

    This allows Trainer to recognize the dataset as iterable and use the
    appropriate DataLoader path (no sampler, direct iteration).

    Supports two modes:
      - lazy=True (default): True streaming. Yields directly from HF iterator
        without downloading/materializing the full dataset. Suitable for large
        remote datasets (e.g. fineweb-edu). Shuffle uses HF's buffer shuffle.
      - lazy=False: Legacy V1-compatible mode. Materializes all data into memory
        for epoch-based full-array shuffle. Only suitable for small datasets.

    Args:
        hf_iterable: A HuggingFace IterableDataset instance (e.g. from
            load_dataset(..., streaming=True) with .map()/.filter() applied).
        shuffle: Whether to shuffle data each epoch.
        seed: Random seed for shuffle.
        lazy: If True, iterate directly from HF without materialization (true streaming).
            If False, materialize all data for epoch-based shuffle (V1 compat).
    """

    def __init__(self, hf_iterable, shuffle: bool = False, seed: int = 0, lazy: bool = True):
        super().__init__()
        self._hf = hf_iterable
        self._shuffle = shuffle
        self._seed = seed
        self._lazy = lazy
        self._data = None  # only used when lazy=False

    def _materialize(self):
        """Load all data from HF iterable into memory for epoch-based shuffle."""
        if self._data is None:
            self._data = list(self._hf)

    def __iter__(self):
        if self._lazy:
            # True streaming: yield directly from HF iterator.
            # HF IterableDataset fetches parquet chunks on-demand, never downloads all.
            # For shuffle, rely on HF's .shuffle(buffer_size=...) applied upstream.
            yield from self._hf
        else:
            # Legacy mode: materialize all data, then epoch-based shuffle.
            self._materialize()
            data = self._data
            n = len(data)
            indices = list(range(n))
            rng = np.random.RandomState(self._seed) if self._shuffle else None

            while True:
                if rng is not None:
                    rng.shuffle(indices)
                else:
                    indices = list(range(n))

                last_item = None
                for idx in indices:
                    last_item = data[idx]
                    yield last_item

                # Align with V1 IteratorSFTDataset._generate_sequences behavior:
                # V1 yields the last element twice per epoch due to a trailing
                # `if len(batch_sequence) > 0: yield batch_sequence` after the loop.
                if last_item is not None:
                    yield last_item
