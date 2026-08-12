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

"""Lazy-encoding dataset wrapper for training.

Wraps an HF Dataset + encode function into a map-style Dataset suitable for
DataLoader. Encoding happens lazily in __getitem__, with automatic retry
on failure (bad samples get skipped via random fallback).
"""

import logging
import traceback
from typing import Any, Callable, Dict, Optional

import numpy as np
import paddle

from ..datapipe.encode import EncodedSample

logger = logging.getLogger(__name__)


class LazyEncodeDataset(paddle.io.Dataset):
    """Map-style dataset that lazily encodes HF Dataset rows.

    Each __getitem__ call fetches a row from the underlying HF Dataset,
    applies encode_func to produce an EncodedSample. If encoding fails,
    retries with random fallback indices.

    Inherits from paddle.io.Dataset for compatibility with PaddleFormers Trainer.

    Args:
        dataset: HuggingFace Dataset with 'messages' column.
        encode_func: Callable that takes a row dict → EncodedSample or None.
        n_try_fetch: Max retry count on encoding failure.
        seed: Random seed for reproducible fallback selection.
    """

    def __init__(
        self,
        dataset: Any,
        encode_func: Callable[[Dict[str, Any]], Optional[EncodedSample]],
        n_try_fetch: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.encode_func = encode_func
        self.n_try_fetch = min(n_try_fetch, max(1, len(dataset)))
        self._rng = np.random.RandomState(seed)
        self._fallback_indices = self._rng.permutation(len(dataset)).tolist()
        self._fallback_ptr = 0
        self._traceback_counter = 0
        self._traceback_limit = 10

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> EncodedSample:
        for attempt in range(self.n_try_fetch):
            i = idx if attempt == 0 else self._next_fallback()
            try:
                row = self.dataset[i]
                result = self.encode_func(row)
                if result is not None:
                    return result
            except Exception:
                if self._traceback_counter < self._traceback_limit:
                    logger.warning(
                        f"Encoding failed for index {i} (attempt {attempt + 1}):\n" f"{traceback.format_exc()}"
                    )
                    self._traceback_counter += 1

        raise RuntimeError(f"Failed to encode sample after {self.n_try_fetch} attempts " f"(started at index {idx}).")

    def _next_fallback(self) -> int:
        idx = self._fallback_indices[self._fallback_ptr % len(self._fallback_indices)]
        self._fallback_ptr += 1
        return idx
