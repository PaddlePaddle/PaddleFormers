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

"""Dataset operations: sample, split, concat, interleave, shuffle.

Thin wrappers over HuggingFace datasets utilities with added support
for oversampling, streaming (IterableDataset), and unified API.
"""

from typing import List, Optional, Tuple

import numpy as np
from datasets import Dataset as HfMapDataset
from datasets import IterableDataset as HfIterableDataset  # noqa: F401
from datasets import concatenate_datasets as hf_concat
from datasets import interleave_datasets as hf_interleave

from .schema import DATASET_TYPE


def sample_dataset(
    dataset: HfMapDataset,
    n: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> HfMapDataset:
    """Sample n rows from dataset. Supports oversampling (n > len(dataset)).

    Args:
        dataset: must be a Map-style Dataset (not IterableDataset)
        n: target number of rows. If > len(dataset), repeats the dataset.
        shuffle: whether to shuffle the sampled indices
        seed: random seed
    """
    if not isinstance(dataset, HfMapDataset):
        raise TypeError("sample_dataset only supports Map-style Dataset, not IterableDataset.")
    if n <= 0:
        raise ValueError("Cannot sample non-positive number of rows")
    if not isinstance(n, int):
        raise ValueError("Cannot sample non-integer number of rows")

    rng = np.random.RandomState(seed)
    length = len(dataset)

    if n <= length:
        if shuffle:
            idx = rng.permutation(length)[:n].tolist()
        else:
            idx = list(range(n))
    else:
        # Oversampling: tile full dataset + remainder
        repeats = n // length
        remainder = n % length
        idx = np.tile(np.arange(length), repeats)
        if remainder > 0:
            extra = rng.permutation(length)[:remainder] if shuffle else np.arange(remainder)
            idx = np.concatenate([idx, extra])
        if shuffle:
            rng.shuffle(idx)
        idx = idx.tolist()

    return dataset.select(idx)


def split_dataset(
    dataset: DATASET_TYPE,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[DATASET_TYPE, DATASET_TYPE]:
    """Split dataset into train and validation sets.

    Args:
        dataset: HF Dataset or IterableDataset
        test_ratio: fraction for validation (0.0 ~ 1.0)
        shuffle: whether to shuffle before splitting (Map-style only)
        seed: random seed

    Returns:
        (train_dataset, val_dataset)
    """
    if isinstance(dataset, HfMapDataset):
        split = dataset.train_test_split(test_size=test_ratio, shuffle=shuffle, seed=seed)
        return split["train"], split["test"]
    else:
        # IterableDataset: use take/skip (requires known length or estimate)
        # For streaming, caller should provide dataset_sample to know the size
        raise ValueError(
            "split_dataset on IterableDataset requires known size. "
            "Use sample first or manually call .take()/.skip()."
        )


def concat_datasets(datasets: List[DATASET_TYPE]) -> Optional[DATASET_TYPE]:
    """Concatenate multiple datasets into one."""
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return hf_concat(datasets)


def interleave(
    datasets: List[DATASET_TYPE],
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    stopping_strategy: str = "first_exhausted",
) -> Optional[DATASET_TYPE]:
    """Interleave multiple datasets with optional probability weights.

    Args:
        datasets: list of datasets to interleave
        probabilities: sampling probability for each dataset (must sum to 1)
        seed: random seed
        stopping_strategy: 'first_exhausted' or 'all_exhausted'
    """
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return hf_interleave(
        datasets,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )


def shuffle_dataset(
    dataset: DATASET_TYPE,
    seed: int = 42,
    buffer_size: int = 1000,
) -> DATASET_TYPE:
    """Shuffle a dataset.

    For Map-style Dataset: full in-memory shuffle.
    For IterableDataset: streaming shuffle with buffer.
    """
    if isinstance(dataset, HfMapDataset):
        return dataset.shuffle(seed=seed)
    else:
        return dataset.shuffle(seed=seed, buffer_size=buffer_size)
