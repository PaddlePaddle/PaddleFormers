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

"""Tests for datasets_v2/ops.py.

Run with: python -m pytest tests/datasets_v2/test_ops.py -v
"""

import importlib

# Workaround: broken torchcodec residual in env
_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    if name == "torchcodec":
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _patched_find_spec

import pytest
from datasets import Dataset

from paddleformers.datasets_v2.ops import (
    concat_datasets,
    interleave,
    sample_dataset,
    shuffle_dataset,
    split_dataset,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def small_dataset():
    """A small 5-row dataset."""
    return Dataset.from_dict({"text": [f"row_{i}" for i in range(5)]})


@pytest.fixture
def empty_dataset():
    """An empty dataset."""
    return Dataset.from_dict({"text": []})


# ============================================================
# sample_dataset
# ============================================================


class TestSampleDataset:
    def test_sample_less_than_length(self, small_dataset):
        """Sample fewer rows than available."""
        result = sample_dataset(small_dataset, n=3, seed=42)
        assert len(result) == 3

    def test_sample_equal_to_length(self, small_dataset):
        """Sample exactly the dataset size."""
        result = sample_dataset(small_dataset, n=5, seed=42)
        assert len(result) == 5

    def test_sample_with_shuffle(self, small_dataset):
        """Shuffled sample should (usually) differ from sequential."""
        result = sample_dataset(small_dataset, n=5, shuffle=True, seed=42)
        # The order should be a permutation
        texts = set(result["text"])
        assert texts == {f"row_{i}" for i in range(5)}

    def test_sample_without_shuffle(self, small_dataset):
        """Non-shuffled sample returns first N rows in order."""
        result = sample_dataset(small_dataset, n=3, shuffle=False)
        assert result["text"] == ["row_0", "row_1", "row_2"]

    def test_oversample(self, small_dataset):
        """Oversample: n > len(dataset) should repeat."""
        result = sample_dataset(small_dataset, n=12, seed=42)
        assert len(result) == 12
        # All values should be from the original dataset
        assert all(t.startswith("row_") for t in result["text"])

    def test_oversample_without_shuffle(self, small_dataset):
        """Oversample without shuffle should tile deterministically."""
        result = sample_dataset(small_dataset, n=12, shuffle=False)
        assert len(result) == 12
        # First 5 should be row_0..row_4, next 5 same, then row_0, row_1
        assert result["text"][:5] == [f"row_{i}" for i in range(5)]
        assert result["text"][5:10] == [f"row_{i}" for i in range(5)]
        assert result["text"][10:] == ["row_0", "row_1"]

    def test_reproducibility(self, small_dataset):
        """Same seed should give same result."""
        r1 = sample_dataset(small_dataset, n=3, seed=123)
        r2 = sample_dataset(small_dataset, n=3, seed=123)
        assert r1["text"] == r2["text"]

    def test_different_seeds(self, small_dataset):
        """Different seeds should (very likely) give different results."""
        r1 = sample_dataset(small_dataset, n=5, shuffle=True, seed=1)
        r2 = sample_dataset(small_dataset, n=5, shuffle=True, seed=99)
        # Extremely unlikely to be the same permutation
        assert r1["text"] != r2["text"]

    def test_zero_n_raises(self, small_dataset):
        """n=0 should raise ValueError."""
        with pytest.raises(ValueError, match="non-positive"):
            sample_dataset(small_dataset, n=0)

    def test_negative_n_raises(self, small_dataset):
        """Negative n should raise ValueError."""
        with pytest.raises(ValueError, match="non-positive"):
            sample_dataset(small_dataset, n=-1)

    def test_float_n_raises(self, small_dataset):
        """Float n should raise ValueError."""
        with pytest.raises(ValueError, match="non-integer"):
            sample_dataset(small_dataset, n=2.5)

    def test_empty_dataset_raises(self, empty_dataset):
        """Sampling from empty dataset should raise (or handle gracefully)."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            sample_dataset(empty_dataset, n=5)

    def test_iterable_dataset_raises(self, small_dataset):
        """IterableDataset should raise TypeError."""
        iterable = small_dataset.to_iterable_dataset()
        with pytest.raises(TypeError, match="Map-style"):
            sample_dataset(iterable, n=3)


# ============================================================
# split_dataset
# ============================================================


class TestSplitDataset:
    def test_basic_split(self, small_dataset):
        """Split should produce two non-empty datasets."""
        train, test = split_dataset(small_dataset, test_ratio=0.4, seed=42)
        assert len(train) + len(test) == 5
        assert len(test) >= 1
        assert len(train) >= 1

    def test_split_ratio(self):
        """Split ratio should approximately match."""
        ds = Dataset.from_dict({"x": list(range(100))})
        train, test = split_dataset(ds, test_ratio=0.2, seed=42)
        assert len(test) == 20
        assert len(train) == 80

    def test_split_reproducibility(self, small_dataset):
        """Same seed gives same split."""
        t1, v1 = split_dataset(small_dataset, test_ratio=0.4, seed=42)
        t2, v2 = split_dataset(small_dataset, test_ratio=0.4, seed=42)
        assert t1["text"] == t2["text"]
        assert v1["text"] == v2["text"]

    def test_split_iterable_raises(self, small_dataset):
        """IterableDataset should raise ValueError."""
        iterable = small_dataset.to_iterable_dataset()
        with pytest.raises(ValueError, match="IterableDataset"):
            split_dataset(iterable)


# ============================================================
# concat_datasets
# ============================================================


class TestConcatDatasets:
    def test_empty_list(self):
        """Empty list returns None."""
        assert concat_datasets([]) is None

    def test_single_dataset(self, small_dataset):
        """Single dataset returned as-is."""
        result = concat_datasets([small_dataset])
        assert result is small_dataset

    def test_multiple_datasets(self):
        """Multiple datasets concatenated."""
        ds1 = Dataset.from_dict({"text": ["a", "b"]})
        ds2 = Dataset.from_dict({"text": ["c", "d", "e"]})
        result = concat_datasets([ds1, ds2])
        assert len(result) == 5
        assert result["text"] == ["a", "b", "c", "d", "e"]

    def test_three_datasets(self):
        """Three datasets."""
        ds1 = Dataset.from_dict({"x": [1]})
        ds2 = Dataset.from_dict({"x": [2]})
        ds3 = Dataset.from_dict({"x": [3]})
        result = concat_datasets([ds1, ds2, ds3])
        assert len(result) == 3
        assert result["x"] == [1, 2, 3]


# ============================================================
# interleave
# ============================================================


class TestInterleave:
    def test_empty_list(self):
        """Empty list returns None."""
        assert interleave([]) is None

    def test_single_dataset(self, small_dataset):
        """Single dataset returned as-is."""
        result = interleave([small_dataset])
        assert result is small_dataset

    def test_two_datasets(self):
        """Two datasets interleaved."""
        ds1 = Dataset.from_dict({"text": ["a", "b", "c"]})
        ds2 = Dataset.from_dict({"text": ["x", "y", "z"]})
        result = interleave([ds1, ds2], seed=42)
        assert len(result) == 6  # first_exhausted: both have 3, result has 6

    def test_with_probabilities(self):
        """Interleave with probability weights."""
        ds1 = Dataset.from_dict({"text": [f"a{i}" for i in range(10)]})
        ds2 = Dataset.from_dict({"text": [f"b{i}" for i in range(10)]})
        result = interleave([ds1, ds2], probabilities=[0.8, 0.2], seed=42)
        assert len(result) > 0


# ============================================================
# shuffle_dataset
# ============================================================


class TestShuffleDataset:
    def test_map_dataset(self, small_dataset):
        """Shuffling a map dataset."""
        result = shuffle_dataset(small_dataset, seed=42)
        assert len(result) == 5
        assert set(result["text"]) == set(small_dataset["text"])

    def test_iterable_dataset(self, small_dataset):
        """Shuffling an iterable dataset (buffer-based)."""
        iterable = small_dataset.to_iterable_dataset()
        result = shuffle_dataset(iterable, seed=42, buffer_size=10)
        rows = list(result)
        assert len(rows) == 5

    def test_reproducibility(self, small_dataset):
        """Same seed gives same shuffle."""
        r1 = shuffle_dataset(small_dataset, seed=42)
        r2 = shuffle_dataset(small_dataset, seed=42)
        assert r1["text"] == r2["text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
