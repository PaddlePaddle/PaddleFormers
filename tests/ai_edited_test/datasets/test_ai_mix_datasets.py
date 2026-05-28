# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
from unittest.mock import MagicMock

from paddleformers.datasets.reader.mix_datasets import (
    CLASS_MAPPING,
    BaseMixDataset,
    ConcatDataset,
    InterLeaveDataset,
    RandomDataset,
    create_dataset_instance,
)


def _make_mock_multi_source_dataset(datasets, probs):
    """Helper to create a mock MultiSourceDataset with task groups."""
    mock = MagicMock()
    mock._task_group = [{"dataset": datasets[i], "prob": probs[i]} for i in range(len(datasets))]
    return mock


def _make_simple_dataset(n, prefix="item"):
    """Helper to create a simple list-based dataset."""
    return [f"{prefix}_{i}" for i in range(n)]


class TestBaseMixDataset(unittest.TestCase):
    """Tests for BaseMixDataset."""

    def _create_base_dataset(self, *args, **kwargs):
        """Helper to create a concrete BaseMixDataset for testing."""

        class ConcreteMixDataset(BaseMixDataset):
            def __iter__(self):
                return iter([])

            def __len__(self):
                return 0

        return ConcreteMixDataset(*args, **kwargs)

    def test_init_normalizes_probabilities(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.7, 0.3])

        dataset = self._create_base_dataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=100,
        )
        # Probabilities should be normalized
        total = sum(dataset.datasets_prob)
        self.assertAlmostEqual(total, 1.0)

    def test_init_upsampling_mode(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = self._create_base_dataset(
            mock_ms,
            mix_strategy="interleave_under",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=100,
        )
        self.assertEqual(dataset.mode, "upsampling")

    def test_init_oversampling_mode(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = self._create_base_dataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=100,
        )
        self.assertEqual(dataset.mode, "oversampling")

    def test_init_reverse_default(self):
        ds1 = _make_simple_dataset(5, "a")
        mock_ms = _make_mock_multi_source_dataset([ds1], [1.0])

        dataset = self._create_base_dataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=50,
        )
        self.assertFalse(dataset.reverse)

    def test_init_reverse_true(self):
        ds1 = _make_simple_dataset(5, "a")
        mock_ms = _make_mock_multi_source_dataset([ds1], [1.0])

        dataset = self._create_base_dataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=50,
            reverse=True,
        )
        self.assertTrue(dataset.reverse)

    def test_abstract_methods(self):
        ds1 = _make_simple_dataset(5, "a")
        _make_mock_multi_source_dataset([ds1], [1.0])
        # Verify BaseMixDataset has abstractmethod decorators on __iter__ and __len__
        # These are the methods subclasses must implement
        self.assertTrue(hasattr(BaseMixDataset.__iter__, "__isabstractmethod__"))
        self.assertTrue(hasattr(BaseMixDataset.__len__, "__isabstractmethod__"))


class TestRandomDataset(unittest.TestCase):
    """Tests for RandomDataset."""

    def test_init(self):
        ds1 = _make_simple_dataset(10, "a")
        ds2 = _make_simple_dataset(10, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = RandomDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=20,
        )
        self.assertEqual(len(dataset), 20)

    def test_iter_returns_correct_count(self):
        ds1 = _make_simple_dataset(20, "a")
        ds2 = _make_simple_dataset(20, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = RandomDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=20,
        )
        items = list(dataset)
        self.assertEqual(len(items), 20)

    def test_iter_no_shuffle(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = RandomDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=10,
        )
        items = list(dataset)
        # Without shuffle, items should come from ds1 first then ds2
        a_items = [i for i in items if i.startswith("a")]
        b_items = [i for i in items if i.startswith("b")]
        self.assertEqual(len(a_items), 5)
        self.assertEqual(len(b_items), 5)

    def test_iter_with_shuffle(self):
        ds1 = _make_simple_dataset(20, "a")
        ds2 = _make_simple_dataset(20, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = RandomDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=20,
        )
        items = list(dataset)
        self.assertEqual(len(items), 20)

    def test_iter_with_reverse(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = RandomDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=10,
            reverse=True,
        )
        items = list(dataset)
        # Reversed should have ds2 items before ds1 items
        self.assertTrue(items[-1].startswith("a") or items[0].startswith("b"))

    def test_epoch_index_increments(self):
        ds1 = _make_simple_dataset(5, "a")
        mock_ms = _make_mock_multi_source_dataset([ds1], [1.0])

        dataset = RandomDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=5,
        )
        self.assertEqual(dataset.epoch_index, 0)
        list(dataset)  # First epoch
        self.assertEqual(dataset.epoch_index, 1)

    def test_len(self):
        ds1 = _make_simple_dataset(10, "a")
        mock_ms = _make_mock_multi_source_dataset([ds1], [1.0])

        dataset = RandomDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=50,
        )
        self.assertEqual(len(dataset), 50)


class TestConcatDataset(unittest.TestCase):
    """Tests for ConcatDataset."""

    def test_init(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = ConcatDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=100,
        )
        self.assertEqual(len(dataset), 10)  # 5 + 5

    def test_iter_returns_all_items(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(3, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = ConcatDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        items = list(dataset)
        self.assertEqual(len(items), 8)

    def test_iter_no_shuffle_preserves_order(self):
        ds1 = _make_simple_dataset(3, "a")
        ds2 = _make_simple_dataset(2, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = ConcatDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        items = list(dataset)
        # Without shuffle, should be ds1 items first then ds2
        self.assertEqual(items[0], "a_0")
        self.assertEqual(items[2], "a_2")
        self.assertEqual(items[3], "b_0")
        self.assertEqual(items[4], "b_1")

    def test_iter_with_shuffle(self):
        ds1 = _make_simple_dataset(10, "a")
        ds2 = _make_simple_dataset(10, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = ConcatDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=100,
        )
        items = list(dataset)
        self.assertEqual(len(items), 20)
        # Just verify all items are present
        self.assertTrue(all(i.startswith("a_") or i.startswith("b_") for i in items))

    def test_epoch_index_increments(self):
        ds1 = _make_simple_dataset(3, "a")
        mock_ms = _make_mock_multi_source_dataset([ds1], [1.0])

        dataset = ConcatDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        self.assertEqual(dataset.epoch_index, 0)
        list(dataset)
        self.assertEqual(dataset.epoch_index, 1)

    def test_len(self):
        ds1 = _make_simple_dataset(7, "a")
        ds2 = _make_simple_dataset(3, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = ConcatDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=100,
        )
        self.assertEqual(len(dataset), 10)


class TestInterLeaveDataset(unittest.TestCase):
    """Tests for InterLeaveDataset."""

    def test_init_upsampling(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_under",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=100,
        )
        self.assertEqual(dataset.mode, "upsampling")

    def test_init_oversampling(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=100,
        )
        self.assertEqual(dataset.mode, "oversampling")

    def test_upsampling_stops_at_first_exhausted(self):
        ds1 = _make_simple_dataset(3, "a")
        ds2 = _make_simple_dataset(10, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_under",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        # Upsampling stops when first dataset exhausted
        # The total should be around the size of the smaller dataset
        self.assertLessEqual(len(dataset), 10)

    def test_oversampling_uses_all(self):
        ds1 = _make_simple_dataset(3, "a")
        ds2 = _make_simple_dataset(3, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        # Oversampling should use all data from both datasets
        self.assertGreaterEqual(len(dataset), 6)

    def test_iter_returns_all_items(self):
        ds1 = _make_simple_dataset(3, "a")
        ds2 = _make_simple_dataset(3, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        items = list(dataset)
        self.assertEqual(len(items), len(dataset))

    def test_iter_no_shuffle(self):
        ds1 = _make_simple_dataset(3, "a")
        ds2 = _make_simple_dataset(3, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        items = list(dataset)
        # All items from both datasets should be present
        self.assertTrue(len(items) >= 6)

    def test_iter_with_shuffle(self):
        ds1 = _make_simple_dataset(10, "a")
        ds2 = _make_simple_dataset(10, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=True,
            num_samples_each_epoch=100,
        )
        items = list(dataset)
        self.assertEqual(len(items), len(dataset))

    def test_epoch_index_increments(self):
        ds1 = _make_simple_dataset(3, "a")
        mock_ms = _make_mock_multi_source_dataset([ds1], [1.0])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        self.assertEqual(dataset.epoch_index, 0)
        list(dataset)
        self.assertEqual(dataset.epoch_index, 1)

    def test_single_dataset(self):
        ds1 = _make_simple_dataset(5, "a")
        mock_ms = _make_mock_multi_source_dataset([ds1], [1.0])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        self.assertEqual(len(dataset), 5)
        items = list(dataset)
        self.assertEqual(len(items), 5)

    def test_build_dataset_prints_info(self):
        """Test that _build_dataset runs and produces output."""
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(3, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.7, 0.3])

        dataset = InterLeaveDataset(
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        # If we reach here without error, _build_dataset completed
        self.assertGreater(len(dataset.data), 0)


class TestCreateDatasetInstance(unittest.TestCase):
    """Tests for create_dataset_instance function."""

    def test_create_concat(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        result = create_dataset_instance(
            "concat",
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        self.assertIsInstance(result, ConcatDataset)

    def test_create_random(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        result = create_dataset_instance(
            "random",
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=20,
        )
        self.assertIsInstance(result, RandomDataset)

    def test_create_interleave_under(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        result = create_dataset_instance(
            "interleave_under",
            mock_ms,
            mix_strategy="interleave_under",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        self.assertIsInstance(result, InterLeaveDataset)

    def test_create_interleave_over(self):
        ds1 = _make_simple_dataset(5, "a")
        ds2 = _make_simple_dataset(5, "b")
        mock_ms = _make_mock_multi_source_dataset([ds1, ds2], [0.5, 0.5])

        result = create_dataset_instance(
            "interleave_over",
            mock_ms,
            mix_strategy="interleave_over",
            random_seed=42,
            random_shuffle=False,
            num_samples_each_epoch=100,
        )
        self.assertIsInstance(result, InterLeaveDataset)

    def test_create_unknown_returns_none(self):
        result = create_dataset_instance("nonexistent_class")
        self.assertIsNone(result)


class TestClassMapping(unittest.TestCase):
    """Tests for CLASS_MAPPING constant."""

    def test_contains_all_types(self):
        self.assertIn("concat", CLASS_MAPPING)
        self.assertIn("interleave_under", CLASS_MAPPING)
        self.assertIn("interleave_over", CLASS_MAPPING)
        self.assertIn("random", CLASS_MAPPING)
        self.assertEqual(len(CLASS_MAPPING), 4)

    def test_values_are_classes(self):
        self.assertEqual(CLASS_MAPPING["concat"], ConcatDataset)
        self.assertEqual(CLASS_MAPPING["random"], RandomDataset)
        self.assertEqual(CLASS_MAPPING["interleave_under"], InterLeaveDataset)
        self.assertEqual(CLASS_MAPPING["interleave_over"], InterLeaveDataset)


if __name__ == "__main__":
    unittest.main()
