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

import os
import tempfile
import unittest

import numpy as np

from paddleformers.datasets.loader import create_dataset
from paddleformers.datasets.template.template import get_template_and_fix_tokenizer
from paddleformers.transformers import AutoTokenizer
from tests.testing_utils import get_tests_dir

MODEL_NAME = "/home/models/PaddleFormers/tiny-random-qwen3"


def _make_base_config(tokenizer, *, packing=False, binpacking=False, mix_strategy="concat", is_valid=False):

    cfg = {
        "tokenizer": tokenizer,
        "max_seq_len": 512,
        "random_seed": 42,
        "num_replicas": 1,
        "rank": 0,
        "num_samples_each_epoch": 6000000,
        "random_shuffle": False,
        "greedy_intokens": False,
        "packing": packing,
        "binpacking": binpacking,
        "use_template": True,
        "packing_interval": 100,
        "mix_strategy": mix_strategy,
        "encode_one_turn": True,
        "use_template": True,
        "template_backend": "custom",
        "is_pretraining": False,
        "truncate_packing": False,
        "dataset_type": "map",
        "stage": "SFT",
        "template": "qwen3_nothink",
        "tool_format": None,
        "default_system": None,
        "is_valid": is_valid,
    }
    if cfg["template_backend"] == "custom":
        template_instance = get_template_and_fix_tokenizer(cfg)
    else:
        template_instance = None
    cfg["template_instance"] = template_instance
    return cfg


class TestMapSFTDatasetBasic(unittest.TestCase):
    """Basic structural tests for MapSFTDataset (non-packing mode)."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.dataset_path = os.path.join(get_tests_dir(os.path.join("fixtures", "dummy")), "sft", "train.jsonl")

    def _create_dataset(self, **extra):
        cfg = _make_base_config(self.tokenizer, **extra)
        return create_dataset(
            task_group=self.dataset_path,
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **cfg,
        )

    def test_isinstance_paddle_dataset(self):
        """MapSFTDataset must be a paddle.io.Dataset (not IterableDataset)."""
        from paddle.io import Dataset, IterableDataset

        ds = self._create_dataset()
        self.assertIsInstance(ds, Dataset)
        self.assertNotIsInstance(ds, IterableDataset)

    def test_raw_data_loaded(self):
        """raw_data should be a non-empty list after init."""
        ds = self._create_dataset()
        self.assertIsInstance(ds.raw_data, list)
        self.assertGreater(len(ds.raw_data), 0)

    def test_len_equals_raw_data(self):
        """Without packing, __len__ should equal len(raw_data)."""
        ds = self._create_dataset()
        self.assertEqual(len(ds), len(ds.raw_data))

    def test_len_concat_two_sources(self):
        """concat mix_strategy with two identical sources doubles the sample count."""
        cfg = _make_base_config(self.tokenizer, mix_strategy="concat")
        ds = create_dataset(
            task_group=", ".join([self.dataset_path, self.dataset_path]),
            task_group_prob="1.0,1.0",
            sub_dataset_type="erniekit,erniekit",
            **cfg,
        )
        cfg_single = _make_base_config(self.tokenizer, mix_strategy="concat")
        ds_single = create_dataset(
            task_group=self.dataset_path,
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **cfg_single,
        )
        self.assertEqual(len(ds.raw_data), len(ds_single.raw_data) * 2)

    def test_packed_idx_is_none_without_packing(self):
        """packed_idx must be None when packing=False."""
        ds = self._create_dataset()
        self.assertIsNone(ds.packed_idx)

    def test_getitem_returns_list(self):
        """__getitem__ should return a list (of Sequence objects)."""
        ds = self._create_dataset()
        item = ds[0]
        self.assertIsInstance(item, list)
        self.assertGreater(len(item), 0)

    def test_getitem_sequence_fields(self):
        """Each element in the returned list must have token_ids, labels, position_ids."""
        ds = self._create_dataset()
        item = ds[0]
        seq = item[0]
        self.assertTrue(hasattr(seq, "token_ids"))
        self.assertTrue(hasattr(seq, "labels"))
        self.assertTrue(hasattr(seq, "position_ids"))

    def test_getitem_token_ids_within_max_seq_len(self):
        """token_ids length must not exceed max_seq_len."""
        ds = self._create_dataset()
        for idx in range(min(10, len(ds))):
            item = ds[idx]
            for seq in item:
                self.assertLessEqual(len(seq.token_ids), ds.max_seq_len)

    def test_getitem_labels_same_length_as_token_ids(self):
        """labels length must equal token_ids length."""
        ds = self._create_dataset()
        for idx in range(min(10, len(ds))):
            item = ds[idx]
            for seq in item:
                self.assertEqual(len(seq.token_ids), len(seq.labels))

    def test_getitem_position_ids_same_length_as_token_ids(self):
        """position_ids length must equal token_ids length."""
        ds = self._create_dataset()
        for idx in range(min(10, len(ds))):
            item = ds[idx]
            for seq in item:
                self.assertEqual(len(seq.token_ids), len(seq.position_ids))

    def test_getitem_single_returns_one_sequence(self):
        """In non-packing mode each __getitem__ call returns exactly one sequence."""
        ds = self._create_dataset()
        item = ds[0]
        self.assertEqual(len(item), 1)

    def test_is_valid_mode(self):
        """is_valid=True dataset should have finite length and be iterable by index."""
        ds = self._create_dataset(is_valid=True)
        self.assertGreater(len(ds), 0)
        item = ds[0]
        self.assertIsInstance(item, list)


class TestMapSFTDatasetPacking(unittest.TestCase):
    """Tests for MapSFTDataset in binpacking mode."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.dataset_path = os.path.join(get_tests_dir(os.path.join("fixtures", "dummy")), "sft", "train.jsonl")

    def _create_packed_dataset(self, **extra):
        cfg = _make_base_config(self.tokenizer, packing=True, binpacking=True, **extra)
        return create_dataset(
            task_group=self.dataset_path,
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **cfg,
        )

    def test_packed_idx_not_none(self):
        """packed_idx must be set after init with packing+binpacking."""
        ds = self._create_packed_dataset()
        self.assertIsNotNone(ds.packed_idx)

    def test_packed_idx_is_list_of_lists(self):
        """packed_idx must be a list of lists of integer indices."""
        ds = self._create_packed_dataset()
        self.assertIsInstance(ds.packed_idx, list)
        for group in ds.packed_idx:
            self.assertIsInstance(group, list)
            for idx in group:
                self.assertIsInstance(idx, (int, np.integer))

    def test_len_equals_packed_groups(self):
        """__len__ should equal number of packed groups."""
        ds = self._create_packed_dataset()
        self.assertEqual(len(ds), len(ds.packed_idx))

    def test_len_packed_le_raw_data(self):
        """Packed dataset length should be <= raw_data length (packing merges samples)."""
        ds = self._create_packed_dataset()
        self.assertLessEqual(len(ds), len(ds.raw_data))

    def test_getitem_packed_returns_list(self):
        """__getitem__ in packed mode returns a non-empty list of sequences."""
        ds = self._create_packed_dataset()
        item = ds[0]
        self.assertIsInstance(item, list)
        self.assertGreater(len(item), 0)

    def test_getitem_packed_sequences_within_max_seq_len(self):
        """Total token length of a packed group must fit within max_seq_len."""
        ds = self._create_packed_dataset()
        for idx in range(min(5, len(ds))):
            item = ds[idx]
            total_len = sum(len(seq.token_ids) for seq in item)
            self.assertLessEqual(total_len, ds.max_seq_len)

    def test_packing_only_true_raises(self):
        """packing=True without binpacking=True must raise ValueError."""
        cfg = _make_base_config(self.tokenizer, packing=True, binpacking=False)
        with self.assertRaises(ValueError):
            create_dataset(
                task_group=self.dataset_path,
                task_group_prob="1.0",
                sub_dataset_type="erniekit",
                **cfg,
            )

    def test_all_packed_indices_valid(self):
        """Every raw index referenced in packed_idx must be in range [0, len(raw_data))."""
        ds = self._create_packed_dataset()
        for group in ds.packed_idx:
            for idx in group:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(ds.raw_data))


class TestMapSFTDatasetPackedIdxCache(unittest.TestCase):
    """Tests for packed_idx save/load cache functionality."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.dataset_path = os.path.join(get_tests_dir(os.path.join("fixtures", "dummy")), "sft", "train.jsonl")

    def _make_cfg(self, cache_dir, is_valid=False):
        cfg = _make_base_config(self.tokenizer, packing=True, binpacking=True, is_valid=is_valid)
        cfg["packed_idx_cache_dir"] = cache_dir
        return cfg

    def test_cache_file_created(self):
        """A train_packed_idx.npz cache file should be written when packed_idx_cache_dir is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            create_dataset(
                task_group=self.dataset_path,
                task_group_prob="1.0",
                sub_dataset_type="erniekit",
                **cfg,
            )
            expected_cache = os.path.join(tmpdir, "train_packed_idx.npz")
            self.assertTrue(os.path.isfile(expected_cache), f"Cache file not found: {expected_cache}")

    def test_cache_loaded_on_second_init(self):
        """Second init with same cache_dir should load from cache and produce identical packed_idx."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds1 = create_dataset(
                task_group=self.dataset_path,
                task_group_prob="1.0",
                sub_dataset_type="erniekit",
                **self._make_cfg(tmpdir),
            )
            ds2 = create_dataset(
                task_group=self.dataset_path,
                task_group_prob="1.0",
                sub_dataset_type="erniekit",
                **self._make_cfg(tmpdir),
            )
            self.assertEqual(len(ds1.packed_idx), len(ds2.packed_idx))
            for g1, g2 in zip(ds1.packed_idx, ds2.packed_idx):
                self.assertEqual(sorted(g1), sorted(g2))

    def test_eval_cache_file_name(self):
        """is_valid=True should write cache under eval_packed_idx.npz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir, is_valid=True)
            create_dataset(
                task_group=self.dataset_path,
                task_group_prob="1.0",
                sub_dataset_type="erniekit",
                **cfg,
            )
            expected_cache = os.path.join(tmpdir, "eval_packed_idx.npz")
            self.assertTrue(os.path.isfile(expected_cache), f"Eval cache file not found: {expected_cache}")

    def test_corrupt_cache_falls_back_to_recompute(self):
        """A corrupt .npz cache should be silently ignored and packed_idx recomputed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "train_packed_idx.npz")
            with open(cache_file, "wb") as f:
                f.write(b"not a valid npz file")

            ds = create_dataset(
                task_group=self.dataset_path,
                task_group_prob="1.0",
                sub_dataset_type="erniekit",
                **self._make_cfg(tmpdir),
            )
            self.assertIsNotNone(ds.packed_idx)
            self.assertGreater(len(ds.packed_idx), 0)
