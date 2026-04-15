# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

from paddleformers.datasets.loader import create_dataset
from paddleformers.datasets.reader.mix_datasets import ConcatDataset, InterLeaveDataset
from paddleformers.transformers import AutoTokenizer
from tests.testing_utils import get_tests_dir

MODEL_NAME = "/home/models/PaddleFormers/tiny-random-glm4moe-bf16/"
MAX_SEQ_LEN = 8192
SEED = 42
NUM_SAMPLES_EACH_EPOCH = 6000000


def _get_dataset_path(subdir):
    """Return the path to the dummy train.jsonl under the given subdirectory."""
    dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
    return os.path.join(dataset_dir, subdir, "train.jsonl")


# ===========================================================================
# Integration tests — require tokenizer and real data files
# ===========================================================================


class TestPTDataset(unittest.TestCase):
    def _build_dataset(self, mix_strategy, multi_source=False):
        """Build a PT dataset with the given mix strategy."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        dataset_path = _get_dataset_path("pt")

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": MAX_SEQ_LEN,
            "random_seed": SEED,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": NUM_SAMPLES_EACH_EPOCH,
            "random_shuffle": True,
            "greedy_intokens": True,
            "packing": False,
            "mix_strategy": mix_strategy,
            "encode_one_turn": True,
            "use_template": True,
            "is_pretraining": True,
            "truncate_packing": True,
            "stage": "PT",
        }

        if multi_source:
            task_group = ", ".join([dataset_path, dataset_path])
            task_group_prob = "1.0,1.0"
            sub_dataset_type = "erniekit,erniekit"
        else:
            task_group = dataset_path
            task_group_prob = "1.0"
            sub_dataset_type = "erniekit"

        return create_dataset(
            task_group=task_group,
            task_group_prob=task_group_prob,
            sub_dataset_type=sub_dataset_type,
            **dataset_config,
        )

    def test_random_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="random")
        self.assertEqual(len(train_dataset.mix_datasets), NUM_SAMPLES_EACH_EPOCH)

    def test_concat_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="concat", multi_source=True)
        self.assertEqual(len(train_dataset.mix_datasets), 40)

    def test_interleave_under_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="interleave_under", multi_source=True)
        self.assertEqual(len(train_dataset.mix_datasets), 39)

    def test_interleave_over_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="interleave_over", multi_source=True)
        self.assertEqual(len(train_dataset.mix_datasets), 40)


class TestSFTDataset(unittest.TestCase):
    def _build_dataset(self, mix_strategy, multi_source=False):
        """Build a SFT dataset with the given mix strategy."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        dataset_path = _get_dataset_path("sft")

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": MAX_SEQ_LEN,
            "random_seed": SEED,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": NUM_SAMPLES_EACH_EPOCH,
            "random_shuffle": True,
            "greedy_intokens": True,
            "packing": False,
            "mix_strategy": mix_strategy,
            "encode_one_turn": True,
            "use_template": True,
            "is_pretraining": False,
            "truncate_packing": True,
            "stage": "SFT",
        }

        if multi_source:
            task_group = ", ".join([dataset_path, dataset_path])
            task_group_prob = "1.0,1.0"
            sub_dataset_type = "erniekit,erniekit"
        else:
            task_group = dataset_path
            task_group_prob = "1.0"
            sub_dataset_type = "erniekit"

        return create_dataset(
            task_group=task_group,
            task_group_prob=task_group_prob,
            sub_dataset_type=sub_dataset_type,
            **dataset_config,
        )

    def test_random_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="random")
        self.assertEqual(len(train_dataset.mix_datasets), NUM_SAMPLES_EACH_EPOCH)

    def test_concat_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="concat", multi_source=True)
        self.assertEqual(len(train_dataset.mix_datasets), 20)

    def test_interleave_under_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="interleave_under", multi_source=True)
        self.assertEqual(len(train_dataset.mix_datasets), 17)

    def test_interleave_over_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="interleave_over", multi_source=True)
        self.assertEqual(len(train_dataset.mix_datasets), 26)


class TestDPODataset(unittest.TestCase):
    def _build_dataset(self, mix_strategy, multi_source=False):
        """Build a DPO dataset with the given mix strategy."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        dataset_path = _get_dataset_path("dpo")

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": MAX_SEQ_LEN,
            "max_prompt_len": 2048,
            "random_seed": SEED,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": NUM_SAMPLES_EACH_EPOCH,
            "random_shuffle": True,
            "greedy_intokens": True,
            "buffer_size": 500,
            "use_attn_mask_startend_row_indices": True,
            "packing": False,
            "mix_strategy": mix_strategy,
            "encode_one_turn": True,
            "stage": "DPO",
        }

        if multi_source:
            task_group = ", ".join([dataset_path, dataset_path])
            task_group_prob = "1.0,1.0"
            sub_dataset_type = "erniekit,erniekit"
        else:
            task_group = dataset_path
            task_group_prob = "1.0"
            sub_dataset_type = "erniekit"

        return create_dataset(
            task_group=task_group,
            task_group_prob=task_group_prob,
            sub_dataset_type=sub_dataset_type,
            **dataset_config,
        )

    def test_random_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="random")
        self.assertEqual(len(train_dataset.mix_datasets), NUM_SAMPLES_EACH_EPOCH)

    def test_concat_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="concat", multi_source=True)
        self.assertEqual(len(train_dataset.mix_datasets), 20)

    def test_interleave_under_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="interleave_under", multi_source=True)
        self.assertEqual(len(train_dataset.mix_datasets), 17)

    def test_interleave_over_dataset_len(self):
        train_dataset = self._build_dataset(mix_strategy="interleave_over", multi_source=True)
        self.assertEqual(len(train_dataset.mix_datasets), 26)


# ===========================================================================
# Unit tests — no tokenizer or model required
# ===========================================================================


def _make_list_dataset(items):
    """Wrap a plain list so it behaves like an iterable dataset."""

    class _ListDataset:
        def __iter__(self):
            return iter(items)

    return _ListDataset()


def _make_mock_multi_source(datasets_and_probs):
    """Build a minimal multi_source_dataset stub accepted by BaseMixDataset."""

    class _MultiSource:
        pass

    ms = _MultiSource()
    ms._task_group = [{"dataset": ds, "prob": prob} for ds, prob in datasets_and_probs]
    return ms


BASE_CONFIG = {
    "random_seed": 42,
    "random_shuffle": False,
    "num_samples_each_epoch": 10,
    "reverse": False,
}


class TestConcatDatasetContent(unittest.TestCase):
    def test_all_items_present(self):
        """ConcatDataset should contain every item from every source dataset."""
        ds_a = _make_list_dataset([{"id": 0}, {"id": 1}])
        ds_b = _make_list_dataset([{"id": 2}, {"id": 3}])
        ms = _make_mock_multi_source([(ds_a, 0.5), (ds_b, 0.5)])
        concat = ConcatDataset(ms, **{**BASE_CONFIG, "mix_strategy": "concat"})
        ids = sorted(r["id"] for r in concat)
        self.assertEqual(ids, [0, 1, 2, 3])

    def test_length_equals_sum_of_sources(self):
        ds_a = _make_list_dataset([1, 2, 3])
        ds_b = _make_list_dataset([4, 5])
        ms = _make_mock_multi_source([(ds_a, 0.5), (ds_b, 0.5)])
        concat = ConcatDataset(ms, **{**BASE_CONFIG, "mix_strategy": "concat"})
        self.assertEqual(len(concat), 5)


class TestProbabilityNormalization(unittest.TestCase):
    def test_unnormalized_probs_are_normalized(self):
        """Probabilities that don't sum to 1.0 should be auto-normalized."""
        ds_a = _make_list_dataset([1, 2])
        ds_b = _make_list_dataset([3, 4])
        ms = _make_mock_multi_source([(ds_a, 1.0), (ds_b, 1.0)])  # sum = 2.0
        concat = ConcatDataset(ms, **{**BASE_CONFIG, "mix_strategy": "concat"})
        self.assertAlmostEqual(sum(concat.datasets_prob), 1.0, places=6)

    def test_already_normalized_probs_unchanged(self):
        ds_a = _make_list_dataset([1])
        ds_b = _make_list_dataset([2])
        ms = _make_mock_multi_source([(ds_a, 0.3), (ds_b, 0.7)])
        concat = ConcatDataset(ms, **{**BASE_CONFIG, "mix_strategy": "concat"})
        self.assertAlmostEqual(concat.datasets_prob[0], 0.3, places=6)
        self.assertAlmostEqual(concat.datasets_prob[1], 0.7, places=6)


class TestInterleaveDatasetContent(unittest.TestCase):
    def test_interleave_under_stops_at_first_exhausted(self):
        """interleave_under should stop as soon as any source is exhausted once."""
        ds_a = _make_list_dataset([{"src": "a", "i": i} for i in range(2)])
        ds_b = _make_list_dataset([{"src": "b", "i": i} for i in range(10)])
        ms = _make_mock_multi_source([(ds_a, 0.5), (ds_b, 0.5)])
        ds = InterLeaveDataset(ms, **{**BASE_CONFIG, "mix_strategy": "interleave_under"})
        self.assertLessEqual(len(ds), 10)

    def test_interleave_over_uses_all_items_from_both_sources(self):
        """interleave_over keeps going until all sources exhausted; both fully used."""
        ds_a = _make_list_dataset([{"src": "a", "i": i} for i in range(3)])
        ds_b = _make_list_dataset([{"src": "b", "i": i} for i in range(3)])
        ms = _make_mock_multi_source([(ds_a, 0.5), (ds_b, 0.5)])
        ds = InterLeaveDataset(ms, **{**BASE_CONFIG, "mix_strategy": "interleave_over"})
        sources = [r["src"] for r in ds]
        self.assertGreaterEqual(sources.count("a"), 3)
        self.assertGreaterEqual(sources.count("b"), 3)
