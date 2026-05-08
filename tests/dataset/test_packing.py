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

import os
import unittest

from paddleformers.cli.utils.process import add_new_special_tokens
from paddleformers.datasets.loader import create_dataset as create_dataset_sft
from paddleformers.datasets.template.template import get_template_and_fix_tokenizer
from paddleformers.transformers import (
    AutoProcessor,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
)
from tests.testing_utils import get_tests_dir

MODEL_NAME_OR_PATH = "/home/models/PaddleFormers/tiny-random-glm4moe-bf16/"
MAX_SEQ_LEN = 8192
SEED = 23


class TestPacking(unittest.TestCase):
    def _build_dataset(self, packing, binpacking=False, greedy_intokens=False):
        """Build a dataset with the given packing config and return the dataset and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        add_new_special_tokens(tokenizer, None)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if isinstance(tokenizer, (LlamaTokenizer, Llama3Tokenizer)):
            tokenizer.pad_token_id = tokenizer.eos_token_id

        processor = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, use_fast=None)

        dataset_config = {
            "tokenizer": tokenizer,
            "processor": processor,
            "max_seq_len": MAX_SEQ_LEN,
            "random_seed": SEED,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": 6000000,
            "random_shuffle": False,
            "greedy_intokens": greedy_intokens,
            "packing": packing,
            "mix_strategy": "concat",
            "encode_one_turn": False,
            "use_template": True,
            "is_pretraining": False,
            "truncate_packing": False,
            "stage": "sft",
            "template_backend": "custom",
            "split_multi_turn": False,
            "dataset_type": "iterable",
            "truncation_strategy": "delete",
            "dtype": "bfloat16",
            "dataset_num_proc": 1,
            "binpacking": binpacking,
            "packing_interval": 1000,
            "dataloader_num_workers": 0,
            "template": "glm4_moe",
            "tool_format": None,
            "default_system": None,
        }

        if dataset_config["template_backend"] == "custom":
            template_instance = get_template_and_fix_tokenizer(dataset_config)
        else:
            template_instance = None
        dataset_config["template_instance"] = template_instance

        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "sft", "train.jsonl")

        train_dataset = create_dataset_sft(
            task_group=dataset_path,
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **dataset_config,
        )
        return train_dataset, tokenizer

    def test_no_packing_single_sample_per_batch(self):
        """Without packing, each batch should contain exactly one sample."""
        dataset, _ = self._build_dataset(packing=False)
        batch = next(iter(dataset))
        self.assertEqual(len(batch), 1)

    def test_base_packing_multiple_samples_per_batch(self):
        """With base packing (no binpacking, no greedy), multiple samples should be packed."""
        dataset, _ = self._build_dataset(packing=True, binpacking=False, greedy_intokens=False)
        batch = next(iter(dataset))
        self.assertGreater(len(batch), 1)

    def test_greedy_packing_multiple_samples_per_batch(self):
        """With greedy packing, multiple samples should be packed into one batch."""
        dataset, _ = self._build_dataset(packing=True, binpacking=False, greedy_intokens=True)
        batch = next(iter(dataset))
        self.assertGreater(len(batch), 1)

    def test_binpacking_multiple_samples_per_batch(self):
        """With binpacking, a batch should contain multiple samples when they fit within max_seq_len."""
        dataset, _ = self._build_dataset(packing=True, binpacking=True)
        batch = next(iter(dataset))
        # Test data has ~10 short samples (~50 tokens each), all fit within max_seq_len=8192
        self.assertGreater(len(batch), 1)

    def test_greedy_packing_total_tokens_within_limit(self):
        """Total tokens in a greedy-packed batch should not exceed max_seq_len."""
        dataset, _ = self._build_dataset(packing=True, binpacking=True)
        batch = next(iter(dataset))
        total_tokens = sum(len(sample.token_ids) for sample in batch)
        self.assertLessEqual(total_tokens, MAX_SEQ_LEN)

    def test_greedy_packing_position_ids_reset_per_sample(self):
        """Each sample in a greedy-packed batch should have position_ids starting from 0."""
        dataset, _ = self._build_dataset(packing=True, binpacking=True)
        batch = next(iter(dataset))
        for sample in batch:
            self.assertEqual(sample.position_ids[0], 0)
            expected_position_ids = list(range(len(sample.position_ids)))
            self.assertEqual(sample.position_ids, expected_position_ids)

    def test_packing_preserves_content(self):
        """Packing should not alter the content of individual samples."""
        no_pack_ds, _ = self._build_dataset(packing=False)
        no_pack_sample = next(iter(no_pack_ds))[0]

        pack_ds, _ = self._build_dataset(packing=True, binpacking=False, greedy_intokens=False)
        pack_first_sample = next(iter(pack_ds))[0]

        self.assertEqual(pack_first_sample.token_ids, no_pack_sample.token_ids)
        self.assertEqual(pack_first_sample.labels, no_pack_sample.labels)
        self.assertEqual(pack_first_sample.position_ids, no_pack_sample.position_ids)
