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

from paddleformers.datasets.dpo import create_dataset as create_dataset_dpo
from paddleformers.datasets.finetuning import create_dataset as create_dataset_sft
from paddleformers.transformers import AutoTokenizer
from tests.testing_utils import get_tests_dir


class TestSFTDataset(unittest.TestCase):
    def test_random_dataset_len(self):

        ernie_dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        ernie_dataset_path = os.path.join(ernie_dataset_dir, "ernie", "sft-train.jsonl")

        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")
        num_samples_each_epoch = 6000000

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": 8192,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": num_samples_each_epoch,
            "random_shuffle": True,
            "greedy_intokens": True,
            "packing": False,
            "mix_strategy": "random",
            "encode_one_turn": True,
            "use_template": True,
        }

        train_dataset = create_dataset_sft(
            task_group=ernie_dataset_path,
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **dataset_config,
        )

        self.assertEqual(len(train_dataset.mix_datasets), num_samples_each_epoch)

    def test_concat_dataset_len(self):
        ernie_dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        ernie_dataset_path = os.path.join(ernie_dataset_dir, "ernie", "sft-train.jsonl")

        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": 8192,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": 6000000,
            "random_shuffle": True,
            "greedy_intokens": True,
            "packing": False,
            "mix_strategy": "concat",
            "encode_one_turn": True,
            "use_template": True,
        }

        train_dataset = create_dataset_sft(
            task_group=", ".join([ernie_dataset_path, ernie_dataset_path]),
            task_group_prob="1.0,1.0",
            sub_dataset_type="erniekit,erniekit",
            **dataset_config,
        )

        self.assertEqual(len(train_dataset.mix_datasets), 20)

    def test_interleave_under_dataset_len(self):
        ernie_dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        ernie_dataset_path = os.path.join(ernie_dataset_dir, "ernie", "sft-train.jsonl")

        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": 8192,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": 6000000,
            "random_shuffle": True,
            "greedy_intokens": True,
            "packing": False,
            "mix_strategy": "interleave_under",
            "encode_one_turn": True,
            "use_template": True,
        }

        train_dataset = create_dataset_sft(
            task_group=", ".join([ernie_dataset_path, ernie_dataset_path]),
            task_group_prob="1.0,1.0",
            sub_dataset_type="erniekit,erniekit",
            **dataset_config,
        )

        self.assertEqual(len(train_dataset.mix_datasets), 17)

    def test_interleave_over_dataset_len(self):
        ernie_dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        ernie_dataset_path = os.path.join(ernie_dataset_dir, "ernie", "sft-train.jsonl")

        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": 8192,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": 6000000,
            "random_shuffle": True,
            "greedy_intokens": True,
            "packing": False,
            "mix_strategy": "interleave_over",
            "encode_one_turn": True,
            "use_template": True,
        }

        train_dataset = create_dataset_sft(
            task_group=", ".join([ernie_dataset_path, ernie_dataset_path]),
            task_group_prob="1.0,1.0",
            sub_dataset_type="erniekit,erniekit",
            **dataset_config,
        )

        self.assertEqual(len(train_dataset.mix_datasets), 26)


class TestDPODataset(unittest.TestCase):
    def test_random_dataset_len(self):

        ernie_dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        ernie_dataset_path = os.path.join(ernie_dataset_dir, "ernie", "dpo-train.jsonl")

        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")

        num_samples_each_epoch = 6000000

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": 8192,
            "max_prompt_len": 2048,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": num_samples_each_epoch,
            "random_shuffle": True,
            "greedy_intokens": True,
            "buffer_size": 500,
            "use_attn_mask_startend_row_indices": True,
            "mask_out_eos_token": True,
            "packing": False,
            "mix_strategy": "random",
            "encode_one_turn": True,
        }

        train_dataset = create_dataset_dpo(
            task_group=ernie_dataset_path,
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **dataset_config,
        )

        self.assertEqual(len(train_dataset.mix_datasets), num_samples_each_epoch)

    def test_concat_dataset_len(self):

        ernie_dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        ernie_dataset_path = os.path.join(ernie_dataset_dir, "ernie", "dpo-train.jsonl")

        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")

        num_samples_each_epoch = 6000000

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": 8192,
            "max_prompt_len": 2048,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": num_samples_each_epoch,
            "random_shuffle": True,
            "greedy_intokens": True,
            "buffer_size": 500,
            "use_attn_mask_startend_row_indices": True,
            "mask_out_eos_token": True,
            "packing": False,
            "mix_strategy": "concat",
            "encode_one_turn": True,
        }

        train_dataset = create_dataset_dpo(
            task_group=", ".join([ernie_dataset_path, ernie_dataset_path]),
            task_group_prob="1.0,1.0",
            sub_dataset_type="erniekit,erniekit",
            **dataset_config,
        )

        self.assertEqual(len(train_dataset.mix_datasets), 20)

    def test_interleave_under_dataset_len(self):

        ernie_dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        ernie_dataset_path = os.path.join(ernie_dataset_dir, "ernie", "dpo-train.jsonl")

        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")

        num_samples_each_epoch = 6000000

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": 8192,
            "max_prompt_len": 2048,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": num_samples_each_epoch,
            "random_shuffle": True,
            "greedy_intokens": True,
            "buffer_size": 500,
            "use_attn_mask_startend_row_indices": True,
            "mask_out_eos_token": True,
            "packing": False,
            "mix_strategy": "interleave_under",
            "encode_one_turn": True,
        }

        train_dataset = create_dataset_dpo(
            task_group=", ".join([ernie_dataset_path, ernie_dataset_path]),
            task_group_prob="1.0,1.0",
            sub_dataset_type="erniekit,erniekit",
            **dataset_config,
        )

        self.assertEqual(len(train_dataset.mix_datasets), 17)

    def test_interleave_over_dataset_len(self):

        ernie_dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        ernie_dataset_path = os.path.join(ernie_dataset_dir, "ernie", "dpo-train.jsonl")

        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")

        num_samples_each_epoch = 6000000

        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": 8192,
            "max_prompt_len": 2048,
            "random_seed": 42,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": num_samples_each_epoch,
            "random_shuffle": True,
            "greedy_intokens": True,
            "buffer_size": 500,
            "use_attn_mask_startend_row_indices": True,
            "mask_out_eos_token": True,
            "packing": False,
            "mix_strategy": "interleave_over",
            "encode_one_turn": True,
        }

        train_dataset = create_dataset_dpo(
            task_group=", ".join([ernie_dataset_path, ernie_dataset_path]),
            task_group_prob="1.0,1.0",
            sub_dataset_type="erniekit,erniekit",
            **dataset_config,
        )

        self.assertEqual(len(train_dataset.mix_datasets), 26)
