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

import sys
import types
import unittest
from dataclasses import fields

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod

from paddleformers.cli.train.dpo.data_config import DataConfig


class TestDataConfig(unittest.TestCase):
    """Tests for DataConfig dataclass"""

    def test_default_values(self):
        config = DataConfig()
        self.assertIsNone(config.dataset_name_or_path)
        self.assertIsNone(config.train_dataset_type)
        self.assertIsNone(config.train_dataset_path)
        self.assertIsNone(config.train_dataset_prob)
        self.assertEqual(config.eval_dataset_type, "erniekit")
        self.assertEqual(config.eval_dataset_path, "examples/data/sft-eval.jsonl")
        self.assertEqual(config.eval_dataset_prob, "1.0")
        self.assertEqual(config.dataset_type, "iterable")
        self.assertIsNone(config.input_dir)
        self.assertEqual(config.split, "950,50")
        self.assertEqual(config.mix_strategy, "concat")
        self.assertTrue(config.use_template)
        self.assertTrue(config.encode_one_turn)
        self.assertFalse(config.packing)
        self.assertTrue(config.greedy_intokens)
        self.assertTrue(config.random_shuffle)
        self.assertEqual(config.num_samples_each_epoch, 6000000)
        self.assertIsNone(config.task_name)
        self.assertIsNone(config.pad_to_multiple_of)
        self.assertFalse(config.eval_with_do_generation)
        self.assertFalse(config.save_generation_output)
        self.assertFalse(config.lazy)
        self.assertIsNone(config.chat_template)
        self.assertFalse(config.pad_to_max_length)
        self.assertFalse(config.autoregressive)
        self.assertFalse(config.use_pose_convert)

    def test_custom_values(self):
        config = DataConfig(
            dataset_name_or_path="my_dataset",
            train_dataset_type="erniekit",
            train_dataset_path="./train.jsonl",
            train_dataset_prob="1.0",
            dataset_type="map",
            split="800,200",
            mix_strategy="random",
            use_template=False,
            encode_one_turn=False,
            packing=True,
            greedy_intokens=False,
            random_shuffle=False,
            num_samples_each_epoch=1000,
            eval_with_do_generation=True,
            save_generation_output=True,
            lazy=True,
            chat_template="custom_template",
            pad_to_max_length=True,
            autoregressive=True,
            use_pose_convert=True,
        )
        self.assertEqual(config.dataset_name_or_path, "my_dataset")
        self.assertEqual(config.train_dataset_type, "erniekit")
        self.assertEqual(config.train_dataset_path, "./train.jsonl")
        self.assertEqual(config.dataset_type, "map")
        self.assertEqual(config.split, "800,200")
        self.assertEqual(config.mix_strategy, "random")
        self.assertFalse(config.use_template)
        self.assertFalse(config.encode_one_turn)
        self.assertTrue(config.packing)
        self.assertFalse(config.greedy_intokens)
        self.assertFalse(config.random_shuffle)
        self.assertEqual(config.num_samples_each_epoch, 1000)
        self.assertTrue(config.eval_with_do_generation)
        self.assertTrue(config.save_generation_output)
        self.assertTrue(config.lazy)
        self.assertEqual(config.chat_template, "custom_template")
        self.assertTrue(config.pad_to_max_length)
        self.assertTrue(config.autoregressive)
        self.assertTrue(config.use_pose_convert)

    def test_fields_count(self):
        all_fields = [f.name for f in fields(DataConfig)]
        # The exact count may vary, just check it has many fields
        self.assertGreaterEqual(len(all_fields), 20)

    def test_all_in_module(self):
        self.assertEqual(DataConfig.__module__, "paddleformers.cli.train.dpo.data_config")

    def test_multiple_dataset_paths(self):
        config = DataConfig(
            train_dataset_path="./sft-1.jsonl,./sft-2.jsonl",
            train_dataset_prob="0.8,0.2",
            train_dataset_type="erniekit,erniekit",
        )
        self.assertEqual(config.train_dataset_path, "./sft-1.jsonl,./sft-2.jsonl")
        self.assertEqual(config.train_dataset_prob, "0.8,0.2")
        self.assertEqual(config.train_dataset_type, "erniekit,erniekit")


if __name__ == "__main__":
    unittest.main()
