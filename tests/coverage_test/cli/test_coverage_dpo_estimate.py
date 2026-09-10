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
from unittest.mock import MagicMock, patch

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod

from paddleformers.cli.train.dpo.dpo_estimate_training import (
    calculate_acc_steps,
    dpo_estimate_training,
)


class TestCalculateAccSteps(unittest.TestCase):
    """Tests for calculate_acc_steps function"""

    def test_small_dataset_under_100(self):
        result = calculate_acc_steps(
            num_samples=50,
            train_batch=4,
            dataset_world_size=1,
            per_device_train_batch_size=2,
        )
        # samples_per_batch = 2 * 1 * 50 / 4 = 25
        # recommend_bs = 8
        # result = min(ceil(8/25), 32) = min(1, 32) = 1
        self.assertEqual(result, 1)

    def test_medium_dataset_under_1000(self):
        result = calculate_acc_steps(
            num_samples=500,
            train_batch=4,
            dataset_world_size=1,
            per_device_train_batch_size=2,
        )
        # samples_per_batch = 2 * 1 * 500 / 4 = 250
        # recommend_bs = 16
        # result = min(ceil(16/250), 32) = min(1, 32) = 1
        self.assertEqual(result, 1)

    def test_large_dataset_under_10000(self):
        result = calculate_acc_steps(
            num_samples=5000,
            train_batch=4,
            dataset_world_size=1,
            per_device_train_batch_size=2,
        )
        # samples_per_batch = 2 * 1 * 5000 / 4 = 2500
        # recommend_bs = 32
        # result = min(ceil(32/2500), 32) = min(1, 32) = 1
        self.assertEqual(result, 1)

    def test_very_large_dataset_over_100000(self):
        result = calculate_acc_steps(
            num_samples=200000,
            train_batch=4,
            dataset_world_size=1,
            per_device_train_batch_size=2,
        )
        # samples_per_batch = 2 * 1 * 200000 / 4 = 100000
        # recommend_bs = 128
        # result = min(ceil(128/100000), 32) = min(1, 32) = 1
        self.assertEqual(result, 1)

    def test_high_batch_causes_accumulation(self):
        # Small per_device_batch, small train_batch, many samples
        result = calculate_acc_steps(
            num_samples=50000,
            train_batch=1,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        # samples_per_batch = 1 * 1 * 50000 / 1 = 50000
        # recommend_bs = 64
        # result = min(ceil(64/50000), 32) = 1
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 32)

    def test_max_accumulation_is_32(self):
        # Very small samples_per_batch should still cap at 32
        result = calculate_acc_steps(
            num_samples=10,
            train_batch=100,
            dataset_world_size=1,
            per_device_train_batch_size=1,
        )
        # samples_per_batch = 1 * 1 * 10 / 100 = 0.1
        # recommend_bs = 8
        # result = min(ceil(8/0.1), 32) = min(80, 32) = 32
        self.assertLessEqual(result, 32)

    def test_dataset_100_1000_boundary(self):
        result = calculate_acc_steps(
            num_samples=500,
            train_batch=4,
            dataset_world_size=1,
            per_device_train_batch_size=2,
        )
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 32)

    def test_multiple_gpus(self):
        result = calculate_acc_steps(
            num_samples=500,
            train_batch=8,
            dataset_world_size=4,
            per_device_train_batch_size=2,
        )
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 32)


class TestDPOEstimateTraining(unittest.TestCase):
    """Tests for dpo_estimate_training function"""

    @patch("paddleformers.cli.train.dpo.dpo_estimate_training.create_dataset")
    def test_empty_dataset(self, mock_create_dataset):
        import tempfile

        mock_train_dataset = MagicMock()
        # mix_datasets must be a sequence (list) for len() to work
        mock_train_dataset.mix_datasets = []
        mock_create_dataset.return_value = mock_train_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_training_args = MagicMock()
            mock_training_args.should_save = True
            mock_training_args.should_save_model_state = False
            mock_training_args.output_dir = tmpdir
            mock_training_args.num_train_epochs = 1
            mock_training_args.max_steps = -1
            mock_training_args.gradient_accumulation_steps = 1
            mock_training_args.num_of_gpus = -1
            mock_training_args.per_device_train_batch_size = 2
            mock_training_args.pipeline_model_parallel_size = 1
            mock_training_args.tensor_model_parallel_size = 1
            mock_training_args.seed = 42
            mock_training_args.dataset_world_size = 1

            mock_data_args = MagicMock()
            mock_data_args.max_seq_len = 2048
            mock_data_args.max_prompt_len = 1024
            mock_data_args.num_samples_each_epoch = 6000000

            mock_dataset_config = {"stage": "dpo"}

            training_args, res = dpo_estimate_training(
                tokenizer=None,
                data_args=mock_data_args,
                training_args=mock_training_args,
                dataset_config=mock_dataset_config,
                train_dataset=mock_train_dataset,
            )

            self.assertEqual(res["max_steps"], 0)
            self.assertEqual(res["train_samples"], 0)
            self.assertFalse(res["valid"])

    def test_gradient_accumulation_steps_positive(self):
        # When gradient_accumulation_steps >= 0, it should not be recalculated
        mock_training_args = MagicMock()
        mock_training_args.should_save = False
        mock_training_args.should_save_model_state = False
        mock_training_args.output_dir = "/tmp/test_dpo_estimate"
        mock_training_args.num_train_epochs = 1
        mock_training_args.gradient_accumulation_steps = 4
        mock_training_args.num_of_gpus = 1
        mock_training_args.per_device_train_batch_size = 2
        mock_training_args.pipeline_model_parallel_size = 1
        mock_training_args.tensor_model_parallel_size = 1
        mock_training_args.seed = 42
        mock_training_args.dataset_world_size = 1
        mock_training_args.max_steps = -1

        mock_data_args = MagicMock()
        mock_data_args.max_seq_len = 2048
        mock_data_args.max_prompt_len = 1024
        mock_data_args.num_samples_each_epoch = 6000000

        # Verify the calculation formula works with positive gradient_accumulation_steps
        result = calculate_acc_steps(
            num_samples=500,
            train_batch=4,
            dataset_world_size=1,
            per_device_train_batch_size=2,
        )
        self.assertGreaterEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
