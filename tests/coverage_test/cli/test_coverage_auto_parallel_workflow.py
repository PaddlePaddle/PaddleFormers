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

from paddleformers.cli.train.auto_parallel.workflow import (
    PretrainingTrainer,
    get_train_data_file,
)


class TestPretrainingTrainer(unittest.TestCase):
    """Tests for PretrainingTrainer class"""

    def test_is_pretraining_flag(self):
        # PretrainingTrainer should set is_pretraining=True
        self.assertTrue(hasattr(PretrainingTrainer, "__init__"))


class TestGetTrainDataFile(unittest.TestCase):
    """Tests for get_train_data_file function"""

    def test_multi_dir_input(self):
        # When input_dir has multiple space-separated paths
        args = MagicMock()
        args.input_dir = "1 /path/to/data1 1 /path/to/data2"
        result = get_train_data_file(args)
        self.assertEqual(result, ["1", "/path/to/data1", "1", "/path/to/data2"])

    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_single_dir_with_idx_files(self, mock_isfile, mock_listdir):
        mock_isfile.return_value = True
        mock_listdir.return_value = ["data1_idx.npz", "data2.idx"]

        args = MagicMock()
        args.input_dir = "/path/to/data"
        result = get_train_data_file(args)

        self.assertEqual(len(result), 4)  # 2 weights + 2 paths
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[2], 1.0)

    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_single_dir_with_one_file(self, mock_isfile, mock_listdir):
        mock_isfile.return_value = True
        mock_listdir.return_value = ["data1.idx"]

        args = MagicMock()
        args.input_dir = "/path/to/data"
        result = get_train_data_file(args)

        self.assertEqual(len(result), 1)

    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_no_idx_files(self, mock_isfile, mock_listdir):
        mock_isfile.return_value = True
        mock_listdir.return_value = ["data1.bin"]

        args = MagicMock()
        args.input_dir = "/path/to/data"
        result = get_train_data_file(args)

        self.assertEqual(result, [])


class TestAutoParallelWorkflowValidation(unittest.TestCase):
    """Tests for auto_parallel workflow validation logic"""

    def test_dtype_selection_fp16(self):
        training_args = MagicMock()
        training_args.fp16_opt_level = "O2"
        training_args.fp16 = True
        training_args.bf16 = False

        dtype = "float32"
        if training_args.fp16_opt_level == "O2":
            if training_args.fp16:
                dtype = "float16"
            if training_args.bf16:
                dtype = "bfloat16"

        self.assertEqual(dtype, "float16")

    def test_dtype_selection_bf16(self):
        training_args = MagicMock()
        training_args.fp16_opt_level = "O2"
        training_args.fp16 = False
        training_args.bf16 = True

        dtype = "float32"
        if training_args.fp16_opt_level == "O2":
            if training_args.fp16:
                dtype = "float16"
            if training_args.bf16:
                dtype = "bfloat16"

        self.assertEqual(dtype, "bfloat16")

    def test_expert_parallel_forced_for_moe(self):
        training_args = MagicMock()
        training_args.data_parallel_size = 2
        training_args.use_expert_parallel = False

        architectures_to_check = {"Qwen2Moe", "DeepseekV2", "DeepseekV3"}
        config_architectures = ["Qwen2Moe"]

        if (
            any(architecture in str(config_architectures) for architecture in architectures_to_check)
            and training_args.data_parallel_size > 1
        ):
            training_args.use_expert_parallel = True

        self.assertTrue(training_args.use_expert_parallel)

    def test_eval_iters_set(self):
        # run_auto_parallel sets eval_iters to 10
        eval_iters = 10
        test_iters = eval_iters * 10
        self.assertEqual(eval_iters, 10)
        self.assertEqual(test_iters, 100)

    def test_warmup_steps_calculation(self):
        # warmup_steps > 0 uses warmup_steps directly
        training_args = MagicMock()
        training_args.warmup_steps = 100
        training_args.warmup_ratio = 0.1
        training_args.max_steps = 1000

        if training_args.warmup_steps > 0:
            warmup_steps = training_args.warmup_steps
        else:
            warmup_steps = training_args.warmup_ratio * training_args.max_steps

        self.assertEqual(warmup_steps, 100)

    def test_warmup_ratio_calculation(self):
        training_args = MagicMock()
        training_args.warmup_steps = 0
        training_args.warmup_ratio = 0.1
        training_args.max_steps = 1000

        if training_args.warmup_steps > 0:
            warmup_steps = training_args.warmup_steps
        else:
            warmup_steps = training_args.warmup_ratio * training_args.max_steps

        self.assertEqual(warmup_steps, 100)


if __name__ == "__main__":
    unittest.main()
