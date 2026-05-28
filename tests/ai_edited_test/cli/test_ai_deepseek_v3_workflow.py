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

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

# Direct import to avoid __init__.py triggering workflow.py which requires AutoTokenizer
_MODULE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "paddleformers", "cli", "train", "deepseek_v3_pretrain"
)
_MODULE_DIR = os.path.abspath(_MODULE_DIR)


def _load_module(name, path, full_name=None):
    """Load a module directly from file path without going through __init__.py."""
    if full_name is None:
        full_name = f"paddleformers.cli.train.deepseek_v3_pretrain.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-populate the package in sys.modules to prevent __init__.py from loading
_pkg_name = "paddleformers.cli.train.deepseek_v3_pretrain"
if _pkg_name not in sys.modules:
    _pkg_mod = types.ModuleType(_pkg_name)
    _pkg_mod.__path__ = [_MODULE_DIR]
    _pkg_mod.__package__ = _pkg_name
    sys.modules[_pkg_name] = _pkg_mod

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod

# Pre-load configuration module (dependency of workflow)
_config_mod = _load_module("configuration", os.path.join(_MODULE_DIR, "configuration.py"))

# Pre-load moe_utils module (dependency of workflow)
_moe_utils_mod = _load_module("moe_utils", os.path.join(_MODULE_DIR, "moe_utils.py"))

# Now load workflow module directly
_workflow_mod = _load_module("workflow", os.path.join(_MODULE_DIR, "workflow.py"))

PreTrainingArguments = _workflow_mod.PreTrainingArguments
DataArguments = _workflow_mod.DataArguments
ModelArguments = _workflow_mod.ModelArguments
PretrainingTrainer = _workflow_mod.PretrainingTrainer
get_train_data_file = _workflow_mod.get_train_data_file


class TestPreTrainingArguments(unittest.TestCase):
    """Test PreTrainingArguments dataclass."""

    def test_default_values(self):
        """Test default values of PreTrainingArguments."""
        self.assertEqual(PreTrainingArguments.min_learning_rate, 1e-5)
        self.assertIsNone(PreTrainingArguments.decay_steps)
        self.assertFalse(PreTrainingArguments.enable_linear_fused_grad_add)
        self.assertFalse(PreTrainingArguments.autotuner_benchmark)
        self.assertTrue(PreTrainingArguments.unified_checkpoint)

    def test_custom_values(self):
        """Test PreTrainingArguments with custom values."""
        args = PreTrainingArguments(
            output_dir="/tmp/test_output",
            min_learning_rate=5e-6,
            decay_steps=1000,
            enable_linear_fused_grad_add=True,
            autotuner_benchmark=True,
            bf16=True,
        )
        self.assertEqual(args.min_learning_rate, 5e-6)
        self.assertEqual(args.decay_steps, 1000)
        self.assertTrue(args.enable_linear_fused_grad_add)
        self.assertTrue(args.autotuner_benchmark)

    def test_autotuner_benchmark_post_init(self):
        """Test that autotuner_benchmark modifies settings in __post_init__."""
        args = PreTrainingArguments(
            output_dir="/tmp/test_output",
            autotuner_benchmark=True,
            bf16=True,
        )
        self.assertEqual(args.max_steps, 5)
        self.assertTrue(args.do_train)
        self.assertFalse(args.do_export)
        self.assertFalse(args.do_predict)
        self.assertFalse(args.do_eval)
        self.assertTrue(args.overwrite_output_dir)
        self.assertFalse(args.load_best_model_at_end)
        self.assertFalse(args.unified_checkpoint)


class TestDataArguments(unittest.TestCase):
    """Test DataArguments dataclass."""

    def test_default_values(self):
        """Test default values of DataArguments."""
        args = DataArguments()
        self.assertIsNone(args.input_dir)
        self.assertEqual(args.split, "949,50,1")
        self.assertEqual(args.max_seq_length, 1024)
        self.assertFalse(args.share_folder)
        self.assertEqual(args.data_impl, "mmap")
        self.assertTrue(args.skip_warmup)
        self.assertIsNone(args.data_cache)

    def test_custom_values(self):
        """Test DataArguments with custom values."""
        args = DataArguments(
            input_dir="/data/train",
            split="800,200,0",
            max_seq_length=4096,
            share_folder=True,
            data_impl="lazy",
            skip_warmup=False,
            data_cache="/cache",
        )
        self.assertEqual(args.input_dir, "/data/train")
        self.assertEqual(args.split, "800,200,0")
        self.assertEqual(args.max_seq_length, 4096)
        self.assertTrue(args.share_folder)
        self.assertEqual(args.data_impl, "lazy")
        self.assertFalse(args.skip_warmup)
        self.assertEqual(args.data_cache, "/cache")


class TestModelArguments(unittest.TestCase):
    """Test ModelArguments dataclass."""

    def test_default_values(self):
        """Test default values of ModelArguments."""
        args = ModelArguments()
        self.assertEqual(args.model_name_or_path, "__internal_testing__/tiny-random-llama")
        self.assertIsNone(args.tokenizer_name_or_path)
        self.assertFalse(args.use_fast_layer_norm)
        self.assertEqual(args.hidden_dropout_prob, 0.1)
        self.assertEqual(args.attention_probs_dropout_prob, 0.1)
        self.assertFalse(args.continue_training)

    def test_custom_values(self):
        """Test ModelArguments with custom values."""
        args = ModelArguments(
            model_name_or_path="deepseek-ai/deepseek-v3",
            tokenizer_name_or_path="deepseek-ai/deepseek-v3",
            use_fast_layer_norm=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            continue_training=True,
        )
        self.assertEqual(args.model_name_or_path, "deepseek-ai/deepseek-v3")
        self.assertEqual(args.tokenizer_name_or_path, "deepseek-ai/deepseek-v3")
        self.assertTrue(args.use_fast_layer_norm)
        self.assertEqual(args.hidden_dropout_prob, 0.0)
        self.assertEqual(args.attention_probs_dropout_prob, 0.0)
        self.assertTrue(args.continue_training)


class TestGetTrainDataFile(unittest.TestCase):
    """Test get_train_data_file function."""

    def test_multi_input_dir(self):
        """Test with multi-path input_dir (space-separated)."""
        args = MagicMock()
        args.input_dir = "/data1 /data2"
        result = get_train_data_file(args)
        self.assertEqual(result, ["/data1", "/data2"])

    def test_single_dir_with_idx_files(self):
        """Test with a directory containing .idx files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["data1.idx", "data2.idx", "config.json"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("test")

            args = MagicMock()
            args.input_dir = tmpdir
            result = get_train_data_file(args)
            # Multi-file directory returns weighted list: [1.0, path1, 1.0, path2]
            self.assertEqual(len(result), 4)
            self.assertEqual(result[0], 1.0)
            self.assertEqual(result[2], 1.0)
            for path in [result[1], result[3]]:
                self.assertNotIn(".idx", path)

    def test_single_dir_with_npz_idx_files(self):
        """Test with a directory containing _idx.npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["data1_idx.npz", "data2_idx.npz"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("test")

            args = MagicMock()
            args.input_dir = tmpdir
            result = get_train_data_file(args)
            # Multi-file directory returns weighted list: [1.0, path1, 1.0, path2]
            self.assertEqual(len(result), 4)
            self.assertEqual(result[0], 1.0)
            self.assertEqual(result[2], 1.0)
            for path in [result[1], result[3]]:
                self.assertNotIn("_idx.npz", path)

    def test_single_dir_returns_list(self):
        """Test that single-file directory returns a list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.idx"), "w") as f:
                f.write("test")

            args = MagicMock()
            args.input_dir = tmpdir
            result = get_train_data_file(args)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)

    def test_multi_dataset_returns_weighted_list(self):
        """Test that multi-file directory returns weighted list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["data1.idx", "data2.idx", "data3.idx"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("test")

            args = MagicMock()
            args.input_dir = tmpdir
            result = get_train_data_file(args)
            # Should return [1.0, path1, 1.0, path2, 1.0, path3]
            self.assertEqual(len(result), 6)
            self.assertEqual(result[0], 1.0)
            self.assertEqual(result[2], 1.0)
            self.assertEqual(result[4], 1.0)


class TestPretrainingTrainer(unittest.TestCase):
    """Test PretrainingTrainer class."""

    def test_is_pretraining_flag(self):
        """Test that PretrainingTrainer has the evaluate method."""
        self.assertTrue(hasattr(PretrainingTrainer, "__init__"))
        self.assertTrue(hasattr(PretrainingTrainer, "evaluate"))


class TestEnvironmentVariable(unittest.TestCase):
    """Test environment variable set by workflow module."""

    def test_use_casual_mask_env(self):
        """Test that USE_CASUAL_MASK is set to True."""
        self.assertEqual(os.environ.get("USE_CASUAL_MASK"), "True")


if __name__ == "__main__":
    unittest.main()
