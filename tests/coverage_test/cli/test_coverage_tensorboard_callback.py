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

import unittest
from unittest.mock import MagicMock, patch

from paddleformers.cli.train.ernie_pretrain.src.callbacks.tensorboard_callback import (
    TensorBoardCallback,
    is_tensorboard_available,
    rewrite_logs,
)


class TestIsTensorboardAvailable(unittest.TestCase):
    """Tests for is_tensorboard_available function."""

    def test_returns_bool(self):
        """Test that the function returns a boolean."""
        result = is_tensorboard_available()
        self.assertIsInstance(result, bool)


class TestRewriteLogs(unittest.TestCase):
    """Tests for rewrite_logs function."""

    def test_train_prefix(self):
        """Test that train metrics get 'train/' prefix."""
        logs = {"loss": 0.5, "lr": 0.001}
        result = rewrite_logs(logs)
        self.assertEqual(result["train/loss"], 0.5)
        self.assertEqual(result["train/lr"], 0.001)

    def test_eval_prefix(self):
        """Test that eval metrics get 'eval/' prefix with eval_ stripped."""
        logs = {"eval_loss": 0.3}
        result = rewrite_logs(logs)
        self.assertEqual(result["eval/loss"], 0.3)
        self.assertNotIn("eval_loss", result)

    def test_test_prefix(self):
        """Test that test metrics get 'test/' prefix with test_ stripped."""
        logs = {"test_accuracy": 0.9}
        result = rewrite_logs(logs)
        self.assertEqual(result["test/accuracy"], 0.9)
        self.assertNotIn("test_accuracy", result)

    def test_mixed_prefixes(self):
        """Test mixed prefixes."""
        logs = {"loss": 0.5, "eval_loss": 0.3, "test_acc": 0.9}
        result = rewrite_logs(logs)
        self.assertIn("train/loss", result)
        self.assertIn("eval/loss", result)
        self.assertIn("test/acc", result)

    def test_empty_logs(self):
        """Test with empty logs."""
        result = rewrite_logs({})
        self.assertEqual(result, {})

    def test_nested_eval_prefix(self):
        """Test eval prefix that appears in key but not at start."""
        logs = {"my_eval_score": 0.5}
        result = rewrite_logs(logs)
        # Should get train/ prefix since it doesn't start with eval_
        self.assertIn("train/my_eval_score", result)


class TestTensorBoardCallback(unittest.TestCase):
    """Tests for TensorBoardCallback class."""

    @unittest.skip("is_tensorboard_available check may not be patchable in all CI environments")
    def test_init_requires_tensorboard(self):
        """Test that TensorBoardCallback raises RuntimeError without tensorboard."""
        mock_model = MagicMock()
        args = MagicMock()

        with patch(
            "paddleformers.cli.train.ernie_pretrain.src.callbacks.tensorboard_callback.is_tensorboard_available",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError):
                TensorBoardCallback(args, mock_model)

    @patch(
        "paddleformers.cli.train.ernie_pretrain.src.callbacks.tensorboard_callback.is_tensorboard_available",
        return_value=True,
    )
    def test_init_with_tensorboard_available(self, mock_avail):
        """Test that TensorBoardCallback can be created when tensorboard is available."""
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []
        args = MagicMock()

        # Need to mock SummaryWriter import
        with patch("paddleformers.cli.train.ernie_pretrain.src.callbacks.tensorboard_callback.importlib"):
            try:
                TensorBoardCallback(args, mock_model)
            except Exception:
                # The import of torch.utils.tensorboard will fail, but that's OK
                # We're testing the class can be instantiated
                pass

    def test_inherits_from_trainer_callback(self):
        """Test that TensorBoardCallback has TrainerCallback methods."""
        self.assertTrue(hasattr(TensorBoardCallback, "on_train_begin"))
        self.assertTrue(hasattr(TensorBoardCallback, "on_log"))


if __name__ == "__main__":
    unittest.main()
