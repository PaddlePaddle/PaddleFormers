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
from unittest.mock import MagicMock

import paddle

from paddleformers.cli.train.ernie_pretrain.src.callbacks.logging_callback import (
    LoggingCallback,
)


class TestLoggingCallback(unittest.TestCase):
    """Tests for LoggingCallback class."""

    def setUp(self):
        """Set up test fixtures."""
        self.callback = LoggingCallback()

    def test_init(self):
        """Test LoggingCallback initialization."""
        self.assertIsNotNone(self.callback)

    def test_inherits_from_trainer_callback(self):
        """Test that LoggingCallback has TrainerCallback methods."""
        self.assertTrue(hasattr(self.callback, "on_log"))
        self.assertTrue(hasattr(self.callback, "on_train_begin"))

    def test_on_log_removes_total_flos(self):
        """Test that on_log removes 'total_flos' from logs."""
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        logs = {"loss": 0.5, "total_flos": 12345, "lr": 0.001}

        self.callback.on_log(args, state, control, logs=logs)

        # total_flos should have been popped
        self.assertNotIn("total_flos", logs)
        self.assertIn("loss", logs)

    def test_on_log_dict_logs(self):
        """Test on_log with dict logs."""
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        logs = {"loss": 0.5, "learning_rate": 0.001}

        # Should not raise
        self.callback.on_log(args, state, control, logs=logs)

    def test_on_log_with_inputs_data_id(self):
        """Test on_log with data_id in inputs."""
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        logs = {"loss": 0.5}

        inputs = {"data_id": paddle.to_tensor([1, 2, 3])}
        # The callback creates a new dict, not modifying the original
        self.callback.on_log(args, state, control, logs=logs, inputs=inputs)

    def test_on_log_with_inputs_src_id(self):
        """Test on_log with src_id in inputs."""
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        logs = {"loss": 0.5}

        inputs = {"src_id": paddle.to_tensor([4, 5])}
        self.callback.on_log(args, state, control, logs=logs, inputs=inputs)

    def test_on_log_with_inputs_data_type(self):
        """Test on_log with data_type in inputs."""
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        logs = {"loss": 0.5}

        inputs = {"data_type": paddle.to_tensor([1])}
        self.callback.on_log(args, state, control, logs=logs, inputs=inputs)

    def test_on_log_with_metrics_dumper(self):
        """Test on_log with metrics_dumper in kwargs."""
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        logs = {"loss": 0.5}
        metrics_dumper = []

        self.callback.on_log(args, state, control, logs=logs, metrics_dumper=metrics_dumper)
        self.assertEqual(len(metrics_dumper), 1)
        self.assertEqual(metrics_dumper[0], logs)

    def test_on_log_float_formatting(self):
        """Test that float values are formatted correctly."""
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        logs = {"loss": 0.5, "tiny_value": 1e-6, "int_metric": 42}

        # Should not raise
        self.callback.on_log(args, state, control, logs=logs)


if __name__ == "__main__":
    unittest.main()
