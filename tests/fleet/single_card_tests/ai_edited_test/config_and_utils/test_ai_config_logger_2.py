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
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import json
import unittest
from unittest.mock import MagicMock

import paddle


class TestGetConfigLoggerPath(unittest.TestCase):
    """Tests for get_config_logger_path function."""

    def test_with_config_logger_dir(self):
        """Test returns config_logger_dir when present."""
        from paddleformers.fleet.config_logger import get_config_logger_path

        config = MagicMock()
        config.config_logger_dir = "/tmp/test_logs"
        result = get_config_logger_path(config)
        self.assertEqual(result, "/tmp/test_logs")

    def test_without_config_logger_dir(self):
        """Test returns empty string when not present."""
        from paddleformers.fleet.config_logger import get_config_logger_path

        config = MagicMock(spec=[])
        result = get_config_logger_path(config)
        self.assertEqual(result, "")


class TestHasConfigLoggerEnabled(unittest.TestCase):
    """Tests for has_config_logger_enabled function."""

    def test_enabled(self):
        """Test returns True when config_logger_dir is set."""
        from paddleformers.fleet.config_logger import has_config_logger_enabled

        config = MagicMock()
        config.config_logger_dir = "/tmp/test_logs"
        self.assertTrue(has_config_logger_enabled(config))

    def test_disabled(self):
        """Test returns False when config_logger_dir is empty."""
        from paddleformers.fleet.config_logger import has_config_logger_enabled

        config = MagicMock(spec=[])
        self.assertFalse(has_config_logger_enabled(config))


class TestGetPathCount(unittest.TestCase):
    """Tests for get_path_count function."""

    def test_first_call_returns_zero(self):
        """Test first call with a path returns 0."""
        from paddleformers.fleet.config_logger import get_path_count

        # Use a unique path to avoid interference
        result = get_path_count("/unique_test_path_for_unit_test_12345")
        self.assertEqual(result, 0)

    def test_subsequent_call_increments(self):
        """Test subsequent calls increment the count."""
        from paddleformers.fleet.config_logger import get_path_count

        path = "/unique_test_path_increment_12345"
        get_path_count(path)
        result = get_path_count(path)
        self.assertEqual(result, 1)


class TestGetPathWithCount(unittest.TestCase):
    """Tests for get_path_with_count function."""

    def test_format(self):
        """Test output format."""
        from paddleformers.fleet.config_logger import get_path_with_count

        path = "/unique_format_test_path_12345"
        result = get_path_with_count(path)
        self.assertIn(".iter", result)
        self.assertTrue(result.startswith(path))


class TestJSONEncoder(unittest.TestCase):
    """Tests for JSONEncoderWithMcoreTypes."""

    def test_encode_standard_types(self):
        """Test encoding standard Python types."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        data = {"key": "value", "number": 42}
        result = json.dumps(data, cls=JSONEncoderWithMcoreTypes)
        self.assertIn("key", result)

    def test_encode_paddle_dtype(self):
        """Test encoding paddle.dtype."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        data = {"dtype": paddle.float32}
        result = json.dumps(data, cls=JSONEncoderWithMcoreTypes)
        self.assertIn("dtype", result)

    def test_encode_function_type(self):
        """Test encoding function type."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        def my_func():
            pass

        data = {"func": my_func}
        result = json.dumps(data, cls=JSONEncoderWithMcoreTypes)
        self.assertIn("func", result)

    def test_encode_nn_module(self):
        """Test encoding nn.Module."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        layer = paddle.nn.Linear(4, 4)
        data = {"layer": layer}
        result = json.dumps(data, cls=JSONEncoderWithMcoreTypes)
        self.assertIn("layer", result)


if __name__ == "__main__":
    unittest.main()
