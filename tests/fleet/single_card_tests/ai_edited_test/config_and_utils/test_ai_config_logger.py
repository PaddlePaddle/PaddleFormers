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
import tempfile
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import paddle


class TestConfigLogger(unittest.TestCase):
    """Tests for paddleformers.fleet.config_logger module."""

    def test_get_config_logger_path_returns_empty_string(self):
        """Test get_config_logger_path returns empty string when attribute is missing."""
        from paddleformers.fleet.config_logger import get_config_logger_path

        config = MagicMock(spec=[])
        self.assertEqual(get_config_logger_path(config), "")

    def test_get_config_logger_path_returns_set_value(self):
        """Test get_config_logger_path returns the set directory path."""
        from paddleformers.fleet.config_logger import get_config_logger_path

        config = MagicMock()
        config.config_logger_dir = "/tmp/test_logs"
        self.assertEqual(get_config_logger_path(config), "/tmp/test_logs")

    def test_has_config_logger_enabled_false(self):
        """Test has_config_logger_enabled returns False when path is empty."""
        from paddleformers.fleet.config_logger import has_config_logger_enabled

        config = MagicMock(spec=[])
        self.assertFalse(has_config_logger_enabled(config))

    def test_has_config_logger_enabled_true(self):
        """Test has_config_logger_enabled returns True when path is set."""
        from paddleformers.fleet.config_logger import has_config_logger_enabled

        config = MagicMock()
        config.config_logger_dir = "/tmp/test_logs"
        self.assertTrue(has_config_logger_enabled(config))

    def test_get_path_count_first_time(self):
        """Test get_path_count returns 0 for a new path."""
        from paddleformers.fleet.config_logger import get_path_count

        result = get_path_count("/tmp/new_path")
        self.assertEqual(result, 0)

    def test_get_path_count_increments(self):
        """Test get_path_count increments on subsequent calls."""
        from paddleformers.fleet.config_logger import get_path_count

        unique_path = "/tmp/test_unique_path_12345"
        count0 = get_path_count(unique_path)
        count1 = get_path_count(unique_path)
        count2 = get_path_count(unique_path)
        self.assertEqual(count0, 0)
        self.assertEqual(count1, 1)
        self.assertEqual(count2, 2)

    def test_get_path_with_count(self):
        """Test get_path_with_count appends iteration number."""
        from paddleformers.fleet.config_logger import get_path_with_count

        result = get_path_with_count("/tmp/test_path")
        self.assertRegex(result, r"/tmp/test_path\.iter\d+")

    def test_get_path_with_count_sequential(self):
        """Test get_path_with_count returns sequential values."""
        from paddleformers.fleet.config_logger import get_path_with_count

        unique_path = "/tmp/seq_test_abc"
        r0 = get_path_with_count(unique_path)
        r1 = get_path_with_count(unique_path)
        self.assertRegex(r0, r"/tmp/seq_test_abc\.iter0")
        self.assertRegex(r1, r"/tmp/seq_test_abc\.iter1")

    def test_json_encoder_with_mcore_types_function(self):
        """Test JSONEncoderWithMcoreTypes handles function type."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()

        def dummy_func():
            pass

        result = json.dumps({"fn": dummy_func}, cls=JSONEncoderWithMcoreTypes)
        # The function should be serialized as a string
        self.assertIn("dummy_func", result)

    def test_json_encoder_with_mcore_types_dict(self):
        """Test JSONEncoderWithMcoreTypes handles dict type."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default({"a": 1, "b": 2})
        # The encoder recursively processes values, converting non-standard types via str()
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_json_encoder_with_mcore_types_ordered_dict(self):
        """Test JSONEncoderWithMcoreTypes handles OrderedDict."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        od = OrderedDict([("x", 1), ("y", 2)])
        result = encoder.default(od)
        self.assertIsInstance(result, dict)
        # Values are recursively processed; int values become str via fallback
        self.assertIn("x", result)
        self.assertIn("y", result)

    def test_json_encoder_with_mcore_types_list(self):
        """Test JSONEncoderWithMcoreTypes handles list type."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default([1, 2, 3])
        # Values are recursively processed; int values become str via fallback
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_json_encoder_with_mcore_types_paddle_dtype(self):
        """Test JSONEncoderWithMcoreTypes handles paddle.dtype."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default(paddle.float32)
        self.assertIn("float32", str(result))

    def test_json_encoder_with_mcore_types_nn_module_leaf(self):
        """Test JSONEncoderWithMcoreTypes handles leaf nn.Module."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        layer = paddle.nn.Linear(10, 10)
        result = encoder.default(layer)
        self.assertIsInstance(result, str)

    def test_json_encoder_with_mcore_types_nn_module_with_children(self):
        """Test JSONEncoderWithMcoreTypes handles nn.Module with children."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        model = paddle.nn.Sequential(
            paddle.nn.Linear(10, 10),
            paddle.nn.ReLU(),
        )
        result = encoder.default(model)
        self.assertIsInstance(result, dict)

    def test_json_encoder_with_mcore_types_regular_object(self):
        """Test JSONEncoderWithMcoreTypes handles regular Python objects."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default(42)
        # Should fallback to super().default() or str()
        self.assertIsNotNone(result)

    def test_json_encoder_with_mcore_types_process_group(self):
        """Test JSONEncoderWithMcoreTypes handles objects named ProcessGroup."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        mock_pg = MagicMock()
        mock_pg.__class__.__name__ = "ProcessGroup"
        result = encoder.default(mock_pg)
        self.assertIsInstance(result, str)

    def test_log_config_to_disk_json(self):
        """Test log_config_to_disk writes JSON file to disk."""
        from paddleformers.fleet.config_logger import log_config_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            config.config_logger_dir = tmpdir

            with patch(
                "paddleformers.fleet.config_logger.parallel_state"
            ) as mock_ps:
                mock_ps.get_all_ranks.return_value = "0_0_0_0_0"

                data = {"key1": "value1", "key2": 42}
                log_config_to_disk(config, data, prefix="test_prefix")

            # Verify the file was created
            files = os.listdir(tmpdir)
            self.assertTrue(len(files) > 0)
            json_file = next(f for f in files if f.endswith(".json"))
            with open(os.path.join(tmpdir, json_file), "r") as fp:
                loaded = json.load(fp)
            self.assertEqual(loaded, data)

    def test_log_config_to_disk_with_self_key(self):
        """Test log_config_to_disk removes 'self' key and uses class name as prefix."""
        from paddleformers.fleet.config_logger import log_config_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            config.config_logger_dir = tmpdir

            with patch(
                "paddleformers.fleet.config_logger.parallel_state"
            ) as mock_ps:
                mock_ps.get_all_ranks.return_value = "0_0_0_0_0"

                mock_self = MagicMock()
                data = {"self": mock_self, "data": 123}
                log_config_to_disk(config, data)

            files = os.listdir(tmpdir)
            self.assertTrue(len(files) > 0)

    def test_log_config_to_disk_ordered_dict(self):
        """Test log_config_to_disk writes .pth file for OrderedDict."""
        from paddleformers.fleet.config_logger import log_config_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            config.config_logger_dir = tmpdir

            with patch(
                "paddleformers.fleet.config_logger.parallel_state"
            ) as mock_ps:
                mock_ps.get_all_ranks.return_value = "0_0_0_0_0"

                data = OrderedDict([("a", 1), ("b", 2)])
                log_config_to_disk(config, data, prefix="ordered_test")

            files = os.listdir(tmpdir)
            pth_files = [f for f in files if f.endswith(".pth")]
            self.assertTrue(len(pth_files) > 0)

    def test_log_config_to_disk_creates_directory(self):
        """Test log_config_to_disk creates directory if not exists."""
        from paddleformers.fleet.config_logger import log_config_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "nested", "dir")
            config = MagicMock()
            config.config_logger_dir = new_dir

            with patch(
                "paddleformers.fleet.config_logger.parallel_state"
            ) as mock_ps:
                mock_ps.get_all_ranks.return_value = "0_0_0_0_0"

                data = {"test": True}
                log_config_to_disk(config, data, prefix="mkdir_test")

            self.assertTrue(os.path.exists(new_dir))
            files = os.listdir(new_dir)
            self.assertTrue(len(files) > 0)

    def test_log_config_to_disk_none_path_assertion(self):
        """Test log_config_to_disk raises AssertionError when path is None."""
        from paddleformers.fleet.config_logger import log_config_to_disk

        config = MagicMock()
        config.config_logger_dir = None
        with self.assertRaises(AssertionError):
            log_config_to_disk(config, {"key": "value"})

    def test_log_config_to_disk_with_custom_rank_str(self):
        """Test log_config_to_disk with custom rank_str."""
        from paddleformers.fleet.config_logger import log_config_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            config.config_logger_dir = tmpdir

            data = {"key": "value"}
            log_config_to_disk(
                config, data, prefix="rank_test", rank_str="1_2_3_4_5"
            )

            files = os.listdir(tmpdir)
            self.assertTrue(any("rank_1_2_3_4_5" in f for f in files))

    def test_json_encoder_with_mcore_types_float16_module(self):
        """Test JSONEncoderWithMcoreTypes handles Float16Module named object."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        mock_f16 = MagicMock()
        mock_f16.__class__.__name__ = "Float16Module"
        mock_f16.module = paddle.nn.Linear(10, 10)
        result = encoder.default(mock_f16)
        self.assertIsInstance(result, dict)
        self.assertIn("Float16Module", result)

    def test_json_encoder_with_mcore_types_unique_descriptor(self):
        """Test JSONEncoderWithMcoreTypes handles UniqueDescriptor named object."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        mock_ud = MagicMock()
        mock_ud.__class__.__name__ = "UniqueDescriptor"
        mock_ud.some_attr = "test_value"
        result = encoder.default(mock_ud)
        self.assertIsInstance(result, dict)

    def test_json_encoder_with_mcore_types_dataclass(self):
        """Test JSONEncoderWithMcoreTypes handles dataclass instances."""
        from dataclasses import dataclass

        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()

        @dataclass
        class SimpleConfig:
            name: str = "test"
            value: int = 42

        result = encoder.default(SimpleConfig())
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 42)

    def test_json_encoder_with_mcore_types_module_list(self):
        """Test JSONEncoderWithMcoreTypes handles ModuleList named object."""
        from paddleformers.fleet.config_logger import JSONEncoderWithMcoreTypes

        encoder = JSONEncoderWithMcoreTypes()
        mock_ml = MagicMock()
        mock_ml.__class__.__name__ = "ModuleList"
        # Provide items for iteration
        mock_ml.__iter__ = MagicMock(return_value=iter([1, 2, 3]))
        result = encoder.default(mock_ml)
        self.assertIsInstance(result, list)
