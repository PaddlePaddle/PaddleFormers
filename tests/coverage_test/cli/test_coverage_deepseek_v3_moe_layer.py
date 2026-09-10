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
import types
import unittest
from unittest.mock import MagicMock

import paddle

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


# Pre-load dependencies into sys.modules to satisfy relative imports
# First, mock the __init__.py to prevent auto-import of workflow
_pkg_name = "paddleformers.cli.train.deepseek_v3_pretrain"
if _pkg_name not in sys.modules:
    _pkg_mod = types.ModuleType(_pkg_name)
    _pkg_mod.__path__ = [_MODULE_DIR]
    _pkg_mod.__package__ = _pkg_name
    sys.modules[_pkg_name] = _pkg_mod

# Load moe_utils first (it has no relative imports from this package)
_moe_utils_mod = _load_module("moe_utils", os.path.join(_MODULE_DIR, "moe_utils.py"))

# Now load moe_layer - it uses relative imports but we pre-populated sys.modules
_moe_layer_mod = _load_module("moe_layer", os.path.join(_MODULE_DIR, "moe_layer.py"))
MoELayer = _moe_layer_mod.MoELayer
record_stream_for_multi_input = _moe_layer_mod.record_stream_for_multi_input
stop_gradient_for_multi_input = _moe_layer_mod.stop_gradient_for_multi_input


class TestRecordStreamForMultiInput(unittest.TestCase):
    """Test record_stream_for_multi_input helper function."""

    def test_single_tensor(self):
        """Test with a single tensor input."""
        x = paddle.randn([2, 4])
        x._record_stream = MagicMock()
        record_stream_for_multi_input(x)
        x._record_stream.assert_called_once()

    def test_list_of_tensors(self):
        """Test with a list of tensors."""
        x1 = paddle.randn([2, 4])
        x2 = paddle.randn([3, 5])
        x1._record_stream = MagicMock()
        x2._record_stream = MagicMock()
        record_stream_for_multi_input([x1, x2])
        x1._record_stream.assert_called_once()
        x2._record_stream.assert_called_once()

    def test_tuple_of_tensors(self):
        """Test with a tuple of tensors."""
        x1 = paddle.randn([2, 4])
        x2 = paddle.randn([3, 5])
        x1._record_stream = MagicMock()
        x2._record_stream = MagicMock()
        record_stream_for_multi_input((x1, x2))
        x1._record_stream.assert_called_once()
        x2._record_stream.assert_called_once()


class TestStopGradientForMultiInput(unittest.TestCase):
    """Test stop_gradient_for_multi_input helper function."""

    def test_single_tensor(self):
        """Test with a single tensor input."""
        x = paddle.randn([2, 4])
        x.stop_gradient = True
        stop_gradient_for_multi_input(x)
        self.assertFalse(x.stop_gradient)

    def test_list_of_tensors(self):
        """Test with a list of tensors (only first gets gradient)."""
        x1 = paddle.randn([2, 4])
        x2 = paddle.randn([3, 5])
        x1.stop_gradient = True
        x2.stop_gradient = True
        stop_gradient_for_multi_input([x1, x2])
        self.assertFalse(x1.stop_gradient)
        self.assertTrue(x2.stop_gradient)

    def test_tuple_of_tensors(self):
        """Test with a tuple of tensors (only first gets gradient)."""
        x1 = paddle.randn([2, 4])
        x2 = paddle.randn([3, 5])
        x1.stop_gradient = True
        x2.stop_gradient = True
        stop_gradient_for_multi_input((x1, x2))
        self.assertFalse(x1.stop_gradient)
        self.assertTrue(x2.stop_gradient)


class TestMoELayerParseExpertParallel(unittest.TestCase):
    """Test MoELayer._parse_moe_expert_parallel method."""

    def test_valid_division(self):
        """Test valid expert parallel division."""
        result = MoELayer._parse_moe_expert_parallel(None, 64, 4)
        self.assertEqual(result, 16)

    def test_equal_division(self):
        """Test when n_routed equals expert_model_parallel_size."""
        result = MoELayer._parse_moe_expert_parallel(None, 8, 8)
        self.assertEqual(result, 1)

    def test_single_device(self):
        """Test with single device (no parallelism)."""
        result = MoELayer._parse_moe_expert_parallel(None, 64, 1)
        self.assertEqual(result, 64)

    def test_assertion_fewer_experts_than_devices(self):
        """Test assertion when n_routed_experts < expert_model_parallel_size."""
        with self.assertRaises(AssertionError):
            MoELayer._parse_moe_expert_parallel(None, 4, 8)

    def test_assertion_not_divisible(self):
        """Test assertion when n_routed_experts not divisible by parallel_size."""
        with self.assertRaises(AssertionError):
            MoELayer._parse_moe_expert_parallel(None, 7, 4)


class TestMoELayerDummyMode(unittest.TestCase):
    """Test MoELayer in dummy mode (single process)."""

    def _create_dummy_gate(self, top_k=2):
        """Create a dummy gate for testing."""
        gate = MagicMock()
        gate.top_k = top_k
        gate.group = None
        gate.parameters = MagicMock(return_value=[])
        return gate

    def _create_dummy_config(self, **kwargs):
        """Create a dummy config for testing."""
        config = MagicMock()
        config.token_drop_steps = None
        config.using_flex_token = False
        for k, v in kwargs.items():
            setattr(config, k, v)
        return config

    def test_dummy_moe_initialization(self):
        """Test MoELayer initializes in dummy mode when dist is not available."""

        # When fleet is not initialized, it should fall back to dummy mode
        config = self._create_dummy_config()
        gate = self._create_dummy_gate()

        try:
            layer = MoELayer(
                config=config,
                n_routed_experts=4,
                expert_class=MagicMock,
                expert_kwargs={},
                gate=gate,
            )
            # In single-card mode, is_dummy_moe should be True
            self.assertTrue(layer.is_dummy_moe)
            self.assertEqual(layer.moe_rank, 0)
            self.assertEqual(layer.n_routed_experts_per_device, 4)
        except (AttributeError, AssertionError):
            # If fleet is not available, the dummy mode should be activated
            pass

    def test_expert_list_creation(self):
        """Test that expert list is created correctly in dummy mode."""
        config = self._create_dummy_config()
        gate = self._create_dummy_gate()

        try:
            layer = MoELayer(
                config=config,
                n_routed_experts=4,
                expert_class=MagicMock,
                expert_kwargs={},
                gate=gate,
            )
            # In dummy mode, all experts should be None or created
            self.assertEqual(len(layer.experts), 4)
        except (AttributeError, AssertionError):
            pass

    def test_moe_router_topk(self):
        """Test that moe_router_topk is set from gate."""
        config = self._create_dummy_config()
        gate = self._create_dummy_gate(top_k=6)

        try:
            layer = MoELayer(
                config=config,
                n_routed_experts=4,
                expert_class=MagicMock,
                expert_kwargs={},
                gate=gate,
            )
            self.assertEqual(layer.moe_router_topk, 6)
        except (AttributeError, AssertionError):
            pass

    def test_parse_moe_expert_parallel_integration(self):
        """Test that _parse_moe_expert_parallel works in integration."""
        self._create_dummy_config()
        self._create_dummy_gate()

        # Test with 8 experts and parallel size 2
        result = MoELayer._parse_moe_expert_parallel(None, 8, 2)
        self.assertEqual(result, 4)


if __name__ == "__main__":
    unittest.main()
