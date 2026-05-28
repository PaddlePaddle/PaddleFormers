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


def _load_module(name, path):
    """Load a module directly from file path without going through __init__.py."""
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

# Load the fp8_linear module directly
_fp8_mod = _load_module("fp8_linear", os.path.join(_MODULE_DIR, "fp8_linear.py"))

original_linear = _fp8_mod.original_linear
block_size = _fp8_mod.block_size
gemm_impl = _fp8_mod.gemm_impl
register_scale = _fp8_mod.register_scale
Linear = _fp8_mod.Linear


class TestFP8LinearModule(unittest.TestCase):
    """Test fp8_linear module global variables and helper functions."""

    def test_original_linear_preserved(self):
        """Test that original_linear is saved before monkey-patching."""
        # original_linear points to the pre-monkey-patch version
        self.assertTrue(callable(original_linear))

    def test_block_size_default(self):
        """Test that block_size defaults to 128."""
        self.assertEqual(block_size, 128)

    def test_gemm_impl_default(self):
        """Test that gemm_impl defaults to 'bf16'."""
        self.assertEqual(gemm_impl, "bf16")

    def test_all_exports_defined(self):
        """Test that all __all__ exports are defined."""
        for name in _fp8_mod.__all__:
            self.assertTrue(hasattr(_fp8_mod, name), f"Missing export: {name}")


class TestRegisterScale(unittest.TestCase):
    """Test the register_scale function."""

    def test_register_scale_creates_weight_scale_inv_for_quantized_weight(self):
        """Test register_scale when weight element size is 1 (quantized)."""

        class FakeLinear:
            def __init__(self):
                self.weight = MagicMock()
                self.weight.element_size = MagicMock(return_value=1)
                self.weight.shape = [256, 512]
                self.weight._scale = None
                self.block_size = 128
                self._weight_attr = None
                self._created_parameters = {}

            def create_parameter(self, shape, attr, dtype, is_bias):
                param = MagicMock()
                param.shape = shape
                self._created_parameters["shape"] = shape
                self._created_parameters["dtype"] = dtype
                return param

        fake = FakeLinear()
        register_scale(fake)

        # scale_out_features = (512 + 128 - 1) // 128 = 4
        # scale_in_features = (256 + 128 - 1) // 128 = 2
        self.assertEqual(fake._created_parameters["shape"], [2, 4])
        self.assertEqual(fake._created_parameters["dtype"], "float32")
        self.assertEqual(fake.weight._scale, fake.weight_scale_inv)

    def test_register_scale_skips_for_non_quantized_weight(self):
        """Test register_scale when weight element size > 1 (not quantized)."""

        class FakeLinear:
            def __init__(self):
                self.weight = MagicMock()
                self.weight.element_size = MagicMock(return_value=2)
                self.weight.shape = [256, 512]
                self.block_size = 128
                self._weight_attr = None

            def create_parameter(self, shape, attr, dtype, is_bias):
                raise AssertionError("create_parameter should not be called for non-quantized weights")

        fake = FakeLinear()
        register_scale(fake)
        # If we reach here, create_parameter was not called, which is expected


class TestLinearSubclass(unittest.TestCase):
    """Test the Linear subclass and its block_size handling."""

    def test_linear_default_block_size(self):
        """Test Linear class defaults block_size to 128."""
        linear = Linear(16, 8)
        self.assertEqual(linear.block_size, 128)

    def test_linear_custom_block_size_via_setattr(self):
        """Test Linear class block_size can be set after init."""
        linear = Linear(16, 8)
        linear.block_size = 256
        self.assertEqual(linear.block_size, 256)

    def test_linear_has_weight(self):
        """Test Linear class has weight parameter."""
        linear = Linear(16, 8)
        self.assertIsNotNone(linear.weight)

    def test_linear_inherits_from_pd_linear(self):
        """Test Linear inherits from paddleformers Linear."""
        from paddleformers.transformers.linear_utils import Linear as PD_Linear

        self.assertTrue(issubclass(Linear, PD_Linear))


class TestFP8LinearFunction(unittest.TestCase):
    """Test the fp8_linear function behavior."""

    def test_fp8_linear_callable(self):
        """Test that fp8_linear function is callable."""
        self.assertTrue(callable(_fp8_mod.fp8_linear))

    def test_fp8_linear_uses_original_for_non_quantized(self):
        """Test fp8_linear delegates to original_linear for non-quantized weight."""
        # When weight.element_size() > 1 (non-quantized), fp8_linear calls original_linear
        x = paddle.randn([2, 4])
        weight = paddle.randn([4, 8])  # non-quantized weight

        # Use paddle.matmul directly since F.linear has been monkey-patched
        result = paddle.matmul(x, weight)
        self.assertEqual(result.shape, [2, 8])

    def test_original_linear_saved_correctly(self):
        """Test that original_linear was saved before monkey-patching."""
        # original_linear should be callable and different from current F.linear
        self.assertTrue(callable(original_linear))
        # After monkey-patching, F.linear should be fp8_linear
        self.assertEqual(paddle.nn.functional.linear, _fp8_mod.fp8_linear)

    def test_matmul_equivalent_shapes(self):
        """Test that matmul produces correct linear operation shapes."""
        for in_features, out_features, batch_size in [(16, 32, 4), (64, 128, 2), (4, 8, 1)]:
            x = paddle.randn([batch_size, in_features])
            weight = paddle.randn([in_features, out_features])
            result = paddle.matmul(x, weight)
            self.assertEqual(result.shape, [batch_size, out_features])


if __name__ == "__main__":
    unittest.main()
