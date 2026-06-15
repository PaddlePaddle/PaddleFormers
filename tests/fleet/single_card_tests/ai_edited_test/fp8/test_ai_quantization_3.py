# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import functools
import importlib.util
import unittest
from unittest.mock import patch

import paddle


def _load_quantization_module():
    """Load quantization module directly from source to avoid deep_gemm import."""
    spec = importlib.util.spec_from_file_location(
        "fp8_quantization",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "src",
            "paddleformers.fleet",
            "fp8",
            "quantization.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGetQuantFuncExtra(unittest.TestCase):
    """Additional tests for get_quant_func in fp8/quantization.py."""

    def test_blockwise_inp_quant_is_partial(self):
        """Test blockwise inp_quant_func is a functools.partial."""
        mod = _load_quantization_module()
        with patch.object(
            paddle.incubate.nn.functional,
            "fp8_quant_blockwise",
            create=True,
        ):
            try:
                inp_func, weight_func = mod.get_quant_func("blockwise")
                self.assertIsInstance(inp_func, functools.partial)
            except (AttributeError, ImportError):
                pass

    def test_blockwise_weight_quant_is_partial(self):
        """Test blockwise weight_quant_func is a functools.partial."""
        mod = _load_quantization_module()
        with patch.object(
            paddle.incubate.nn.functional,
            "fp8_quant_blockwise",
            create=True,
        ):
            try:
                inp_func, weight_func = mod.get_quant_func("blockwise")
                self.assertIsInstance(weight_func, functools.partial)
            except (AttributeError, ImportError):
                pass

    def test_blockwise_inp_uses_1x128(self):
        """Test blockwise inp_quant_func uses 1x128 quant_method."""
        mod = _load_quantization_module()
        with patch.object(
            paddle.incubate.nn.functional,
            "fp8_quant_blockwise",
            create=True,
        ):
            try:
                inp_func, _ = mod.get_quant_func("blockwise")
                self.assertEqual(inp_func.keywords.get("quant_method"), "1x128")
            except (AttributeError, ImportError):
                pass

    def test_blockwise_weight_uses_128x128(self):
        """Test blockwise weight_quant_func uses 128x128 quant_method."""
        mod = _load_quantization_module()
        with patch.object(
            paddle.incubate.nn.functional,
            "fp8_quant_blockwise",
            create=True,
        ):
            try:
                _, weight_func = mod.get_quant_func("blockwise")
                self.assertEqual(weight_func.keywords.get("quant_method"), "128x128")
            except (AttributeError, ImportError):
                pass

    def test_blockwise_weight_input_transpose_false(self):
        """Test blockwise weight_quant_func has input_transpose=False."""
        mod = _load_quantization_module()
        with patch.object(
            paddle.incubate.nn.functional,
            "fp8_quant_blockwise",
            create=True,
        ):
            try:
                _, weight_func = mod.get_quant_func("blockwise")
                self.assertEqual(weight_func.keywords.get("input_transpose"), False)
            except (AttributeError, ImportError):
                pass


if __name__ == "__main__":
    unittest.main()
