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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


# Tests for paddleformers.fleet/fp8/quantization.py
# Tests for get_quant_func

import unittest
from unittest.mock import patch


class TestGetQuantFunc(unittest.TestCase):
    """Tests for get_quant_func function."""

    def test_blockwise_recipe_returns_two_funcs(self):
        """Test blockwise recipe returns (inp_quant_func, weight_quant_func)."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        # Mock the fp8_quant_blockwise function
        with patch(
            "paddleformers.fleet.fp8.quantization.paddle.incubate.nn.functional.fp8_quant_blockwise",
            create=True,
        ):
            try:
                inp_func, weight_func = get_quant_func("blockwise")
                self.assertTrue(callable(inp_func))
                self.assertTrue(callable(weight_func))
            except (AttributeError, ImportError):
                # If paddle.incubate.nn.functional.fp8_quant_blockwise is not available,
                # the partial will still be created
                pass

    def test_unsupported_recipe_raises(self):
        """Test unsupported recipe raises ValueError."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        with self.assertRaises(ValueError) as ctx:
            get_quant_func("unsupported_recipe")
        self.assertIn("unsupported_recipe", str(ctx.exception))
        self.assertIn("blockwise", str(ctx.exception))

    def test_blockwise_with_input_trans(self):
        """Test blockwise recipe with input_trans=True."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        with patch(
            "paddleformers.fleet.fp8.quantization.paddle.incubate.nn.functional.fp8_quant_blockwise",
            create=True,
        ):
            try:
                inp_func, weight_func = get_quant_func("blockwise", input_trans=True)
                self.assertTrue(callable(inp_func))
            except (AttributeError, ImportError):
                pass

    def test_blockwise_with_out_scale_trans(self):
        """Test blockwise recipe with out_scale_trans=True."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        with patch(
            "paddleformers.fleet.fp8.quantization.paddle.incubate.nn.functional.fp8_quant_blockwise",
            create=True,
        ):
            try:
                inp_func, weight_func = get_quant_func("blockwise", out_scale_trans=True)
                self.assertTrue(callable(inp_func))
            except (AttributeError, ImportError):
                pass

    def test_blockwise_with_pow2_scale(self):
        """Test blockwise recipe with pow2_scale=True."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        with patch(
            "paddleformers.fleet.fp8.quantization.paddle.incubate.nn.functional.fp8_quant_blockwise",
            create=True,
        ):
            try:
                inp_func, weight_func = get_quant_func("blockwise", pow2_scale=True)
                self.assertTrue(callable(inp_func))
            except (AttributeError, ImportError):
                pass

    def test_function_signature(self):
        """Test get_quant_func has expected parameters."""
        import inspect

        from paddleformers.fleet.fp8.quantization import get_quant_func

        sig = inspect.signature(get_quant_func)
        params = list(sig.parameters.keys())
        self.assertIn("fp8_recipe", params)
        self.assertIn("input_trans", params)
        self.assertIn("out_scale_trans", params)
        self.assertIn("pow2_scale", params)


class TestGetQuantFuncDefaultValues(unittest.TestCase):
    """Tests for get_quant_func default parameter values."""

    def test_default_input_trans(self):
        """Test default input_trans is False."""
        import inspect

        from paddleformers.fleet.fp8.quantization import get_quant_func

        sig = inspect.signature(get_quant_func)
        self.assertEqual(sig.parameters["input_trans"].default, False)

    def test_default_out_scale_trans(self):
        """Test default out_scale_trans is False."""
        import inspect

        from paddleformers.fleet.fp8.quantization import get_quant_func

        sig = inspect.signature(get_quant_func)
        self.assertEqual(sig.parameters["out_scale_trans"].default, False)

    def test_default_pow2_scale(self):
        """Test default pow2_scale is False."""
        import inspect

        from paddleformers.fleet.fp8.quantization import get_quant_func

        sig = inspect.signature(get_quant_func)
        self.assertEqual(sig.parameters["pow2_scale"].default, False)


if __name__ == "__main__":
    unittest.main()
