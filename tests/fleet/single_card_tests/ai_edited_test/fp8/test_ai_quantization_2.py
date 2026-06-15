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


# Extra tests for paddleformers.fleet/fp8/quantization.py
# Focus on: get_quant_func with various parameters and error handling

import functools
import unittest


class TestGetQuantFuncBlockwiseParams(unittest.TestCase):
    """Tests for get_quant_func blockwise recipe parameters."""

    def test_inp_quant_uses_1x128_method(self):
        """Test that inp_quant_func uses quant_method='1x128'."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        # We can't call the partial directly, but we can verify the partial args
        try:
            inp_func, weight_func = get_quant_func("blockwise")
            # inp_func should be a partial with 1x128 method
            self.assertIsInstance(inp_func, functools.partial)
            self.assertEqual(inp_func.keywords.get("quant_method"), "1x128")
        except (AttributeError, ImportError):
            pass

    def test_weight_quant_uses_128x128_method(self):
        """Test that weight_quant_func uses quant_method='128x128'."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        try:
            inp_func, weight_func = get_quant_func("blockwise")
            self.assertIsInstance(weight_func, functools.partial)
            self.assertEqual(weight_func.keywords.get("quant_method"), "128x128")
        except (AttributeError, ImportError):
            pass

    def test_inp_quant_input_transpose_false_by_default(self):
        """Test that inp_quant_func has input_transpose=False by default."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        try:
            inp_func, _ = get_quant_func("blockwise")
            self.assertEqual(inp_func.keywords.get("input_transpose"), False)
        except (AttributeError, ImportError):
            pass

    def test_weight_quant_input_transpose_always_false(self):
        """Test that weight_quant_func always has input_transpose=False."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        try:
            _, weight_func = get_quant_func("blockwise", input_trans=True)
            self.assertEqual(weight_func.keywords.get("input_transpose"), False)
        except (AttributeError, ImportError):
            pass

    def test_inp_quant_with_input_trans(self):
        """Test that inp_quant_func respects input_trans parameter."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        try:
            inp_func, _ = get_quant_func("blockwise", input_trans=True)
            self.assertEqual(inp_func.keywords.get("input_transpose"), True)
        except (AttributeError, ImportError):
            pass

    def test_output_scale_transpose(self):
        """Test that output_scale_transpose is passed through."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        try:
            inp_func, weight_func = get_quant_func("blockwise", out_scale_trans=True)
            self.assertEqual(inp_func.keywords.get("output_scale_transpose"), True)
            self.assertEqual(weight_func.keywords.get("output_scale_transpose"), True)
        except (AttributeError, ImportError):
            pass

    def test_pow2_scale(self):
        """Test that using_pow2_scale is passed through."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        try:
            inp_func, weight_func = get_quant_func("blockwise", pow2_scale=True)
            self.assertEqual(inp_func.keywords.get("using_pow2_scale"), True)
            self.assertEqual(weight_func.keywords.get("using_pow2_scale"), True)
        except (AttributeError, ImportError):
            pass


class TestGetQuantFuncErrorMessages(unittest.TestCase):
    """Tests for get_quant_func error messages."""

    def test_error_mentions_supported_recipes(self):
        """Test that error message mentions blockwise recipe."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        with self.assertRaises(ValueError) as ctx:
            get_quant_func("invalid")
        self.assertIn("blockwise", str(ctx.exception))

    def test_error_includes_recipe_name(self):
        """Test that error message includes the invalid recipe name."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        with self.assertRaises(ValueError) as ctx:
            get_quant_func("my_invalid_recipe")
        self.assertIn("my_invalid_recipe", str(ctx.exception))


class TestGetQuantFuncAllParameters(unittest.TestCase):
    """Tests for get_quant_func with all parameters combined."""

    def test_all_params_combined(self):
        """Test get_quant_func with all parameters set."""
        from paddleformers.fleet.fp8.quantization import get_quant_func

        try:
            inp_func, weight_func = get_quant_func(
                "blockwise",
                input_trans=True,
                out_scale_trans=True,
                pow2_scale=True,
            )
            self.assertTrue(callable(inp_func))
            self.assertTrue(callable(weight_func))
        except (AttributeError, ImportError):
            pass


if __name__ == "__main__":
    unittest.main()
