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


import types
import unittest
from unittest.mock import MagicMock

import paddle

# Check if deep_gemm is available (requires GPU compute capability >= 9.0)
try:
    from paddlefleet_ops import is_deep_gemm_available

    _DEEP_GEMM_AVAILABLE = is_deep_gemm_available()
except Exception:
    _DEEP_GEMM_AVAILABLE = False

# Mock deep_gemm if not available so that paddleformers.fleet.fp8.linear can import.
# fp8.linear does `from paddlefleet_ops import deep_gemm` at module level.
if not _DEEP_GEMM_AVAILABLE:
    # Ensure paddlefleet_ops exists and has a deep_gemm attribute
    # Use try/except because paddlefleet_ops.__getattr__ raises RuntimeError
    # for unsupported modules
    try:
        ops_mod = sys.modules.get("paddlefleet_ops")
        if ops_mod is None:
            ops_mod = types.ModuleType("paddlefleet_ops")
            sys.modules["paddlefleet_ops"] = ops_mod
        # Check if deep_gemm already exists without triggering __getattr__
        if "deep_gemm" not in dir(ops_mod):
            mock_dg = MagicMock()
            mock_dg.fp8_gemm_nt = MagicMock()
            ops_mod.deep_gemm = mock_dg
    except (RuntimeError, AttributeError):
        ops_mod = types.ModuleType("paddlefleet_ops")
        mock_dg = MagicMock()
        mock_dg.fp8_gemm_nt = MagicMock()
        ops_mod.deep_gemm = mock_dg
        sys.modules["paddlefleet_ops"] = ops_mod

    # Clear any cached fp8 modules so they re-import with the mock
    for _key in list(sys.modules.keys()):
        if _key.startswith("paddleformers.fleet.fp8"):
            del sys.modules[_key]


class TestFP8Linear(unittest.TestCase):
    """Tests for FP8Linear in linear.py."""

    def test_fp8_linear_import(self):
        """Test that FP8Linear can be imported from the fp8 package."""
        from paddleformers.fleet.fp8 import FP8Linear

        self.assertTrue(callable(FP8Linear))

    def test_fp8_linear_is_subclass_of_column_parallel_linear(self):
        """Test that FP8Linear inherits from ColumnParallelLinear."""
        from paddleformers.fleet.fp8 import FP8Linear
        from paddleformers.fleet.tensor_parallel import ColumnParallelLinear

        self.assertTrue(issubclass(FP8Linear, ColumnParallelLinear))


class TestFp8GemmForward(unittest.TestCase):
    """Tests for _FP8Gemm static forward method."""

    def test_fp8_gemm_unexpected_length(self):
        """Test that _FP8Gemm.forward raises ValueError for unexpected quant result length."""
        from paddleformers.fleet.fp8.linear import _FP8Gemm

        # Mock inputs
        inp = paddle.randn([2, 4], dtype=paddle.float32)
        weight = paddle.randn([8, 4], dtype=paddle.float32)

        # Create a quant func that returns wrong length
        mock_inp_quant = MagicMock(return_value=(1, 2, 3, 4, 5))
        mock_weight_quant = MagicMock(
            return_value=(
                paddle.zeros([8, 4], dtype=paddle.float8_e4m3fn),
                paddle.ones([1], dtype=paddle.float32),
            )
        )

        with self.assertRaises(ValueError) as ctx:
            _FP8Gemm.forward(
                None, inp, weight, mock_inp_quant, mock_weight_quant
            )
        self.assertIn("Unexpected length", str(ctx.exception))


class TestFP8Init(unittest.TestCase):
    """Tests for fp8 __init__.py."""

    def test_fp8_import_fp8linear(self):
        """Test that FP8Linear can be imported via paddleformers.fleet.fp8."""
        from paddleformers.fleet.fp8 import FP8Linear

        self.assertIsNotNone(FP8Linear)


if __name__ == "__main__":
    unittest.main()
