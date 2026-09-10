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


def _can_import_triton():
    """Check if triton can be imported and has an active GPU driver."""
    try:
        import triton  # noqa: F401

        # Also check that triton has an active driver (GPU support)
        try:
            triton.runtime.driver.active.get_current_device()
            return True
        except (RuntimeError, AttributeError):
            return False
    except (ImportError, ModuleNotFoundError):
        return False


def _triton_supports_fp8e4nv():
    """Check if triton supports fp8e4nv on the current GPU architecture."""
    try:
        import triton

        device = triton.runtime.driver.active.get_current_device()
        cc = triton.runtime.driver.active.get_device_properties(device)["cc"]
        # fp8e4nv requires sm89+ (Ada Lovelace or newer)
        return cc >= 89
    except Exception:
        return False


_TRITON_AVAILABLE = _can_import_triton()
_TRITON_FP8E4NV_AVAILABLE = _triton_supports_fp8e4nv()

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

# Try to load the kernel module (requires triton)
_kernel_mod = None
_kernel_load_error = None
try:
    _kernel_mod = _load_module("kernel", os.path.join(_MODULE_DIR, "kernel.py"))
except Exception as e:
    _kernel_load_error = str(e)


@unittest.skipIf(not _TRITON_AVAILABLE, "Triton GPU driver not available")
class TestKernelModule(unittest.TestCase):
    """Test kernel module imports and function signatures."""

    def test_act_quant_function_exists(self):
        """Test that act_quant function can be referenced."""
        self.assertTrue(callable(_kernel_mod.act_quant))

    def test_weight_dequant_function_exists(self):
        """Test that weight_dequant function can be referenced."""
        self.assertTrue(callable(_kernel_mod.weight_dequant))

    def test_fp8_gemm_function_exists(self):
        """Test that fp8_gemm function can be referenced."""
        self.assertTrue(callable(_kernel_mod.fp8_gemm))


@unittest.skipIf(not _TRITON_AVAILABLE, "Triton GPU driver not available")
@unittest.skipIf(
    not _TRITON_FP8E4NV_AVAILABLE, "Triton fp8e4nv not supported on this GPU architecture (requires sm89+)"
)
class TestActQuant(unittest.TestCase):
    """Test act_quant function."""

    def test_act_quant_output_types(self):
        """Test act_quant returns correct output types."""
        import paddle

        x = paddle.randn([2, 128])
        y, s = _kernel_mod.act_quant(x, block_size=128)
        self.assertEqual(y.dtype, paddle.float8_e4m3fn)
        self.assertEqual(s.dtype, paddle.float32)

    def test_act_quant_scale_shape(self):
        """Test act_quant returns correct scale shape."""
        import paddle

        x = paddle.randn([4, 256])
        y, s = _kernel_mod.act_quant(x, block_size=128)
        # scale shape should be x.shape[:-1] + (x.shape[-1] // block_size,)
        self.assertEqual(s.shape, [4, 2])

    def test_act_quant_assertion_block_size(self):
        """Test act_quant asserts last dim divisible by block_size."""
        import paddle

        x = paddle.randn([2, 100])  # 100 is not divisible by 128
        with self.assertRaises(AssertionError):
            _kernel_mod.act_quant(x, block_size=128)


@unittest.skipIf(not _TRITON_AVAILABLE, "Triton GPU driver not available")
@unittest.skipIf(
    not _TRITON_FP8E4NV_AVAILABLE, "Triton fp8e4nv not supported on this GPU architecture (requires sm89+)"
)
class TestWeightDequant(unittest.TestCase):
    """Test weight_dequant function."""

    def test_weight_dequant_output_shape(self):
        """Test weight_dequant returns correct output shape."""
        import paddle

        x = paddle.randn([128, 256]).cast(paddle.float8_e4m3fn)
        s = paddle.randn([1, 2])
        y = _kernel_mod.weight_dequant(x, s, block_size=128)
        self.assertEqual(y.shape, [128, 256])

    def test_weight_dequant_asserts_2d(self):
        """Test weight_dequant asserts input tensors have 2 dimensions."""
        import paddle

        x = paddle.randn([128, 256]).cast(paddle.float8_e4m3fn)
        s = paddle.randn([1, 2, 3])  # 3D scale tensor
        with self.assertRaises(AssertionError):
            _kernel_mod.weight_dequant(x, s, block_size=128)

    def test_weight_dequant_asserts_contiguous(self):
        """Test weight_dequant asserts tensors are contiguous."""
        import paddle

        x = paddle.randn([128, 256]).cast(paddle.float8_e4m3fn)
        s = paddle.randn([1, 2])
        # Contiguous input should work
        y = _kernel_mod.weight_dequant(x, s, block_size=128)
        self.assertIsNotNone(y)


if __name__ == "__main__":
    unittest.main()
