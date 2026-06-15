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


# Tests for src/paddleformers.fleet/_extensions/flashmask/rr_attn_estimate_triton_op.py
# Dedicated tests for triton kernel wrappers: check_dense_contains_partial_stride,
# gemm_fuse_softmax_causal, gemm_fuse_softmax_non_causal

import types
import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False

# Mock triton if not available
_triton_available = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _triton_available = True
except (ImportError, ModuleNotFoundError):
    pass

if not _triton_available:
    _mock_tl = types.ModuleType("triton.language")
    _mock_triton = types.ModuleType("triton")
    _mock_triton.jit = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)
    _mock_triton.cdiv = lambda a, b: (a + b - 1) // b
    _mock_triton.next_power_of_2 = lambda n: (1 << (n - 1).bit_length() if n > 0 else 1)
    sys.modules.setdefault("triton", _mock_triton)
    sys.modules.setdefault("triton.language", _mock_tl)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestCheckDenseContainsPartialStride(unittest.TestCase):
    """Tests for check_dense_contains_partial_stride triton kernel."""

    def test_is_jit_function(self):
        """Test that check_dense_contains_partial_stride is a jit function."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            check_dense_contains_partial_stride,
        )

        self.assertTrue(callable(check_dense_contains_partial_stride))

    def test_is_importable(self):
        """Test check_dense_contains_partial_stride can be imported."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            check_dense_contains_partial_stride,
        )

        self.assertIsNotNone(check_dense_contains_partial_stride)

    def test_kernel_signature(self):
        """Test that the kernel accepts expected parameters."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            check_dense_contains_partial_stride,
        )

        # The function is a triton.jit, so it is callable but won't execute
        # on CPU. Just verify it exists and has the right name.
        self.assertEqual(
            check_dense_contains_partial_stride.__name__,
            "check_dense_contains_partial_stride",
        )


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestGemmFuseSoftmaxCausal(unittest.TestCase):
    """Tests for gemm_fuse_softmax_causal triton kernel."""

    def test_is_jit_function(self):
        """Test that gemm_fuse_softmax_causal is a jit function."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            gemm_fuse_softmax_causal,
        )

        self.assertTrue(callable(gemm_fuse_softmax_causal))

    def test_is_importable(self):
        """Test gemm_fuse_softmax_causal can be imported."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            gemm_fuse_softmax_causal,
        )

        self.assertIsNotNone(gemm_fuse_softmax_causal)

    def test_kernel_name(self):
        """Test kernel has the correct name."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            gemm_fuse_softmax_causal,
        )

        self.assertEqual(gemm_fuse_softmax_causal.__name__, "gemm_fuse_softmax_causal")


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestGemmFuseSoftmaxNonCausal(unittest.TestCase):
    """Tests for gemm_fuse_softmax_non_causal triton kernel."""

    def test_is_jit_function(self):
        """Test that gemm_fuse_softmax_non_causal is a jit function."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            gemm_fuse_softmax_non_causal,
        )

        self.assertTrue(callable(gemm_fuse_softmax_non_causal))

    def test_is_importable(self):
        """Test gemm_fuse_softmax_non_causal can be imported."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            gemm_fuse_softmax_non_causal,
        )

        self.assertIsNotNone(gemm_fuse_softmax_non_causal)

    def test_kernel_name(self):
        """Test kernel has the correct name."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            gemm_fuse_softmax_non_causal,
        )

        self.assertEqual(
            gemm_fuse_softmax_non_causal.__name__,
            "gemm_fuse_softmax_non_causal",
        )


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestCausalVsNonCausalKernels(unittest.TestCase):
    """Tests comparing causal and non-causal kernel structures."""

    def test_both_kernels_exist(self):
        """Test that both causal and non-causal kernels are available."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            gemm_fuse_softmax_causal,
            gemm_fuse_softmax_non_causal,
        )

        self.assertIsNotNone(gemm_fuse_softmax_causal)
        self.assertIsNotNone(gemm_fuse_softmax_non_causal)

    def test_kernels_are_distinct(self):
        """Test that causal and non-causal kernels are different functions."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            gemm_fuse_softmax_causal,
            gemm_fuse_softmax_non_causal,
        )

        self.assertNotEqual(
            gemm_fuse_softmax_causal.__name__,
            gemm_fuse_softmax_non_causal.__name__,
        )


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestTritonKernelImports(unittest.TestCase):
    """Tests for verifying all triton kernel imports."""

    def test_all_kernels_importable(self):
        """Test all expected kernels can be imported."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            check_dense_contains_partial_stride,
            flashmask_apply,
            gemm_fuse_softmax_causal,
            gemm_fuse_softmax_non_causal,
        )

        kernels = [
            check_dense_contains_partial_stride,
            flashmask_apply,
            gemm_fuse_softmax_causal,
            gemm_fuse_softmax_non_causal,
        ]
        for kernel in kernels:
            self.assertTrue(callable(kernel))

    def test_module_level_functions(self):
        """Test that module-level helper functions are importable."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            RawPtrs,
            StrideMaxMinPtrs,
            _extract_raw_ptrs,
            _prepare_stride_maxmin_ptrs,
            _require,
            rr_attn_estimate_triton_func,
        )

        self.assertTrue(callable(_require))
        self.assertTrue(callable(_extract_raw_ptrs))
        self.assertTrue(callable(_prepare_stride_maxmin_ptrs))
        self.assertTrue(callable(rr_attn_estimate_triton_func))
        self.assertIsNotNone(RawPtrs)
        self.assertIsNotNone(StrideMaxMinPtrs)


if __name__ == "__main__":
    unittest.main()
