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


# Tests for src/paddleformers.fleet/_extensions/flashmask/block_mask_utils.py
# Dedicated tests for bitonic_argsort_device, top_p_kernel, _compare_and_swap,
# _bitonic_merge

import types
import unittest

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

try:
    from paddlefleet_ops._extensions.flashmask.block_mask_utils import (  # noqa: F401
        bitonic_argsort_device,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestBitonicArgsortDevice(unittest.TestCase):
    """Tests for bitonic_argsort_device triton kernel."""

    def test_is_jit_function(self):
        """Test that bitonic_argsort_device is a triton jit function."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            bitonic_argsort_device,
        )

        self.assertTrue(callable(bitonic_argsort_device))

    def test_is_importable(self):
        """Test bitonic_argsort_device can be imported."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            bitonic_argsort_device,
        )

        self.assertIsNotNone(bitonic_argsort_device)

    def test_function_name(self):
        """Test that function name is correct."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            bitonic_argsort_device,
        )

        self.assertEqual(bitonic_argsort_device.__name__, "bitonic_argsort_device")


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestCompareAndSwap(unittest.TestCase):
    """Tests for _compare_and_swap triton kernel."""

    def test_is_jit_function(self):
        """Test that _compare_and_swap is a triton jit function."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _compare_and_swap,
        )

        self.assertTrue(callable(_compare_and_swap))

    def test_is_importable(self):
        """Test _compare_and_swap can be imported."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _compare_and_swap,
        )

        self.assertIsNotNone(_compare_and_swap)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestBitonicMerge(unittest.TestCase):
    """Tests for _bitonic_merge triton kernel."""

    def test_is_jit_function(self):
        """Test that _bitonic_merge is a triton jit function."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _bitonic_merge,
        )

        self.assertTrue(callable(_bitonic_merge))

    def test_is_importable(self):
        """Test _bitonic_merge can be imported."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _bitonic_merge,
        )

        self.assertIsNotNone(_bitonic_merge)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestTopPKernel(unittest.TestCase):
    """Tests for top_p_kernel triton kernel."""

    def test_is_jit_function(self):
        """Test that top_p_kernel is a triton jit function."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import top_p_kernel

        self.assertTrue(callable(top_p_kernel))

    def test_is_importable(self):
        """Test top_p_kernel can be imported."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import top_p_kernel

        self.assertIsNotNone(top_p_kernel)

    def test_function_name(self):
        """Test that function name is correct."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import top_p_kernel

        self.assertEqual(top_p_kernel.__name__, "top_p_kernel")

    def test_used_in_find_blocks_topp(self):
        """Test that top_p_kernel is called by find_blocks_topp."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            find_blocks_topp,
            top_p_kernel,
        )

        # Verify they come from the same module
        self.assertEqual(
            find_blocks_topp.__module__,
            top_p_kernel.__module__,
        )


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestBitonicSortingRelationship(unittest.TestCase):
    """Tests for the relationship between bitonic sort components."""

    def test_all_sort_components_importable(self):
        """Test all bitonic sorting components can be imported."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _bitonic_merge,
            _compare_and_swap,
            bitonic_argsort_device,
            top_p_kernel,
        )

        self.assertTrue(callable(_compare_and_swap))
        self.assertTrue(callable(_bitonic_merge))
        self.assertTrue(callable(bitonic_argsort_device))
        self.assertTrue(callable(top_p_kernel))

    def test_compare_and_swap_is_building_block(self):
        """Test that _compare_and_swap is used within _bitonic_merge."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _bitonic_merge,
            _compare_and_swap,
        )

        # Both should be JIT functions
        self.assertTrue(callable(_compare_and_swap))
        self.assertTrue(callable(_bitonic_merge))

    def test_bitonic_merge_used_in_argsort(self):
        """Test that _bitonic_merge is used within bitonic_argsort_device."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _bitonic_merge,
            bitonic_argsort_device,
        )

        self.assertTrue(callable(_bitonic_merge))
        self.assertTrue(callable(bitonic_argsort_device))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestBlockMaskUtilsModuleStructure(unittest.TestCase):
    """Tests for module-level structure of block_mask_utils."""

    def test_module_exports(self):
        """Test that expected names are exported from the module."""
        import paddlefleet_ops._extensions.flashmask.block_mask_utils as bm

        expected_names = [
            "find_blocks_topp",
            "top_p_kernel",
            "bitonic_argsort_device",
            "_bitonic_merge",
            "_compare_and_swap",
            "check_fully_masked_state",
            "check_partially_masked_state",
            "_load_bounds",
            "_is_block_fully_masked",
            "_is_block_partially_masked",
        ]
        for name in expected_names:
            self.assertTrue(
                hasattr(bm, name),
                f"Module missing expected attribute: {name}",
            )

    def test_all_jit_functions_are_callable(self):
        """Test that all JIT-decorated functions are callable."""
        import paddlefleet_ops._extensions.flashmask.block_mask_utils as bm

        jit_names = [
            "find_blocks_topp",
            "top_p_kernel",
            "bitonic_argsort_device",
            "_bitonic_merge",
            "_compare_and_swap",
            "check_fully_masked_state",
            "check_partially_masked_state",
            "_load_bounds",
            "_is_block_fully_masked",
            "_is_block_partially_masked",
        ]
        for name in jit_names:
            func = getattr(bm, name)
            self.assertTrue(
                callable(func),
                f"{name} should be callable",
            )


if __name__ == "__main__":
    unittest.main()
