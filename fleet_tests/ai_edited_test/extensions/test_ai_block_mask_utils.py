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


# Tests for src/paddlefleet/_extensions/flashmask/block_mask_utils.py
# Triton kernels are mocked since they require GPU.

import types
import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False
from unittest import mock

# Mock triton and triton.language if not available
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
    _mock_triton.jit = lambda fn=None, **kw: (
        fn if fn is not None else lambda f: f
    )
    _mock_triton.cdiv = lambda a, b: (a + b - 1) // b
    _mock_triton.next_power_of_2 = lambda n: (
        1 << (n - 1).bit_length() if n > 0 else 1
    )
    sys.modules.setdefault("triton", _mock_triton)
    sys.modules.setdefault("triton.language", _mock_tl)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFindBlocksTopp(unittest.TestCase):
    """Tests for find_blocks_topp function."""

    def test_find_blocks_topp_basic(self):
        """Test find_blocks_topp basic functionality."""
        import paddle
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            find_blocks_topp,
        )

        x = paddle.randn([1, 1, 2, 4], dtype="float32")
        # Mock the triton kernel
        mock_mask = paddle.ones([1, 1, 2, 4], dtype="bool")
        with (
            mock.patch(
                "paddlefleet_ops._extensions.flashmask.block_mask_utils.top_p_kernel"
            ),
            mock.patch("triton.next_power_of_2", return_value=4),
        ):
            result = find_blocks_topp(x, p=0.9)
            self.assertEqual(result.shape, [1, 1, 2, 4])

    def test_find_blocks_topp_reshape(self):
        """Test find_blocks_topp reshapes correctly."""
        import paddle
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            find_blocks_topp,
        )

        x = paddle.randn([2, 4, 2, 8], dtype="float32")
        with (
            mock.patch(
                "paddlefleet_ops._extensions.flashmask.block_mask_utils.top_p_kernel"
            ),
            mock.patch("triton.next_power_of_2", return_value=8),
        ):
            result = find_blocks_topp(x, p=0.5)
            self.assertEqual(result.shape, [2, 4, 2, 8])

    def test_find_blocks_topp_small_n(self):
        """Test find_blocks_topp with n < 1."""
        import paddle
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            find_blocks_topp,
        )

        x = paddle.randn([1, 1, 2, 1], dtype="float32")
        mock_mask = paddle.ones([1, 1, 2, 1], dtype="bool")
        with (
            mock.patch(
                "paddlefleet_ops._extensions.flashmask.block_mask_utils.top_p_kernel"
            ),
            mock.patch("triton.next_power_of_2", return_value=1),
        ):
            result = find_blocks_topp(x, p=0.9)
            self.assertEqual(result.shape, [1, 1, 2, 1])


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestBlockMaskUtilsImports(unittest.TestCase):
    """Test that triton jit functions exist."""

    def test_load_bounds_exists(self):
        """Test _load_bounds is defined."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _load_bounds,
        )

        self.assertIsNotNone(_load_bounds)

    def test_is_block_fully_masked_exists(self):
        """Test _is_block_fully_masked is defined."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _is_block_fully_masked,
        )

        self.assertIsNotNone(_is_block_fully_masked)

    def test_check_fully_masked_state_exists(self):
        """Test check_fully_masked_state is defined."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            check_fully_masked_state,
        )

        self.assertIsNotNone(check_fully_masked_state)

    def test_is_block_partially_masked_exists(self):
        """Test _is_block_partially_masked is defined."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _is_block_partially_masked,
        )

        self.assertIsNotNone(_is_block_partially_masked)

    def test_check_partially_masked_state_exists(self):
        """Test check_partially_masked_state is defined."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            check_partially_masked_state,
        )

        self.assertIsNotNone(check_partially_masked_state)

    def test_compare_and_swap_exists(self):
        """Test _compare_and_swap is defined."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _compare_and_swap,
        )

        self.assertIsNotNone(_compare_and_swap)

    def test_bitonic_merge_exists(self):
        """Test _bitonic_merge is defined."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            _bitonic_merge,
        )

        self.assertIsNotNone(_bitonic_merge)

    def test_bitonic_argsort_device_exists(self):
        """Test bitonic_argsort_device is defined."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            bitonic_argsort_device,
        )

        self.assertIsNotNone(bitonic_argsort_device)

    def test_top_p_kernel_exists(self):
        """Test top_p_kernel is defined."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            top_p_kernel,
        )

        self.assertIsNotNone(top_p_kernel)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestBlockMaskUtilsUsedByTritonOp(unittest.TestCase):
    """Test that functions used by rr_attn_estimate_triton_op are importable."""

    def test_imports_in_triton_op(self):
        """Test that triton_op imports from block_mask_utils."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            check_fully_masked_state as cfm,
            check_partially_masked_state as cpm,
            find_blocks_topp as fbt,
        )
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            check_fully_masked_state,
            check_partially_masked_state,
            find_blocks_topp,
        )

        self.assertIs(check_fully_masked_state, cfm)
        self.assertIs(check_partially_masked_state, cpm)
        self.assertIs(find_blocks_topp, fbt)


if __name__ == "__main__":
    unittest.main()
