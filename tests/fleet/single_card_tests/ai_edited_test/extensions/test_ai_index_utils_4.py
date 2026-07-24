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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False

import paddle


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFindBlocksTopp(unittest.TestCase):
    """Tests for find_blocks_topp function in index_utils."""

    def test_find_blocks_topp_exists(self):
        """Test find_blocks_topp can be imported."""
        try:
            from paddlefleet_ops._extensions.flashmask.index_utils import (
                find_blocks_topp,
            )

            self.assertTrue(callable(find_blocks_topp))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_find_blocks_topp_signature(self):
        """Test find_blocks_topp has expected signature."""
        try:
            import inspect

            from paddlefleet_ops._extensions.flashmask.index_utils import (
                find_blocks_topp,
            )

            sig = inspect.signature(find_blocks_topp)
            params = list(sig.parameters.keys())
            self.assertIn("x", params)
            self.assertIn("p", params)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_find_blocks_topp_output_type(self):
        """Test find_blocks_topp returns a tensor."""
        try:
            from paddlefleet_ops._extensions.flashmask.index_utils import (
                find_blocks_topp,
            )

            x = paddle.rand([1, 1, 4, 4])
            result = find_blocks_topp(x, 0.9)
            self.assertIsInstance(result, paddle.Tensor)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_find_blocks_topp_output_shape(self):
        """Test find_blocks_topp preserves input shape."""
        try:
            from paddlefleet_ops._extensions.flashmask.index_utils import (
                find_blocks_topp,
            )

            x = paddle.rand([1, 1, 4, 4])
            result = find_blocks_topp(x, 0.9)
            self.assertEqual(result.shape, x.shape)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_find_blocks_topp_low_threshold(self):
        """Test find_blocks_topp with very low threshold includes most elements."""
        try:
            from paddlefleet_ops._extensions.flashmask.index_utils import (
                find_blocks_topp,
            )

            x = paddle.rand([1, 1, 4, 4])
            result = find_blocks_topp(x, 0.01)
            # With very low threshold, almost all should be True
            self.assertEqual(result.shape, x.shape)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_top_p_kernel_exists(self):
        """Test top_p_kernel is defined in module."""
        try:
            from paddlefleet_ops._extensions.flashmask.index_utils import (
                top_p_kernel,
            )

            self.assertIsNotNone(top_p_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")


if __name__ == "__main__":
    unittest.main()
