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

import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False

import paddle


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestPrepareMaxmin(unittest.TestCase):
    """Tests for prepare_maxmin function in block_mask_utils."""

    def test_prepare_maxmin_exists(self):
        """Test prepare_maxmin function can be imported."""
        try:
            from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
                prepare_maxmin,
            )

            self.assertTrue(callable(prepare_maxmin))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_prepare_maxmin_signature(self):
        """Test prepare_maxmin has expected signature."""
        try:
            import inspect

            from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
                prepare_maxmin,
            )

            sig = inspect.signature(prepare_maxmin)
            params = list(sig.parameters.keys())
            self.assertIn("input", params)
            self.assertIn("chunk_size", params)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_prepare_maxmin_output_shapes(self):
        """Test prepare_maxmin returns correct output shapes."""
        try:
            from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
                prepare_maxmin,
            )

            x = paddle.zeros([2, 4, 64], dtype=paddle.int32)
            chunk_size = 16
            out_max, out_min = prepare_maxmin(x, chunk_size)
            # num_chunks = ceil(64/16) = 4
            self.assertEqual(out_max.shape, [2, 4, 4])
            self.assertEqual(out_min.shape, [2, 4, 4])
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_prepare_maxmin_small_input(self):
        """Test prepare_maxmin with small input."""
        try:
            from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
                prepare_maxmin,
            )

            x = paddle.zeros([1, 2, 32], dtype=paddle.int32)
            chunk_size = 16
            out_max, out_min = prepare_maxmin(x, chunk_size)
            self.assertEqual(out_max.shape[0], 1)
            self.assertEqual(out_max.shape[1], 2)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_scan_maxmin_chunked_exists(self):
        """Test scan_maxmin_chunked kernel exists."""
        try:
            from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
                scan_maxmin_chunked,
            )

            self.assertIsNotNone(scan_maxmin_chunked)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")


if __name__ == "__main__":
    unittest.main()
