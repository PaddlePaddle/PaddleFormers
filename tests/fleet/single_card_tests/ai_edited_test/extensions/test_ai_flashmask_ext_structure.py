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


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFlashmaskExtensions(unittest.TestCase):
    """Tests for flashmask _extensions module structure."""

    def test_flashmask_module_import(self):
        """Test the flashmask module can be imported."""
        try:
            from paddlefleet_ops._extensions import flashmask

            self.assertIsNotNone(flashmask)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_flashmask_has_block_mask_utils(self):
        """Test flashmask has block_mask_utils submodule."""
        try:
            from paddlefleet_ops._extensions.flashmask import block_mask_utils

            self.assertIsNotNone(block_mask_utils)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_flashmask_has_index_utils(self):
        """Test flashmask has index_utils submodule."""
        try:
            from paddlefleet_ops._extensions.flashmask import index_utils

            self.assertIsNotNone(index_utils)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_bitonic_argsort_device_exists(self):
        """Test bitonic_argsort_device is defined in index_utils."""
        try:
            from paddlefleet_ops._extensions.flashmask.index_utils import (
                bitonic_argsort_device,
            )

            self.assertIsNotNone(bitonic_argsort_device)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_compare_and_swap_exists(self):
        """Test _compare_and_swap is defined in index_utils."""
        try:
            from paddlefleet_ops._extensions.flashmask.index_utils import (
                _compare_and_swap,
            )

            self.assertIsNotNone(_compare_and_swap)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_bitonic_merge_exists(self):
        """Test _bitonic_merge is defined in index_utils."""
        try:
            from paddlefleet_ops._extensions.flashmask.index_utils import _bitonic_merge

            self.assertIsNotNone(_bitonic_merge)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")


if __name__ == "__main__":
    unittest.main()
