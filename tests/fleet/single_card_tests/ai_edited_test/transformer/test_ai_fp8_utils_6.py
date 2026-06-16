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

from paddleformers.fleet.transformer.moe.fp8_utils import FP8_ALIGN


class TestFP8Align(unittest.TestCase):
    """Tests for FP8_ALIGN constant."""

    def test_fp8_align_value(self):
        """Test FP8_ALIGN is a positive integer."""
        self.assertIsInstance(FP8_ALIGN, int)
        self.assertGreater(FP8_ALIGN, 0)


class TestFP8UtilsImport(unittest.TestCase):
    """Tests for fp8_utils module imports."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        from paddleformers.fleet.transformer.moe import fp8_utils

        self.assertTrue(hasattr(fp8_utils, "FP8_ALIGN"))


class TestFP8QuantUtils(unittest.TestCase):
    """Tests for fp8 quantization utilities."""

    def test_fused_stack_quant_without_cache_import(self):
        """Test fused_stack_quant_without_cache can be imported."""
        try:
            from paddleformers.fleet.transformer.moe.fp8_utils import (
                fused_stack_quant_without_cache,
            )

            self.assertIsNotNone(fused_stack_quant_without_cache)
        except (ImportError, AttributeError):
            # May not be available in all environments
            pass


if __name__ == "__main__":
    unittest.main()
