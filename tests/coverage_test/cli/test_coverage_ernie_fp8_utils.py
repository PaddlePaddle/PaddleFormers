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

import unittest

from paddleformers.cli.train.ernie_pretrain.models.moe.token_dispatcher.fp8_utils import (
    _get_fp8_weight_and_scale,
)


class TestGetFp8WeightAndScale(unittest.TestCase):
    """Tests for _get_fp8_weight_and_scale function."""

    def test_function_exists(self):
        """Test that the function exists."""
        self.assertTrue(callable(_get_fp8_weight_and_scale))

    def test_import_succeeds(self):
        """Test that fp8_utils can be imported."""
        try:
            from paddleformers.cli.train.ernie_pretrain.models.moe.token_dispatcher.fp8_utils import (
                ExpertsGroupGemmContiguousNode,
                ExpertsGroupGemmNode,
            )

            self.assertTrue(ExpertsGroupGemmNode is not None)
            self.assertTrue(ExpertsGroupGemmContiguousNode is not None)
        except ImportError:
            self.skipTest("fp8_utils requires GPU FP8 support")


class TestFp8UtilsModule(unittest.TestCase):
    """Tests for the fp8_utils module."""

    def test_module_all_exports(self):
        """Test that __all__ exports are defined."""
        from paddleformers.cli.train.ernie_pretrain.models.moe.token_dispatcher import (
            fp8_utils,
        )

        self.assertIn("ExpertsGroupGemmNode", fp8_utils.__all__)
        self.assertIn("ExpertsGroupGemmContiguousNode", fp8_utils.__all__)


if __name__ == "__main__":
    unittest.main()
