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


class TestAdamWTritonImport(unittest.TestCase):
    """Test that the adamw_triton module can be imported or gracefully handled."""

    def test_import_or_skip(self):
        # The module requires triton, which may not be available
        try:
            from paddleformers.utils.adamw_triton import DTYPE_MAPPING

            self.assertIsInstance(DTYPE_MAPPING, dict)
        except (ImportError, RuntimeError):
            self.skipTest("Triton is not available in this environment")


if __name__ == "__main__":
    unittest.main()
