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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


# Tests for src/paddleformers.fleet/distributed/__init__.py

import unittest


class TestDistributedInit(unittest.TestCase):
    """Tests for the distributed package __init__.py"""

    def test_distributed_init_import(self):
        """Test that the distributed __init__.py can be imported."""
        import paddleformers.fleet.distributed

        # The __init__.py is mostly just the license header, but importing
        # should not fail.
        self.assertTrue(hasattr(paddleformers.fleet.distributed, "__file__"))


if __name__ == "__main__":
    unittest.main()
