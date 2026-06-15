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


class TestDistributedModelAMP(unittest.TestCase):
    """Tests for distributed_model with AMP settings."""


class TestDistributedModelPipelineParallel(unittest.TestCase):
    """Tests for distributed_model with pipeline parallel."""


class TestDistributedModelInterleave(unittest.TestCase):
    """Tests for distributed_model interleave pipeline selection."""


class TestDistributedInitModule(unittest.TestCase):
    """Tests for the distributed __init__ module."""

    def test_import_distributed(self):
        import paddleformers.fleet.distributed

        self.assertIsNotNone(paddleformers.fleet.distributed)


if __name__ == "__main__":
    unittest.main()
