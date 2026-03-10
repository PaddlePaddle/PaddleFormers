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

"""
Muon optimizer smoke tests — exercise V2 and V3 sharding paths on 2 GPUs.
"""

import os
import unittest

import paddle

from tests.parallel_launch import TestMultipleGpus


class TestMuonSmoke(TestMultipleGpus):
    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or paddle.device.cuda.get_device_capability()[0] < 8,
        "BF16 matmul requires GPU compute capability >= 80 (Ampere+)",
    )
    def test_muon_sharding_v2(self):
        """Muon + Sharding V2, 2 GPUs."""
        self.run_2gpu("tests/muon/muon_smoke_test_worker.py")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or paddle.device.cuda.get_device_capability()[0] < 8,
        "BF16 matmul requires GPU compute capability >= 80 (Ampere+)",
    )
    def test_muon_sharding_v3(self):
        """Muon + Sharding V3, 2 GPUs."""
        os.environ["FLAGS_sharding_v3"] = "1"
        self.run_2gpu("tests/muon/muon_smoke_test_worker.py")
        os.environ.pop("FLAGS_sharding_v3", None)


if __name__ == "__main__":
    unittest.main()
