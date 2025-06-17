# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

from paddleformers.datasets import load_dataset
from tests.testing_utils import get_tests_dir


class TestIterDataset(unittest.TestCase):
    def test_skip(self):
        fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
        first_ds = load_dataset("json", data_files=os.path.join(fixture_path, "tnews", "train.json"), lazy=True)[0]
        first_ds = first_ds.skip(5)
        first_data = next(iter(first_ds))
        second_ds = iter(
            load_dataset("json", data_files=os.path.join(fixture_path, "tnews", "train.json"), lazy=True)[0]
        )
        for i in range(5):
            next(second_ds)
        second_data = next(second_ds)
        self.assertEqual(first_data, second_data)
