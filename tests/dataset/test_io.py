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

from paddleformers.datasets.reader.io import load_json, load_parquet
from tests.testing_utils import get_tests_dir


class TestDatasetIO(unittest.TestCase):
    def test_jsonl_io(self):
        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "io", "train.jsonl")
        res = load_json(dataset_path)
        # load_json returns generator for JSONL files
        res = list(res)
        self.assertEqual(len(res), 3)

    def test_parquet_io(self):
        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "io", "train.parquet")
        res = load_parquet(dataset_path)
        self.assertEqual(len(res), 3)
