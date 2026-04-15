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

from paddleformers.datasets.reader.io import load_csv, load_json, load_parquet, load_txt
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

    def test_txt_io(self):
        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "io", "train.txt")
        res = load_txt(dataset_path)
        self.assertIsInstance(res, str)
        self.assertGreater(len(res), 0)

    def test_csv_io(self):
        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "io", "train.csv")
        res = load_csv(dataset_path)
        self.assertIsInstance(res, list)
        # header row + 3 data rows
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0], ["question", "answer"])

    def test_jsonl_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            list(load_json("/nonexistent/path/to/file.jsonl"))

    def test_txt_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_txt("/nonexistent/path/to/file.txt")

    def test_csv_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_csv("/nonexistent/path/to/file.csv")

    def test_parquet_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_parquet("/nonexistent/path/to/file.parquet")

    def test_jsonl_parse_error(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"key": "valid"}\n')
            f.write("this is not valid json\n")
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                list(load_json(tmp_path))
        finally:
            os.unlink(tmp_path)
