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

from paddleformers.datasets.reader.file_reader import FileListReader, FileReader
from tests.testing_utils import get_tests_dir

output_data = {
    "messages": [
        {"role": "user", "content": "针对产品发布提出五种营销策略。"},
        {"role": "assistant", "content": "1. 社交媒体活动。\n2. 电子邮件营销。\n3. 在线和离线广告。\n4. 推荐和评论。\n5. 合作名人推销。"},
    ],
    "label": [1],
    "system": "",
}


class TestDatasetFileReader(unittest.TestCase):
    def test_file_reader(self):
        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "io", "train.jsonl")
        file_reader = FileReader(dataset_path, "erniekit")
        dataset_iterator = iter(file_reader)
        example = next(dataset_iterator)
        self.assertEqual(example, output_data)

    def test_filelist_reader(self):
        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "io")
        filelist_reader = FileListReader(dataset_path, "erniekit")
        dataset_iterator = iter(filelist_reader)
        example = next(dataset_iterator)
        self.assertEqual(example, output_data)

    def test_file_samplenum_undersample(self):
        """file_samplenum < total rows: only the first N samples are returned."""
        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "io", "train.jsonl")
        file_reader = FileReader(dataset_path, "erniekit", file_samplenum=2)
        samples = [s for s in iter(file_reader)]
        self.assertEqual(len(samples), 2)

    def test_file_samplenum_oversample(self):
        """file_samplenum > total rows: data is repeated to reach the target count."""
        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "io", "train.jsonl")
        # 3 rows in file, request 7 -> 3*2 + 1 = 7
        file_reader = FileReader(dataset_path, "erniekit", file_samplenum=7)
        samples = [s for s in iter(file_reader)]
        self.assertEqual(len(samples), 7)
        # First sample repeats at index 3
        self.assertEqual(samples[0], samples[3])

    def test_unsupported_file_type_raises(self):
        """Passing an unsupported file_type raises ValueError."""
        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "io", "train.jsonl")
        file_reader = FileReader(dataset_path, "unknown_format")
        with self.assertRaises(ValueError):
            for _ in iter(file_reader):
                pass

    def test_unsupported_extension_raises(self):
        """A file with unsupported extension raises ValueError when iterated."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"some content")
            tmp_path = f.name
        try:
            file_reader = FileReader(tmp_path, "erniekit")
            with self.assertRaises(ValueError):
                for _ in iter(file_reader):
                    pass
        finally:
            os.unlink(tmp_path)

    def test_filelist_reader_invalid_dir_raises(self):
        """FileListReader raises ValueError when given a non-directory path."""
        with self.assertRaises(ValueError):
            FileListReader("/nonexistent/directory", "erniekit")
