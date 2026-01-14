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
