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

from __future__ import annotations

import hashlib
import os
import unittest
from tempfile import TemporaryDirectory


class LockFileTest(unittest.TestCase):
    test_url = (
        "https://bj.bcebos.com/paddlenlp/models/transformers/roformerv2/roformer_v2_chinese_char_small/vocab.txt"
    )

    def test_downloading_with_exist_file(self):

        from paddlenlp.utils.downloader import get_path_from_url_with_filelock

        with TemporaryDirectory() as tempdir:
            lock_file_name = hashlib.md5((self.test_url + tempdir).encode("utf-8")).hexdigest()
            lock_file_path = os.path.join(tempdir, ".lock", lock_file_name)
            os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)

            # create lock file
            with open(lock_file_path, "w", encoding="utf-8") as f:
                f.write("temp test")

            # downloading with exist lock file
            config_file = get_path_from_url_with_filelock(self.test_url, root_dir=tempdir)
            self.assertIsNotNone(config_file)

    def test_downloading_with_opened_exist_file(self):

        from paddlenlp.utils.downloader import get_path_from_url_with_filelock

        with TemporaryDirectory() as tempdir:
            lock_file_name = hashlib.md5((self.test_url + tempdir).encode("utf-8")).hexdigest()
            lock_file_path = os.path.join(tempdir, ".lock", lock_file_name)
            os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)

            # create lock file
            with open(lock_file_path, "w", encoding="utf-8") as f:
                f.write("temp test")

            # downloading with opened lock file
            open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
            _ = os.open(lock_file_path, open_mode)
            config_file = get_path_from_url_with_filelock(self.test_url, root_dir=tempdir)
            self.assertIsNotNone(config_file)
