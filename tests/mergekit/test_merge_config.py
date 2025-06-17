# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from tempfile import TemporaryDirectory

from paddlenlp.mergekit import MergeConfig


class TestMergeConfig(unittest.TestCase):
    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            merge_config = MergeConfig()
            merge_config.to_dict()
            merge_config.save_pretrained(tempdir)
            loaded_merge_config = MergeConfig.from_pretrained(tempdir)
            self.assertEqual(merge_config, loaded_merge_config)
            with self.assertRaises(ValueError):
                MergeConfig.from_pretrained("./rand")

    def test_raise_exception(self):
        with self.assertRaises(ValueError):
            MergeConfig(merge_method="linear1")
        with self.assertRaises(ValueError):
            MergeConfig(model_path_list=["./model1"])
        with self.assertRaises(ValueError):
            MergeConfig(model_path_list=["./model1", "./model2"], weight_list=[0.1])
        with self.assertRaises(ValueError):
            MergeConfig(reserve_p=1.1)
        with self.assertRaises(ValueError):
            MergeConfig(
                sparsify_type="magprune",
                reserve_p=0.8,
                epsilon=0.8,
            )
