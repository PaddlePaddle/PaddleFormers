# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Hugging Face inc.
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

import copy
import json
import os
import tempfile
import unittest

from paddlenlp.transformers import AutoModel, BertModel
from paddlenlp.utils.env import CONFIG_NAME, PADDLE_WEIGHTS_NAME


class AutoModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert")

    def test_from_pretrained_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.model.save_pretrained(tmp_dir)
            model = AutoModel.from_pretrained(tmp_dir)
            self.assertIsInstance(model, BertModel)

    def test_from_pretrained_no_init_class_with_model_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = copy.deepcopy(self.model)
            # when init_class is not found, we rely on the filename to get the import class
            model_save_path = os.path.join(tmp_dir, "tiny-random-bert")
            model.save_pretrained(model_save_path)
            config = model.config.to_dict()
            config.pop("architectures")
            with open(os.path.join(model_save_path, "config.json"), "w", encoding="utf-8") as writer:
                writer.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
            reloaded_model = AutoModel.from_pretrained(model_save_path)
            self.assertIsInstance(reloaded_model, BertModel)

    def test_from_pretrained_no_init_class_no_model_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = copy.deepcopy(self.model)
            model.save_pretrained(tmp_dir)
            config = model.config.to_dict()
            config.pop("architectures")
            with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as writer:
                writer.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
            with self.assertRaises(AttributeError):
                AutoModel.from_pretrained(tmp_dir)

    def test_model_from_pretrained_cache_dir(self):
        model_name = "__internal_testing__/tiny-random-bert"
        with tempfile.TemporaryDirectory() as tempdir:
            AutoModel.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, PADDLE_WEIGHTS_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))

    @unittest.skip("skipping due to connection error!")
    def test_from_hf_hub(self):
        model = AutoModel.from_pretrained("PaddleCI/tiny-random-bert", from_hf_hub=True, convert_from_torch=False)
        self.assertIsInstance(model, BertModel)

    @unittest.skip("skipping due to connection error!")
    def test_from_aistudio(self):
        model = AutoModel.from_pretrained("PaddleNLP/tiny-random-bert", from_aistudio=True)
        self.assertIsInstance(model, BertModel)
