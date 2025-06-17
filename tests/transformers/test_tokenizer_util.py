# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import json
import os
import tempfile
import unittest

from paddlenlp.transformers import BertTokenizer
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.utils.env import TOKENIZER_CONFIG_NAME


class EmptyTokenizer(PretrainedTokenizer):
    def __init__(self, a=1, b=2):
        pass


class SubEmptyTokenizer(EmptyTokenizer):
    def __init__(self, c=3, d=4):
        super().__init__(a=c, b=d)


class TokenizerUtilsTest(unittest.TestCase):
    def test_multi_inherit(self):
        tokenizer = SubEmptyTokenizer()

        self.assertIn("c", tokenizer.init_kwargs)
        self.assertEqual(tokenizer.init_kwargs["c"], 3)

    def test_config(self):
        tmpdirname = tempfile.mkdtemp()

        tokenizer = SubEmptyTokenizer()
        tokenizer.save_pretrained(tmpdirname)

        with open(os.path.join(tmpdirname, "tokenizer_config.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("c", data)
        self.assertEqual(data["c"], 3)
        self.assertEqual(data["tokenizer_class"], "SubEmptyTokenizer")

    def test_from_pretrained_cache_dir(self):
        model_name = "__internal_testing__/tiny-random-bert"
        with tempfile.TemporaryDirectory() as tempdir:
            BertTokenizer.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, TOKENIZER_CONFIG_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))
