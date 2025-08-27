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
import os
import shutil
import sys
import unittest

import paddle as paddle_shadow

sys.modules["paddle"] = None
from paddleformers.transformers import AutoTokenizer

sys.modules["paddle"] = paddle_shadow


class TestTokenizer(unittest.TestCase):
    def setUp(self):

        self.test_dirs = ["./slow_tokenizer", "./fast_tokenizer"]
        for test_dir in self.test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def tearDown(self):
        for test_dir in self.test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_slow_tokenizer_from_pretrained(self):
        tokenizer = AutoTokenizer.from_pretrained("PaddleNLP/Qwen2-7B", use_fast=False)
        if hasattr(tokenizer, "is_fast"):
            self.assertFalse(tokenizer.is_fast)
        else:
            self.assertNotIn("Fast", tokenizer.__class__.__name__)

    def test_slow_tokenizer_save_pretrained(self):
        tokenizer = AutoTokenizer.from_pretrained("PaddleNLP/Qwen2-7B", use_fast=False)
        special_tokens_dict = {"additional_special_tokens": ["[ENT_START]", "[ENT_END]"]}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.add_tokens(["new_word", "another_word"])
        tokenizer.model_max_length = 512
        tokenizer.save_pretrained("./slow_tokenizer")
        self.assertTrue(os.path.exists("./slow_tokenizer/tokenizer_config.json"))

    def test_fast_tokenizer_from_pretrained(self):
        tokenizer = AutoTokenizer.from_pretrained("PaddleNLP/Qwen2-7B")
        if hasattr(tokenizer, "is_fast"):
            self.assertTrue(tokenizer.is_fast)
        else:
            self.assertIn("Fast", tokenizer.__class__.__name__)

    def test_fast_tokenizer_save_pretrained(self):
        tokenizer = AutoTokenizer.from_pretrained("PaddleNLP/Qwen2-7B")
        special_tokens_dict = {"additional_special_tokens": ["[ENT_START]", "[ENT_END]"]}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.add_tokens(["new_word", "another_word"])
        tokenizer.model_max_length = 512
        tokenizer.save_pretrained("./fast_tokenizer")
        self.assertTrue(os.path.exists("./fast_tokenizer/tokenizer_config.json"))
