# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
import tempfile
import unittest

from paddleformers.transformers.intern_lm2.tokenizer import InternLM2Tokenizer

model_path = "learncat/internlm2_tiny_paddle"

class TestTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = InternLM2Tokenizer.from_pretrained(model_path)


    def test_slow_tokenizer_from_pretrained(self):
        self.assertTrue(self.tokenizer is not None)

    def test_slow_tokenizer_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            special_tokens_dict = {"additional_special_tokens": ["[ENT_START]", "[ENT_END]"]}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.tokenizer.add_tokens(["new_word", "another_word"])
            self.tokenizer.model_max_length = 512
            self.tokenizer.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tokenizer_config.json")))

    def test_tokenize(self):
        text = "hello world, this is a tokenizer test"
        output_dict = self.tokenizer(text)
        decode_text = self.tokenizer.decode(output_dict["input_ids"], skip_special_tokens=True)
        self.assertEqual(text, decode_text)
