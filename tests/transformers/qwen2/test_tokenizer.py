# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Team. All rights reserved.
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

from paddleformers.transformers import Qwen2Tokenizer, Qwen2TokenizerFast


class TestTokenizer(unittest.TestCase):
    from_pretrained_id = "paddleformers_test/tiny-random-qwen2"
    tokenizer_class = Qwen2Tokenizer
    rust_tokenizer_class = Qwen2TokenizerFast
    test_slow_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    test_seq2seq = False

    def test_slow_tokenizer_from_pretrained(self):
        tokenizer = Qwen2Tokenizer.from_pretrained(self.from_pretrained_id, download_hub="aistudio")
        self.assertTrue(tokenizer is not None)

    def test_slow_tokenizer_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = Qwen2Tokenizer.from_pretrained(self.from_pretrained_id)
            special_tokens_dict = {"additional_special_tokens": ["[ENT_START]", "[ENT_END]"]}
            tokenizer.add_special_tokens(special_tokens_dict)
            tokenizer.add_tokens(["new_word", "another_word"])
            tokenizer.model_max_length = 512
            tokenizer.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tokenizer_config.json")))

    def test_fast_tokenizer_from_pretrained(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.from_pretrained_id, download_hub="aistudio")
        self.assertTrue(tokenizer is not None)

    def test_fast_tokenizer_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = Qwen2TokenizerFast.from_pretrained(self.from_pretrained_id, download_hub="aistudio")
            special_tokens_dict = {"additional_special_tokens": ["[ENT_START]", "[ENT_END]"]}
            tokenizer.add_special_tokens(special_tokens_dict)
            tokenizer.add_tokens(["new_word", "another_word"])
            tokenizer.model_max_length = 512
            tokenizer.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tokenizer_config.json")))

    def test_tokenize(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.from_pretrained_id, download_hub="aistudio")
        text = "hello world, this is a tokenizer test"
        output_dict = tokenizer(text)
        decode_text = tokenizer.decode(output_dict["input_ids"], skip_special_tokens=True)
        self.assertEqual(text, decode_text)
