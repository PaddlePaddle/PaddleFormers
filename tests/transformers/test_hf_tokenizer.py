# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.transformers import AutoTokenizer, Qwen2Tokenizer


@unittest.skip("multi source download CI not support")
class TestHFTokenizer(unittest.TestCase):
    def encode(self, tokenizer):
        input_text = "hello world, 你好"
        output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_text))
        true_ids = [14990, 1879, 11, 220, 108386]
        self.assertEqual(output_ids, true_ids)

    def test_ai_studio(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", from_aistudio=True)
        self.encode(tokenizer)
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", from_aistudio=True)
        self.encode(tokenizer)

    def test_model_scope(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", from_modelscope=True)
        self.encode(tokenizer)
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", from_modelscope=True)
        self.encode(tokenizer)

    def test_hf_hub(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", from_hf_hub=True)
        self.encode(tokenizer)
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", from_hf_hub=True)
        self.encode(tokenizer)

    def test_default(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.encode(tokenizer)
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.encode(tokenizer)

    def test_ernie_4_5_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT", from_hf_hub=True)
        input_text = "hello world, 你好"
        output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_text))
        true_ids = [18830, 3135, 93938, 93919, 5300]
        self.assertEqual(output_ids, true_ids)

    def test_auto_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/micro-random-llama")
        input_text = "hello world, 你好"
        output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_text))
        true_ids = [22172, 3186, 29892, 29871, 30919, 31076]
        self.assertEqual(output_ids, true_ids)
