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

import paddle

from paddleformers.transformers import AutoTokenizer, Qwen2Tokenizer
from paddleformers.utils.download import DownloadSource
from tests.testing_utils import set_proxy


class TestHFMultiSourceTokenizer(unittest.TestCase):
    def encode(self, tokenizer):
        input_text = "hello world, 你好"
        output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_text))
        true_ids = [14990, 1879, 11, 220, 108386]
        self.assertEqual(output_ids, true_ids)

    @set_proxy(DownloadSource.AISTUDIO)
    def test_ai_studio(self):
        tokenizer = AutoTokenizer.from_pretrained("ModelHub/Qwen2.5-7B-Instruct", download_hub="aistudio")
        self.encode(tokenizer)
        tokenizer = Qwen2Tokenizer.from_pretrained("ModelHub/Qwen2.5-7B-Instruct", download_hub="aistudio")
        self.encode(tokenizer)

    @set_proxy(DownloadSource.MODELSCOPE)
    def test_model_scope(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", download_hub="modelscope")
        self.encode(tokenizer)
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", download_hub="modelscope")
        self.encode(tokenizer)

    @set_proxy(DownloadSource.HUGGINGFACE)
    def test_hf_hub(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", download_hub="huggingface")
        self.encode(tokenizer)
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", download_hub="huggingface")
        self.encode(tokenizer)

    def test_default(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.encode(tokenizer)
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.encode(tokenizer)

    @set_proxy(DownloadSource.HUGGINGFACE)
    def test_ernie_4_5_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT", download_hub="huggingface")
        input_text = "hello world, 你好"
        output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_text))
        true_ids = [18830, 3135, 93938, 93919, 5300]
        self.assertEqual(output_ids, true_ids)

    def test_auto_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("test_paddleformers/tiny-random-llama")
        input_text = "hello world, 你好"
        output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_text))
        true_ids = [12199, 3186, 29892, 29871, 30919, 31076]
        self.assertEqual(output_ids, true_ids)


class TestHFTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("PaddleNLP/Qwen2.5-7B")

    def test_encode(self):
        input_text = "hello world, this is paddle format checker"
        output_ids = self.tokenizer.encode(input_text, return_tensors="pd")[0]
        self.assertIsInstance(output_ids, paddle.Tensor)
        decode_text = self.tokenizer.decode(output_ids)
        self.assertEqual(input_text, decode_text)

    def test_encode_plus(self):
        input_text = "hello world, this is paddle format checker"
        output_dict = self.tokenizer.encode_plus(input_text, return_tensors="pd")
        true_dict = {
            "input_ids": [[14990, 1879, 11, 419, 374, 39303, 3561, 40915]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],
        }
        self.assertEqual(output_dict["input_ids"].tolist(), true_dict["input_ids"])
        self.assertEqual(output_dict["attention_mask"].tolist(), true_dict["attention_mask"])

    def test_batch_encode_plus(self):
        input_text = ["hello world, this is paddle format checker", "covert to decode to check"]
        output_dict = self.tokenizer.batch_encode_plus(input_text, return_tensors="pd", padding=True)
        true_dict = {
            "input_ids": [
                [14990, 1879, 11, 419, 374, 39303, 3561, 40915],
                [1015, 1621, 311, 16895, 311, 1779, 151643, 151643],
            ],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]],
        }
        self.assertEqual(output_dict["input_ids"].tolist(), true_dict["input_ids"])
        self.assertEqual(output_dict["attention_mask"].tolist(), true_dict["attention_mask"])

    def test_single_apply_chat_template(self):
        input_text = "hello world, this is paddle format checker"
        true_chat_str = self.tokenizer.apply_chat_template(input_text, tokenize=False)
        output_ids = self.tokenizer.apply_chat_template(input_text, return_tensors="pd")
        decode_str = self.tokenizer.decode(output_ids[0])
        self.assertEqual(true_chat_str, decode_str)

    def test_dict_apply_chat_template(self):
        input_text_dict_list = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "hello!"},
        ]
        true_chat_str = self.tokenizer.apply_chat_template(
            input_text_dict_list, tokenize=False, add_generation_prompt=True
        )
        output_ids = self.tokenizer.apply_chat_template(
            input_text_dict_list, return_tensors="pd", add_generation_prompt=True
        )
        decode_str = self.tokenizer.decode(output_ids[0])
        self.assertEqual(true_chat_str, decode_str)


class TestPaddleTokenizerMethod(unittest.TestCase):
    def test_tokenizer_decode_token(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("PaddleNLP/Qwen2.5-7B", download_hub="aistudio")
        test_cases = ["1. 百度 2. 腾讯", "hello world! I like eating banana", "🤓😖", "🤓😖testtest"]
        for test_case in test_cases:
            input_ids = tokenizer(test_case)["input_ids"]
            decoded_text = tokenizer.decode(input_ids)
            stream_decoded_text = ""
            offset = 0
            token_offset = 0
            for i in range(len(input_ids)):
                token_text, offset, token_offset = tokenizer.decode_token(input_ids[: i + 1], offset, token_offset)
                stream_decoded_text += token_text
            self.assertEqual(decoded_text, stream_decoded_text)
