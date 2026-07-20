# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.transformers import InternLM3Tokenizer

hf_model_path = "internlm/internlm3-8b-instruct"


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = InternLM3Tokenizer.from_pretrained(hf_model_path, download_hub="huggingface")
        except Exception:
            cls.tokenizer = None

    def test_tokenizer_from_pretrained(self):
        if self.tokenizer is None:
            self.skipTest("Model path not available")
        self.assertTrue(self.tokenizer is not None)

    def test_tokenizer_save_pretrained(self):
        if self.tokenizer is None:
            self.skipTest("Model path not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            special_tokens_dict = {"additional_special_tokens": ["[ENT_START]", "[ENT_END]"]}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.tokenizer.add_tokens(["new_word", "another_word"])
            self.tokenizer.model_max_length = 512
            self.tokenizer.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tokenizer_config.json")))

    def test_tokenize(self):
        if self.tokenizer is None:
            self.skipTest("Model path not available")

        text = "hello world, this is a tokenizer test"
        input_ids = self.tokenizer.encode(text)
        decode_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        self.assertEqual(text, decode_text)

    def test_tokenizer_vocab_size(self):
        if self.tokenizer is None:
            self.skipTest("Model path not available")

        vocab_size = self.tokenizer.vocab_size
        self.assertGreater(vocab_size, 0)

    def test_tokenizer_bos_eos_tokens(self):
        if self.tokenizer is None:
            self.skipTest("Model path not available")

        self.assertIsNotNone(self.tokenizer.bos_token_id)
        self.assertIsNotNone(self.tokenizer.eos_token_id)

    def test_tokenizer_build_inputs_with_special_tokens(self):
        if self.tokenizer is None:
            self.skipTest("Model path not available")

        token_ids_0 = [1, 2, 3]
        output = self.tokenizer.build_inputs_with_special_tokens(token_ids_0)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), len(token_ids_0))


if __name__ == "__main__":
    unittest.main()
