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
from pathlib import Path

from paddleformers.transformers import AutoTokenizer, IndexTokenizer

INDEX_TOKENIZER_DIR = Path(os.environ["INDEX_TOKENIZER_DIR"]) if "INDEX_TOKENIZER_DIR" in os.environ else None


class IndexTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if INDEX_TOKENIZER_DIR is None or not (INDEX_TOKENIZER_DIR / "tokenizer.model").is_file():
            raise unittest.SkipTest("Set INDEX_TOKENIZER_DIR to run local Index tokenizer integration tests")
        try:
            cls.tokenizer = IndexTokenizer.from_pretrained(
                INDEX_TOKENIZER_DIR, local_files_only=True, download_hub="huggingface"
            )
        except RuntimeError as error:
            raise unittest.SkipTest(f"Index tokenizer model is unavailable: {error}")

    def test_local_model_encode_save_load_and_auto_routing(self):
        input_ids = self.tokenizer.encode("你好，Index", add_special_tokens=True)

        self.assertFalse(self.tokenizer.add_bos_token)
        self.assertFalse(self.tokenizer.add_eos_token)
        self.assertEqual(input_ids, self.tokenizer.encode("你好，Index", add_special_tokens=False))
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("reserved_0"), 3)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("reserved_1"), 4)
        self.assertIn("reserved_0", self.tokenizer.added_tokens_encoder)
        self.assertIn("reserved_1", self.tokenizer.added_tokens_encoder)
        self.assertIn(
            "reserved_0", self.tokenizer.apply_chat_template([{"role": "user", "content": "你好"}], tokenize=False)
        )

        with tempfile.TemporaryDirectory() as directory:
            self.tokenizer.save_pretrained(directory)
            reloaded = IndexTokenizer.from_pretrained(directory, local_files_only=True, download_hub="huggingface")
            self.assertEqual(reloaded.encode("你好，Index"), input_ids)

        auto_tokenizer = AutoTokenizer.from_pretrained(
            INDEX_TOKENIZER_DIR, local_files_only=True, download_hub="huggingface"
        )
        self.assertIsInstance(auto_tokenizer, IndexTokenizer)
