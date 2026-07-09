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

import os
import tempfile
import unittest

from paddleformers.cli.train.ernie_pretrain.src.tokenizers.tokenization_eb_v2 import (
    ErnieBotTokenizer,
)

# Path to a minimal test sentencepiece model
_TEST_MODEL_DIR = "/tmp/test_tokenizer_model"
_TEST_MODEL_FILE = os.path.join(_TEST_MODEL_DIR, "tokenizer.model")


def _model_available():
    return os.path.isfile(_TEST_MODEL_FILE)


@unittest.skipIf(not _model_available(), "Test sentencepiece model not available")
class TestErnieBotTokenizer(unittest.TestCase):
    """Tests for ErnieBotTokenizer class."""

    @classmethod
    def setUpClass(cls):
        """Set up tokenizer for testing."""
        cls.tokenizer = ErnieBotTokenizer(_TEST_MODEL_FILE)

    def test_vocab_size(self):
        """Test vocab_size property."""
        vocab_size = self.tokenizer.vocab_size
        self.assertIsInstance(vocab_size, int)
        self.assertGreater(vocab_size, 0)

    def test_space_token(self):
        """Test space_token property."""
        self.assertEqual(self.tokenizer.space_token, "<mask:1>")

    def test_gend_token(self):
        """Test gend_token property."""
        self.assertEqual(self.tokenizer.gend_token, "<mask:7>")

    def test_space_token_id(self):
        """Test space_token_id property."""
        token_id = self.tokenizer.space_token_id
        self.assertIsInstance(token_id, int)

    def test_gend_token_id(self):
        """Test gend_token_id property."""
        token_id = self.tokenizer.gend_token_id
        self.assertIsInstance(token_id, int)

    def test_tokenize(self):
        """Test tokenize method."""
        tokens = self.tokenizer.tokenize("hello world")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_encode_produces_input_ids(self):
        """Test that encoding text produces input_ids."""
        tokens = self.tokenizer.tokenize("hello")
        input_ids = [self.tokenizer._convert_token_to_id(t) for t in tokens]
        self.assertIsInstance(input_ids, list)
        self.assertGreater(len(input_ids), 0)
        for id in input_ids:
            self.assertIsInstance(id, int)

    def test_convert_tokens_to_string(self):
        """Test convert_tokens_to_string method."""
        tokens = self.tokenizer.tokenize("hello")
        result = self.tokenizer.convert_tokens_to_string(tokens)
        self.assertIsInstance(result, str)

    def test_get_vocab(self):
        """Test get_vocab method."""
        vocab = self.tokenizer.get_vocab()
        self.assertIsInstance(vocab, dict)
        self.assertGreater(len(vocab), 0)

    def test_model_input_names(self):
        """Test model_input_names."""
        self.assertIn("input_ids", self.tokenizer.model_input_names)
        self.assertIn("position_ids", self.tokenizer.model_input_names)
        self.assertIn("attention_mask", self.tokenizer.model_input_names)

    def test_padding_side(self):
        """Test padding_side is right."""
        self.assertEqual(self.tokenizer.padding_side, "right")

    def test_bos_token(self):
        """Test bos_token is set."""
        self.assertEqual(self.tokenizer.bos_token, "<s>")

    def test_eos_token(self):
        """Test eos_token is set."""
        self.assertEqual(self.tokenizer.eos_token, "</s>")

    def test_pad_token(self):
        """Test pad_token is set."""
        self.assertEqual(self.tokenizer.pad_token, "<pad>")

    def test_unk_token(self):
        """Test unk_token is set."""
        self.assertEqual(self.tokenizer.unk_token, "<unk>")

    def test_save_vocabulary(self):
        """Test save_vocabulary method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.tokenizer.save_vocabulary(tmpdir)
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
            for f in result:
                self.assertTrue(os.path.exists(f))

    def test_convert_token_to_id(self):
        """Test _convert_token_to_id method."""
        token_id = self.tokenizer._convert_token_to_id("<s>")
        self.assertIsInstance(token_id, int)

    def test_convert_id_to_token(self):
        """Test _convert_id_to_token method."""
        token = self.tokenizer._convert_id_to_token(1)
        self.assertIsInstance(token, str)


if __name__ == "__main__":
    unittest.main()
