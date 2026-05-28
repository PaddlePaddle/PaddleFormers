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

from paddleformers.data.tokenizer import BaseTokenizer, get_idx_from_word


class TestGetIdxFromWord(unittest.TestCase):
    """Tests for get_idx_from_word helper function."""

    def test_known_word(self):
        """Test that known word returns correct index."""
        word_to_idx = {"hello": 1, "world": 2, "<unk>": 0}
        result = get_idx_from_word("hello", word_to_idx, "<unk>")
        self.assertEqual(result, 1)

    def test_unknown_word(self):
        """Test that unknown word returns unk index."""
        word_to_idx = {"hello": 1, "world": 2, "<unk>": 0}
        result = get_idx_from_word("unknown", word_to_idx, "<unk>")
        self.assertEqual(result, 0)

    def test_unk_word_itself(self):
        """Test that the unk word itself returns its own index."""
        word_to_idx = {"hello": 1, "<unk>": 0}
        result = get_idx_from_word("<unk>", word_to_idx, "<unk>")
        self.assertEqual(result, 0)

    def test_empty_vocab(self):
        """Test with empty vocab raises KeyError on unk."""
        word_to_idx = {}
        with self.assertRaises(KeyError):
            get_idx_from_word("hello", word_to_idx, "<unk>")


class TestBaseTokenizer(unittest.TestCase):
    """Tests for BaseTokenizer class."""

    def test_import(self):
        """Test that BaseTokenizer can be imported."""
        self.assertIsNotNone(BaseTokenizer)

    def test_instantiation(self):
        """Test that BaseTokenizer can be instantiated with a vocab."""
        vocab = {"hello": 1, "world": 2, "<unk>": 0}
        tokenizer = BaseTokenizer(vocab)
        self.assertIsInstance(tokenizer, BaseTokenizer)

    def test_vocab_attribute(self):
        """Test that vocab attribute is set correctly."""
        vocab = {"hello": 1, "world": 2}
        tokenizer = BaseTokenizer(vocab)
        self.assertEqual(tokenizer.vocab, vocab)

    def test_get_tokenizer_returns_none(self):
        """Test that get_tokenizer returns None (no tokenizer attribute set)."""
        tokenizer = BaseTokenizer({"a": 0})
        # No tokenizer attribute is set by default
        with self.assertRaises(AttributeError):
            tokenizer.get_tokenizer()

    def test_cut_is_noop(self):
        """Test that cut method is a no-op (pass)."""
        tokenizer = BaseTokenizer({"a": 0})
        # cut should not raise
        result = tokenizer.cut("hello world")
        self.assertIsNone(result)

    def test_encode_is_noop(self):
        """Test that encode method is a no-op (pass)."""
        tokenizer = BaseTokenizer({"a": 0})
        # encode should not raise
        result = tokenizer.encode("hello world")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
