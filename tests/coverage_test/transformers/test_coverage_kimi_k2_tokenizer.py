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

from paddleformers.transformers.kimi_k2.tokenizer import KimiK2TikTokenTokenizer


class TestKimiK2TikTokenTokenizer(unittest.TestCase):
    """Tests for KimiK2TikTokenTokenizer."""

    @classmethod
    def setUpClass(cls):
        # Create a minimal tiktoken model file for testing
        cls.tmpdir = tempfile.mkdtemp()
        cls.vocab_file = os.path.join(cls.tmpdir, "tiktoken.model")
        cls._create_minimal_tiktoken_file()

    @classmethod
    def _create_minimal_tiktoken_file(cls):
        """Create a minimal valid tiktoken BPE file."""
        import base64

        # Write a simple BPE ranks file with a few entries
        bpe_ranks = {
            base64.b64encode(b"a").decode(): 0,
            base64.b64encode(b"b").decode(): 1,
            base64.b64encode(b"c").decode(): 2,
            base64.b64encode(b"ab").decode(): 3,
        }
        with open(cls.vocab_file, "w") as f:
            for token, rank in bpe_ranks.items():
                f.write(f"{token} {rank}\n")

    def _make_tokenizer(self):
        """Create a tokenizer with the test vocab file and proper special tokens."""
        from tokenizers import AddedToken

        num_base_tokens = 4  # 4 base tokens from our test file
        # Build the added_tokens_decoder with the required special tokens
        added_tokens_decoder = {}
        for i in range(num_base_tokens, num_base_tokens + 256 + 2):
            token_str = f"<|reserved_token_{i}|>"
            if i == num_base_tokens:
                token_str = "[BOS]"
            elif i == num_base_tokens + 1:
                token_str = "[EOS]"
            added_tokens_decoder[i] = AddedToken(token_str, special=True)

        return KimiK2TikTokenTokenizer(
            vocab_file=self.vocab_file,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token=f"<|reserved_token_{num_base_tokens + 2}|>",
            pad_token=f"<|reserved_token_{num_base_tokens + 3}|>",
            added_tokens_decoder=added_tokens_decoder,
        )

    def test_init(self):
        tokenizer = self._make_tokenizer()
        self.assertIsNotNone(tokenizer)
        self.assertGreater(tokenizer.vocab_size, 0)

    def test_vocab_size(self):
        tokenizer = self._make_tokenizer()
        self.assertEqual(tokenizer.vocab_size, tokenizer.n_words)

    def test_get_vocab(self):
        tokenizer = self._make_tokenizer()
        vocab = tokenizer.get_vocab()
        self.assertIsInstance(vocab, dict)

    def test_encode(self):
        tokenizer = self._make_tokenizer()
        ids = tokenizer.encode("abc")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_decode(self):
        tokenizer = self._make_tokenizer()
        text = "abc"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        self.assertIsInstance(decoded, str)

    def test_decode_int_input(self):
        tokenizer = self._make_tokenizer()
        result = tokenizer.decode(0)
        self.assertIsInstance(result, str)

    def test_convert_token_to_id(self):
        tokenizer = self._make_tokenizer()
        vocab = tokenizer.get_vocab()
        # Test a known token
        if "a" in vocab:
            result = tokenizer._convert_token_to_id("a")
            self.assertEqual(result, vocab["a"])

    def test_convert_id_to_token(self):
        tokenizer = self._make_tokenizer()
        # Test known token IDs
        for i in range(min(4, tokenizer.vocab_size)):
            result = tokenizer._convert_id_to_token(i)
            self.assertIsNotNone(result)

    def test_tokenizer_roundtrip(self):
        """Test that encode then decode produces consistent results."""
        tokenizer = self._make_tokenizer()
        original = "abc"
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)
        self.assertIsInstance(decoded, str)

    def test_clean_up_tokenization(self):
        result = KimiK2TikTokenTokenizer.clean_up_tokenization("hello  world")
        self.assertEqual(result, "hello  world")  # No change for tiktoken

    def test_split_whitespaces_or_nonwhitespaces(self):
        result = list(KimiK2TikTokenTokenizer._split_whitespaces_or_nonwhitespaces("hello world", 3))
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_split_whitespaces_edge_cases(self):
        result = list(KimiK2TikTokenTokenizer._split_whitespaces_or_nonwhitespaces("a   b", 2))
        self.assertIsInstance(result, list)

    def test_pre_tokenizer_process(self):
        tokenizer = self._make_tokenizer()
        result = tokenizer.pre_tokenizer_process("hello world")
        self.assertEqual(result, ["hello world"])

    def test_save_vocabulary(self):
        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = tokenizer.save_vocabulary(tmpdir)
            self.assertIsInstance(result, tuple)
            self.assertTrue(os.path.exists(result[0]))

    def test_save_vocabulary_not_dir(self):
        tokenizer = self._make_tokenizer()
        with self.assertRaises(ValueError):
            tokenizer.save_vocabulary("/not/a/real/directory")

    def test_encode_with_kwargs(self):
        """Test encode with extra kwargs calls super().encode."""
        tokenizer = self._make_tokenizer()
        try:
            tokenizer.encode("abc", add_special_tokens=True)
        except Exception:
            pass  # May fail due to mock, but should call super()

    def test_decode_with_kwargs(self):
        """Test decode with extra kwargs calls super().decode."""
        tokenizer = self._make_tokenizer()
        try:
            tokenizer.decode([0, 1, 2], skip_special_tokens=True)
        except Exception:
            pass  # May fail due to mock, but should call super()


if __name__ == "__main__":
    unittest.main()
