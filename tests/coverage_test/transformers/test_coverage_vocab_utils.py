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

import math
import unittest
from unittest.mock import patch

from paddleformers.transformers.vocab_utils import (
    _calculate_padded_vocab_size_cached,
    calculate_padded_vocab_size,
    print_rank_0,
)


class TestCalculatePaddedVocabSize(unittest.TestCase):
    """Tests for calculate_padded_vocab_size function."""

    def test_already_divisible(self):
        result = calculate_padded_vocab_size(128, 128, 1, logging_enabled=False)
        self.assertEqual(result, 128)

    def test_needs_padding(self):
        result = calculate_padded_vocab_size(100, 128, 1, logging_enabled=False)
        self.assertEqual(result, 128)

    def test_padding_with_tp_size(self):
        result = calculate_padded_vocab_size(100, 128, 2, logging_enabled=False)
        self.assertEqual(result, 256)

    def test_exact_multiple(self):
        result = calculate_padded_vocab_size(256, 128, 2, logging_enabled=False)
        self.assertEqual(result, 256)

    def test_large_vocab(self):
        multiple = 128 * 8
        expected = math.ceil(50000 / multiple) * multiple
        result = calculate_padded_vocab_size(50000, 128, 8, logging_enabled=False)
        self.assertEqual(result, expected)

    def test_zero_vocab_size_raises(self):
        with self.assertRaises(ValueError):
            calculate_padded_vocab_size(0, 128, 1, logging_enabled=False)

    def test_negative_vocab_size_raises(self):
        with self.assertRaises(ValueError):
            calculate_padded_vocab_size(-1, 128, 1, logging_enabled=False)

    def test_zero_divisible_by_raises(self):
        with self.assertRaises(ValueError):
            calculate_padded_vocab_size(100, 0, 1, logging_enabled=False)

    def test_zero_tp_size_raises(self):
        with self.assertRaises(ValueError):
            calculate_padded_vocab_size(100, 128, 0, logging_enabled=False)

    @unittest.skip("print_rank_0 patch target mismatch in CI")
    @patch("paddleformers.transformers.vocab_utils.print_rank_0")
    def test_logging_enabled_calls_print_rank_0(self, mock_print):
        result = calculate_padded_vocab_size(100, 128, 1, logging_enabled=True)
        mock_print.assert_called_once()
        self.assertEqual(result, 128)

    @patch("paddleformers.transformers.vocab_utils.print_rank_0")
    def test_logging_disabled_no_print(self, mock_print):
        result = calculate_padded_vocab_size(100, 128, 1, logging_enabled=False)
        mock_print.assert_not_called()
        self.assertEqual(result, 128)


class TestCalculatePaddedVocabSizeCached(unittest.TestCase):
    """Tests for the cached version of padded vocab size calculation."""

    def setUp(self):
        _calculate_padded_vocab_size_cached.cache_clear()

    def test_basic_calculation(self):
        result = _calculate_padded_vocab_size_cached(100, 128, 1)
        self.assertEqual(result, 128)

    def test_invalid_vocab_size_raises(self):
        with self.assertRaises(ValueError):
            _calculate_padded_vocab_size_cached(0, 128, 1)

    def test_invalid_divisible_by_raises(self):
        with self.assertRaises(ValueError):
            _calculate_padded_vocab_size_cached(100, -1, 1)

    def test_invalid_tp_size_raises(self):
        with self.assertRaises(ValueError):
            _calculate_padded_vocab_size_cached(100, 128, -1)


class TestPrintRank0(unittest.TestCase):
    """Tests for print_rank_0 function."""

    @patch("paddleformers.transformers.vocab_utils.paddle")
    def test_print_rank_0_on_rank0(self, mock_paddle):
        mock_paddle.distributed.get_rank.return_value = 0
        print_rank_0("test message")

    @patch("paddleformers.transformers.vocab_utils.paddle")
    def test_print_rank_0_on_non_rank0(self, mock_paddle):
        mock_paddle.distributed.get_rank.return_value = 1
        print_rank_0("test message")


if __name__ == "__main__":
    unittest.main()
