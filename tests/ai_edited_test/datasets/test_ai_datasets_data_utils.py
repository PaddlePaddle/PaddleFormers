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

import unittest
from unittest.mock import MagicMock, patch

from paddleformers.datasets.data_utils import (
    calculate_matched_group,
    convert_to_input_ids,
    convert_to_tokens_for_pt,
    generate_greedy_packs_from_sequences,
    get_worker_sliced_iterator,
    print_debug_info,
    round_up_to_multiple_of_8,
)


class TestRoundUpToMultipleOf8(unittest.TestCase):
    """Test round_up_to_multiple_of_8 function."""

    def test_zero(self):
        """Test with zero."""
        self.assertEqual(round_up_to_multiple_of_8(0), 0)

    def test_already_multiple(self):
        """Test with values already multiples of 8."""
        for n in [8, 16, 24, 128, 256]:
            self.assertEqual(round_up_to_multiple_of_8(n), n)

    def test_round_up(self):
        """Test with values that need rounding up."""
        self.assertEqual(round_up_to_multiple_of_8(1), 8)
        self.assertEqual(round_up_to_multiple_of_8(5), 8)
        self.assertEqual(round_up_to_multiple_of_8(7), 8)
        self.assertEqual(round_up_to_multiple_of_8(9), 16)
        self.assertEqual(round_up_to_multiple_of_8(15), 16)
        self.assertEqual(round_up_to_multiple_of_8(17), 24)

    def test_large_numbers(self):
        """Test with large numbers."""
        self.assertEqual(round_up_to_multiple_of_8(1000), 1000)
        self.assertEqual(round_up_to_multiple_of_8(1001), 1008)
        self.assertEqual(round_up_to_multiple_of_8(1023), 1024)


class TestPrintDebugInfo(unittest.TestCase):
    """Test print_debug_info function."""

    def test_with_valid_data(self):
        """Test with valid tokenizer and data."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "decoded text"
        # Should not raise
        print_debug_info(mock_tokenizer, [1, 2, 3], "test_label")
        mock_tokenizer.decode.assert_called_once_with([1, 2, 3])

    def test_with_decode_error(self):
        """Test with tokenizer decode error."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.side_effect = TypeError("decode error")
        # Should not raise, just log the error
        print_debug_info(mock_tokenizer, [1, 2, 3], "test_label")


class TestConvertToTokensForPT(unittest.TestCase):
    """Test convert_to_tokens_for_pt function."""

    def test_basic_conversion(self):
        """Test basic PT token conversion."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["token1", "token2", "token3"]
        mock_tokenizer.convert_tokens_to_ids.return_value = [1, 2, 3]

        dial = [{"content": "hello world"}]
        tokens = convert_to_tokens_for_pt(dial, mock_tokenizer, max_src_len=1024)

        # Should join content and tokenize
        mock_tokenizer.tokenize.assert_called_once_with("hello world")
        self.assertEqual(tokens, ["token1", "token2", "token3"])

    def test_multi_content_join(self):
        """Test that multiple content entries are joined with newline."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["t1", "t2"]

        dial = [{"content": "line1"}, {"content": "line2"}, {"content": "line3"}]
        convert_to_tokens_for_pt(dial, mock_tokenizer, max_src_len=1024)

        mock_tokenizer.tokenize.assert_called_once_with("line1\nline2\nline3")

    def test_truncation_when_too_long(self):
        """Test truncation when tokens exceed max_src_len."""
        mock_tokenizer = MagicMock()
        # Return tokens longer than max_src_len
        tokens = [f"t{i}" for i in range(20)]
        mock_tokenizer.tokenize.return_value = tokens

        dial = [{"content": "long text"}]
        result = convert_to_tokens_for_pt(dial, mock_tokenizer, max_src_len=10)

        # Should be truncated (head + tail strategy)
        self.assertLessEqual(len(result), 10 + 10 // 2)


class TestConvertToInputIds(unittest.TestCase):
    """Test convert_to_input_ids function."""

    def test_base_format(self):
        """Test with 'base' data format."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["t1", "t2"]
        mock_tokenizer.convert_tokens_to_ids.return_value = [10, 20]

        dials = [[{"content": "hello"}]]
        input_ids, num_tokens = convert_to_input_ids(dials, mock_tokenizer, "base", 1024)

        self.assertEqual(input_ids, [[10, 20]])
        self.assertEqual(num_tokens, 2)

    def test_chat_format(self):
        """Test with 'chat' data format."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "some_template"
        mock_tokenizer.encode_chat_inputs.return_value = [([1, 2], [3, 4])]

        dials = [[{"role": "user", "content": "hello"}]]
        input_ids, num_tokens = convert_to_input_ids(dials, mock_tokenizer, "chat", 1024)

        self.assertEqual(len(input_ids), 1)
        self.assertGreater(num_tokens, 0)

    def test_invalid_format(self):
        """Test with invalid data format raises ValueError."""
        mock_tokenizer = MagicMock()
        dials = [[{"content": "hello"}]]
        with self.assertRaises(ValueError):
            convert_to_input_ids(dials, mock_tokenizer, "invalid", 1024)

    def test_multiple_dialogues(self):
        """Test with multiple dialogues."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["t1"]
        mock_tokenizer.convert_tokens_to_ids.return_value = [5]

        dials = [[{"content": "a"}], [{"content": "b"}]]
        input_ids, num_tokens = convert_to_input_ids(dials, mock_tokenizer, "base", 1024)

        self.assertEqual(len(input_ids), 2)
        self.assertEqual(num_tokens, 2)


class TestCalculateMatchedGroup(unittest.TestCase):
    """Test calculate_matched_group function."""

    def test_empty_sequences(self):
        """Test with empty sequences list."""
        sequences, ret_sequences = calculate_matched_group([], 1024)
        self.assertEqual(sequences, [])
        self.assertEqual(ret_sequences, [])

    def test_basic_packing(self):
        """Test basic bin packing."""
        # Each sequence is (token_ids, length, ...)
        sequences = [
            ("id1", 100, None),
            ("id2", 200, None),
            ("id3", 300, None),
        ]
        result, ret_sequences = calculate_matched_group(sequences, 500)
        # binpacking should pack sequences within packing_length
        self.assertIsInstance(result, list)

    def test_is_finished_true(self):
        """Test with is_finished=True (default)."""
        sequences = [
            ("id1", 100, None),
            ("id2", 200, None),
        ]
        result, ret_sequences = calculate_matched_group(sequences, 500, is_finished=True)
        self.assertEqual(ret_sequences, [])

    def test_is_finished_false(self):
        """Test with is_finished=False keeps last as ret_sequences."""
        sequences = [
            ("id1", 100, None),
            ("id2", 200, None),
            ("id3", 300, None),
        ]
        result, ret_sequences = calculate_matched_group(sequences, 500, is_finished=False)
        # When not finished, the last packed group is returned as ret_sequences
        self.assertIsInstance(ret_sequences, list)


class TestGenerateGreedyPacksFromSequences(unittest.TestCase):
    """Test generate_greedy_packs_from_sequences function."""

    def _make_sequence(self, length):
        """Create a mock sequence with given token length."""

        class MockSequence:
            def __init__(self, length):
                self.token_ids = list(range(length))

        return MockSequence(length)

    def test_single_sequence(self):
        """Test with a single sequence that fits."""
        sequences = [self._make_sequence(4)]
        result = generate_greedy_packs_from_sequences(8, sequences)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)

    def test_multiple_sequences_fit(self):
        """Test with multiple sequences that fit in one pack."""
        sequences = [self._make_sequence(2), self._make_sequence(3)]
        result = generate_greedy_packs_from_sequences(8, sequences)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

    def test_sequences_need_multiple_packs(self):
        """Test with sequences that need multiple packs."""
        sequences = [self._make_sequence(6), self._make_sequence(6)]
        result = generate_greedy_packs_from_sequences(8, sequences)
        self.assertEqual(len(result), 2)

    def test_empty_sequences(self):
        """Test with no sequences raises IndexError (known behavior)."""
        sequences = []
        with self.assertRaises(IndexError):
            generate_greedy_packs_from_sequences(8, sequences)

    def test_exact_fit(self):
        """Test with sequences that exactly fill a pack."""
        sequences = [self._make_sequence(4), self._make_sequence(4)]
        result = generate_greedy_packs_from_sequences(8, sequences)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)


class TestGetWorkerSlicedIterator(unittest.TestCase):
    """Test get_worker_sliced_iterator function."""

    def test_basic_iteration(self):
        """Test basic iteration without multi-worker."""
        dataset = [1, 2, 3, 4, 5]
        iterator = get_worker_sliced_iterator(dataset)
        # Should be an iterator
        result = []
        for i, val in enumerate(iterator):
            result.append(val)
            if i >= 4:
                break
        self.assertEqual(result, [1, 2, 3, 4, 5])

    @patch("paddleformers.datasets.data_utils.paddle.io.get_worker_info")
    def test_with_worker_info(self, mock_get_worker_info):
        """Test with worker info (multi-worker mode)."""
        mock_worker = MagicMock()
        mock_worker.id = 0
        mock_worker.num_workers = 2
        mock_get_worker_info.return_value = mock_worker

        dataset = [0, 1, 2, 3, 4, 5]
        iterator = get_worker_sliced_iterator(dataset)
        # Worker 0 should get elements 0, 2, 4
        result = []
        for i, val in enumerate(iterator):
            result.append(val)
            if i >= 2:
                break
        self.assertEqual(result, [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
