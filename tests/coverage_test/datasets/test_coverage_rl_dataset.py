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
from unittest.mock import MagicMock, patch

import numpy as np

from paddleformers.datasets.rlhf_datasets.rl_dataset import (
    RLHFDataset,
    collate_fn,
    left_padding,
    padding_batch_data,
)


class TestLeftPadding(unittest.TestCase):
    """Tests for left_padding function."""

    def test_basic_padding(self):
        sequences = [np.array([1, 2, 3]), np.array([4, 5])]
        result = left_padding(sequences, padding_value=0)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 3)  # max length
        np.testing.assert_array_equal(result[0], [1, 2, 3])
        np.testing.assert_array_equal(result[1], [0, 4, 5])

    def test_with_max_length(self):
        sequences = [np.array([1, 2])]
        result = left_padding(sequences, padding_value=0, max_length=5)
        self.assertEqual(result.shape[1], 5)

    def test_single_sequence(self):
        sequences = [np.array([1, 2, 3, 4, 5])]
        result = left_padding(sequences, padding_value=0)
        self.assertEqual(result.shape, (1, 5))


class TestPaddingBatchData(unittest.TestCase):
    """Tests for padding_batch_data function."""

    def test_basic_without_label(self):
        samples = [
            {"input_ids": np.array([1, 2, 3])},
            {"input_ids": np.array([4, 5])},
        ]
        result = padding_batch_data(samples, pad_token_id=0, requires_label=False, max_prompt_len=5)
        self.assertIn("input_ids", result)
        self.assertIn("raw_prompt_len", result)
        self.assertNotIn("label_ids", result)

    def test_with_label(self):
        samples = [
            {"input_ids": np.array([1, 2, 3]), "label_ids": np.array([4, 5, 6])},
            {"input_ids": np.array([7, 8]), "label_ids": np.array([9, 10])},
        ]
        result = padding_batch_data(samples, pad_token_id=0, requires_label=True, max_prompt_len=5)
        self.assertIn("input_ids", result)
        self.assertIn("raw_prompt_len", result)
        self.assertIn("label_ids", result)
        self.assertIn("raw_label_ids_len", result)


class TestCollateFn(unittest.TestCase):
    """Tests for collate_fn function."""

    def test_basic_collate(self):
        samples = [
            {"input_ids": np.array([1, 2, 3])},
            {"input_ids": np.array([4, 5])},
        ]
        result = collate_fn(samples, pad_token_id=0, requires_label=False, max_prompt_len=5)
        self.assertIn("input_ids", result)
        self.assertIn("raw_prompt_len", result)

    def test_collate_with_labels(self):
        samples = [
            {"input_ids": np.array([1, 2, 3]), "label_ids": np.array([4, 5, 6])},
            {"input_ids": np.array([7, 8]), "label_ids": np.array([9, 10])},
        ]
        result = collate_fn(samples, pad_token_id=0, requires_label=True, max_prompt_len=5)
        self.assertIn("label_ids", result)


@unittest.skip("load_dataset from HuggingFace datasets cannot be reliably patched in CI")
class TestRLHFDataset(unittest.TestCase):
    """Tests for RLHFDataset."""

    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.os.path.exists", return_value=True)
    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.load_dataset")
    def test_init(self, mock_load_dataset, mock_exists):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_load_dataset.return_value = mock_dataset

        tokenizer = MagicMock()
        dataset = RLHFDataset(
            dataset_name_or_path="/tmp/test_data.json",
            tokenizer=tokenizer,
        )
        self.assertEqual(len(dataset), 10)

    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.os.path.exists", return_value=True)
    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.load_dataset")
    def test_getitem(self, mock_load_dataset, mock_exists):
        mock_rawdata = MagicMock()
        mock_rawdata.__len__ = MagicMock(return_value=1)
        mock_rawdata.__getitem__ = MagicMock(return_value={"src": "hello"})
        mock_load_dataset.return_value = mock_rawdata

        tokenizer = MagicMock()
        tokenizer.model_max_length = 1024
        tokenizer.return_value = {"input_ids": np.array([[1, 2, 3]])}

        dataset = RLHFDataset(
            dataset_name_or_path="/tmp/test_data.json",
            tokenizer=tokenizer,
        )
        item = dataset[0]
        self.assertIn("input_ids", item)

    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.os.path.exists", return_value=True)
    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.load_dataset")
    def test_getitem_cached(self, mock_load_dataset, mock_exists):
        """Test that repeated access returns cached data."""
        mock_rawdata = MagicMock()
        mock_rawdata.__len__ = MagicMock(return_value=1)
        mock_rawdata.__getitem__ = MagicMock(return_value={"src": "hello"})
        mock_load_dataset.return_value = mock_rawdata

        tokenizer = MagicMock()
        tokenizer.model_max_length = 1024
        tokenizer.return_value = {"input_ids": np.array([[1, 2, 3]])}

        dataset = RLHFDataset(
            dataset_name_or_path="/tmp/test_data.json",
            tokenizer=tokenizer,
        )
        dataset[0]
        dataset[0]
        # Second access should use cached data, not call tokenize again
        self.assertEqual(tokenizer.call_count, 1)

    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.os.path.exists", return_value=True)
    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.load_dataset")
    def test_getitem_with_labels(self, mock_load_dataset, mock_exists):
        mock_rawdata = MagicMock()
        mock_rawdata.__len__ = MagicMock(return_value=1)
        mock_rawdata.__getitem__ = MagicMock(return_value={"src": "hello", "response": "world"})
        mock_load_dataset.return_value = mock_rawdata

        tokenizer = MagicMock()
        tokenizer.model_max_length = 1024
        tokenizer.return_value = {"input_ids": np.array([[1, 2, 3]])}

        dataset = RLHFDataset(
            dataset_name_or_path="/tmp/test_data.json",
            tokenizer=tokenizer,
            requires_label=True,
            response_key="response",
        )
        item = dataset[0]
        self.assertIn("label_ids", item)

    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.os.path.exists", return_value=True)
    @patch("paddleformers.datasets.rlhf_datasets.rl_dataset.load_dataset")
    def test_getitem_with_chat_template(self, mock_load_dataset, mock_exists):
        mock_rawdata = MagicMock()
        mock_rawdata.__len__ = MagicMock(return_value=1)
        mock_rawdata.__getitem__ = MagicMock(return_value={"src": [{"role": "user", "content": "hello"}]})
        mock_load_dataset.return_value = mock_rawdata

        tokenizer = MagicMock()
        tokenizer.model_max_length = 1024
        tokenizer.chat_template = "some_template"
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.return_value = {"input_ids": np.array([[1, 2, 3]])}

        dataset = RLHFDataset(
            dataset_name_or_path="/tmp/test_data.json",
            tokenizer=tokenizer,
            apply_chat_template=True,
        )
        item = dataset[0]
        self.assertIn("input_ids", item)


if __name__ == "__main__":
    unittest.main()
