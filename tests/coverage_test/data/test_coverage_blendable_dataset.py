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
from unittest.mock import MagicMock

import numpy as np
import paddle


class TestBlendableDataset(unittest.TestCase):
    """Tests for BlendableDataset class."""

    def _make_mock_dataset(self, size=10, num_datasets=2):
        """Create a BlendableDataset-like object with manually set indices."""
        from paddleformers.data.blendable_dataset import BlendableDataset

        mock_datasets = []
        for i in range(num_datasets):
            ds = MagicMock()
            ds.desc = f"Dataset {i}"
            ds.__getitem__ = MagicMock(return_value={"text": f"sample_{i}"})
            mock_datasets.append(ds)

        # Create object without calling __init__ (which needs fast_dataindex)
        dataset = BlendableDataset.__new__(BlendableDataset)
        dataset.datasets = mock_datasets
        dataset.size = size
        dataset.desc = "test dataset"

        # Set up indices manually
        dataset.dataset_index = np.random.randint(0, num_datasets, size=size).astype(np.uint8)
        dataset.dataset_sample_index = np.random.randint(0, 100, size=size).astype(np.int64)

        return dataset

    def test_import(self):
        """Test that BlendableDataset can be imported."""
        from paddleformers.data.blendable_dataset import BlendableDataset

        self.assertIsNotNone(BlendableDataset)

    def test_is_paddle_dataset(self):
        """Test that BlendableDataset is a subclass of paddle.io.Dataset."""
        from paddleformers.data.blendable_dataset import BlendableDataset

        self.assertTrue(issubclass(BlendableDataset, paddle.io.Dataset))

    def test_len(self):
        """Test __len__ returns correct size."""
        dataset = self._make_mock_dataset(size=20)
        self.assertEqual(len(dataset), 20)

    def test_getitem(self):
        """Test __getitem__ returns correct structure."""
        dataset = self._make_mock_dataset(size=10)
        item = dataset[0]
        self.assertIn("dataset_idx", item)

    def test_getitem_valid_index(self):
        """Test __getitem__ with valid index."""
        dataset = self._make_mock_dataset(size=10)
        for i in range(10):
            item = dataset[i]
            self.assertIn("dataset_idx", item)

    def test_getitem_out_of_range(self):
        """Test __getitem__ with out-of-range index raises IndexError."""
        dataset = self._make_mock_dataset(size=10)
        with self.assertRaises(IndexError):
            dataset[10]


class TestPrintRank0(unittest.TestCase):
    """Tests for print_rank_0 function."""

    def test_import(self):
        """Test that print_rank_0 can be imported."""
        from paddleformers.data.blendable_dataset import print_rank_0

        self.assertTrue(callable(print_rank_0))


if __name__ == "__main__":
    unittest.main()
