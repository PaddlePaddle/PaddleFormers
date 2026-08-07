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
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.tensor_parallel.utils import (
    VocabUtility,
    gather_split_1d_tensor,
    split_tensor_along_last_dim,
    split_tensor_into_1d_equal_chunks,
)


class TestSplitTensorAlongLastDim(unittest.TestCase):
    """Tests for split_tensor_along_last_dim."""

    def test_split_2_partitions(self):
        """Test splitting into 2 equal partitions."""
        tensor = paddle.randn([2, 8])
        result = split_tensor_along_last_dim(tensor, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, [2, 4])
        self.assertEqual(result[1].shape, [2, 4])

    def test_split_4_partitions(self):
        """Test splitting into 4 equal partitions."""
        tensor = paddle.randn([3, 16])
        result = split_tensor_along_last_dim(tensor, 4)
        self.assertEqual(len(result), 4)
        for r in result:
            self.assertEqual(r.shape, [3, 4])

    def test_contiguous_split_chunks(self):
        """Test contiguous_split_chunks flag."""
        tensor = paddle.randn([2, 8])
        result = split_tensor_along_last_dim(
            tensor, 2, contiguous_split_chunks=True
        )
        self.assertEqual(len(result), 2)
        for r in result:
            self.assertTrue(r.is_contiguous())

    def test_1d_tensor(self):
        """Test splitting 1D tensor."""
        tensor = paddle.randn([8])
        result = split_tensor_along_last_dim(tensor, 4)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].shape, [2])


class TestSplitTensorInto1DEqualChunks(unittest.TestCase):
    """Tests for split_tensor_into_1d_equal_chunks."""

    @patch(
        "paddleformers.fleet.tensor_parallel.utils.get_tensor_model_parallel_group_if_none"
    )
    def test_split_with_new_buffer(self, mock_get_group):
        """Test splitting with new_buffer=True."""
        group = MagicMock()
        group.world_size = 2
        group.rank = 0
        mock_get_group.return_value = group

        tensor = paddle.randn([4, 4])
        result = split_tensor_into_1d_equal_chunks(tensor, new_buffer=True)
        self.assertEqual(result.shape, [8])

    @patch(
        "paddleformers.fleet.tensor_parallel.utils.get_tensor_model_parallel_group_if_none"
    )
    def test_split_view(self, mock_get_group):
        """Test splitting returning a view."""
        group = MagicMock()
        group.world_size = 2
        group.rank = 1
        mock_get_group.return_value = group

        tensor = paddle.randn([4, 4])
        result = split_tensor_into_1d_equal_chunks(tensor, new_buffer=False)
        self.assertEqual(result.shape, [8])


class TestGatherSplit1DTensor(unittest.TestCase):
    """Tests for gather_split_1d_tensor."""

    def test_asserts_1d(self):
        """Test assertion that input must be 1D."""
        tensor = paddle.randn([2, 4])
        with self.assertRaises(AssertionError):
            gather_split_1d_tensor(tensor)

    @patch(
        "paddleformers.fleet.tensor_parallel.utils.get_tensor_model_parallel_group_if_none"
    )
    @patch("paddleformers.fleet.tensor_parallel.utils.dist.stream.all_gather")
    def test_gather_returns_correct_size(self, mock_all_gather, mock_get_group):
        """Test gathered tensor has correct total size."""
        group = MagicMock()
        group.world_size = 2
        mock_get_group.return_value = group
        mock_all_gather.return_value = None

        tensor = paddle.randn([4])
        result = gather_split_1d_tensor(tensor)
        self.assertEqual(result.shape, [8])


class TestVocabUtilityVocabRangeFromPerPartition(unittest.TestCase):
    """Tests for VocabUtility.vocab_range_from_per_partition_vocab_size."""

    def test_rank_0(self):
        """Test vocab range for rank 0."""
        result = VocabUtility.vocab_range_from_per_partition_vocab_size(
            100, 0, 2
        )
        self.assertEqual(result, (0, 100))

    def test_rank_1(self):
        """Test vocab range for rank 1."""
        result = VocabUtility.vocab_range_from_per_partition_vocab_size(
            100, 1, 2
        )
        self.assertEqual(result, (100, 200))

    def test_rank_2(self):
        """Test vocab range for rank 2."""
        result = VocabUtility.vocab_range_from_per_partition_vocab_size(
            50, 2, 4
        )
        self.assertEqual(result, (100, 150))

    def test_single_rank(self):
        """Test vocab range with single rank."""
        result = VocabUtility.vocab_range_from_per_partition_vocab_size(
            1000, 0, 1
        )
        self.assertEqual(result, (0, 1000))


class TestVocabUtilityVocabRangeFromGlobal(unittest.TestCase):
    """Tests for VocabUtility.vocab_range_from_global_vocab_size."""

    def test_even_division(self):
        """Test vocab range with evenly divisible global vocab size."""
        result = VocabUtility.vocab_range_from_global_vocab_size(100, 0, 2)
        self.assertEqual(result, (0, 50))

    def test_even_division_rank_1(self):
        """Test vocab range rank 1 with even division."""
        result = VocabUtility.vocab_range_from_global_vocab_size(100, 1, 2)
        self.assertEqual(result, (50, 100))

    def test_larger_world_size(self):
        """Test vocab range with larger world size."""
        result = VocabUtility.vocab_range_from_global_vocab_size(1000, 3, 8)
        per_partition = 1000 // 8
        self.assertEqual(result, (3 * per_partition, 4 * per_partition))

    def test_single_rank_global(self):
        """Test vocab range from global with single rank."""
        result = VocabUtility.vocab_range_from_global_vocab_size(1000, 0, 1)
        self.assertEqual(result, (0, 1000))


if __name__ == "__main__":
    unittest.main()
