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
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle


class TestSplitTensorAlongLastDim(unittest.TestCase):
    """Tests for split_tensor_along_last_dim."""

    def test_split_2d_tensor(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_along_last_dim,
        )

        tensor = paddle.randn([4, 8])
        result = split_tensor_along_last_dim(tensor, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, [4, 4])
        self.assertEqual(result[1].shape, [4, 4])

    def test_split_3d_tensor(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_along_last_dim,
        )

        tensor = paddle.randn([2, 4, 12])
        result = split_tensor_along_last_dim(tensor, 3)
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertEqual(chunk.shape, [2, 4, 4])

    def test_split_1d_tensor(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_along_last_dim,
        )

        tensor = paddle.randn([10])
        result = split_tensor_along_last_dim(tensor, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, [5])
        self.assertEqual(result[1].shape, [5])

    def test_split_contiguous_chunks(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_along_last_dim,
        )

        tensor = paddle.randn([4, 8])
        result = split_tensor_along_last_dim(tensor, 2, contiguous_split_chunks=True)
        self.assertEqual(len(result), 2)
        for chunk in result:
            self.assertTrue(chunk.is_contiguous())

    def test_split_single_partition(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_along_last_dim,
        )

        tensor = paddle.randn([4, 8])
        result = split_tensor_along_last_dim(tensor, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, [4, 8])

    def test_split_not_divisible_raises(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_along_last_dim,
        )

        tensor = paddle.randn([4, 7])
        with self.assertRaises(AssertionError):
            split_tensor_along_last_dim(tensor, 2)

    def test_split_4d_tensor(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_along_last_dim,
        )

        tensor = paddle.randn([2, 3, 4, 8])
        result = split_tensor_along_last_dim(tensor, 4)
        self.assertEqual(len(result), 4)
        for chunk in result:
            self.assertEqual(chunk.shape, [2, 3, 4, 2])


class TestSplitTensorInto1DEqualChunks(unittest.TestCase):
    """Tests for split_tensor_into_1d_equal_chunks."""

    def test_split_with_mock_group(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_into_1d_equal_chunks,
        )

        mock_group = MagicMock()
        mock_group.world_size = 2
        mock_group.rank = 0

        tensor = paddle.arange(10, dtype=paddle.float32)

        with patch(
            "paddleformers.fleet.tensor_parallel.utils.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            result = split_tensor_into_1d_equal_chunks(tensor)

        self.assertEqual(result.shape[0], 5)

    def test_split_with_new_buffer(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_into_1d_equal_chunks,
        )

        mock_group = MagicMock()
        mock_group.world_size = 2
        mock_group.rank = 1

        tensor = paddle.arange(10, dtype=paddle.float32)

        with patch(
            "paddleformers.fleet.tensor_parallel.utils.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            result = split_tensor_into_1d_equal_chunks(tensor, new_buffer=True)

        self.assertEqual(result.shape[0], 5)

    def test_split_rank1(self):
        from paddleformers.fleet.tensor_parallel.utils import (
            split_tensor_into_1d_equal_chunks,
        )

        mock_group = MagicMock()
        mock_group.world_size = 2
        mock_group.rank = 1

        tensor = paddle.arange(10, dtype=paddle.float32)

        with patch(
            "paddleformers.fleet.tensor_parallel.utils.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            result = split_tensor_into_1d_equal_chunks(tensor)

        # Rank 1 should get indices [5, 10)
        expected = paddle.arange(10, dtype=paddle.float32)[5:10]
        np.testing.assert_allclose(result, expected)


class TestGatherSplit1DTensor(unittest.TestCase):
    """Tests for gather_split_1d_tensor."""

    def test_gather_asserts_1d(self):
        from paddleformers.fleet.tensor_parallel.utils import gather_split_1d_tensor

        tensor = paddle.randn([2, 3])
        with self.assertRaises(AssertionError):
            gather_split_1d_tensor(tensor)

    def test_gather_with_mock(self):
        from paddleformers.fleet.tensor_parallel.utils import gather_split_1d_tensor

        mock_group = MagicMock()
        mock_group.world_size = 2

        tensor = paddle.arange(5, dtype=paddle.float32)

        with patch(  # noqa: SIM117
            "paddleformers.fleet.tensor_parallel.utils.get_tensor_model_parallel_group_if_none",
            return_value=mock_group,
        ):
            with patch("paddle.distributed.stream.all_gather"):
                result = gather_split_1d_tensor(tensor)

        # Should return a tensor of size 10 (5 * 2)
        self.assertEqual(result.shape[0], 10)


class TestVocabUtility(unittest.TestCase):
    """Tests for VocabUtility."""

    def test_vocab_range_from_per_partition_vocab_size(self):
        from paddleformers.fleet.tensor_parallel.utils import VocabUtility

        # rank 0, world_size 2, per_partition=100
        start, end = VocabUtility.vocab_range_from_per_partition_vocab_size(100, 0, 2)
        self.assertEqual(start, 0)
        self.assertEqual(end, 100)

        # rank 1, world_size 2, per_partition=100
        start, end = VocabUtility.vocab_range_from_per_partition_vocab_size(100, 1, 2)
        self.assertEqual(start, 100)
        self.assertEqual(end, 200)

        # rank 0, world_size 1
        start, end = VocabUtility.vocab_range_from_per_partition_vocab_size(500, 0, 1)
        self.assertEqual(start, 0)
        self.assertEqual(end, 500)

    def test_vocab_range_from_global_vocab_size(self):
        from paddleformers.fleet.tensor_parallel.utils import VocabUtility

        # global_vocab=200, rank=0, world_size=2
        start, end = VocabUtility.vocab_range_from_global_vocab_size(200, 0, 2)
        self.assertEqual(start, 0)
        self.assertEqual(end, 100)

        # global_vocab=200, rank=1, world_size=2
        start, end = VocabUtility.vocab_range_from_global_vocab_size(200, 1, 2)
        self.assertEqual(start, 100)
        self.assertEqual(end, 200)

        # global_vocab=300, rank=0, world_size=3
        start, end = VocabUtility.vocab_range_from_global_vocab_size(300, 0, 3)
        self.assertEqual(start, 0)
        self.assertEqual(end, 100)

    def test_vocab_range_not_divisible_raises(self):
        from paddleformers.fleet.tensor_parallel.utils import VocabUtility

        with self.assertRaises(AssertionError):
            VocabUtility.vocab_range_from_global_vocab_size(7, 0, 2)
