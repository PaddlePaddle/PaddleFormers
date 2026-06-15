# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on applicable law or agreed to in writing, software
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
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.tensor_parallel.mappings import (
    _AllGatherFromTensorParallelRegion,
    _AllToAll,
    _gather_along_first_dim,
    _gather_along_last_dim,
    _reduce,
    _reduce_scatter_along_first_dim,
    _ReduceScatterToTensorParallelRegion,
)


def _tensors_equal(a, b):
    """Helper to compare paddle tensors for equality."""
    return bool(paddle.equal(a, b).numpy().all())


class TestReduceFunction(unittest.TestCase):
    """Tests for _reduce helper function."""

    def test_returns_input_when_world_size_1(self):
        """Should return input when world_size is 1."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        x = paddle.randn([4, 8])
        result = _reduce(x, mock_group)
        self.assertTrue(_tensors_equal(result, x))

    def test_raises_when_group_is_none(self):
        """Should raise AssertionError when group is None."""
        with self.assertRaises(AssertionError):
            _reduce(paddle.randn([4, 8]), None)


class TestGatherAlongFirstDimSingleGPU(unittest.TestCase):
    """Tests for _gather_along_first_dim with single GPU."""

    def test_returns_input_when_world_size_1(self):
        """Should return input when world_size is 1."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        x = paddle.randn([4, 8])
        result = _gather_along_first_dim(x, mock_group)
        self.assertTrue(_tensors_equal(result, x))

    def test_raises_when_group_is_none(self):
        """Should raise AssertionError when group is None."""
        with self.assertRaises(AssertionError):
            _gather_along_first_dim(paddle.randn([4, 8]), None)


class TestGatherAlongLastDimSingleGPU(unittest.TestCase):
    """Tests for _gather_along_last_dim with single GPU."""

    def test_returns_input_when_world_size_1(self):
        """Should return input when world_size is 1."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        x = paddle.randn([4, 8])
        result = _gather_along_last_dim(x, mock_group)
        self.assertTrue(_tensors_equal(result, x))


class TestReduceScatterAlongFirstDimSingleGPU(unittest.TestCase):
    """Tests for _reduce_scatter_along_first_dim with single GPU."""

    def test_returns_input_when_world_size_1(self):
        """Should return input when world_size is 1."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        x = paddle.randn([4, 8])
        result = _reduce_scatter_along_first_dim(x, mock_group)
        self.assertTrue(_tensors_equal(result, x))


class TestReduceScatterAlongLastDim(unittest.TestCase):
    """Tests for _reduce_scatter_along_last_dim shape calculation."""

    def test_shape_divisibility_check(self):
        """Last dim should be divisible by world_size for reduce_scatter."""
        x = paddle.randn([4, 8])
        target_shape = list(x.shape)
        self.assertEqual(target_shape[-1] % 2, 0)
        expected_last_dim = target_shape[-1] // 2
        self.assertEqual(expected_last_dim, 4)


class TestAllGatherFromTensorParallelRegion(unittest.TestCase):
    """Tests for _AllGatherFromTensorParallelRegion."""

    def test_forward_with_none_group(self):
        """Forward with None group should return input."""
        ctx = MagicMock()
        x = paddle.randn([4, 8])
        result = _AllGatherFromTensorParallelRegion.forward(ctx, x, None)
        self.assertTrue(_tensors_equal(result, x))


class TestReduceScatterToTensorParallelRegion(unittest.TestCase):
    """Tests for _ReduceScatterToTensorParallelRegion."""

    def test_forward_with_none_group(self):
        """Forward with None group should return input."""
        ctx = MagicMock()
        x = paddle.randn([4, 8])
        result = _ReduceScatterToTensorParallelRegion.forward(ctx, x, None)
        self.assertTrue(_tensors_equal(result, x))


class TestAllToAllSingleGPU(unittest.TestCase):
    """Tests for _AllToAll with single GPU."""

    def test_returns_input_when_world_size_1(self):
        """Should return input when world_size is 1."""
        ctx = MagicMock()
        mock_group = MagicMock()
        mock_group.world_size = 1
        x = paddle.randn([4, 8])
        result = _AllToAll.forward(ctx, mock_group, x, None, None)
        self.assertTrue(_tensors_equal(result, x))


if __name__ == "__main__":
    unittest.main()
