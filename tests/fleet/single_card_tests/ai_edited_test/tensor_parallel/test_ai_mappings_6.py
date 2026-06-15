# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
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

import paddle

from paddleformers.fleet.tensor_parallel.mappings import (
    _GatherFromSequenceParallelRegion,
    _ScatterToSequenceParallelRegion,
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)


def _tensors_equal(a, b):
    """Helper to compare paddle tensors for equality."""
    return bool(paddle.equal(a, b).numpy().all())


class TestHelperFunctionsWithSingleGPU(unittest.TestCase):
    """Tests for high-level helper functions with single GPU."""

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_copy_to_tensor_model_parallel_region(self, mock_get_group):
        """copy_to_tensor_model_parallel_region should work with single GPU."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        mock_get_group.return_value = mock_group
        x = paddle.randn([4, 8])
        result = copy_to_tensor_model_parallel_region(x)
        self.assertTrue(_tensors_equal(result, x))

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_reduce_from_tensor_model_parallel_region_single(self, mock_get_group):
        """reduce_from_tensor_model_parallel_region should work with single GPU."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_get_group.return_value = mock_group
        x = paddle.randn([4, 8])
        result = reduce_from_tensor_model_parallel_region(x)
        self.assertTrue(_tensors_equal(result, x))

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_scatter_to_tensor_model_parallel_region_single(self, mock_get_group):
        """scatter_to_tensor_model_parallel_region should work with single GPU."""
        mock_group = MagicMock()
        mock_group.ranks = [0]
        mock_group.world_size = 1
        mock_get_group.return_value = mock_group
        x = paddle.randn([4, 8])
        result = scatter_to_tensor_model_parallel_region(x)
        self.assertTrue(_tensors_equal(result, x))

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_gather_from_tensor_model_parallel_region_single(self, mock_get_group):
        """gather_from_tensor_model_parallel_region should work with single GPU."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        mock_get_group.return_value = mock_group
        x = paddle.randn([4, 8])
        result = gather_from_tensor_model_parallel_region(x)
        self.assertTrue(_tensors_equal(result, x))

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_scatter_to_sequence_parallel_region_single(self, mock_get_group):
        """scatter_to_sequence_parallel_region should work with single GPU."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        mock_get_group.return_value = mock_group
        x = paddle.randn([4, 8])
        result = scatter_to_sequence_parallel_region(x)
        self.assertTrue(_tensors_equal(result, x))

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_gather_from_sequence_parallel_region_single(self, mock_get_group):
        """gather_from_sequence_parallel_region should work with single GPU."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        mock_get_group.return_value = mock_group
        x = paddle.randn([4, 8])
        result = gather_from_sequence_parallel_region(x)
        self.assertTrue(_tensors_equal(result, x))


class TestScatterToSequenceParallelRegionBackward(unittest.TestCase):
    """Tests for _ScatterToSequenceParallelRegion backward."""

    def test_backward_with_none_group(self):
        """Backward with None group should return grad_output."""
        ctx = MagicMock()
        ctx.group = None
        grad = paddle.randn([4, 8])
        result = _ScatterToSequenceParallelRegion.backward(ctx, grad)
        self.assertTrue(_tensors_equal(result, grad))


class TestGatherFromSequenceParallelRegionBackward(unittest.TestCase):
    """Tests for _GatherFromSequenceParallelRegion backward."""

    def test_backward_with_none_group(self):
        """Backward with None group should return grad_output."""
        ctx = MagicMock()
        ctx.group = None
        grad = paddle.randn([4, 8])
        result = _GatherFromSequenceParallelRegion.backward(ctx, grad)
        self.assertTrue(_tensors_equal(result, grad))


if __name__ == "__main__":
    unittest.main()
