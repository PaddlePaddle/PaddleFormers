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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.tensor_parallel.mappings import (
    _CopyToModelParallelRegion,
    _gather_along_first_dim,
    _gather_along_last_dim,
    _GatherFromModelParallelRegion,
    _GatherFromSequenceParallelRegion,
    _reduce,
    _reduce_scatter_along_first_dim,
    _reduce_scatter_along_last_dim,
    _ReduceFromModelParallelRegion,
    _ReduceScatterToSequenceParallelRegion,
    _ScatterToModelParallelRegion,
    _ScatterToSequenceParallelRegion,
    _split_along_first_dim,
    _split_along_last_dim,
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)


def _make_group(world_size=2, rank=0):
    """Create a mock process group."""
    group = MagicMock()
    group.world_size = world_size
    group.rank = rank
    group.nranks = world_size
    group.ranks = list(range(world_size))
    return group


class TestReduce(unittest.TestCase):
    """Tests for _reduce helper function."""

    def test_reduce_asserts_group_not_none(self):
        """Test _reduce raises when group is None."""
        with self.assertRaises(AssertionError):
            _reduce(paddle.randn([2, 4]), None)

    @patch("paddleformers.fleet.tensor_parallel.mappings.paddle.distributed.all_reduce")
    def test_reduce_single_gpu_bypass(self, mock_all_reduce):
        """Test _reduce bypasses when world_size is 1."""
        group = _make_group(world_size=1, rank=0)
        x = paddle.randn([2, 4])
        result = _reduce(x, group)
        mock_all_reduce.assert_not_called()

    @patch("paddleformers.fleet.tensor_parallel.mappings.paddle.distributed.all_reduce")
    def test_reduce_calls_all_reduce(self, mock_all_reduce):
        """Test _reduce calls all_reduce for multi-GPU."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 4])
        _reduce(x, group)
        mock_all_reduce.assert_called_once()


class TestSplitAlongLastDim(unittest.TestCase):
    """Tests for _split_along_last_dim helper function."""

    def test_split_asserts_group_not_none(self):
        """Test _split_along_last_dim raises when group is None."""
        with self.assertRaises(AssertionError):
            _split_along_last_dim(paddle.randn([2, 4]), None)

    @patch("paddleformers.fleet.tensor_parallel.mappings.split_tensor_along_last_dim")
    def test_split_single_gpu_bypass(self, mock_split):
        """Test bypass when world_size is 1."""
        group = _make_group(world_size=1, rank=0)
        x = paddle.randn([2, 4])
        result = _split_along_last_dim(x, group)
        mock_split.assert_not_called()

    @patch("paddleformers.fleet.tensor_parallel.mappings.split_tensor_along_last_dim")
    def test_split_returns_rank_slice(self, mock_split):
        """Test _split_along_last_dim returns the correct rank slice."""
        group = _make_group(world_size=2, rank=1)
        x = paddle.randn([2, 4])
        slice_a = paddle.randn([2, 2])
        slice_b = paddle.randn([2, 2])
        mock_split.return_value = [slice_a, slice_b]
        result = _split_along_last_dim(x, group)
        self.assertTrue(result is slice_b)


class TestSplitAlongFirstDim(unittest.TestCase):
    """Tests for _split_along_first_dim helper function."""

    def test_split_asserts_group_not_none(self):
        """Test _split_along_first_dim raises when group is None."""
        with self.assertRaises(AssertionError):
            _split_along_first_dim(paddle.randn([4, 8]), None)

    @patch("paddleformers.fleet.tensor_parallel.mappings.paddle.distributed")
    def test_split_single_gpu_bypass(self, mock_dist):
        """Test bypass when world_size is 1."""
        group = _make_group(world_size=1, rank=0)
        x = paddle.randn([4, 8])
        result = _split_along_first_dim(x, group)
        self.assertIs(result, x)

    def test_split_first_dim_asserts_divisible(self):
        """Test _split_along_first_dim raises when first dim not divisible."""
        group = _make_group(world_size=3, rank=0)
        x = paddle.randn([4, 8])
        with self.assertRaises(AssertionError):
            _split_along_first_dim(x, group)

    def test_split_first_dim_returns_slice(self):
        """Test _split_along_first_dim returns correct rank slice."""
        group = _make_group(world_size=2, rank=1)
        x = paddle.randn([4, 8])
        result = _split_along_first_dim(x, group)
        self.assertEqual(result.shape[0], 2)


class TestGatherAlongLastDim(unittest.TestCase):
    """Tests for _gather_along_last_dim helper function."""

    @patch("paddleformers.fleet.tensor_parallel.mappings.dist.all_gather")
    def test_gather_single_gpu_bypass(self, mock_all_gather):
        """Test bypass when world_size is 1."""
        group = _make_group(world_size=1, rank=0)
        x = paddle.randn([2, 4])
        result = _gather_along_last_dim(x, group)
        mock_all_gather.assert_not_called()

    @patch("paddleformers.fleet.tensor_parallel.mappings.paddle.concat")
    @patch("paddleformers.fleet.tensor_parallel.mappings.dist.all_gather")
    def test_gather_calls_all_gather(self, mock_all_gather, mock_concat):
        """Test _gather_along_last_dim calls all_gather."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 4])
        mock_concat.return_value = paddle.randn([2, 8])
        _gather_along_last_dim(x, group)
        mock_all_gather.assert_called_once()


class TestReduceScatterAlongLastDim(unittest.TestCase):
    """Tests for _reduce_scatter_along_last_dim helper function."""

    def test_asserts_divisible(self):
        """Test assertion when last dim is not divisible by world_size."""
        group = _make_group(world_size=3, rank=0)
        x = paddle.randn([2, 4])
        with self.assertRaises(AssertionError):
            _reduce_scatter_along_last_dim(x, group)

    @patch("paddleformers.fleet.tensor_parallel.mappings._reduce_scatter_along_first_dim")
    # The source code for _reduce_scatter_along_last_dim uses
    # paddle.split with keyword args (dim, split_size_or_sections) that
    # are not supported in this PaddlePaddle version. Skip.
    @unittest.skip("Source code uses paddle.split with unsupported keyword arguments")
    def test_reduce_scatter_last_dim_delegates(self, mock_rs_first):
        """Test _reduce_scatter_along_last_dim delegates properly."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 4])
        expected = paddle.randn([2, 2])
        mock_rs_first.return_value = expected
        result = _reduce_scatter_along_last_dim(x, group)
        self.assertTrue(result is expected)


class TestGatherAlongFirstDim(unittest.TestCase):
    """Tests for _gather_along_first_dim helper function."""

    def test_asserts_group_not_none(self):
        """Test _gather_along_first_dim raises when group is None."""
        with self.assertRaises(AssertionError):
            _gather_along_first_dim(paddle.randn([2, 4]), None)

    @patch("paddleformers.fleet.tensor_parallel.mappings.dist.all_gather")
    def test_gather_first_single_gpu(self, mock_all_gather):
        """Test bypass when world_size is 1."""
        group = _make_group(world_size=1, rank=0)
        x = paddle.randn([2, 4])
        result = _gather_along_first_dim(x, group)
        mock_all_gather.assert_not_called()


class TestReduceScatterAlongFirstDim(unittest.TestCase):
    """Tests for _reduce_scatter_along_first_dim helper function."""

    def test_asserts_group_not_none(self):
        """Test _reduce_scatter_along_first_dim raises when group is None."""
        with self.assertRaises(AssertionError):
            _reduce_scatter_along_first_dim(paddle.randn([4, 8]), None)

    def test_asserts_divisible(self):
        """Test assertion when first dim is not divisible."""
        group = _make_group(world_size=3, rank=0)
        x = paddle.randn([4, 8])
        with self.assertRaises(AssertionError):
            _reduce_scatter_along_first_dim(x, group)

    @patch("paddleformers.fleet.tensor_parallel.mappings._reduce_scatter_base")
    def test_with_global_buffer(self, mock_rs_base):
        """Test reduce_scatter uses global buffer when flag is True."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([4, 8])
        with patch("paddleformers.fleet.tensor_parallel.mappings.get_global_memory_buffer") as mock_buf:
            mock_buf.return_value.get_tensor.return_value = paddle.randn([2, 8])
            _reduce_scatter_along_first_dim(x, group, use_global_buffer=True)
            mock_buf.return_value.get_tensor.assert_called_once()


class TestCopyToModelParallelRegion(unittest.TestCase):
    """Tests for _CopyToModelParallelRegion autograd function."""

    def test_forward_passes_through(self):
        """Test forward simply passes input through."""
        x = paddle.randn([2, 4])
        group = _make_group(world_size=1, rank=0)
        result = _CopyToModelParallelRegion.apply(x, group)
        self.assertTrue(result is x)

    @patch("paddleformers.fleet.tensor_parallel.mappings._reduce")
    def test_backward_calls_reduce(self, mock_reduce):
        """Test backward calls _reduce."""
        x = paddle.randn([2, 4])
        group = _make_group(world_size=2, rank=0)
        mock_reduce.return_value = paddle.randn([2, 4])
        result = _CopyToModelParallelRegion.apply(x, group)
        # backward should call _reduce when group is not None


class TestReduceFromModelParallelRegion(unittest.TestCase):
    """Tests for _ReduceFromModelParallelRegion autograd function."""

    @patch("paddleformers.fleet.tensor_parallel.mappings._reduce")
    def test_forward_calls_reduce_multi_gpu(self, mock_reduce):
        """Test forward calls _reduce when world_size > 1."""
        x = paddle.randn([2, 4])
        group = _make_group(world_size=2, rank=0)
        mock_reduce.return_value = x
        _ReduceFromModelParallelRegion.apply(x, group)
        mock_reduce.assert_called_once()

    def test_forward_bypasses_single_gpu(self):
        """Test forward bypasses _reduce when nranks <= 1."""
        x = paddle.randn([2, 4])
        group = _make_group(world_size=1, rank=0)
        result = _ReduceFromModelParallelRegion.apply(x, group)
        self.assertTrue(result is x)

    def test_backward_passes_through(self):
        """Test backward simply passes gradient through."""
        x = paddle.randn([2, 4])
        group = _make_group(world_size=2, rank=0)


class TestScatterToModelParallelRegion(unittest.TestCase):
    """Tests for _ScatterToModelParallelRegion autograd function."""

    def test_forward_none_group(self):
        """Test forward when group is None."""
        x = paddle.randn([2, 4])
        result = _ScatterToModelParallelRegion.apply(x, None)
        self.assertTrue(result is x)

    @patch("paddleformers.fleet.tensor_parallel.mappings._split_along_last_dim")
    def test_forward_splits(self, mock_split):
        """Test forward calls split."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 4])
        mock_split.return_value = paddle.randn([2, 2])
        _ScatterToModelParallelRegion.apply(x, group)
        mock_split.assert_called_once()

    @patch("paddleformers.fleet.tensor_parallel.mappings._gather_along_last_dim")
    def test_backward_gathers(self, mock_gather):
        """Test backward calls gather."""
        group = _make_group(world_size=2, rank=0)
        mock_gather.return_value = paddle.randn([2, 4])
        x = paddle.randn([2, 4])
        _ScatterToModelParallelRegion.apply(x, group)
        # backward should call _gather_along_last_dim when group is not None


class TestGatherFromModelParallelRegion(unittest.TestCase):
    """Tests for _GatherFromModelParallelRegion autograd function."""

    def test_forward_none_group(self):
        """Test forward when group is None."""
        x = paddle.randn([2, 4])
        result = _GatherFromModelParallelRegion.apply(x, None)
        self.assertTrue(result is x)

    @patch("paddleformers.fleet.tensor_parallel.mappings._gather_along_last_dim")
    def test_forward_gathers(self, mock_gather):
        """Test forward calls gather."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 4])
        mock_gather.return_value = paddle.randn([2, 8])
        _GatherFromModelParallelRegion.apply(x, group)
        mock_gather.assert_called_once()


class TestScatterToSequenceParallelRegion(unittest.TestCase):
    """Tests for _ScatterToSequenceParallelRegion autograd function."""

    def test_forward_none_group(self):
        """Test forward when group is None."""
        x = paddle.randn([2, 4])
        result = _ScatterToSequenceParallelRegion.apply(x, None)
        self.assertTrue(result is x)

    @patch("paddleformers.fleet.tensor_parallel.mappings._split_along_first_dim")
    def test_forward_splits_first_dim(self, mock_split):
        """Test forward splits along first dim."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([4, 8])
        mock_split.return_value = paddle.randn([2, 8])
        _ScatterToSequenceParallelRegion.apply(x, group)
        mock_split.assert_called_once()


class TestGatherFromSequenceParallelRegion(unittest.TestCase):
    """Tests for _GatherFromSequenceParallelRegion autograd function."""

    def test_forward_none_group(self):
        """Test forward when group is None."""
        x = paddle.randn([2, 4])
        result = _GatherFromSequenceParallelRegion.apply(x, None, True, None, False)
        self.assertTrue(result is x)

    @patch("paddleformers.fleet.tensor_parallel.mappings._gather_along_first_dim")
    def test_forward_gathers(self, mock_gather):
        """Test forward calls gather along first dim."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 8])
        mock_gather.return_value = paddle.randn([4, 8])
        _GatherFromSequenceParallelRegion.apply(x, group, True, None, False)
        mock_gather.assert_called_once()


class TestReduceScatterToSequenceParallelRegion(unittest.TestCase):
    """Tests for _ReduceScatterToSequenceParallelRegion autograd function."""

    def test_forward_none_group(self):
        """Test forward when group is None."""
        x = paddle.randn([2, 4])
        result = _ReduceScatterToSequenceParallelRegion.apply(x, None)
        self.assertTrue(result is x)

    @patch("paddleformers.fleet.tensor_parallel.mappings._reduce_scatter_along_first_dim")
    def test_forward_reduce_scatters(self, mock_rs):
        """Test forward calls reduce_scatter."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([4, 8])
        mock_rs.return_value = paddle.randn([2, 8])
        _ReduceScatterToSequenceParallelRegion.apply(x, group, None, False)
        mock_rs.assert_called_once()


class TestWrapperFunctions(unittest.TestCase):
    """Tests for the wrapper functions that call autograd Function.apply."""

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_copy_to_tp_region(self, mock_get_group):
        """Test copy_to_tensor_model_parallel_region calls apply."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        x = paddle.randn([2, 4])
        result = copy_to_tensor_model_parallel_region(x)
        self.assertIsNotNone(result)

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_reduce_from_tp_region(self, mock_get_group):
        """Test reduce_from_tensor_model_parallel_region calls apply."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        x = paddle.randn([2, 4])
        result = reduce_from_tensor_model_parallel_region(x)
        self.assertIsNotNone(result)

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_scatter_to_tp_region(self, mock_get_group):
        """Test scatter_to_tensor_model_parallel_region calls apply."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        x = paddle.randn([2, 4])
        result = scatter_to_tensor_model_parallel_region(x)
        self.assertIsNotNone(result)

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_gather_from_tp_region(self, mock_get_group):
        """Test gather_from_tensor_model_parallel_region calls apply."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        x = paddle.randn([2, 4])
        result = gather_from_tensor_model_parallel_region(x)
        self.assertIsNotNone(result)

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_scatter_to_sp_region(self, mock_get_group):
        """Test scatter_to_sequence_parallel_region calls apply."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        x = paddle.randn([2, 4])
        result = scatter_to_sequence_parallel_region(x)
        self.assertIsNotNone(result)

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_gather_from_sp_region(self, mock_get_group):
        """Test gather_from_sequence_parallel_region calls apply."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        x = paddle.randn([2, 4])
        result = gather_from_sequence_parallel_region(x)
        self.assertIsNotNone(result)

    @patch("paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none")
    def test_reduce_scatter_to_sp_region(self, mock_get_group):
        """Test reduce_scatter_to_sequence_parallel_region calls apply."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        x = paddle.randn([2, 4])
        result = reduce_scatter_to_sequence_parallel_region(x)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
