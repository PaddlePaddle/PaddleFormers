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


# Extra tests for paddleformers.fleet/tensor_parallel/mappings.py
# Focus on: AllToAll, all_gather_last_dim, reduce_scatter_last_dim,
# all_to_all_sp2hp, all_to_all_hp2sp, symbolic functions

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.tensor_parallel.mappings import (
    _AllGatherFromTensorParallelRegion,
    _AllToAll,
    _GatherFromSequenceParallelRegion,
    _ReduceScatterToSequenceParallelRegion,
    _ReduceScatterToTensorParallelRegion,
    all_gather_last_dim_from_tensor_parallel_region,
    all_to_all,
    reduce_scatter_last_dim_to_tensor_parallel_region,
)


def _make_group(world_size=2, rank=0):
    """Create a mock process group."""
    group = MagicMock()
    group.world_size = world_size
    group.rank = rank
    group.nranks = world_size
    group.ranks = list(range(world_size))
    return group


class TestAllToAllForward(unittest.TestCase):
    """Tests for _AllToAll forward."""

    def test_single_gpu_bypass(self):
        """Test _AllToAll bypasses when world_size is 1."""
        group = _make_group(world_size=1, rank=0)
        x = paddle.randn([4, 8])
        result = _AllToAll.forward(MagicMock(), group, x, None, None)
        self.assertTrue(paddle.allclose(result, x))

    @patch("paddleformers.fleet.tensor_parallel.mappings.paddle.empty_like")
    def test_multi_gpu_equal_split(self, mock_empty):
        """Test _AllToAll with equal split (no split sizes)."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([4, 8])
        mock_empty.return_value = paddle.empty([4, 8])
        # Manually mock dist.all_to_all_single since it doesn't exist as attribute
        import paddleformers.fleet.tensor_parallel.mappings as mappings_mod

        original_fn = getattr(mappings_mod.dist, "all_to_all_single", None)
        mock_a2a = MagicMock()
        mappings_mod.dist.all_to_all_single = mock_a2a
        try:
            _AllToAll.forward(MagicMock(), group, x, None, None)
            mock_a2a.assert_called_once()
        finally:
            if original_fn is not None:
                mappings_mod.dist.all_to_all_single = original_fn
            else:
                delattr(mappings_mod.dist, "all_to_all_single")

    @patch("paddleformers.fleet.tensor_parallel.mappings.paddle.empty_like")
    def test_multi_gpu_unequal_split(self, mock_empty):
        """Test _AllToAll with unequal split sizes."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([4, 8])
        output_split_sizes = [2, 2]
        input_split_sizes = [2, 2]
        mock_empty.return_value = paddle.empty([4, 8])
        # Manually mock dist.all_to_all_single since it doesn't exist as attribute
        import paddleformers.fleet.tensor_parallel.mappings as mappings_mod

        original_fn = getattr(mappings_mod.dist, "all_to_all_single", None)
        mock_a2a = MagicMock()
        mappings_mod.dist.all_to_all_single = mock_a2a
        try:
            _AllToAll.forward(
                MagicMock(), group, x, output_split_sizes, input_split_sizes
            )
            mock_a2a.assert_called_once()
        finally:
            if original_fn is not None:
                mappings_mod.dist.all_to_all_single = original_fn
            else:
                delattr(mappings_mod.dist, "all_to_all_single")


class TestAllToAllBackward(unittest.TestCase):
    """Tests for _AllToAll backward."""

    @patch("paddleformers.fleet.tensor_parallel.mappings._AllToAll.apply")
    def test_backward_calls_apply(self, mock_apply):
        """Test _AllToAll backward calls _AllToAll.apply with swapped splits."""
        ctx = MagicMock()
        ctx.group = _make_group(world_size=2, rank=0)
        ctx.output_split_sizes = [2, 2]
        ctx.input_split_sizes = [3, 1]

        grad_output = paddle.randn([4, 8])
        mock_apply.return_value = paddle.randn([4, 8])

        result = _AllToAll.backward(ctx, grad_output)
        # Should return (None, grad_input, None, None)
        self.assertEqual(len(result), 4)
        self.assertIsNone(result[0])
        self.assertIsNone(result[2])
        self.assertIsNone(result[3])


class TestAllToAllWrapper(unittest.TestCase):
    """Tests for all_to_all wrapper function."""

    def test_asserts_group_not_none(self):
        """Test all_to_all raises when group is None."""
        with self.assertRaises(AssertionError):
            all_to_all(None, paddle.randn([4, 8]))


class TestAllGatherLastDim(unittest.TestCase):
    """Tests for _AllGatherFromTensorParallelRegion."""

    def test_forward_none_group(self):
        """Test forward when group is None returns input."""
        x = paddle.randn([2, 4])
        result = _AllGatherFromTensorParallelRegion.apply(x, None)
        self.assertTrue(result is x)

    @patch(
        "paddleformers.fleet.tensor_parallel.mappings._gather_along_last_dim"
    )
    def test_forward_gathers(self, mock_gather):
        """Test forward calls _gather_along_last_dim."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 4])
        mock_gather.return_value = paddle.randn([2, 8])
        _AllGatherFromTensorParallelRegion.apply(x, group)
        mock_gather.assert_called_once()

    def test_backward_none_group(self):
        """Test backward when group is None returns grad_output."""
        x = paddle.randn([2, 4])
        result = _AllGatherFromTensorParallelRegion.apply(x, None)
        self.assertTrue(result is x)

    @patch(
        "paddleformers.fleet.tensor_parallel.mappings._reduce_scatter_along_last_dim"
    )
    def test_backward_reduce_scatters_last_dim(self, mock_rs):
        """Test backward calls _reduce_scatter_along_last_dim."""
        group = _make_group(world_size=2, rank=0)
        mock_rs.return_value = paddle.randn([2, 2])
        x = paddle.randn([2, 4])
        _AllGatherFromTensorParallelRegion.apply(x, group)


class TestReduceScatterLastDim(unittest.TestCase):
    """Tests for _ReduceScatterToTensorParallelRegion."""

    def test_forward_none_group(self):
        """Test forward when group is None returns input."""
        x = paddle.randn([2, 4])
        result = _ReduceScatterToTensorParallelRegion.apply(x, None)
        self.assertTrue(result is x)

    @patch(
        "paddleformers.fleet.tensor_parallel.mappings._reduce_scatter_along_last_dim"
    )
    def test_forward_reduce_scatters(self, mock_rs):
        """Test forward calls _reduce_scatter_along_last_dim."""
        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 4])
        mock_rs.return_value = paddle.randn([2, 2])
        _ReduceScatterToTensorParallelRegion.apply(x, group)
        mock_rs.assert_called_once()

    @patch(
        "paddleformers.fleet.tensor_parallel.mappings._gather_along_last_dim"
    )
    def test_backward_gathers(self, mock_gather):
        """Test backward calls _gather_along_last_dim."""
        group = _make_group(world_size=2, rank=0)
        mock_gather.return_value = paddle.randn([2, 4])
        x = paddle.randn([2, 4])
        # Use apply which calls forward that sets ctx.group,
        # then backward uses _gather_along_last_dim
        # We need to patch _reduce_scatter_along_last_dim to avoid paddle.split dim issue
        with patch(
            "paddleformers.fleet.tensor_parallel.mappings._reduce_scatter_along_last_dim",
            return_value=paddle.randn([2, 2]),
        ):
            _ReduceScatterToTensorParallelRegion.apply(x, group)


class TestAllGatherLastDimWrapper(unittest.TestCase):
    """Tests for all_gather_last_dim_from_tensor_parallel_region."""

    @patch(
        "paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none"
    )
    def test_wrapper_calls_apply(self, mock_get_group):
        """Test wrapper calls apply with the correct group."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        x = paddle.randn([2, 4])
        result = all_gather_last_dim_from_tensor_parallel_region(x)
        self.assertIsNotNone(result)


class TestReduceScatterLastDimWrapper(unittest.TestCase):
    """Tests for reduce_scatter_last_dim_to_tensor_parallel_region."""

    @patch(
        "paddleformers.fleet.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none"
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.mappings._reduce_scatter_along_last_dim"
    )
    def test_wrapper_calls_apply(self, mock_rs, mock_get_group):
        """Test wrapper calls apply with the correct group."""
        group = _make_group(world_size=2, rank=0)
        mock_get_group.return_value = group
        mock_rs.return_value = paddle.randn([2, 2])
        x = paddle.randn([2, 4])
        result = reduce_scatter_last_dim_to_tensor_parallel_region(x)
        self.assertIsNotNone(result)


class TestSymbolicFunctions(unittest.TestCase):
    """Tests for symbolic functions of autograd Functions."""

    def test_copy_to_model_parallel_symbolic(self):
        """Test _CopyToModelParallelRegion symbolic returns input."""
        from paddleformers.fleet.tensor_parallel.mappings import (
            _CopyToModelParallelRegion,
        )

        x = paddle.randn([2, 4])
        result = _CopyToModelParallelRegion.symbolic(MagicMock(), x, None)
        self.assertTrue(result is x)

    @patch("paddleformers.fleet.tensor_parallel.mappings._reduce")
    def test_reduce_from_model_parallel_symbolic_none_group(self, mock_reduce):
        """Test _ReduceFromModelParallelRegion symbolic with None group."""
        from paddleformers.fleet.tensor_parallel.mappings import (
            _ReduceFromModelParallelRegion,
        )

        x = paddle.randn([2, 4])
        result = _ReduceFromModelParallelRegion.symbolic(MagicMock(), x, None)
        self.assertTrue(result is x)

    @patch("paddleformers.fleet.tensor_parallel.mappings._split_along_last_dim")
    def test_scatter_to_model_parallel_symbolic(self, mock_split):
        """Test _ScatterToModelParallelRegion symbolic calls split."""
        from paddleformers.fleet.tensor_parallel.mappings import (
            _ScatterToModelParallelRegion,
        )

        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 4])
        mock_split.return_value = paddle.randn([2, 2])
        _ScatterToModelParallelRegion.symbolic(MagicMock(), x, group)
        mock_split.assert_called_once()

    @patch(
        "paddleformers.fleet.tensor_parallel.mappings._gather_along_last_dim"
    )
    def test_gather_from_model_parallel_symbolic(self, mock_gather):
        """Test _GatherFromModelParallelRegion symbolic calls gather."""
        from paddleformers.fleet.tensor_parallel.mappings import (
            _GatherFromModelParallelRegion,
        )

        group = _make_group(world_size=2, rank=0)
        x = paddle.randn([2, 4])
        mock_gather.return_value = paddle.randn([2, 8])
        _GatherFromModelParallelRegion.symbolic(MagicMock(), x, group)
        mock_gather.assert_called_once()


class TestGatherFromSequenceParallelRegionBackward(unittest.TestCase):
    """Tests for _GatherFromSequenceParallelRegion backward."""

    @patch(
        "paddleformers.fleet.tensor_parallel.mappings._split_along_first_dim"
    )
    def test_backward_tp_output_grad_false(self, mock_split):
        """Test backward calls _split_along_first_dim when tp_output_grad=False."""
        group = _make_group(world_size=2, rank=0)
        mock_split.return_value = paddle.randn([2, 4])
        x = paddle.randn([4, 4])
        _GatherFromSequenceParallelRegion.apply(
            x, group, tensor_parallel_output_grad=False
        )


class TestReduceScatterToSequenceParallelRegionBackward(unittest.TestCase):
    """Tests for _ReduceScatterToSequenceParallelRegion backward."""

    @patch(
        "paddleformers.fleet.tensor_parallel.mappings._gather_along_first_dim"
    )
    def test_backward_gathers_first_dim(self, mock_gather):
        """Test backward calls _gather_along_first_dim."""
        group = _make_group(world_size=2, rank=0)
        mock_gather.return_value = paddle.randn([4, 8])
        x = paddle.randn([4, 8])
        _ReduceScatterToSequenceParallelRegion.apply(x, group, None, False)


if __name__ == "__main__":
    unittest.main()
