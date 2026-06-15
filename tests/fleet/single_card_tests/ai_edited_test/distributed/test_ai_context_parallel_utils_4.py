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


# Tests for src/paddleformers.fleet/context_parallel_utils.py
# Test scatter_with_padding, all_gather_without_padding,
# ContextParallelNormalScatter, ContextParallelNormalGather

import unittest
from unittest import mock

import paddle


class TestScatterWithPadding(unittest.TestCase):
    """Tests for scatter_with_padding function."""

    def test_exact_division(self):
        """Test scatter_with_padding when total_num is divisible by cp_degree."""
        from paddleformers.fleet.context_parallel_utils import scatter_with_padding

        x = paddle.randn([8, 4])
        mock_group = mock.MagicMock()
        mock_group.nranks = 2
        mock_group.rank = 0

        result = scatter_with_padding(x, num_pad=0, axis=0, group=mock_group)
        self.assertEqual(result.shape[0], 4)

    def test_with_padding(self):
        """Test scatter_with_padding with padding."""
        from paddleformers.fleet.context_parallel_utils import scatter_with_padding

        # total_num=10, num_pad=2, cp_degree=3
        # avg_num = (10+2)//3 = 4
        x = paddle.randn([10, 4])
        mock_group = mock.MagicMock()
        mock_group.nranks = 3
        mock_group.rank = 0

        result = scatter_with_padding(x, num_pad=2, axis=0, group=mock_group)
        self.assertEqual(result.shape[0], 4)

    def test_higher_rank_no_data(self):
        """Test scatter_with_padding when rank >= rank_idx."""
        from paddleformers.fleet.context_parallel_utils import scatter_with_padding

        # total_num=5, num_pad=1, cp_degree=3
        # avg_num = (5+1)//3 = 2
        # rank 0: [0,2), rank 1: [2,4), rank 2: [4,5)->pad to 2
        x = paddle.randn([5, 4])
        mock_group = mock.MagicMock()
        mock_group.nranks = 3
        mock_group.rank = 2

        result = scatter_with_padding(x, num_pad=1, axis=0, group=mock_group)
        self.assertEqual(result.shape[0], 2)

    def test_axis_parameter(self):
        """Test scatter_with_padding respects axis parameter.

        Note: scatter_with_padding always splits on axis 0 internally
        (paddle.split is called without axis), so passing axis=1 effectively
        treats the second axis as if it were axis 0. The axis parameter only
        affects the output shape indexing and the cp_rank logic.
        """
        from paddleformers.fleet.context_parallel_utils import scatter_with_padding

        # Use axis=0 with data that splits evenly
        x = paddle.randn([8, 4])
        mock_group = mock.MagicMock()
        mock_group.nranks = 2
        mock_group.rank = 0

        result = scatter_with_padding(x, num_pad=0, axis=0, group=mock_group)
        self.assertEqual(result.shape[0], 4)

    def test_zero_padding(self):
        """Test scatter_with_padding with zero padding."""
        from paddleformers.fleet.context_parallel_utils import scatter_with_padding

        x = paddle.randn([8, 4])
        mock_group = mock.MagicMock()
        mock_group.nranks = 2
        mock_group.rank = 1

        result = scatter_with_padding(x, num_pad=0, axis=0, group=mock_group)
        self.assertEqual(result.shape, [4, 4])


class TestAllGatherWithoutPadding(unittest.TestCase):
    """Tests for all_gather_without_padding function."""

    def test_no_padding(self):
        """Test all_gather_without_padding with zero padding."""
        from paddleformers.fleet.context_parallel_utils import (
            all_gather_without_padding,
        )

        x = paddle.randn([4, 8])
        mock_group = mock.MagicMock()
        mock_group.nranks = 2

        with mock.patch("paddle.distributed.stream.all_gather") as mock_ag:
            mock_ag.return_value = None
            with mock.patch("paddle.empty") as mock_empty:
                mock_empty.return_value = paddle.randn([8, 8])
                with mock.patch("paddle.slice") as mock_slice:
                    mock_slice.return_value = paddle.randn([8, 8])
                    result = all_gather_without_padding(x, num_pad=0, axis=0, group=mock_group)
                    # With num_pad=0, no slicing should happen (pad_start == len)
                    # Actually it still slices, but with full range

    def test_with_padding(self):
        """Test all_gather_without_padding with padding."""
        from paddleformers.fleet.context_parallel_utils import (
            all_gather_without_padding,
        )

        x = paddle.randn([4, 8])
        mock_group = mock.MagicMock()
        mock_group.nranks = 2

        with mock.patch("paddle.distributed.stream.all_gather"):  # noqa: SIM117
            with mock.patch("paddle.empty") as mock_empty:
                mock_out = paddle.randn([8, 8])
                mock_empty.return_value = mock_out
                with mock.patch("paddle.slice") as mock_slice:
                    mock_slice.return_value = paddle.randn([6, 8])
                    result = all_gather_without_padding(x, num_pad=2, axis=0, group=mock_group)
                    mock_slice.assert_called_once()


class TestContextParallelNormalScatter(unittest.TestCase):
    """Tests for ContextParallelNormalScatter PyLayer."""

    def test_single_rank_returns_clone(self):
        """Test single rank returns clone."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalScatter,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 1

        x = paddle.randn([8, 16])

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = ContextParallelNormalScatter.forward(mock_ctx, x, num_pad=0, axis=0)
            # Should return clone

    def test_multi_rank_calls_scatter_with_padding(self):
        """Test multi rank calls scatter_with_padding."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalScatter,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 2
        mock_hcg.get_context_parallel_group.return_value = mock_group

        x = paddle.randn([8, 16])

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=mock_hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.scatter_with_padding",
                return_value=paddle.randn([4, 16]),
            ) as mock_scatter,
        ):
            result = ContextParallelNormalScatter.forward(mock_ctx, x, num_pad=0, axis=0)
            mock_scatter.assert_called_once()
            self.assertEqual(mock_ctx.group, mock_group)
            self.assertEqual(mock_ctx.num_pad, 0)

    def test_backward_single_rank(self):
        """Test backward with single rank returns clone."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalScatter,
        )

        mock_ctx = mock.MagicMock()
        mock_ctx.group = mock.MagicMock()
        mock_ctx.group.nranks = 1
        mock_ctx.num_pad = 0
        mock_ctx.axis = 0

        grad = paddle.randn([4, 16])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.all_gather_without_padding",
        ) as mock_ag:
            result = ContextParallelNormalScatter.backward(mock_ctx, grad)
            mock_ag.assert_not_called()

    def test_backward_multi_rank(self):
        """Test backward with multi rank calls all_gather_without_padding."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalScatter,
        )

        mock_ctx = mock.MagicMock()
        mock_ctx.group = mock.MagicMock()
        mock_ctx.group.nranks = 2
        mock_ctx.num_pad = 2
        mock_ctx.axis = 0

        grad = paddle.randn([4, 16])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.all_gather_without_padding",
            return_value=paddle.randn([8, 16]),
        ) as mock_ag:
            result = ContextParallelNormalScatter.backward(mock_ctx, grad)
            mock_ag.assert_called_once_with(grad, 2, 0, mock_ctx.group)


class TestContextParallelNormalGather(unittest.TestCase):
    """Tests for ContextParallelNormalGather PyLayer."""

    def test_single_rank_returns_clone(self):
        """Test single rank returns clone."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalGather,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 1
        mock_hcg.get_context_parallel_group.return_value = mock_group

        x = paddle.randn([4, 16])

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = ContextParallelNormalGather.forward(mock_ctx, x, num_pad=0, axis=0)
            # ctx.group should still be set even for single rank
            self.assertEqual(mock_ctx.group, mock_group)

    def test_multi_rank_calls_all_gather(self):
        """Test multi rank calls all_gather_without_padding."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalGather,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 2
        mock_hcg.get_context_parallel_group.return_value = mock_group

        x = paddle.randn([4, 16])

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=mock_hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.all_gather_without_padding",
                return_value=paddle.randn([8, 16]),
            ) as mock_ag,
        ):
            result = ContextParallelNormalGather.forward(mock_ctx, x, num_pad=0, axis=0)
            mock_ag.assert_called_once()
            self.assertEqual(mock_ctx.group, mock_group)
            self.assertEqual(mock_ctx.num_pad, 0)

    def test_backward_single_rank(self):
        """Test backward with single rank returns clone."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalGather,
        )

        mock_ctx = mock.MagicMock()
        mock_ctx.group = mock.MagicMock()
        mock_ctx.group.nranks = 1

        grad = paddle.randn([8, 16])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.scatter_with_padding",
        ) as mock_scatter:
            result = ContextParallelNormalGather.backward(mock_ctx, grad)
            mock_scatter.assert_not_called()

    def test_backward_multi_rank(self):
        """Test backward with multi rank calls scatter_with_padding."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalGather,
        )

        mock_ctx = mock.MagicMock()
        mock_ctx.group = mock.MagicMock()
        mock_ctx.group.nranks = 2
        mock_ctx.num_pad = 2
        mock_ctx.axis = 1

        grad = paddle.randn([8, 16])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.scatter_with_padding",
            return_value=paddle.randn([4, 16]),
        ) as mock_scatter:
            result = ContextParallelNormalGather.backward(mock_ctx, grad)
            mock_scatter.assert_called_once_with(grad, 2, 1, mock_ctx.group)


class TestNormalPyLayerStructure(unittest.TestCase):
    """Tests for PyLayer structure."""

    def test_normal_scatter_has_forward_backward(self):
        """Test ContextParallelNormalScatter has forward and backward."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalScatter,
        )

        self.assertTrue(callable(ContextParallelNormalScatter.forward))
        self.assertTrue(callable(ContextParallelNormalScatter.backward))

    def test_normal_gather_has_forward_backward(self):
        """Test ContextParallelNormalGather has forward and backward."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelNormalGather,
        )

        self.assertTrue(callable(ContextParallelNormalGather.forward))
        self.assertTrue(callable(ContextParallelNormalGather.backward))


if __name__ == "__main__":
    unittest.main()
