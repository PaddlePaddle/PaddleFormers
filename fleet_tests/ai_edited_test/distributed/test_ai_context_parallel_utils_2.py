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


# Tests for src/paddlefleet/context_parallel_utils.py
# Additional tests for scatter_balance, all_gather_balance, reduce_scatter

import unittest
from unittest import mock

import paddle


class TestScatterBalanceSingleRank(unittest.TestCase):
    """Tests for scatter_balance with single rank."""

    def test_single_rank_returns_clone(self):
        """Test scatter_balance returns clone when parallelism == 1."""
        from paddleformers.fleet.context_parallel_utils import scatter_balance

        x = paddle.randn([8, 16])
        mock_group = mock.MagicMock()
        mock_group.nranks = 1

        result = scatter_balance(x, group=mock_group, axis=0)
        self.assertEqual(result.shape, [8, 16])

    def test_single_rank_returns_different_object(self):
        """Test that single rank returns a different tensor object."""
        from paddleformers.fleet.context_parallel_utils import scatter_balance

        x = paddle.randn([4, 8])
        mock_group = mock.MagicMock()
        mock_group.nranks = 1

        result = scatter_balance(x, group=mock_group, axis=0)
        self.assertIsNot(result, x)


class TestAllGatherBalanceSingleRank(unittest.TestCase):
    """Tests for all_gather_balance with single rank."""

    def test_single_rank_returns_clone(self):
        """Test all_gather_balance returns clone when parallelism == 1."""
        from paddleformers.fleet.context_parallel_utils import all_gather_balance

        x = paddle.randn([8, 16])
        mock_group = mock.MagicMock()
        mock_group.nranks = 1

        result = all_gather_balance(x, group=mock_group, axis=0)
        self.assertEqual(result.shape, [8, 16])

    def test_single_rank_returns_different_object(self):
        """Test that single rank returns a different tensor object."""
        from paddleformers.fleet.context_parallel_utils import all_gather_balance

        x = paddle.randn([4, 8])
        mock_group = mock.MagicMock()
        mock_group.nranks = 1

        result = all_gather_balance(x, group=mock_group, axis=0)
        self.assertIsNot(result, x)


class TestReduceScatterAnyAxisSingleRank(unittest.TestCase):
    """Tests for reduce_scatter_any_axis with single rank."""

    def test_single_rank_returns_clone(self):
        """Test reduce_scatter_any_axis returns clone when parallelism == 1."""
        from paddleformers.fleet.context_parallel_utils import reduce_scatter_any_axis

        x = paddle.randn([8, 16])
        mock_group = mock.MagicMock()
        mock_group.nranks = 1

        result = reduce_scatter_any_axis(x, axis=0, group=mock_group)
        self.assertEqual(result.shape, [8, 16])

    def test_single_rank_returns_different_object(self):
        """Test that single rank returns a different tensor object."""
        from paddleformers.fleet.context_parallel_utils import reduce_scatter_any_axis

        x = paddle.randn([4, 8])
        mock_group = mock.MagicMock()
        mock_group.nranks = 1

        result = reduce_scatter_any_axis(x, axis=0, group=mock_group)
        self.assertIsNot(result, x)


class TestReduceScatterAnyAxisBalanceSingleRank(unittest.TestCase):
    """Tests for reduce_scatter_any_axis_balance with single rank."""

    def test_single_rank_returns_clone(self):
        """Test reduce_scatter_any_axis_balance returns clone when parallelism == 1."""
        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_any_axis_balance,
        )

        x = paddle.randn([8, 16])
        mock_group = mock.MagicMock()
        mock_group.nranks = 1

        result = reduce_scatter_any_axis_balance(x, axis=0, group=mock_group)
        self.assertEqual(result.shape, [8, 16])


class TestScatterBalanceMultiRank(unittest.TestCase):
    """Tests for scatter_balance with multiple ranks (mocked)."""

    def test_assertion_on_undivisible_length(self):
        """Test scatter_balance asserts when seq_len not divisible by parallelism*2."""
        from paddleformers.fleet.context_parallel_utils import scatter_balance

        x = paddle.randn([7, 16])  # 7 not divisible by 2*2=4
        mock_group = mock.MagicMock()
        mock_group.nranks = 2
        mock_group.rank = 0

        with self.assertRaises(AssertionError) as ctx:
            scatter_balance(x, group=mock_group, axis=0)
        self.assertIn("divided exactly", str(ctx.exception))

    def test_scatter_uses_slice_and_concat(self):
        """Test scatter_balance calls slice and concat correctly."""
        from paddleformers.fleet.context_parallel_utils import scatter_balance

        x = paddle.randn([8, 16])  # 8 divisible by 2*2=4
        mock_group = mock.MagicMock()
        mock_group.nranks = 2
        mock_group.rank = 0

        with mock.patch("paddle.slice") as mock_slice:  # noqa: SIM117
            with mock.patch("paddle.concat") as mock_concat:
                mock_slice.return_value = paddle.randn([2, 16])
                mock_concat.return_value = paddle.randn([4, 16])
                with mock.patch("paddle.assign") as mock_assign:
                    mock_assign.return_value = paddle.randn([4, 16])
                    result = scatter_balance(x, group=mock_group, axis=0)
                    # slice should be called twice (start and end chunks)
                    self.assertEqual(mock_slice.call_count, 2)


class TestReduceScatterAnyAxisMultiRank(unittest.TestCase):
    """Tests for reduce_scatter_any_axis with multiple ranks."""

    def test_assertion_on_undivisible_length(self):
        """Test reduce_scatter_any_axis asserts on undivisible axis."""
        from paddleformers.fleet.context_parallel_utils import reduce_scatter_any_axis

        x = paddle.randn([7, 16])
        mock_group = mock.MagicMock()
        mock_group.nranks = 2

        with self.assertRaises(AssertionError) as ctx:
            reduce_scatter_any_axis(x, axis=0, group=mock_group)
        self.assertIn("divided exactly", str(ctx.exception))

    def test_reduce_scatter_balance_assertion(self):
        """Test reduce_scatter_any_axis_balance asserts on undivisible."""
        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_any_axis_balance,
        )

        x = paddle.randn([7, 16])
        mock_group = mock.MagicMock()
        mock_group.nranks = 2

        with self.assertRaises(AssertionError):
            reduce_scatter_any_axis_balance(x, axis=0, group=mock_group)


class TestScatterBalanceDefaultGroup(unittest.TestCase):
    """Tests for scatter_balance with default group (None)."""

    def test_none_group_uses_fleet(self):
        """Test that None group triggers fleet.get_hybrid_communicate_group."""
        from paddleformers.fleet.context_parallel_utils import scatter_balance

        x = paddle.randn([8, 16])
        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_group.nranks = 1
        mock_hcg.get_model_parallel_group.return_value = mock_group

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = scatter_balance(x, group=None, axis=0)
            mock_hcg.get_model_parallel_group.assert_called_once()


class TestAllGatherBalanceDefaultGroup(unittest.TestCase):
    """Tests for all_gather_balance with default group."""

    def test_none_group_uses_fleet(self):
        """Test that None group triggers fleet.get_hybrid_communicate_group."""
        from paddleformers.fleet.context_parallel_utils import all_gather_balance

        x = paddle.randn([8, 16])
        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_group.nranks = 1
        mock_hcg.get_model_parallel_group.return_value = mock_group

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = all_gather_balance(x, group=None, axis=0)
            mock_hcg.get_model_parallel_group.assert_called_once()


class TestReduceScatterDefaultGroup(unittest.TestCase):
    """Tests for reduce_scatter_any_axis with default group."""

    def test_none_group_uses_context_parallel_group(self):
        """Test that None group triggers context parallel group."""
        from paddleformers.fleet.context_parallel_utils import reduce_scatter_any_axis

        x = paddle.randn([8, 16])
        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_group.nranks = 1
        mock_hcg.get_context_parallel_group.return_value = mock_group

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = reduce_scatter_any_axis(x, axis=0, group=None)
            mock_hcg.get_context_parallel_group.assert_called_once()


if __name__ == "__main__":
    unittest.main()
