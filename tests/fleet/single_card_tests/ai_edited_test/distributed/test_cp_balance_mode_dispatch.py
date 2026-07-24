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

"""
Tests for cp_balance_mode dispatch in context_parallel_utils.py and utils.py.

Covers:
  - scatter_contiguous / all_gather_contiguous / reduce_scatter_contiguous
    (all branches: nranks==1, axis==0, axis!=0)
  - ContextParallelScatterOp / GatherOp / AllGatherOp mode dispatch
    (both "dualchunk_allgather" and "contiguous_allgather" paths)
  - get_batch_on_this_cp_rank with cp_balance_mode parameter
  - TransformerConfig.cp_balance_mode field default

Single-card tests using mocked distributed groups.
"""

import os
import sys

# Insert local src/ before site-packages so we test the dev version
_project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
sys.path.insert(0, os.path.join(_project_root, "src"))

import unittest
from unittest import mock

import paddle
import paddle.distributed as dist

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_group(nranks=4, rank=1):
    """Create a mock process group."""
    group = mock.MagicMock()
    group.nranks = nranks
    group.rank = rank
    return group


def _make_mock_hcg(nranks=4, rank=1):
    """Create a mock hybrid communicate group."""
    hcg = mock.MagicMock()
    group = _make_mock_group(nranks, rank)
    hcg.get_context_parallel_group.return_value = group
    hcg.get_context_parallel_world_size.return_value = nranks
    return hcg, group


# ===========================================================================
# Test scatter_contiguous
# ===========================================================================


class TestScatterContiguous(unittest.TestCase):
    """Tests for scatter_contiguous bare function."""

    def test_nranks_1_returns_clone(self):
        """nranks==1 returns a clone (no-op)."""
        from paddleformers.fleet.context_parallel_utils import (
            scatter_contiguous,
        )

        group = _make_mock_group(nranks=1, rank=0)
        x = paddle.arange(12).reshape([3, 4]).cast("float32")
        out = scatter_contiguous(x, group=group, axis=0)
        self.assertTrue(paddle.equal_all(out, x))
        # Must be a copy, not same storage
        out[0, 0] = -1.0
        self.assertNotEqual(x[0, 0].item(), -1.0)

    def test_axis_0_rank_slicing(self):
        """axis=0: rank r gets rows [r*chunk, (r+1)*chunk]."""
        from paddleformers.fleet.context_parallel_utils import (
            scatter_contiguous,
        )

        nranks = 4
        x = paddle.arange(16).reshape([8, 2]).cast("float32")
        for rank in range(nranks):
            group = _make_mock_group(nranks=nranks, rank=rank)
            out = scatter_contiguous(x, group=group, axis=0)
            expected = x[rank * 2 : (rank + 1) * 2, :]
            self.assertTrue(paddle.equal_all(out, expected))

    def test_axis_1_rank_slicing(self):
        """axis=1: rank r gets cols [r*chunk, (r+1)*chunk]."""
        from paddleformers.fleet.context_parallel_utils import (
            scatter_contiguous,
        )

        nranks = 2
        x = paddle.arange(12).reshape([3, 4]).cast("float32")
        for rank in range(nranks):
            group = _make_mock_group(nranks=nranks, rank=rank)
            out = scatter_contiguous(x, group=group, axis=1)
            expected = x[:, rank * 2 : (rank + 1) * 2]
            self.assertTrue(paddle.equal_all(out, expected))

    def test_negative_axis(self):
        """axis=-1 works correctly."""
        from paddleformers.fleet.context_parallel_utils import (
            scatter_contiguous,
        )

        nranks = 2
        x = paddle.arange(24).reshape([2, 3, 4]).cast("float32")
        group = _make_mock_group(nranks=nranks, rank=0)
        out = scatter_contiguous(x, group=group, axis=-1)
        expected = x[:, :, :2]
        self.assertTrue(paddle.equal_all(out, expected))

    def test_group_none_uses_fleet(self):
        """group=None fetches group from fleet."""
        from paddleformers.fleet.context_parallel_utils import (
            scatter_contiguous,
        )

        hcg, group = _make_mock_hcg(nranks=2, rank=0)
        x = paddle.arange(8).reshape([4, 2]).cast("float32")
        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=hcg,
        ):
            out = scatter_contiguous(x, group=None, axis=0)
        expected = x[:2, :]
        self.assertTrue(paddle.equal_all(out, expected))


# ===========================================================================
# Test all_gather_contiguous
# ===========================================================================


class TestAllGatherContiguous(unittest.TestCase):
    """Tests for all_gather_contiguous bare function."""

    def test_nranks_1_returns_clone(self):
        """nranks==1 returns a clone."""
        from paddleformers.fleet.context_parallel_utils import (
            all_gather_contiguous,
        )

        group = _make_mock_group(nranks=1, rank=0)
        x = paddle.randn([2, 4])
        out = all_gather_contiguous(x, group=group, axis=0)
        self.assertTrue(paddle.equal_all(out, x))

    def test_axis_0_calls_all_gather_flat(self):
        """axis=0: uses flat buffer all_gather."""
        from paddleformers.fleet.context_parallel_utils import (
            all_gather_contiguous,
        )

        nranks = 2
        group = _make_mock_group(nranks=nranks, rank=0)
        x = paddle.ones([3, 4])

        def fake_all_gather(output, input_tensor, group, use_calc_stream):
            output[:] = paddle.concat([input_tensor, input_tensor * 2], axis=0)

        with mock.patch.object(
            dist.stream, "all_gather", side_effect=fake_all_gather
        ):
            out = all_gather_contiguous(x, group=group, axis=0)
        self.assertEqual(list(out.shape), [6, 4])

    def test_axis_1_calls_list_all_gather(self):
        """axis=1: uses list-based all_gather + concat."""
        from paddleformers.fleet.context_parallel_utils import (
            all_gather_contiguous,
        )

        nranks = 2
        group = _make_mock_group(nranks=nranks, rank=0)
        x = paddle.ones([2, 3])

        def fake_all_gather(tensor_list, input_tensor, group, use_calc_stream):
            for i, t in enumerate(tensor_list):
                tensor_list[i][:] = input_tensor * (i + 1)

        with mock.patch.object(
            dist.stream, "all_gather", side_effect=fake_all_gather
        ):
            out = all_gather_contiguous(x, group=group, axis=1)
        self.assertEqual(list(out.shape), [2, 6])

    def test_group_none_uses_fleet(self):
        """group=None fetches from fleet."""
        from paddleformers.fleet.context_parallel_utils import (
            all_gather_contiguous,
        )

        hcg, group = _make_mock_hcg(nranks=1, rank=0)
        x = paddle.randn([2, 4])
        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=hcg,
        ):
            out = all_gather_contiguous(x, group=None, axis=0)
        self.assertTrue(paddle.equal_all(out, x))


# ===========================================================================
# Test reduce_scatter_contiguous
# ===========================================================================


class TestReduceScatterContiguous(unittest.TestCase):
    """Tests for reduce_scatter_contiguous bare function."""

    def test_nranks_1_returns_clone(self):
        """nranks==1 returns a clone."""
        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_contiguous,
        )

        group = _make_mock_group(nranks=1, rank=0)
        x = paddle.randn([4, 6])
        out = reduce_scatter_contiguous(x, axis=0, group=group)
        self.assertTrue(paddle.equal_all(out, x))

    def test_axis_0_calls_reduce_scatter(self):
        """axis=0: uses dist.stream.reduce_scatter."""
        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_contiguous,
        )

        nranks = 2
        group = _make_mock_group(nranks=nranks, rank=0)
        x = paddle.ones([4, 6])

        def fake_reduce_scatter(
            output, input_tensor, op, group, use_calc_stream
        ):
            output[:] = input_tensor[: output.shape[0]]

        with mock.patch.object(
            dist.stream, "reduce_scatter", side_effect=fake_reduce_scatter
        ):
            out = reduce_scatter_contiguous(x, axis=0, group=group)
        self.assertEqual(list(out.shape), [2, 6])

    def test_axis_1_calls_alltoall(self):
        """axis=1: uses split + alltoall + fp32 sum."""
        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_contiguous,
        )

        nranks = 2
        group = _make_mock_group(nranks=nranks, rank=0)
        x = paddle.ones([3, 4])

        def fake_alltoall(output_list, input_list, group, use_calc_stream):
            for i, buf in enumerate(output_list):
                buf[:] = input_list[i] * (i + 1)

        with mock.patch.object(
            dist.stream, "alltoall", side_effect=fake_alltoall
        ):
            out = reduce_scatter_contiguous(x, axis=1, group=group)
        # output shape: each chunk is [3, 2], sum of 2 chunks
        self.assertEqual(list(out.shape), [3, 2])
        # dtype should match input
        self.assertEqual(out.dtype, x.dtype)

    def test_group_none_uses_fleet(self):
        """group=None fetches from fleet."""
        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_contiguous,
        )

        hcg, group = _make_mock_hcg(nranks=1, rank=0)
        x = paddle.randn([4, 6])
        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=hcg,
        ):
            out = reduce_scatter_contiguous(x, axis=0, group=None)
        self.assertTrue(paddle.equal_all(out, x))


# ===========================================================================
# Test PyLayer dispatch (ContextParallelScatterOp / GatherOp / AllGatherOp)
# ===========================================================================


class TestContextParallelScatterOpDispatch(unittest.TestCase):
    """Tests for ContextParallelScatterOp mode dispatch."""

    def _call_forward(self, mode, nranks=4, rank=1):
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        hcg, group = _make_mock_hcg(nranks=nranks, rank=rank)
        x = paddle.arange(16).reshape([4, 4]).cast("float32")

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=hcg,
        ):
            ctx = mock.MagicMock()
            out = ContextParallelScatterOp.forward(ctx, x, axis=0, mode=mode)
        return out, ctx, x

    def test_contiguous_mode_calls_scatter_contiguous(self):
        """mode='contiguous_allgather' dispatches to scatter_contiguous."""
        out, ctx, x = self._call_forward("contiguous_allgather")
        # rank=1, nranks=4, chunk=1 row -> row [1:2]
        expected = x[1:2, :]
        self.assertTrue(paddle.equal_all(out, expected))
        self.assertEqual(ctx.mode, "contiguous_allgather")

    def test_dualchunk_mode_calls_scatter_balance(self):
        """mode='dualchunk_allgather' dispatches to scatter_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        hcg, group = _make_mock_hcg(nranks=2, rank=0)
        x = paddle.arange(8).reshape([4, 2]).cast("float32")

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.scatter_balance",
                return_value=x[:2],
            ) as mock_scatter,
        ):
            ctx = mock.MagicMock()
            out = ContextParallelScatterOp.forward(
                ctx, x, axis=0, mode="dualchunk_allgather"
            )
            mock_scatter.assert_called_once()
        self.assertEqual(ctx.mode, "dualchunk_allgather")

    def test_backward_contiguous_calls_all_gather(self):
        """backward with contiguous mode calls all_gather_contiguous."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        ctx = mock.MagicMock()
        ctx.mode = "contiguous_allgather"
        ctx.axis = 0
        ctx.group = _make_mock_group(nranks=1, rank=0)
        grad = paddle.randn([2, 4])

        out = ContextParallelScatterOp.backward(ctx, grad)
        # nranks=1, clone
        self.assertTrue(paddle.equal_all(out, grad))

    def test_backward_dualchunk_calls_all_gather_balance(self):
        """backward with dualchunk mode calls all_gather_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        ctx = mock.MagicMock()
        ctx.mode = "dualchunk_allgather"
        ctx.axis = 0
        ctx.group = _make_mock_group(nranks=2, rank=0)
        grad = paddle.randn([2, 4])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.all_gather_balance",
            return_value=paddle.randn([4, 4]),
        ) as mock_fn:
            out = ContextParallelScatterOp.backward(ctx, grad)
            mock_fn.assert_called_once()


class TestContextParallelGatherOpDispatch(unittest.TestCase):
    """Tests for ContextParallelGatherOp mode dispatch."""

    def test_contiguous_mode_forward(self):
        """forward with contiguous mode calls all_gather_contiguous."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        hcg, group = _make_mock_hcg(nranks=2, rank=0)
        x = paddle.randn([2, 4])
        fake_gathered = paddle.randn([4, 4])

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.all_gather_contiguous",
                return_value=fake_gathered,
            ) as mock_fn,
        ):
            ctx = mock.MagicMock()
            out = ContextParallelGatherOp.forward(
                ctx, x, axis=0, mode="contiguous_allgather"
            )
            mock_fn.assert_called_once()
        self.assertTrue(paddle.equal_all(out, fake_gathered))
        self.assertEqual(ctx.mode, "contiguous_allgather")

    def test_dualchunk_mode_forward(self):
        """forward with dualchunk mode calls all_gather_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        hcg, group = _make_mock_hcg(nranks=2, rank=0)
        x = paddle.randn([2, 4])

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.all_gather_balance",
                return_value=paddle.randn([4, 4]),
            ) as mock_fn,
        ):
            ctx = mock.MagicMock()
            out = ContextParallelGatherOp.forward(
                ctx, x, axis=0, mode="dualchunk_allgather"
            )
            mock_fn.assert_called_once()

    def test_backward_contiguous(self):
        """backward with contiguous mode calls scatter_contiguous."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        ctx = mock.MagicMock()
        ctx.mode = "contiguous_allgather"
        ctx.axis = 0
        ctx.group = _make_mock_group(nranks=2, rank=0)
        grad = paddle.arange(8).reshape([4, 2]).cast("float32")

        out = ContextParallelGatherOp.backward(ctx, grad)
        expected = grad[:2, :]
        self.assertTrue(paddle.equal_all(out, expected))

    def test_backward_dualchunk(self):
        """backward with dualchunk mode calls scatter_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        ctx = mock.MagicMock()
        ctx.mode = "dualchunk_allgather"
        ctx.axis = 0
        ctx.group = _make_mock_group(nranks=2, rank=0)
        grad = paddle.randn([4, 4])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.scatter_balance",
            return_value=paddle.randn([2, 4]),
        ) as mock_fn:
            out = ContextParallelGatherOp.backward(ctx, grad)
            mock_fn.assert_called_once()


class TestContextParallelAllGatherOpDispatch(unittest.TestCase):
    """Tests for ContextParallelAllGatherOp mode dispatch."""

    def test_contiguous_mode_forward(self):
        """forward with contiguous mode calls all_gather_contiguous."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        hcg, group = _make_mock_hcg(nranks=2, rank=0)
        x = paddle.randn([2, 4])
        fake_gathered = paddle.randn([4, 4])

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.all_gather_contiguous",
                return_value=fake_gathered,
            ) as mock_fn,
        ):
            ctx = mock.MagicMock()
            out = ContextParallelAllGatherOp.forward(
                ctx, x, axis=0, mode="contiguous_allgather"
            )
            mock_fn.assert_called_once()
        self.assertTrue(paddle.equal_all(out, fake_gathered))
        self.assertEqual(ctx.mode, "contiguous_allgather")

    def test_dualchunk_mode_forward(self):
        """forward with dualchunk mode calls all_gather_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        hcg, group = _make_mock_hcg(nranks=2, rank=0)
        x = paddle.randn([2, 4])

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.all_gather_balance",
                return_value=paddle.randn([4, 4]),
            ) as mock_fn,
        ):
            ctx = mock.MagicMock()
            out = ContextParallelAllGatherOp.forward(
                ctx, x, axis=0, mode="dualchunk_allgather"
            )
            mock_fn.assert_called_once()

    def test_backward_contiguous(self):
        """backward with contiguous mode calls reduce_scatter_contiguous."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        ctx = mock.MagicMock()
        ctx.mode = "contiguous_allgather"
        ctx.axis = 0
        ctx.group = _make_mock_group(nranks=1, rank=0)
        grad = paddle.randn([4, 4])

        out = ContextParallelAllGatherOp.backward(ctx, grad)
        self.assertTrue(paddle.equal_all(out, grad))  # nranks=1, clone

    def test_backward_dualchunk(self):
        """backward with dualchunk calls reduce_scatter_any_axis_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        ctx = mock.MagicMock()
        ctx.mode = "dualchunk_allgather"
        ctx.axis = 0
        ctx.group = _make_mock_group(nranks=2, rank=0)
        grad = paddle.randn([4, 4])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.reduce_scatter_any_axis_balance",
            return_value=paddle.randn([2, 4]),
        ) as mock_fn:
            out = ContextParallelAllGatherOp.backward(ctx, grad)
            mock_fn.assert_called_once()


# ===========================================================================
# Test get_batch_on_this_cp_rank with cp_balance_mode
# ===========================================================================


class TestGetBatchOnThisCpRank(unittest.TestCase):
    """Tests for get_batch_on_this_cp_rank in utils.py."""

    def _setup_fleet_mock(self, nranks=4, rank=1):
        hcg, group = _make_mock_hcg(nranks=nranks, rank=rank)
        return mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=hcg,
        )

    def test_tensor_input_default_mode(self):
        """Tensor input uses dualchunk by default."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        x = paddle.arange(16).reshape([4, 4]).cast("float32")
        with (
            self._setup_fleet_mock(nranks=2, rank=0),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.scatter_balance",
                return_value=x[:, :2],
            ) as mock_fn,
        ):
            out = get_batch_on_this_cp_rank(x)
            mock_fn.assert_called_once()

    def test_tensor_input_contiguous_mode(self):
        """Tensor input with contiguous mode scatters along axis=-1."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        x = paddle.arange(16).reshape([4, 4]).cast("float32")
        with self._setup_fleet_mock(nranks=2, rank=0):
            out = get_batch_on_this_cp_rank(
                x, cp_balance_mode="contiguous_allgather"
            )
        # axis=-1, rank=0, nranks=2 -> first half of cols
        self.assertEqual(list(out.shape), [4, 2])

    def test_dict_input_splits_known_keys(self):
        """Dict input splits input_ids, position_ids, labels; passes others."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        inputs = {
            "input_ids": paddle.arange(8).reshape([2, 4]).cast("int64"),
            "position_ids": paddle.arange(8).reshape([2, 4]).cast("int64"),
            "labels": paddle.arange(8).reshape([2, 4]).cast("int64"),
            "attention_mask": paddle.ones([2, 4]),
        }
        with self._setup_fleet_mock(nranks=2, rank=0):
            out = get_batch_on_this_cp_rank(
                inputs, cp_balance_mode="contiguous_allgather"
            )
        # Split keys should be halved on axis=-1
        self.assertEqual(list(out["input_ids"].shape), [2, 2])
        self.assertEqual(list(out["position_ids"].shape), [2, 2])
        self.assertEqual(list(out["labels"].shape), [2, 2])
        # Non-split key should be unchanged
        self.assertEqual(list(out["attention_mask"].shape), [2, 4])

    def test_dict_input_default_dualchunk_mode(self):
        """Dict input uses dualchunk by default."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        inputs = {
            "input_ids": paddle.arange(8).reshape([2, 4]).cast("int64"),
            "labels": paddle.arange(8).reshape([2, 4]).cast("int64"),
        }
        with (
            self._setup_fleet_mock(nranks=2, rank=0),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.scatter_balance",
                return_value=paddle.zeros([2, 2], dtype="int64"),
            ) as mock_fn,
        ):
            out = get_batch_on_this_cp_rank(inputs)
            # Called once per split key
            self.assertEqual(mock_fn.call_count, 2)

    def test_list_input_raises(self):
        """List input raises AssertionError."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        with self.assertRaises(AssertionError):
            get_batch_on_this_cp_rank([paddle.ones([2, 4])])

    def test_invalid_type_raises(self):
        """Non-tensor/dict/list input raises ValueError."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        with self.assertRaises(ValueError):
            get_batch_on_this_cp_rank("invalid")


# ===========================================================================
# Test PyLayer assertion guards (nranks <= 1 should raise)
# ===========================================================================


class TestPyLayerAssertionGuards(unittest.TestCase):
    """Test that PyLayer ops raise AssertionError when cp_world_size <= 1."""

    def test_scatter_op_asserts_nranks_gt_1(self):
        """ScatterOp raises when context_parallel_world_size <= 1."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        hcg, _ = _make_mock_hcg(nranks=1, rank=0)
        x = paddle.randn([4, 4])
        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            self.assertRaises(AssertionError),
        ):
            ContextParallelScatterOp.forward(mock.MagicMock(), x, axis=0)

    def test_gather_op_asserts_nranks_gt_1(self):
        """GatherOp raises when context_parallel_world_size <= 1."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        hcg, _ = _make_mock_hcg(nranks=1, rank=0)
        x = paddle.randn([4, 4])
        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            self.assertRaises(AssertionError),
        ):
            ContextParallelGatherOp.forward(mock.MagicMock(), x, axis=0)

    def test_allgather_op_asserts_nranks_gt_1(self):
        """AllGatherOp raises when context_parallel_world_size <= 1."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        hcg, _ = _make_mock_hcg(nranks=1, rank=0)
        x = paddle.randn([4, 4])
        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            self.assertRaises(AssertionError),
        ):
            ContextParallelAllGatherOp.forward(mock.MagicMock(), x, axis=0)


# ===========================================================================
# Test reduce_scatter_contiguous dtype preservation (bf16 path)
# ===========================================================================


class TestReduceScatterContiguousDtype(unittest.TestCase):
    """Test that reduce_scatter_contiguous preserves input dtype through fp32 sum."""

    def test_bf16_input_preserved(self):
        """bf16 input -> fp32 sum -> cast back to bf16."""
        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_contiguous,
        )

        nranks = 2
        group = _make_mock_group(nranks=nranks, rank=0)
        x = paddle.ones([3, 4]).cast("bfloat16")

        def fake_alltoall(output_list, input_list, group, use_calc_stream):
            for i, buf in enumerate(output_list):
                buf[:] = paddle.ones(buf.shape).cast("bfloat16")

        with mock.patch.object(
            dist.stream, "alltoall", side_effect=fake_alltoall
        ):
            out = reduce_scatter_contiguous(x, axis=1, group=group)
        self.assertEqual(out.dtype, paddle.bfloat16)

    def test_float16_input_preserved(self):
        """float16 input -> fp32 sum -> cast back to float16."""
        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_contiguous,
        )

        nranks = 2
        group = _make_mock_group(nranks=nranks, rank=0)
        x = paddle.ones([3, 4]).cast("float16")

        def fake_alltoall(output_list, input_list, group, use_calc_stream):
            for i, buf in enumerate(output_list):
                buf[:] = paddle.ones(buf.shape).cast("float16")

        with mock.patch.object(
            dist.stream, "alltoall", side_effect=fake_alltoall
        ):
            out = reduce_scatter_contiguous(x, axis=1, group=group)
        self.assertEqual(out.dtype, paddle.float16)


# ===========================================================================
# Test scatter_contiguous value correctness for various ranks
# ===========================================================================


class TestScatterContiguousValueCorrectness(unittest.TestCase):
    """Test scatter_contiguous produces correct values for all ranks with 3D tensors."""

    def test_3d_tensor_axis_1(self):
        """3D tensor [B, S, D] scattered on axis=1."""
        from paddleformers.fleet.context_parallel_utils import (
            scatter_contiguous,
        )

        nranks = 4
        B, S, D = 2, 8, 3
        x = paddle.arange(B * S * D).reshape([B, S, D]).cast("float32")
        for rank in range(nranks):
            group = _make_mock_group(nranks=nranks, rank=rank)
            out = scatter_contiguous(x, group=group, axis=1)
            expected = x[:, rank * 2 : (rank + 1) * 2, :]
            self.assertTrue(paddle.equal_all(out, expected))
            self.assertEqual(list(out.shape), [B, 2, D])


# ===========================================================================
# Test TransformerConfig.cp_balance_mode field
# ===========================================================================


class TestTransformerConfigCpBalanceMode(unittest.TestCase):
    """Tests for cp_balance_mode field in TransformerConfig."""

    def test_default_value(self):
        """Default is 'dualchunk_allgather'."""
        from paddleformers.fleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig()
        self.assertEqual(config.cp_balance_mode, "dualchunk_allgather")

    def test_set_contiguous(self):
        """Can be set to 'contiguous_allgather'."""
        from paddleformers.fleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig(cp_balance_mode="contiguous_allgather")
        self.assertEqual(config.cp_balance_mode, "contiguous_allgather")


if __name__ == "__main__":
    unittest.main()
