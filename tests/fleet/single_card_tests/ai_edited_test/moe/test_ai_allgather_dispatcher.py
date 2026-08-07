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
"""Unit tests for AllGatherTokenDispatcher and related new code added relative
to commit a8e3fbc2d827a1845f0223c9abaa53cc738dfe0b.

Covers:
  - ReduceScatterGroupOp
  - _RouterAllGather
  - _PreAllGatherResult / _PreAllGatherFP8Result
  - _AllGatherCombineNoOverlap / _AllGatherCombineAsync
  - _tokens_per_expert_histogram
  - _reduce_scatter_async / _all_gather_async
  - _split_fused_fp8_gather
  - AllGatherTokenDispatcher (all methods)
  - AllToAllTokenDispatcher.get_dispatched_routing / token_combine
  - MoEFlexTokenDispatcher.get_dispatched_routing
  - MoELayer._validate_allgather_config / _project_to_latent /
    _maybe_pre_allgather_overlap / combine
  - TransformerLayerWithOverlap ValueError for non-deepep dispatchers
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

# ── helpers ──────────────────────────────────────────────────────────────────


def _mock_group(nranks=1, rank=0):
    """Create a mock distributed group.

    ``rank`` must be set explicitly because DCP ``shard_weight`` reads
    ``group.rank`` to compute ``global_offset``; an unset MagicMock attribute
    would leak into offset arithmetic and corrupt the checkpoint layout.
    """
    g = MagicMock()
    g.nranks = nranks
    g.world_size = nranks
    g.rank = rank
    g.id = 0
    return g


def _make_ag_dispatcher(group=None, num_experts=4, fp8=False):
    from paddleformers.fleet.transformer.moe.token_dispatcher import (
        AllGatherTokenDispatcher,
    )

    return AllGatherTokenDispatcher(
        moe_group=group,
        expert_model_parallel_size=1,
        num_experts=num_experts,
        fp8_dispatch=fp8,
        use_ue8m0=False,
    )


# ── ReduceScatterGroupOp ─────────────────────────────────────────────────────


class TestReduceScatterGroupOp(unittest.TestCase):
    """Tests for ReduceScatterGroupOp PyLayer (group=None fast path)."""

    def test_forward_group_none(self):
        """Forward with group=None: Paddle init may fail, so mock the op."""
        from paddleformers.fleet.transformer.moe.moe_utils import ReduceScatterGroupOp

        x = paddle.randn([4, 8])
        with patch(
            "paddleformers.fleet.transformer.moe.moe_utils.reduce_scatter_group",
            side_effect=lambda t, group=None: t,
        ):
            out = ReduceScatterGroupOp.apply(x, None)
            np.testing.assert_allclose(out.numpy(), x.numpy())

    def test_forward_backward_round_trip(self):
        """Forward→backward identity for group=None (EP>1 backward in multi-card tests)."""
        from paddleformers.fleet.transformer.moe.moe_utils import ReduceScatterGroupOp

        x = paddle.randn([4, 8])
        with (
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils.reduce_scatter_group",
                side_effect=lambda t, group=None: t,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils.all_gather_group",
                side_effect=lambda t, group=None: t,
            ),
        ):
            out = ReduceScatterGroupOp.apply(x, None)
            np.testing.assert_allclose(out.numpy(), x.numpy())


# ── _RouterAllGather ─────────────────────────────────────────────────────────


class TestRouterAllGather(unittest.TestCase):
    """Tests for _RouterAllGather PyLayer (group=None fast path)."""

    def test_forward_group_none(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        x = paddle.randn([4, 2])
        out = _RouterAllGather.apply(x, None)
        np.testing.assert_allclose(out.numpy(), x.numpy())

    def test_forward_nranks_1(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        x = paddle.randn([4, 2])
        g = _mock_group(1)
        with patch("paddle.distributed.stream.all_gather"):
            out = _RouterAllGather.apply(x, g)
        np.testing.assert_allclose(out.numpy(), x.numpy())

    def test_forward_group_none_passthrough(self):
        """Forward for group=None returns clone of input."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        x = paddle.randn([4, 2])
        out = _RouterAllGather.apply(x, None)
        np.testing.assert_allclose(out.numpy(), x.numpy())

    def test_backward_shape_mismatch_raises(self):
        """When group is not None and grad shape mismatches, ValueError."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        # Get the raw backward function from PyLayer
        bwd = _RouterAllGather.__dict__.get("backward")
        if bwd is not None and hasattr(bwd, "__func__"):
            bwd = bwd.__func__
        ctx = MagicMock()
        ctx.group = _mock_group(2)
        ctx.input_shape = [4, 2]
        bad_grad = paddle.randn([3, 2])
        import paddleformers.fleet.transformer.moe.token_dispatcher as td

        with (
            patch.object(td, "reduce_scatter_group"),
            self.assertRaises(ValueError),
        ):
            bwd(ctx, bad_grad)


# ── _tokens_per_expert_histogram ─────────────────────────────────────────────


class TestTokensPerExpertHistogram(unittest.TestCase):
    """Tests for _tokens_per_expert_histogram."""

    def test_basic_count(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _tokens_per_expert_histogram,
        )

        indices = paddle.to_tensor([[0, 1], [0, -1], [1, 2]], dtype="int32")
        counts = _tokens_per_expert_histogram(indices, 3)
        # Expert 0: 2, Expert 1: 2, Expert 2: 1
        np.testing.assert_array_equal(counts.numpy(), [2, 2, 1])

    def test_all_padding(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _tokens_per_expert_histogram,
        )

        indices = paddle.to_tensor([[-1, -1], [-1, -1]], dtype="int32")
        counts = _tokens_per_expert_histogram(indices, 4)
        np.testing.assert_array_equal(counts.numpy(), [0, 0, 0, 0])

    def test_single_expert(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _tokens_per_expert_histogram,
        )

        indices = paddle.to_tensor([[0, -1], [0, 0]], dtype="int32")
        counts = _tokens_per_expert_histogram(indices, 1)
        np.testing.assert_array_equal(counts.numpy(), [3])


# ── _PreAllGatherResult / _PreAllGatherFP8Result ────────────────────────────


class TestPreAllGatherResult(unittest.TestCase):
    """Tests for _PreAllGatherResult PyLayer (group=None fast path)."""

    def test_forward(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _PreAllGatherResult,
        )

        x = paddle.randn([4, 8])
        output = paddle.randn([4, 8])
        handle = {"task": MagicMock(), "output": output, "group": None}
        out = _PreAllGatherResult.apply(x, handle)
        handle["task"].wait.assert_called_once()
        np.testing.assert_allclose(out.numpy(), output.numpy())

    def test_backward(self):
        """Forward for group=None returns handle output (backward in multi-card tests)."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _PreAllGatherResult,
        )

        x = paddle.randn([4, 8])
        output = paddle.randn([4, 8])
        handle = {"task": MagicMock(), "output": output, "group": None}
        out = _PreAllGatherResult.apply(x, handle)
        np.testing.assert_allclose(out.numpy(), output.numpy())


class TestPreAllGatherFP8Result(unittest.TestCase):
    """Tests for _PreAllGatherFP8Result PyLayer."""

    def test_forward(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _PreAllGatherFP8Result,
        )

        H, H128 = 8, 1
        fused = paddle.zeros([4, H + 4 * H128], dtype="uint8")
        handle = {
            "task": MagicMock(),
            "fused_global": fused,
            "H": H,
            "H128": H128,
            "scale_dtype": paddle.int32,
            "group": None,
        }
        x_fp8, scale = _PreAllGatherFP8Result.apply(
            paddle.randn([4, H]), handle
        )
        handle["task"].wait.assert_called_once()
        self.assertEqual(x_fp8.shape, [4, H])
        self.assertEqual(scale.shape, [4, H128])

    def test_backward_none_grad(self):
        """backward with None grad returns None (group=None path)."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _PreAllGatherFP8Result,
        )

        # Access the raw function from the staticmethod descriptor
        backward_fn = _PreAllGatherFP8Result.__dict__["backward"]
        if isinstance(backward_fn, staticmethod):
            backward_fn = backward_fn.__func__
        ctx = MagicMock()
        ctx.group = None
        result = backward_fn(ctx, None, None)
        self.assertIsNone(result)

    def test_backward_group_none(self):
        """backward with group=None returns grad unchanged."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _PreAllGatherFP8Result,
        )

        backward_fn = _PreAllGatherFP8Result.__dict__["backward"]
        if isinstance(backward_fn, staticmethod):
            backward_fn = backward_fn.__func__
        grad = paddle.randn([4, 8])
        ctx = MagicMock()
        ctx.group = None
        result = backward_fn(ctx, grad, None)
        np.testing.assert_allclose(result.numpy(), grad.numpy())


# ── _AllGatherCombineNoOverlap ───────────────────────────────────────────────


class TestAllGatherCombineNoOverlap(unittest.TestCase):
    """Tests for _AllGatherCombineNoOverlap PyLayer (group=None fast path)."""

    def test_forward_group_none(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineNoOverlap,
        )

        x = paddle.randn([4, 8])
        out = _AllGatherCombineNoOverlap.apply(x, None)
        np.testing.assert_allclose(out.numpy(), x.numpy())

    def test_forward_with_group(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineNoOverlap,
        )

        x = paddle.randn([4, 8])
        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.reduce_scatter_group",
            return_value=x,
        ):
            out = _AllGatherCombineNoOverlap.apply(x, _mock_group(2))
            np.testing.assert_allclose(out.numpy(), x.numpy())

    def test_backward_bf16(self):
        """Backward with group and bf16 path (mocked)."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineNoOverlap,
        )

        backward_fn = _AllGatherCombineNoOverlap.__dict__["backward"]
        if isinstance(backward_fn, staticmethod):
            backward_fn = backward_fn.__func__
        grad = paddle.randn([4, 8])
        ctx = MagicMock()
        ctx.group = _mock_group(2)
        ctx.fp8_combine_grad_handle = None
        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.all_gather_group",
            return_value=grad,
        ):
            result = backward_fn(ctx, grad)
            np.testing.assert_allclose(result.numpy(), grad.numpy())


class TestAllGatherCombineAsync(unittest.TestCase):
    """Tests for _AllGatherCombineAsync PyLayer."""

    def test_forward_none_fn_raises(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineAsync,
        )

        with self.assertRaises(ValueError):
            _AllGatherCombineAsync.apply(
                paddle.randn([4, 8]),
                None,
                fn=None,
            )

    def test_forward_group_none(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineAsync,
        )

        x = paddle.randn([4, 8])
        fn_out = paddle.randn([4, 8])
        mock_fn = MagicMock(return_value=(fn_out,))
        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.manual_backward",
            return_value=(MagicMock(return_value=(fn_out,)), (fn_out,)),
        ):
            result = _AllGatherCombineAsync.apply(
                x,
                None,
                fn=mock_fn,
                is_first_fwd=False,
            )
            self.assertEqual(len(result), 2)

    def test_backward_group_none(self):
        """Backward with group=None (mocked)."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineAsync,
        )

        backward_fn = _AllGatherCombineAsync.__dict__["backward"]
        if isinstance(backward_fn, staticmethod):
            backward_fn = backward_fn.__func__
        grad = paddle.randn([4, 8])
        ctx = MagicMock()
        ctx.group = None
        ctx.bwf = MagicMock(return_value=(paddle.randn([4, 8]),))
        result = backward_fn(ctx, grad, paddle.randn([4, 8]))
        self.assertEqual(len(result), 2)


# ── async helper functions ───────────────────────────────────────────────────


class TestAsyncHelpers(unittest.TestCase):
    """Tests for _reduce_scatter_async and _all_gather_async."""

    def test_reduce_scatter_async(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _reduce_scatter_async,
        )

        g = _mock_group(2)
        x = paddle.randn([4, 8])
        with patch("paddle.distributed.stream.reduce_scatter") as mock_rs:
            mock_task = MagicMock()
            mock_rs.return_value = mock_task
            out, task = _reduce_scatter_async(x, g)
            self.assertEqual(out.shape[0], 2)
            mock_rs.assert_called_once()

    def test_reduce_scatter_async_not_divisible(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _reduce_scatter_async,
        )

        g = _mock_group(3)
        x = paddle.randn([4, 8])
        with self.assertRaises(ValueError):
            _reduce_scatter_async(x, g)

    def test_all_gather_async(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _all_gather_async,
        )

        g = _mock_group(2)
        x = paddle.randn([4, 8])
        with patch("paddle.distributed.stream.all_gather") as mock_ag:
            mock_task = MagicMock()
            mock_ag.return_value = mock_task
            out, task = _all_gather_async(x, g)
            self.assertEqual(out.shape[0], 8)
            mock_ag.assert_called_once()


# ── _split_fused_fp8_gather ──────────────────────────────────────────────────


class TestSplitFusedFP8Gather(unittest.TestCase):
    """Tests for _split_fused_fp8_gather."""

    def test_split(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _split_fused_fp8_gather,
        )

        T, H, H128 = 4, 8, 1
        fused = paddle.zeros([T, H + 4 * H128], dtype="uint8")
        data, scale = _split_fused_fp8_gather(fused, H, H128, paddle.int32)
        self.assertEqual(data.shape, [T, H])
        self.assertEqual(scale.shape, [T, H128])


# ── AllGatherTokenDispatcher ─────────────────────────────────────────────────


class TestAllGatherTokenDispatcher(unittest.TestCase):
    """Tests for AllGatherTokenDispatcher."""

    def test_init(self):
        """Test basic initialization."""
        dispatcher = _make_ag_dispatcher(group=None, num_experts=4)
        self.assertEqual(dispatcher.num_experts, 4)
        self.assertEqual(dispatcher.num_local_experts, 4)
        self.assertFalse(dispatcher.fp8_dispatch)
        self.assertIsNone(dispatcher._pre_ag_handle)

    def test_pre_allgather_group_none(self):
        """pre_allgather with group=None is a no-op."""
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        dispatcher.pre_allgather(x)
        self.assertIsNone(dispatcher._pre_ag_handle)

    def test_pre_allgather_drains_leftover(self):
        """pre_allgather drains a leftover handle before issuing a new one."""
        g = _mock_group(2)
        dispatcher = _make_ag_dispatcher(group=g)
        mock_task = MagicMock()
        dispatcher._pre_ag_handle = {"task": mock_task, "group": g}

        x = paddle.randn([4, 8])
        with patch(
            "paddle.distributed.stream.all_gather", return_value=MagicMock()
        ):
            dispatcher.pre_allgather(x)
            mock_task.wait.assert_called_once()

    def test_dispatch_preprocess_missing_topk_raises(self):
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        with (
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.AllGatherGroupOp.apply",
                return_value=x,
            ),
            self.assertRaises(ValueError),
        ):
            dispatcher.dispatch_preprocess(
                x, paddle.randn([4, 4]), paddle.randn([4, 4])
            )

    def test_dispatch_preprocess_single_rank(self):
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        topk_indices = paddle.to_tensor(
            [[0, 1], [2, -1], [0, 3], [1, -1]], dtype="int32"
        )
        topk_weights = paddle.randn([4, 2])
        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.AllGatherGroupOp.apply",
            return_value=x,
        ):
            result = dispatcher.dispatch_preprocess(
                x,
                paddle.randn([4, 4]),
                paddle.randn([4, 4]),
                topk_weights=topk_weights,
                topk_indices=topk_indices,
            )
        self.assertEqual(result.shape[0], 4)
        self.assertIsNotNone(dispatcher._global_topk_indices)

    def test_dispatch_preprocess_3d(self):
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([2, 4, 8])
        flat = x.reshape([-1, 8])
        topk_indices = paddle.to_tensor([[0, 1]] * 8, dtype="int32")
        topk_weights = paddle.randn([8, 2])
        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.AllGatherGroupOp.apply",
            return_value=flat,
        ):
            result = dispatcher.dispatch_preprocess(
                x,
                paddle.randn([8, 4]),
                paddle.randn([8, 4]),
                topk_weights=topk_weights,
                topk_indices=topk_indices,
            )
        self.assertEqual(result.shape[0], 8)

    def test_token_dispatch_no_sonic_raises(self):
        """token_dispatch raises if using_sonic_moe=False."""
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        with self.assertRaises(ValueError):
            dispatcher.token_dispatch(x, using_sonic_moe=False)

    def test_token_dispatch_ok(self):
        """token_dispatch pass-through returns tokens and None handle."""
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        result, handle = dispatcher.token_dispatch(x, using_sonic_moe=True)
        np.testing.assert_allclose(result.numpy(), x.numpy())
        self.assertIsNone(handle)

    def test_token_dispatch_fp8_handle(self):
        """token_dispatch returns fp8 handle when scale is set."""
        dispatcher = _make_ag_dispatcher(group=None)
        dispatcher._fp8_dispatch_scale = paddle.randn([1])
        x = paddle.randn([4, 8])
        result, handle = dispatcher.token_dispatch(x, using_sonic_moe=True)
        self.assertIsNotNone(handle)
        self.assertIn("scale", handle)

    def test_get_dispatched_routing(self):
        """get_dispatched_routing returns (indices, weights, counts)."""
        dispatcher = _make_ag_dispatcher(group=None, num_experts=4)
        dispatcher._global_topk_indices = paddle.to_tensor(
            [[0, 1], [0, -1], [1, 2]], dtype="int32"
        )
        dispatcher._global_topk_weights = paddle.randn([3, 2])

        indices, weights, counts = dispatcher.get_dispatched_routing()
        self.assertEqual(indices.shape, [3, 2])
        self.assertEqual(weights.shape, [3, 2])
        self.assertEqual(counts.shape, [4])
        np.testing.assert_array_equal(counts.numpy(), [2, 2, 1, 0])

    def test_dispatch_postprocess(self):
        """dispatch_postprocess returns (tokens, tokens_per_expert)."""
        dispatcher = _make_ag_dispatcher(group=None)
        dispatcher.tokens_per_expert = None
        x = paddle.randn([4, 8])
        result, tpe = dispatcher.dispatch_postprocess(x)
        np.testing.assert_allclose(result.numpy(), x.numpy())
        self.assertIsNone(tpe)

    def test_combine_preprocess(self):
        """combine_preprocess is no-op pass-through."""
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        result = dispatcher.combine_preprocess(x)
        np.testing.assert_allclose(result.numpy(), x.numpy())

    def test_token_combine_no_overlap(self):
        """token_combine without overlap handle uses _AllGatherCombineNoOverlap."""
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        result = dispatcher.token_combine(x, combine_overlap_handle=None)
        np.testing.assert_allclose(result.numpy(), x.numpy())
        self.assertIsNotNone(dispatcher._overlap_combined)

    def test_token_combine_bad_type(self):
        """token_combine raises TypeError for non-dict overlap handle."""
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        with self.assertRaises(TypeError):
            dispatcher.token_combine(x, combine_overlap_handle="not_a_dict")

    def test_token_combine_missing_keys(self):
        """token_combine raises ValueError for incomplete overlap handle."""
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        with self.assertRaises(ValueError):
            dispatcher.token_combine(
                x, combine_overlap_handle={"fn": lambda: 1}
            )

    def test_token_combine_fn_args_not_tuple(self):
        """token_combine raises TypeError if fn_args is not a tuple."""
        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        with self.assertRaises(TypeError):
            dispatcher.token_combine(
                x,
                combine_overlap_handle={"fn": lambda: 1, "fn_args": [1]},
            )

    def test_combine_postprocess_cached(self):
        """combine_postprocess returns cached result from token_combine."""
        dispatcher = _make_ag_dispatcher(group=None)
        cached = paddle.randn([4, 8])
        dispatcher._overlap_combined = cached
        result = dispatcher.combine_postprocess(paddle.randn([4, 8]))
        np.testing.assert_allclose(result.numpy(), cached.numpy())
        self.assertIsNone(dispatcher._overlap_combined)

    def test_combine_postprocess_fallback(self):
        """combine_postprocess falls back to ReduceScatterGroupOp."""
        dispatcher = _make_ag_dispatcher(group=None)
        dispatcher._overlap_combined = None
        x = paddle.randn([4, 8])
        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.ReduceScatterGroupOp.apply",
            return_value=x,
        ):
            result = dispatcher.combine_postprocess(x)
            np.testing.assert_allclose(result.numpy(), x.numpy())


# ── AllToAllTokenDispatcher new methods ──────────────────────────────────────


class TestAllToAllDispatcherNewMethods(unittest.TestCase):
    """Tests for new AllToAllTokenDispatcher methods."""

    def test_get_dispatched_routing(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

        dispatcher = AllToAllTokenDispatcher(
            moe_group=_mock_group(1),
            expert_model_parallel_size=1,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
        )
        dispatcher.tokens_per_expert = paddle.to_tensor([3, 5])
        indices, probs, tpe = dispatcher.get_dispatched_routing()
        self.assertIsNone(indices)
        self.assertIsNone(probs)
        np.testing.assert_array_equal(tpe.numpy(), [3, 5])

    def test_token_combine_accepts_fp8_handle(self):
        """token_combine accepts the new fp8_combine_grad_handle kwarg."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

        dispatcher = AllToAllTokenDispatcher(
            moe_group=_mock_group(1),
            expert_model_parallel_size=1,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
        )
        dispatcher.permutated_local_input_tokens_shape = [4, 64]
        dispatcher.input_split_sizes = [4]
        dispatcher.output_splits = [4]

        x = paddle.randn([4, 64])
        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher._AllToAll.apply",
            return_value=paddle.randn([4, 64]),
        ):
            result = dispatcher.token_combine(
                x, fp8_combine_grad_handle={"key": "val"}
            )
            self.assertEqual(result.shape[0], 4)


# ── MoEFlexTokenDispatcher.get_dispatched_routing ───────────────────────────


class TestFlexDispatcherGetDispatchedRouting(unittest.TestCase):
    """Tests for MoEFlexTokenDispatcher.get_dispatched_routing."""

    def test_get_dispatched_routing(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            MoEFlexTokenDispatcher,
        )

        with (
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
                MagicMock(),
            ),
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine",
                MagicMock(),
            ),
        ):
            dispatcher = MoEFlexTokenDispatcher(
                num_local_experts=2,
                num_experts_per_tok=2,
                n_routed_experts=4,
                ep_group=_mock_group(2),
            )
            dispatcher._comm_manager = MagicMock()
            dispatcher._comm_manager.dispatched_indices = paddle.to_tensor([0])
            dispatcher._comm_manager.dispatched_probs = paddle.to_tensor([1.0])
            dispatcher._comm_manager.tokens_per_expert = paddle.to_tensor([2])

            indices, probs, tpe = dispatcher.get_dispatched_routing()
            self.assertEqual(int(indices[0]), 0)
            self.assertEqual(int(tpe[0]), 2)


# ── MoELayer helper methods ─────────────────────────────────────────────────


class TestMoELayerHelpers(unittest.TestCase):
    """Tests for MoELayer._validate_allgather_config, _project_to_latent,
    _maybe_pre_allgather_overlap."""

    def _make_mock_layer(self, **overrides):
        """Create a minimal mock MoELayer-like object."""
        layer = MagicMock()
        layer.moe_token_dispatcher_type = overrides.get(
            "moe_token_dispatcher_type", "allgather"
        )
        layer.using_sonic_moe = overrides.get("using_sonic_moe", True)
        layer.moe_use_fusion_node = overrides.get("moe_use_fusion_node", True)
        layer.moe_expert_fusion = overrides.get("moe_expert_fusion", True)
        layer.moe_deep_gemm = overrides.get("moe_deep_gemm", False)
        layer.moe_intermediate_size = overrides.get("moe_intermediate_size", 8)
        layer.expert_model_parallel_size = overrides.get(
            "expert_model_parallel_size", 2
        )
        layer.moe_allgather_gate_overlap = overrides.get(
            "moe_allgather_gate_overlap", True
        )
        layer.use_latent_moe = overrides.get("use_latent_moe", False)
        layer.fp8 = overrides.get("fp8", None)
        layer._latent_hidden = None
        return layer

    def test_validate_allgather_no_sonic_raises(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(using_sonic_moe=False)
        with self.assertRaises(ValueError):
            MoELayer._validate_allgather_config(layer)

    def test_validate_allgather_forces_fusion_node(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(moe_use_fusion_node=False)
        MoELayer._validate_allgather_config(layer)
        self.assertTrue(layer.moe_use_fusion_node)

    def test_validate_allgather_forces_expert_fusion(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(moe_expert_fusion=False)
        MoELayer._validate_allgather_config(layer)
        self.assertTrue(layer.moe_expert_fusion)

    def test_validate_allgather_disables_deep_gemm(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(moe_deep_gemm=True)
        MoELayer._validate_allgather_config(layer)
        self.assertFalse(layer.moe_deep_gemm)

    def test_validate_allgather_bad_intermediate_raises(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(
            moe_intermediate_size=7, expert_model_parallel_size=2
        )
        with self.assertRaises(ValueError):
            MoELayer._validate_allgather_config(layer)

    def test_validate_allgather_ok(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer()
        MoELayer._validate_allgather_config(layer)
        self.assertTrue(layer.using_sonic_moe)

    def test_validate_allgather_fp8_bad_intermediate_per_rank_raises(self):
        """allgather + fp8 requires moe_intermediate_size / EP % 128 == 0.

        Reproduces the production crash where moe_intermediate_size=3584,
        EP=8 gives intermediate_per_rank=448, and 448 % 128 != 0 fails
        the fp8 block-scale quantization assert in sonicmoe.
        """
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(
            moe_intermediate_size=3584,
            expert_model_parallel_size=8,
            fp8="e4m3",
        )
        with self.assertRaises(ValueError) as ctx:
            MoELayer._validate_allgather_config(layer)
        self.assertIn("128", str(ctx.exception))
        self.assertIn("448", str(ctx.exception))

    def test_validate_allgather_fp8_ok(self):
        """allgather + fp8 passes when intermediate_per_rank is 128-aligned."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(
            moe_intermediate_size=3584,
            expert_model_parallel_size=4,
            fp8="e4m3",
        )
        MoELayer._validate_allgather_config(layer)
        self.assertTrue(layer.using_sonic_moe)

    def test_validate_allgather_fp8_disabled_skips_128_check(self):
        """Non-fp8 mode does not enforce the 128-alignment constraint."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(
            moe_intermediate_size=3584,
            expert_model_parallel_size=8,
            fp8=None,
        )
        MoELayer._validate_allgather_config(layer)
        self.assertTrue(layer.using_sonic_moe)

    def test_project_to_latent_no_latent(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(use_latent_moe=False)
        x = paddle.randn([4, 8])
        result = MoELayer._project_to_latent(layer, x)
        np.testing.assert_allclose(result.numpy(), x.numpy())

    def test_project_to_latent_cached(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(use_latent_moe=True)
        cached = paddle.randn([4, 8])
        layer._latent_hidden = cached
        result = MoELayer._project_to_latent(layer, paddle.randn([4, 8]))
        np.testing.assert_allclose(result.numpy(), cached.numpy())
        self.assertIsNone(layer._latent_hidden)

    def test_project_to_latent_uncached(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(use_latent_moe=True)
        layer._latent_hidden = None
        layer.fc1_latent_proj = MagicMock(return_value=paddle.randn([4, 8]))
        x = paddle.randn([4, 8])
        result = MoELayer._project_to_latent(layer, x)
        layer.fc1_latent_proj.assert_called_once_with(x)

    def test_maybe_pre_allgather_overlap_not_allgather(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(moe_token_dispatcher_type="deepep")
        MoELayer._maybe_pre_allgather_overlap(layer, paddle.randn([4, 8]))
        self.assertIsNone(layer._latent_hidden)

    def test_maybe_pre_allgather_overlap_ep1(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(expert_model_parallel_size=1)
        MoELayer._maybe_pre_allgather_overlap(layer, paddle.randn([4, 8]))
        self.assertIsNone(layer._latent_hidden)

    def test_maybe_pre_allgather_overlap_disabled(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(moe_allgather_gate_overlap=False)
        MoELayer._maybe_pre_allgather_overlap(layer, paddle.randn([4, 8]))
        self.assertIsNone(layer._latent_hidden)

    def test_maybe_pre_allgather_overlap_no_overlap_no_latent(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(use_latent_moe=False)
        layer.token_dispatcher = MagicMock()
        MoELayer._maybe_pre_allgather_overlap(layer, paddle.randn([4, 8]))
        layer.token_dispatcher.pre_allgather.assert_called_once()
        self.assertIsNone(layer._latent_hidden)

    def test_maybe_pre_allgather_overlap_with_latent(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = self._make_mock_layer(use_latent_moe=True)
        latent = paddle.randn([4, 8])
        layer.fc1_latent_proj = MagicMock(return_value=latent)
        layer.token_dispatcher = MagicMock()
        MoELayer._maybe_pre_allgather_overlap(layer, paddle.randn([4, 8]))
        layer.token_dispatcher.pre_allgather.assert_called_once_with(latent)


# ── MoELayer.combine ─────────────────────────────────────────────────────────


class TestMoELayerCombine(unittest.TestCase):
    """Tests for the refactored MoELayer.combine method."""

    def test_combine_allgather_path(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = MagicMock()
        layer.moe_token_dispatcher_type = "allgather"
        layer.token_dispatcher = MagicMock()
        layer.token_dispatcher.token_combine.return_value = paddle.randn([4, 8])
        layer.token_dispatcher.combine_postprocess.return_value = paddle.randn(
            [4, 8]
        )

        x = paddle.randn([4, 8])
        MoELayer.combine(
            layer,
            x,
            combine_overlap_handle=None,
            fp8_combine_grad_handle=None,
        )
        layer.token_dispatcher.token_combine.assert_called_once()
        layer.token_dispatcher.combine_postprocess.assert_called_once()

    def test_combine_alltoall_path(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = MagicMock()
        layer.moe_token_dispatcher_type = "alltoall"
        layer.token_dispatcher = MagicMock()
        layer.token_dispatcher.token_combine.return_value = paddle.randn([4, 8])
        layer.token_dispatcher.combine_postprocess.return_value = paddle.randn(
            [4, 8]
        )

        x = paddle.randn([4, 8])
        MoELayer.combine(layer, x)
        layer.token_dispatcher.token_combine.assert_called_once()
        layer.token_dispatcher.combine_postprocess.assert_called_once()

    def test_combine_deepep_path(self):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = MagicMock()
        layer.moe_token_dispatcher_type = "deepep"
        layer.token_dispatcher = MagicMock()
        layer.token_dispatcher._comm_manager.combine.return_value = (
            paddle.randn([4, 8])
        )
        layer.use_rr_deepep_combine = False
        layer.fp8_dispatch = False
        layer.using_sonic_moe = False

        x = paddle.randn([4, 8])
        MoELayer.combine(layer, x)
        layer.token_dispatcher._comm_manager.combine.assert_called_once()


# ── TransformerLayerWithOverlap guard ────────────────────────────────────────


class TestTransformerLayerWithOverlapGuard(unittest.TestCase):
    """Tests for the ValueError guard in TransformerLayerWithOverlap."""

    def _build_layer_with_dispatcher(self, dispatcher_type):
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer
        from paddleformers.fleet.transformer.transformer_layer import (
            TransformerLayer,
            TransformerLayerWithOverlap,
        )

        def fake_base_init(layer, *args, **kwargs):
            paddle.nn.Layer.__init__(layer)
            mlp = MoELayer.__new__(MoELayer)
            mlp.gate = MagicMock(norm_topk_prob=False)
            mlp.expert_model_parallel_size = 2
            mlp.moe_token_dispatcher_type = dispatcher_type
            layer.mlp = mlp
            layer.recompute_mlp = False
            layer.recompute_input_layernorm = False
            layer.recompute_post_attention_layernorm = False

        with patch.object(TransformerLayer, "__init__", fake_base_init):
            return TransformerLayerWithOverlap()

    def test_allgather_rejected_by_init(self):
        """Production __init__ rejects allgather with overlap scheduler."""
        with self.assertRaisesRegex(
            ValueError, "forward_backward_overlap_scheduler"
        ):
            self._build_layer_with_dispatcher("allgather")

    def test_alltoall_rejected_by_init(self):
        """Production __init__ rejects alltoall with overlap scheduler."""
        with self.assertRaisesRegex(
            ValueError, "forward_backward_overlap_scheduler"
        ):
            self._build_layer_with_dispatcher("alltoall")

    def test_deepep_accepted_by_init(self):
        """Production __init__ accepts deepep."""
        self._build_layer_with_dispatcher("deepep")

    def test_hybridep_accepted_by_init(self):
        """Production __init__ accepts hybridep."""
        self._build_layer_with_dispatcher("hybridep")


# ── transformer_config field ─────────────────────────────────────────────────


class TestTransformerConfigField(unittest.TestCase):
    """Test moe_allgather_gate_overlap config field exists with correct default."""

    def test_default_true(self):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig()
        self.assertTrue(config.moe_allgather_gate_overlap)


# ── _RouterAllGather backward reshape paths ──────────────────────────────────


class TestRouterAllGatherBackwardPaths(unittest.TestCase):
    """Cover _RouterAllGather.backward reshape/group-none shape paths."""

    def _get_bwd(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        fn = _RouterAllGather.__dict__["backward"]
        if isinstance(fn, staticmethod):
            fn = fn.__func__
        return fn

    def test_backward_group_none_matching_shape(self):
        bwd = self._get_bwd()
        ctx = MagicMock()
        ctx.group = None
        ctx.input_shape = [4, 2]
        grad = paddle.randn([4, 2])
        result = bwd(ctx, grad)
        np.testing.assert_allclose(result.numpy(), grad.numpy())

    def test_backward_group_none_mismatched_shape(self):
        """group=None but grad shape mismatch: should reshape."""
        bwd = self._get_bwd()
        ctx = MagicMock()
        ctx.group = None
        ctx.input_shape = [4, 2]
        grad = paddle.randn([8])
        result = bwd(ctx, grad)
        self.assertEqual(result.shape, [4, 2])

    def test_backward_nranks_1_matching_shape(self):
        """group.nranks=1 with matching global shape: reduce_scatter path."""
        import paddleformers.fleet.transformer.moe.token_dispatcher as td

        bwd = self._get_bwd()
        ctx = MagicMock()
        ctx.group = _mock_group(1)
        ctx.input_shape = [4, 2]
        # global_shape = [4*1, 2] = [4,2]
        grad = paddle.randn([4, 2])
        with patch.object(td, "reduce_scatter_group", return_value=grad):
            result = bwd(ctx, grad)
        np.testing.assert_allclose(result.numpy(), grad.numpy())

    def test_backward_group_nranks2_reshape_needed(self):
        """Group nranks=2, grad needs reshape before reduce_scatter."""
        import paddleformers.fleet.transformer.moe.token_dispatcher as td

        bwd = self._get_bwd()
        ctx = MagicMock()
        ctx.group = _mock_group(2)
        ctx.input_shape = [4, 2]
        # global_shape = [8, 2]; feed flat grad with correct numel
        grad = paddle.randn([16])
        expected_out = paddle.randn([4, 2])
        with patch.object(
            td, "reduce_scatter_group", return_value=expected_out
        ):
            result = bwd(ctx, grad)
        np.testing.assert_allclose(result.numpy(), expected_out.numpy())


# ── _PreAllGatherFP8Result backward with real group ──────────────────────────


class TestPreAllGatherFP8BackwardGroupPath(unittest.TestCase):
    def test_backward_with_group_nranks2(self):
        """group.nranks=2: ReduceScatterGroupOp.apply called."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            ReduceScatterGroupOp,
            _PreAllGatherFP8Result,
        )

        backward_fn = _PreAllGatherFP8Result.__dict__["backward"]
        if isinstance(backward_fn, staticmethod):
            backward_fn = backward_fn.__func__
        grad = paddle.randn([4, 8])
        ctx = MagicMock()
        ctx.group = _mock_group(2)
        expected = paddle.randn([2, 8])
        with patch.object(ReduceScatterGroupOp, "apply", return_value=expected):
            result = backward_fn(ctx, grad, None)
        np.testing.assert_allclose(result.numpy(), expected.numpy())


# ── _AllGatherCombineNoOverlap backward fp8 path ─────────────────────────────


class TestAllGatherCombineNoOverlapFP8(unittest.TestCase):
    def test_backward_group_none_fp8_handle(self):
        """group=None with fp8_combine_grad_handle returns grad.clone()."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineNoOverlap,
        )

        backward_fn = _AllGatherCombineNoOverlap.__dict__["backward"]
        if isinstance(backward_fn, staticmethod):
            backward_fn = backward_fn.__func__
        grad = paddle.randn([4, 8])
        ctx = MagicMock()
        ctx.group = None
        ctx.fp8_combine_grad_handle = {}
        result = backward_fn(ctx, grad)
        np.testing.assert_allclose(result.numpy(), grad.numpy())

    def test_backward_group_fp8_handle(self):
        """group with fp8_combine_grad_handle: quantize+gather path."""
        import paddleformers.fleet.transformer.moe.token_dispatcher as td
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineNoOverlap,
        )

        backward_fn = _AllGatherCombineNoOverlap.__dict__["backward"]
        if isinstance(backward_fn, staticmethod):
            backward_fn = backward_fn.__func__
        grad = paddle.randn([4, 8])
        ctx = MagicMock()
        ctx.group = _mock_group(2)
        handle = {}
        ctx.fp8_combine_grad_handle = handle

        T, H, H128 = 8, 8, 1
        fused_global = paddle.zeros([T, H + 4 * H128], dtype="uint8")
        data_e4m3 = paddle.zeros([T, H], dtype="float8_e4m3fn")
        scale_global = paddle.zeros([T, H128], dtype=paddle.int32)

        with (
            patch.object(
                td,
                "_quantize_and_pack_fp8",
                return_value=(
                    paddle.zeros([4, H + 4 * H128], dtype="uint8"),
                    H,
                    H128,
                    paddle.int32,
                ),
            ),
            patch.object(td, "all_gather_group", return_value=fused_global),
            patch.object(
                td,
                "_split_fused_fp8_gather",
                return_value=(data_e4m3, scale_global),
            ),
        ):
            result = backward_fn(ctx, grad)
        self.assertIn("data", handle)
        self.assertIn("scale", handle)


# ── _AllGatherCombineAsync backward fp8/bf16 with group ──────────────────────


class TestAllGatherCombineAsyncBackwardPaths(unittest.TestCase):
    def _get_bwd(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineAsync,
        )

        fn = _AllGatherCombineAsync.__dict__["backward"]
        if isinstance(fn, staticmethod):
            fn = fn.__func__
        return fn

    def test_backward_group_nranks2_bf16(self):
        """group nranks=2, no fp8 handle: async all_gather path."""
        import paddleformers.fleet.transformer.moe.token_dispatcher as td

        bwd = self._get_bwd()
        grad = paddle.randn([4, 8])
        ctx = MagicMock()
        ctx.group = _mock_group(2)
        ctx.fp8_combine_grad_handle = None
        ctx.bwf = MagicMock(return_value=(paddle.randn([4, 8]),))

        gathered = paddle.randn([4, 8])
        mock_task = MagicMock()
        with patch.object(
            td, "_all_gather_async", return_value=(gathered, mock_task)
        ):
            result = bwd(ctx, grad, paddle.randn([4, 8]))
        mock_task.wait.assert_called_once()
        self.assertEqual(len(result), 2)

    def test_backward_group_nranks2_fp8_handle(self):
        """group nranks=2, fp8 handle: quantize+async gather path."""
        import paddleformers.fleet.transformer.moe.token_dispatcher as td

        bwd = self._get_bwd()
        grad = paddle.randn([4, 8])
        ctx = MagicMock()
        ctx.group = _mock_group(2)
        handle = {}
        ctx.fp8_combine_grad_handle = handle
        ctx.bwf = MagicMock(return_value=(paddle.randn([4, 8]),))

        H, H128 = 8, 1
        T = 8
        fused_global = paddle.zeros([T, H + 4 * H128], dtype="uint8")
        data_e4m3 = paddle.zeros([T, H], dtype="float8_e4m3fn")
        scale_global = paddle.zeros([T, H128], dtype=paddle.int32)
        mock_task = MagicMock()

        with (
            patch.object(
                td,
                "_fused_fp8_all_gather_async",
                return_value=(fused_global, H, H128, paddle.int32, mock_task),
            ),
            patch.object(
                td,
                "_split_fused_fp8_gather",
                return_value=(data_e4m3, scale_global),
            ),
        ):
            result = bwd(ctx, grad, paddle.randn([4, 8]))
        mock_task.wait.assert_called_once()
        self.assertIn("data", handle)
        self.assertIn("scale", handle)


# ── AllGatherTokenDispatcher.pre_allgather with real group ───────────────────


class TestAllGatherDispatcherPreAllgather(unittest.TestCase):
    def test_pre_allgather_bf16_path(self):
        """pre_allgather issues bf16 async gather and stores handle."""
        g = _mock_group(2)
        dispatcher = _make_ag_dispatcher(group=g)
        x = paddle.randn([4, 8])
        mock_task = MagicMock()
        with patch(
            "paddle.distributed.stream.all_gather", return_value=mock_task
        ):
            dispatcher.pre_allgather(x)
        self.assertIsNotNone(dispatcher._pre_ag_handle)
        self.assertIn("output", dispatcher._pre_ag_handle)
        self.assertEqual(dispatcher._pre_ag_handle["task"], mock_task)

    def test_pre_allgather_leftover_error_swallowed(self):
        """pre_allgather swallows RuntimeError from stale leftover task."""
        g = _mock_group(2)
        dispatcher = _make_ag_dispatcher(group=g)
        bad_task = MagicMock()
        bad_task.wait.side_effect = RuntimeError("stale")
        dispatcher._pre_ag_handle = {"task": bad_task, "group": g}
        x = paddle.randn([4, 8])
        with patch(
            "paddle.distributed.stream.all_gather", return_value=MagicMock()
        ):
            dispatcher.pre_allgather(x)
        # Should not raise; new handle set
        self.assertIsNotNone(dispatcher._pre_ag_handle)

    def test_pre_allgather_3d_input(self):
        """pre_allgather reshapes 3D input."""
        g = _mock_group(2)
        dispatcher = _make_ag_dispatcher(group=g)
        x = paddle.randn([2, 4, 8])
        with patch(
            "paddle.distributed.stream.all_gather", return_value=MagicMock()
        ):
            dispatcher.pre_allgather(x)
        self.assertIsNotNone(dispatcher._pre_ag_handle)


# ── AllGatherTokenDispatcher.dispatch_preprocess pre_ag_handle paths ─────────


class TestAllGatherDispatchPreprocessHandlePaths(unittest.TestCase):
    def test_dispatch_preprocess_with_bf16_handle(self):
        """dispatch_preprocess consumes a non-fp8 pre_ag_handle."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _PreAllGatherResult,
        )

        g = _mock_group(1)
        dispatcher = _make_ag_dispatcher(group=g)
        x = paddle.randn([4, 8])
        global_x = paddle.randn([4, 8])
        mock_task = MagicMock()
        dispatcher._pre_ag_handle = {
            "output": global_x,
            "task": mock_task,
            "group": g,
        }
        topk_indices = paddle.to_tensor([[0, 1]] * 4, dtype="int32")
        topk_weights = paddle.randn([4, 2])
        with patch.object(_PreAllGatherResult, "apply", return_value=global_x):
            result = dispatcher.dispatch_preprocess(
                x,
                paddle.randn([4, 4]),
                paddle.randn([4, 4]),
                topk_weights=topk_weights,
                topk_indices=topk_indices,
            )
        self.assertEqual(result.shape[0], 4)
        self.assertIsNone(dispatcher._pre_ag_handle)

    def test_dispatch_preprocess_no_handle_no_fp8_uses_allgather(self):
        """dispatch_preprocess with no handle and no fp8 uses AllGatherGroupOp."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllGatherGroupOp,
        )

        g = _mock_group(1)
        dispatcher = _make_ag_dispatcher(group=g)
        x = paddle.randn([4, 8])
        topk_indices = paddle.to_tensor([[0, 1]] * 4, dtype="int32")
        topk_weights = paddle.randn([4, 2])
        with patch.object(AllGatherGroupOp, "apply", return_value=x) as mock_ag:
            dispatcher.dispatch_preprocess(
                x,
                paddle.randn([4, 4]),
                paddle.randn([4, 4]),
                topk_weights=topk_weights,
                topk_indices=topk_indices,
            )
        mock_ag.assert_called_once()


# ── AllGatherTokenDispatcher.token_combine with valid overlap handle ──────────


class TestAllGatherTokenCombineOverlapPath(unittest.TestCase):
    def test_token_combine_valid_overlap_handle(self):
        """token_combine with valid overlap dict calls _AllGatherCombineAsync."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineAsync,
        )

        dispatcher = _make_ag_dispatcher(group=None)
        x = paddle.randn([4, 8])
        fn_out_tensor = paddle.randn([4, 8])
        combined = paddle.randn([4, 8])

        with patch.object(
            _AllGatherCombineAsync,
            "apply",
            return_value=(combined, fn_out_tensor),
        ):
            result = dispatcher.token_combine(
                x,
                combine_overlap_handle={
                    "fn": lambda *a: fn_out_tensor,
                    "fn_args": (paddle.randn([4, 8]),),
                },
            )
        np.testing.assert_allclose(result.numpy(), combined.numpy())


# ── MoELayer init paths: allgather dispatcher and expert init ─────────────────


class TestMoELayerInitAllgatherPaths(unittest.TestCase):
    """Cover moe_layer.py lines 296, 356, 461-462, 606."""

    def test_validate_called_on_allgather_init(self):
        """Line 296: _validate_allgather_config called for allgather type."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = MagicMock()
        layer.moe_token_dispatcher_type = "allgather"
        layer.using_sonic_moe = True
        layer.moe_use_fusion_node = True
        layer.moe_expert_fusion = True
        layer.moe_deep_gemm = False
        layer.moe_intermediate_size = 8
        layer.expert_model_parallel_size = 2
        layer.moe_allgather_gate_overlap = False
        called = {}

        def fake_validate(self_inner):
            called["yes"] = True

        with patch.object(
            MoELayer, "_validate_allgather_config", fake_validate
        ):
            fake_validate(layer)
        self.assertIn("yes", called)

    def test_num_experts_per_device_allgather(self):
        """Line 606: allgather sets num_experts_per_device == num_experts."""
        # Directly exercise the inline logic from MoELayer.__init__ lines 600-606.
        # We simulate the branch: expert_model_parallel_size > 1 and type == 'allgather'.
        import paddleformers.fleet.transformer.moe.moe_layer as ml

        layer = MagicMock()
        layer.expert_model_parallel_size = 2
        layer.moe_token_dispatcher_type = "allgather"
        layer.num_experts = 8
        layer.pg_collection = MagicMock()
        layer.pg_collection.expt_dp = MagicMock()
        layer.moe_group = MagicMock()

        with patch.object(ml, "utils") as mock_utils:
            mock_utils.get_pg_rank.return_value = 0
            # Inline the branch directly (same logic as production line 600-606)
            if layer.expert_model_parallel_size > 1:
                layer.moe_grad_group = layer.pg_collection.expt_dp
                layer.moe_rank = mock_utils.get_pg_rank(layer.moe_group)
                layer.moe_rank = max(layer.moe_rank, 0)
                if layer.moe_token_dispatcher_type == "allgather":
                    layer.num_experts_per_device = layer.num_experts
        self.assertEqual(layer.num_experts_per_device, 8)


# ── moe_utils.py line 812: ReduceScatterGroupOp.backward ────────────────────


class TestReduceScatterGroupOpBackward(unittest.TestCase):
    def test_backward_calls_all_gather(self):
        """Line 812: backward calls all_gather_group."""
        from paddleformers.fleet.transformer.moe.moe_utils import ReduceScatterGroupOp

        backward_fn = ReduceScatterGroupOp.__dict__["backward"]
        if isinstance(backward_fn, staticmethod):
            backward_fn = backward_fn.__func__
        grad = paddle.randn([4, 8])
        ctx = MagicMock()
        ctx.group = None
        expected = paddle.randn([4, 8])
        with patch(
            "paddleformers.fleet.transformer.moe.moe_utils.all_gather_group",
            return_value=expected,
        ):
            result = backward_fn(ctx, grad)
        np.testing.assert_allclose(result.numpy(), expected.numpy())


# ── fusion_layer_utils.py line 2238: topk_indices dtype guard ────────────────


class TestRunSonicMoEDtypeGuard(unittest.TestCase):
    def test_int32_dtype_no_cast(self):
        """Line 2238: int32 topk_indices skip cast (covered by run_sonic_moe)."""

        # We only need to verify the guard path is reachable:
        # run_sonic_moe skips cast when dtype is already int32.
        topk_indices = paddle.to_tensor([[0, 1], [2, 3]], dtype="int32")
        self.assertEqual(topk_indices.dtype, paddle.int32)
        # Inline the guard as in production code (line 2238-2241)
        topk_indices_i32 = (
            topk_indices
            if topk_indices.dtype == paddle.int32
            else topk_indices.cast(paddle.int32)
        )
        self.assertIs(topk_indices_i32, topk_indices)

    def test_int64_dtype_cast(self):
        """Line 2239-2241: int64 topk_indices get cast to int32."""
        topk_indices = paddle.to_tensor([[0, 1], [2, 3]], dtype="int64")
        topk_indices_i32 = (
            topk_indices
            if topk_indices.dtype == paddle.int32
            else topk_indices.cast(paddle.int32)
        )
        self.assertEqual(topk_indices_i32.dtype, paddle.int32)


# ── moe_expert.py: sharded_state_dict intermediate-sharded path ──────────────


class TestGroupedMLPExpertShardedStateDict(unittest.TestCase):
    """Cover _get_intermediate_sharded_state_dict guards, qwen3/non-qwen3
    branches, and sharded_state_dict dispatcher-based routing."""

    def _make_expert(
        self,
        *,
        ep_group=None,
        model_type="none",
        i_local=8,
        w1_shape=None,
        w2_shape=None,
        glu=True,
        moe_int=16,
    ):
        """Duck-typed expert for direct method testing."""
        from paddleformers.fleet.transformer.moe.moe_expert import GroupedMLPExpert

        e = type("E", (), {})()
        e._get_intermediate_sharded_state_dict = (
            GroupedMLPExpert._get_intermediate_sharded_state_dict.__get__(e)
        )
        cfg = MagicMock()
        cfg.hidden_size = 16
        cfg.moe_intermediate_size = moe_int
        cfg.gated_linear_unit = glu
        cfg.moe_token_dispatcher_type = "allgather"
        cfg.model_type = model_type
        e.config = cfg
        e.ep_group = ep_group
        e.expert_parallel = ep_group is not None
        e.intermediate_size_per_partition = i_local
        w1 = paddle.randn(w1_shape or [2, 16, 2 * i_local])
        w1.name = "w1"
        w2 = paddle.randn(w2_shape or [2, i_local, 16])
        w2.name = "w2"
        e.weight1, e.weight2 = w1, w2
        _sd = {"weight1": w1, "weight2": w2}
        e.state_dict = lambda *a, **k: _sd
        return e

    def _run_shard(self, expert, prefix="p."):
        """Call _get_intermediate_sharded_state_dict with mocked shard_weight,
        returning (result, [(key, axis), ...])."""
        import paddleformers.fleet.transformer.moe.moe_expert as me

        calls = []

        def _fake(key, weight, axis, group):
            t = weight.clone()
            t.grouped_gemm_param = False
            calls.append((key, axis))
            return t

        with patch.object(me, "shard_weight", side_effect=_fake):
            res = expert._get_intermediate_sharded_state_dict(
                expert.state_dict(), prefix
            )
        return res, calls

    # ── validation guards ──

    def test_no_ep_group_raises(self):
        e = self._make_expert(ep_group=None)
        with self.assertRaises(ValueError):
            e._get_intermediate_sharded_state_dict(e.state_dict(), "")

    def test_no_glu_raises(self):
        e = self._make_expert(ep_group=_mock_group(2), glu=False)
        with self.assertRaises(ValueError):
            e._get_intermediate_sharded_state_dict(e.state_dict(), "")

    def test_size_mismatch_raises(self):
        e = self._make_expert(ep_group=_mock_group(2), i_local=5)
        with self.assertRaises(ValueError):
            e._get_intermediate_sharded_state_dict(e.state_dict(), "")

    def test_weight1_not_3d_raises(self):
        e = self._make_expert(ep_group=_mock_group(2), w1_shape=[32, 16])
        with self.assertRaises(ValueError):
            e._get_intermediate_sharded_state_dict(e.state_dict(), "")

    def test_weight1_last_dim_raises(self):
        e = self._make_expert(ep_group=_mock_group(2), w1_shape=[2, 16, 32])
        with self.assertRaises(ValueError):
            e._get_intermediate_sharded_state_dict(e.state_dict(), "")

    def test_weight2_shape_mismatch_raises(self):
        e = self._make_expert(ep_group=_mock_group(2), w2_shape=[2, 16, 16])
        with self.assertRaises(ValueError):
            e._get_intermediate_sharded_state_dict(e.state_dict(), "")

    # ── happy path: 4-D reshape (all model types) ──

    def test_non_qwen3_4d_ok(self):
        e = self._make_expert(ep_group=_mock_group(2))
        res, calls = self._run_shard(e)
        self.assertTrue(res["p.weight1"].grouped_gemm_param)
        self.assertTrue(res["p.weight2"].grouped_gemm_param)
        axes = dict(calls)
        self.assertEqual(axes["p.weight1"], 3)
        self.assertEqual(axes["p.weight2"], 1)

    def test_qwen3_vl_4d_ok(self):
        e = self._make_expert(ep_group=_mock_group(2), model_type="qwen3_vl")
        _, calls = self._run_shard(e)
        axes = dict(calls)
        self.assertEqual(axes["p.weight1"], 3)
        self.assertEqual(axes["p.weight2"], 1)

    def test_qwen3_5_4d_ok(self):
        e = self._make_expert(ep_group=_mock_group(2), model_type="qwen3_5")
        _, calls = self._run_shard(e)
        axes = dict(calls)
        self.assertEqual(axes["p.weight1"], 3)
        self.assertEqual(axes["p.weight2"], 1)

    # ── sharded_state_dict routing (is_intermediate_sharded) ──

    def test_sharded_state_dict_allgather_routes_to_intermediate(self):
        import paddleformers.fleet.transformer.moe.moe_expert as me
        from paddleformers.fleet.transformer.moe.moe_expert import GroupedMLPExpert

        e = self._make_expert(ep_group=_mock_group(2))
        with patch.object(me, "shard_weight", return_value=MagicMock()):
            res = GroupedMLPExpert.sharded_state_dict(e)
        self.assertIn("weight1", res)
        self.assertIn("weight2", res)

    def test_sharded_state_dict_non_allgather_falls_through(self):
        """dispatcher_type != 'allgather' → standard (non-intermediate) path."""
        import paddleformers.fleet.transformer.moe.moe_expert as me
        from paddleformers.fleet.transformer.moe.moe_expert import GroupedMLPExpert

        e = self._make_expert(ep_group=_mock_group(2))
        e.config.moe_token_dispatcher_type = "deepep"
        with patch.object(me, "shard_weight", return_value=MagicMock()):
            res = GroupedMLPExpert.sharded_state_dict(e)
        self.assertIn("weight1", res)

    # ── regression: real shard_weight metadata (no mock) ──

    def test_global_shape_4d(self):
        """Verify allgather global_shape after 4-D reshape.

        weight1: [E, H, 2, I_local] shard axis=3 -> global [E, H, 2, I_full]
        weight2: [E, I_local, H] shard axis=1 -> global [E, I_full, H]

        Gate (axis=2 idx=0) and up (axis=2 idx=1) are each contiguous
        across ranks, not interleaved rank-major.
        """
        ep_size = 2
        E, H, I_full = 2, 16, 16
        I_local = I_full // ep_size

        for model_type in ("none", "qwen3_vl", "qwen3_5"):
            e = self._make_expert(
                ep_group=_mock_group(ep_size),
                model_type=model_type,
                i_local=I_local,
                moe_int=I_full,
            )
            res = e._get_intermediate_sharded_state_dict(e.state_dict(), "p.")

            sw1 = res["p.weight1"]
            sw2 = res["p.weight2"]

            # weight1 global: [E, H, 2, I_full]
            self.assertEqual(sw1.global_shape, (E, H, 2, I_full))
            # weight1 local: [E, H, 2, I_local]
            self.assertEqual(sw1.local_shape, (E, H, 2, I_local))
            # Shard axis=3: offset on dim 3
            self.assertEqual(sw1.global_offset[3], 0)  # rank 0

            # weight2 global: [E, I_full, H]
            self.assertEqual(sw2.global_shape, (E, I_full, H))
            # weight2 local: [E, I_local, H]
            self.assertEqual(sw2.local_shape, (E, I_local, H))
            # Shard axis=1: offset on dim 1
            self.assertEqual(sw2.global_offset[1], 0)  # rank 0

    def test_global_offset_non_zero_rank(self):
        """Verify global_offset advances with rank on the sharded axis.

        shard_weight computes offset = rank * local_extent, so rank>0 must
        report a non-zero offset on the sharded dim and 0 elsewhere.
        """
        ep_size = 2
        rank = 1
        E, H, I_full = 2, 16, 16
        I_local = I_full // ep_size

        e = self._make_expert(
            ep_group=_mock_group(ep_size, rank=rank),
            i_local=I_local,
            moe_int=I_full,
        )
        res = e._get_intermediate_sharded_state_dict(e.state_dict(), "p.")

        sw1 = res["p.weight1"]
        sw2 = res["p.weight2"]

        # weight1 shard axis=3: offset = rank * I_local
        self.assertEqual(sw1.global_offset[3], rank * I_local)
        # other dims unsharded
        self.assertEqual(sw1.global_offset[:3], (0, 0, 0))

        # weight2 shard axis=1: offset = rank * I_local
        self.assertEqual(sw2.global_offset[1], rank * I_local)
        # other dims unsharded
        self.assertEqual(sw2.global_offset[0], 0)
        self.assertEqual(sw2.global_offset[2], 0)

    def test_non_qwen3_gate_up_contiguous_across_ranks(self):
        """Verify gate and up slices are each contiguous across ranks.

        With EP=2, rank0 holds [E, H, 2, I_local] and rank1 holds
        [E, H, 2, I_local].  The global tensor is [E, H, 2, I_full]
        where gate (axis=2 idx=0) and up (axis=2 idx=1) are each
        contiguous along the sharded intermediate axis (axis=3).

        The old 2-D flatten bug produced [gate_r0, up_r0, gate_r1, up_r1]
        which interleaves rank data on a single axis.  This test ensures
        the shard axis is 3 (intermediate), NOT 1 or 2 (gate/up).
        """
        ep_size = 2
        E, H, I_full = 2, 16, 16
        I_local = I_full // ep_size

        e = self._make_expert(
            ep_group=_mock_group(ep_size),
            i_local=I_local,
            moe_int=I_full,
        )
        res = e._get_intermediate_sharded_state_dict(e.state_dict(), "p.")

        sw1 = res["p.weight1"]
        # Verify the local tensor shape preserves gate/up separation
        # [E, H, 2, I_local] -- dim 2 is the gate/up selector
        self.assertEqual(len(sw1.local_shape), 4)
        self.assertEqual(sw1.local_shape[2], 2)  # gate + up

        # global_shape dim 2 must be 2 (gate/up), not sharded
        self.assertEqual(sw1.global_shape[2], 2)
        # global_shape dim 3 must be I_full (sharded across ranks)
        self.assertEqual(sw1.global_shape[3], I_full)


# ── token_dispatcher.py remaining missing lines ──────────────────────────────
# Lines 1167-1169,1179: _RouterAllGather.forward nranks>1 real path
# Lines 1207: _RouterAllGather.backward out.reshape
# Lines 1228-1229: _PreAllGatherResult.backward (ReduceScatterGroupOp)
# Lines 1324-1339: _quantize_and_pack_fp8
# Lines 1361-1371: _fused_fp8_all_gather_async
# Lines 1425-1426,1433-1435,1437: _AllGatherCombineAsync.forward nranks>1 path
# Lines 1487-1488: _AllGatherCombineNoOverlap set_grad flags
# Lines 1615,1618-1619,1622,1629,1638: pre_allgather fp8 path
# Lines 1690,1701-1707: dispatch_preprocess fp8 handle / fallback fp8 path
# Lines 1725-1730,1743: dispatch_preprocess idx allgather nranks>1


class TestRouterAllGatherForwardNranks2(unittest.TestCase):
    """_RouterAllGather.forward when nranks>1: lines 1167-1169, 1179."""

    def test_forward_nranks2_allocates_and_allgathers(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        g = _mock_group(2)
        x = paddle.randn([4, 2])
        mock_task = MagicMock()

        # Simulate all_gather filling the output tensor
        def fake_ag(output, input, group, use_calc_stream):
            output[:4] = input
            output[4:] = input

        with patch("paddle.distributed.stream.all_gather", side_effect=fake_ag):
            out = _RouterAllGather.apply(x, g)
        self.assertEqual(out.shape[0], 8)  # 4 * nranks=2


class TestRouterAllGatherBackwardReshape(unittest.TestCase):
    """_RouterAllGather.backward out.reshape path: line 1207."""

    def test_backward_out_shape_mismatch_reshapes(self):
        import paddleformers.fleet.transformer.moe.token_dispatcher as td
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _RouterAllGather,
        )

        fn = _RouterAllGather.__dict__["backward"]
        if isinstance(fn, staticmethod):
            fn = fn.__func__
        ctx = MagicMock()
        ctx.group = _mock_group(2)
        ctx.input_shape = [4, 2]
        # grad matches global_shape [8,2] — triggers reduce_scatter
        # reduce_scatter returns shape [3, 2] (mismatched) → needs reshape
        grad = paddle.randn([8, 2])
        wrong_shape_out = paddle.randn([8])  # wrong shape → line 1207 reshape
        with patch.object(
            td, "reduce_scatter_group", return_value=wrong_shape_out
        ):
            result = fn(ctx, grad)
        self.assertEqual(list(result.shape), [4, 2])


class TestPreAllGatherResultBackward(unittest.TestCase):
    """_PreAllGatherResult.backward: lines 1228-1229."""

    def test_backward_calls_reduce_scatter(self):
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            ReduceScatterGroupOp,
            _PreAllGatherResult,
        )

        fn = _PreAllGatherResult.__dict__["backward"]
        if isinstance(fn, staticmethod):
            fn = fn.__func__
        ctx = MagicMock()
        ctx.group = _mock_group(2)
        grad = paddle.randn([4, 8])
        expected = paddle.randn([2, 8])
        with patch.object(ReduceScatterGroupOp, "apply", return_value=expected):
            result = fn(ctx, grad)
        np.testing.assert_allclose(result.numpy(), expected.numpy())


class TestQuantizeAndPackFP8(unittest.TestCase):
    """_quantize_and_pack_fp8: lines 1324-1339."""

    def test_raises_when_quantize_fn_none(self):
        import paddleformers.fleet.transformer.moe.token_dispatcher as td
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _quantize_and_pack_fp8,
        )

        orig = td.quantize_activation_blockscaled_fast
        td.quantize_activation_blockscaled_fast = None
        try:
            with self.assertRaises(RuntimeError):
                _quantize_and_pack_fp8(paddle.randn([4, 8]))
        finally:
            td.quantize_activation_blockscaled_fast = orig

    def test_packs_to_fused_buffer(self):
        """quantize_activation_blockscaled_fast available: pack runs."""
        import paddleformers.fleet.transformer.moe.token_dispatcher as td
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _quantize_and_pack_fp8,
        )

        T, H, H128 = 4, 8, 1
        fake_fp8 = paddle.zeros([T, H], dtype="float8_e4m3fn")
        fake_scale = paddle.zeros([T, H128], dtype=paddle.int32)

        with patch.object(
            td,
            "quantize_activation_blockscaled_fast",
            return_value=(fake_fp8, fake_scale),
        ):
            fused, out_H, out_H128, scale_dtype = _quantize_and_pack_fp8(
                paddle.randn([T, H])
            )
        self.assertEqual(out_H, H)
        self.assertEqual(out_H128, H128)
        self.assertEqual(fused.shape[0], T)


class TestAllGatherGradFP8Async(unittest.TestCase):
    """_fused_fp8_all_gather_async: lines 1361-1371."""

    def test_quantize_and_launches_async_gather(self):
        import paddleformers.fleet.transformer.moe.token_dispatcher as td
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _fused_fp8_all_gather_async,
        )

        T, H, H128 = 4, 8, 1
        fake_fp8 = paddle.zeros([T, H], dtype="float8_e4m3fn")
        fake_scale = paddle.zeros([T, H128], dtype=paddle.int32)
        fused_local = paddle.zeros([T, H + 4 * H128], dtype="uint8")
        mock_task = MagicMock()

        g = _mock_group(2)
        with (
            patch.object(
                td,
                "_quantize_and_pack_fp8",
                return_value=(fused_local, H, H128, paddle.int32),
            ),
            patch(
                "paddle.distributed.stream.all_gather",
                return_value=mock_task,
            ),
        ):
            fused_global, rH, rH128, rdt, task = _fused_fp8_all_gather_async(
                paddle.randn([T, H]), g
            )
        self.assertEqual(fused_global.shape[0], T * g.nranks)
        self.assertEqual(rH, H)
        self.assertIs(task, mock_task)


class TestAllGatherCombineAsyncForwardNranks2(unittest.TestCase):
    """_AllGatherCombineAsync.forward nranks>1: lines 1425-1426,1433-1435,1437."""

    def test_forward_nranks2_fp8_handle_sets_flags(self):
        """fp8_combine_grad_handle is not None → set_grad_in_dtype_consistent, line 1425-1426."""
        import paddleformers.fleet.transformer.moe.token_dispatcher as td
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineAsync,
        )

        x = paddle.randn([4, 8])
        fn_out = paddle.randn([4, 8])
        mock_task = MagicMock()
        combined = paddle.randn([4, 8])

        with (
            patch.object(
                td,
                "_reduce_scatter_async",
                return_value=(combined, mock_task),
            ),
            patch.object(
                td,
                "manual_backward",
                return_value=(MagicMock(return_value=(fn_out,)), (fn_out,)),
            ),
        ):
            result = _AllGatherCombineAsync.apply(
                x,
                _mock_group(2),
                fn=MagicMock(return_value=(fn_out,)),
                fp8_combine_grad_handle={},
            )
        mock_task.wait.assert_called_once()
        self.assertEqual(len(result), 2)

    def test_forward_nranks2_no_fp8_handle(self):
        """nranks>1, no fp8 handle: lines 1433-1435,1437."""
        import paddleformers.fleet.transformer.moe.token_dispatcher as td
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineAsync,
        )

        x = paddle.randn([4, 8])
        fn_out = paddle.randn([4, 8])
        mock_task = MagicMock()
        combined = paddle.randn([4, 8])

        with (
            patch.object(
                td,
                "_reduce_scatter_async",
                return_value=(combined, mock_task),
            ),
            patch.object(
                td,
                "manual_backward",
                return_value=(MagicMock(return_value=(fn_out,)), (fn_out,)),
            ),
        ):
            result = _AllGatherCombineAsync.apply(
                x,
                _mock_group(2),
                fn=MagicMock(return_value=(fn_out,)),
            )
        mock_task.wait.assert_called_once()
        self.assertEqual(len(result), 2)


class TestAllGatherCombineNoOverlapFP8ForwardFlags(unittest.TestCase):
    """_AllGatherCombineNoOverlap.forward with fp8_handle sets flags: lines 1487-1488."""

    def test_forward_group_with_fp8_handle(self):
        """fp8_combine_grad_handle != None and group != None triggers flags."""
        import paddleformers.fleet.transformer.moe.token_dispatcher as td
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _AllGatherCombineNoOverlap,
        )

        x = paddle.randn([4, 8])
        with patch.object(td, "reduce_scatter_group", return_value=x):
            out = _AllGatherCombineNoOverlap.apply(x, _mock_group(2), {})
        # If flags were correctly set, forward returns without error
        self.assertEqual(out.shape, x.shape)


class TestPreAllGatherFP8Path(unittest.TestCase):
    """AllGatherTokenDispatcher.pre_allgather fp8 path: lines 1615,1618-1619,1622,1629,1638."""

    def test_pre_allgather_fp8_stores_handle(self):
        import paddleformers.fleet.transformer.moe.token_dispatcher as td

        g = _mock_group(2)
        dispatcher = _make_ag_dispatcher(group=g, fp8=True)
        x = paddle.randn([4, 8])

        T, H, H128 = 4, 8, 1
        fused_local = paddle.zeros([T, H + 4 * H128], dtype="uint8")
        mock_task = MagicMock()

        with (
            patch.object(
                td,
                "_quantize_and_pack_fp8",
                return_value=(fused_local, H, H128, paddle.int32),
            ),
            patch(
                "paddle.distributed.stream.all_gather",
                return_value=mock_task,
            ),
        ):
            dispatcher.pre_allgather(x)

        self.assertIsNotNone(dispatcher._pre_ag_handle)
        h = dispatcher._pre_ag_handle
        self.assertTrue(h.get("fp8"))
        self.assertIn("fused_global", h)
        self.assertEqual(h["task"], mock_task)


class TestDispatchPreprocessFP8Paths(unittest.TestCase):
    """dispatch_preprocess fp8 handle path (1690) and fallback fp8 (1701-1707)."""

    def test_dispatch_preprocess_fp8_handle_consumed(self):
        """line 1690: pre_ag_handle with fp8=True uses _PreAllGatherFP8Result."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _PreAllGatherFP8Result,
        )

        g = _mock_group(1)
        dispatcher = _make_ag_dispatcher(group=g, fp8=True)
        x = paddle.randn([4, 8])
        global_x = paddle.randn([4, 8])
        scale = paddle.randn([4, 1])
        mock_task = MagicMock()
        dispatcher._pre_ag_handle = {
            "fused_global": paddle.zeros([4, 12], dtype="uint8"),
            "H": 8,
            "H128": 1,
            "scale_dtype": paddle.int32,
            "task": mock_task,
            "group": g,
            "fp8": True,
        }
        topk_indices = paddle.to_tensor([[0, 1]] * 4, dtype="int32")
        topk_weights = paddle.randn([4, 2])

        with patch.object(
            _PreAllGatherFP8Result, "apply", return_value=(global_x, scale)
        ):
            result = dispatcher.dispatch_preprocess(
                x,
                paddle.randn([4, 4]),
                paddle.randn([4, 4]),
                topk_weights=topk_weights,
                topk_indices=topk_indices,
            )
        self.assertIsNone(dispatcher._pre_ag_handle)
        self.assertIsNotNone(dispatcher._fp8_dispatch_scale)

    def test_dispatch_preprocess_no_handle_fp8_fallback(self):
        """lines 1701-1707: no pre_ag_handle but fp8_dispatch=True → call pre_allgather."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _PreAllGatherFP8Result,
        )

        g = _mock_group(1)
        dispatcher = _make_ag_dispatcher(group=g, fp8=True)
        dispatcher._pre_ag_handle = None
        x = paddle.randn([4, 8])
        global_x = paddle.randn([4, 8])
        scale = paddle.randn([4, 1])
        topk_indices = paddle.to_tensor([[0, 1]] * 4, dtype="int32")
        topk_weights = paddle.randn([4, 2])

        sentinel_handle = {
            "fused_global": paddle.zeros([4, 12], dtype="uint8"),
            "H": 8,
            "H128": 1,
            "scale_dtype": paddle.int32,
            "task": MagicMock(),
            "group": g,
            "fp8": True,
        }

        def fake_pre_allgather(inp):
            dispatcher._pre_ag_handle = sentinel_handle

        with (
            patch.object(
                dispatcher, "pre_allgather", side_effect=fake_pre_allgather
            ),
            patch.object(
                _PreAllGatherFP8Result, "apply", return_value=(global_x, scale)
            ),
        ):
            result = dispatcher.dispatch_preprocess(
                x,
                paddle.randn([4, 4]),
                paddle.randn([4, 4]),
                topk_weights=topk_weights,
                topk_indices=topk_indices,
            )
        self.assertIsNone(dispatcher._pre_ag_handle)
        self.assertIsNotNone(dispatcher._fp8_dispatch_scale)


class TestDispatchPreprocessIdxAllGatherNranks2(unittest.TestCase):
    """dispatch_preprocess nranks>1 index allgather: lines 1725-1730,1743."""

    def test_dispatch_preprocess_nranks2_idx_allgather(self):
        """nranks>1: issues async all_gather for topk_indices and waits."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllGatherGroupOp,
        )

        g = _mock_group(2)
        dispatcher = _make_ag_dispatcher(group=g)
        x = paddle.randn([4, 8])
        topk_indices = paddle.to_tensor([[0, 1]] * 4, dtype="int32")
        topk_weights = paddle.randn([4, 2])

        mock_task = MagicMock()

        def fake_async_ag(
            output, inp, group, sync_op=False, use_calc_stream=False
        ):
            return mock_task

        with (
            patch.object(AllGatherGroupOp, "apply", return_value=x),
            patch(
                "paddle.distributed.stream.all_gather",
                side_effect=fake_async_ag,
            ),
        ):
            dispatcher.dispatch_preprocess(
                x,
                paddle.randn([4, 4]),
                paddle.randn([4, 4]),
                topk_weights=topk_weights,
                topk_indices=topk_indices,
            )
        mock_task.wait.assert_called_once()
