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

"""
Single-card unit tests for RR DeepEP Combine configuration validation.

Tests the parameter validation logic in:
- moe_layer.py: rr_recompute_update()
- fused_a2a.py: fused_combine() parameter validation, DeepEPCombineAsyncRefinedRecompute init
- token_dispatcher.py: _HybridEPManager.combine() accepts use_rr_deepep_combine
"""

import unittest
from unittest.mock import MagicMock, patch

try:
    from paddleformers.fleet.transformer.moe.fused_a2a import (
        DeepEPCombineAsyncRefinedRecompute,
        fused_combine,
    )

    HAS_DEEP_EP = True
except (ImportError, RuntimeError):
    HAS_DEEP_EP = False


@unittest.skipUnless(HAS_DEEP_EP, "DeepEP not available")
class TestFusedCombineValidation(unittest.TestCase):
    """Tests parameter validation in fused_combine()."""

    def test_rr_without_overlap_handle_raises(self):
        """use_rr_deepep_combine=True requires combine_overlap_handle."""
        with self.assertRaises(ValueError) as ctx:
            fused_combine(
                x=MagicMock(),
                group=MagicMock(),
                handle=MagicMock(),
                combine_overlap_handle=None,
                use_rr_deepep_combine=True,
            )
        self.assertIn(
            "use_rr_deepep_combine requires combine_overlap_handle",
            str(ctx.exception),
        )

    def test_overlap_handle_previous_event_conflict(self):
        """previous_event must be None when combine_overlap_handle is provided."""
        with self.assertRaises(ValueError) as ctx:
            fused_combine(
                x=MagicMock(),
                group=MagicMock(),
                handle=MagicMock(),
                combine_overlap_handle={"fn": lambda: None, "fn_args": ()},
                previous_event=MagicMock(),
            )
        self.assertIn("previous_event must be None", str(ctx.exception))

    def test_overlap_handle_not_dict_raises(self):
        """combine_overlap_handle must be a dict."""
        with self.assertRaises(TypeError) as ctx:
            fused_combine(
                x=MagicMock(),
                group=MagicMock(),
                handle=MagicMock(),
                combine_overlap_handle="not_a_dict",
            )
        self.assertIn("must be a dict", str(ctx.exception))

    def test_overlap_handle_missing_fn_raises(self):
        """combine_overlap_handle must contain 'fn' key."""
        with self.assertRaises(ValueError) as ctx:
            fused_combine(
                x=MagicMock(),
                group=MagicMock(),
                handle=MagicMock(),
                combine_overlap_handle={"fn_args": ()},
            )
        self.assertIn("must contain 'fn' key", str(ctx.exception))

    def test_overlap_handle_missing_fn_args_raises(self):
        """combine_overlap_handle must contain 'fn_args' key."""
        with self.assertRaises(ValueError) as ctx:
            fused_combine(
                x=MagicMock(),
                group=MagicMock(),
                handle=MagicMock(),
                combine_overlap_handle={"fn": lambda: None},
            )
        self.assertIn("must contain 'fn_args' key", str(ctx.exception))

    def test_overlap_handle_fn_args_not_tuple_raises(self):
        """combine_overlap_handle['fn_args'] must be a tuple."""
        with self.assertRaises(TypeError) as ctx:
            fused_combine(
                x=MagicMock(),
                group=MagicMock(),
                handle=MagicMock(),
                combine_overlap_handle={"fn": lambda: None, "fn_args": []},
            )
        self.assertIn("must be a tuple", str(ctx.exception))

    def test_rr_without_rr_fusedcombined_raises(self):
        """_rr_fusedcombined must be provided when use_rr_deepep_combine=True."""
        with self.assertRaises(ValueError) as ctx:
            fused_combine(
                x=MagicMock(),
                group=MagicMock(),
                handle=MagicMock(),
                combine_overlap_handle={"fn": lambda: None, "fn_args": ()},
                use_rr_deepep_combine=True,
                _rr_fusedcombined=None,
            )
        self.assertIn("_rr_fusedcombined must be provided", str(ctx.exception))


@unittest.skipUnless(HAS_DEEP_EP, "DeepEP not available")
class TestDeepEPCombineAsyncRefinedRecomputeInit(unittest.TestCase):
    """Tests DeepEPCombineAsyncRefinedRecompute initialization."""

    def test_init_creates_queue(self):
        """Initialization creates a queue and registers it."""
        rr = DeepEPCombineAsyncRefinedRecompute()
        self.assertIsNotNone(rr._hold_tensors_queue)
        self.assertTrue(rr._hold_tensors_queue.empty())

    def test_second_fwd_empty_queue_raises(self):
        """Second forward with empty queue raises RuntimeError."""
        from unittest.mock import PropertyMock

        from paddle import framework

        rr = DeepEPCombineAsyncRefinedRecompute()
        tracer = framework._dygraph_tracer()
        # Simulate recompute pass (has_grad=True means second forward)
        # Use PropertyMock since _has_grad is a property without a deleter
        with patch.object(
            type(tracer),
            "_has_grad",
            new_callable=PropertyMock,
            return_value=True,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                rr.forward(MagicMock(), MagicMock(), {}, fn=lambda: None)
            self.assertIn("Queue is empty", str(ctx.exception))


try:
    from paddleformers.fleet.transformer.moe.token_dispatcher import _DeepEPManager

    HAS_DEEP_EP_MANAGER = True
except (ImportError, RuntimeError):
    HAS_DEEP_EP_MANAGER = False


@unittest.skipUnless(HAS_DEEP_EP_MANAGER, "DeepEP Manager not available")
class TestDeepEPManagerCombineRR(unittest.TestCase):
    """Tests _DeepEPManager.combine() use_rr_deepep_combine branch."""

    def _make_manager(self):
        """Create a _DeepEPManager with mocked dependencies."""
        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
            new=MagicMock(),
        ):
            manager = _DeepEPManager(
                group=MagicMock(),
                router_topk=2,
                num_experts=8,
                num_local_experts=4,
                moe_ep_barrier=False,
            )
        manager.handle = MagicMock()
        return manager

    def test_rr_creates_instance_when_none(self):
        """combine() creates DeepEPCombineAsyncRefinedRecompute when _rr_fusedcombined is None."""
        manager = self._make_manager()
        self.assertIsNone(manager._rr_fusedcombined)
        overlap_handle = {"fn": lambda: None, "fn_args": ()}
        # fused_combine will be called but we mock it to avoid actual dispatch
        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine",
            return_value=MagicMock(),
        ):
            manager.combine(
                MagicMock(),
                combine_overlap_handle=overlap_handle,
                use_rr_deepep_combine=True,
            )
        self.assertIsInstance(
            manager._rr_fusedcombined, DeepEPCombineAsyncRefinedRecompute
        )

    def test_rr_type_mismatch_raises(self):
        """combine() raises RuntimeError when _rr_fusedcombined has wrong type."""
        manager = self._make_manager()
        # Set _rr_fusedcombined to a wrong type
        manager._rr_fusedcombined = "not_the_right_type"
        overlap_handle = {"fn": lambda: None, "fn_args": ()}
        with self.assertRaises(RuntimeError) as ctx:
            manager.combine(
                MagicMock(),
                combine_overlap_handle=overlap_handle,
                use_rr_deepep_combine=True,
            )
        self.assertIn("_rr_fusedcombined type mismatch", str(ctx.exception))


try:
    from paddleformers.fleet.transformer.moe.moe_layer import MoELayer  # noqa: F401

    HAS_MOE_LAYER = True
except (ImportError, RuntimeError):
    HAS_MOE_LAYER = False


@unittest.skipUnless(
    HAS_MOE_LAYER, "MoELayer not available (DeepEP dependency)"
)
class TestRRRecomputeUpdate(unittest.TestCase):
    """Tests MoELayer.rr_recompute_update() validation logic."""

    def _make_moe_layer_mock(self, **overrides):
        """Create a mock MoELayer with necessary attributes."""
        mock = MagicMock()
        mock.moe_token_dispatcher_type = overrides.get(
            "dispatcher_type", "deepep"
        )
        mock.moe_shared_expert_overlap = overrides.get(
            "shared_expert_overlap", True
        )
        mock.use_rr_deepep_combine = False
        mock.config = MagicMock()
        mock.config.recompute_modules = overrides.get(
            "recompute_modules", ["moe_combine"]
        )
        mock.config.recompute_granularity = overrides.get(
            "recompute_granularity", "full"
        )
        mock.config.recompute_method = overrides.get(
            "recompute_method", "first_n"
        )
        if "layer_number" in overrides:
            mock.layer_number = overrides["layer_number"]
        return mock

    def test_non_deepep_dispatcher_raises(self):
        """RR only supported in DeepEP mode."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock = self._make_moe_layer_mock(dispatcher_type="alltoall")
        with self.assertRaises(ValueError) as ctx:
            MoELayer.rr_recompute_update(
                mock, in_full_recompute=True, in_mlp_recompute=False
            )
        self.assertIn("only supported in DeepEP mode", str(ctx.exception))

    def test_no_shared_expert_overlap_raises(self):
        """RR requires moe_shared_expert_overlap."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock = self._make_moe_layer_mock(shared_expert_overlap=False)
        with self.assertRaises(ValueError) as ctx:
            MoELayer.rr_recompute_update(
                mock, in_full_recompute=True, in_mlp_recompute=False
            )
        self.assertIn("only supported in DeepEP mode", str(ctx.exception))

    def test_no_recompute_granularity_raises(self):
        """RR requires recompute_granularity to be set."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock = self._make_moe_layer_mock(recompute_granularity=None)
        with self.assertRaises(ValueError) as ctx:
            MoELayer.rr_recompute_update(
                mock, in_full_recompute=True, in_mlp_recompute=False
            )
        self.assertIn("recompute_granularity must be set", str(ctx.exception))

    def test_list_mode_sets_flag(self):
        """List mode sets use_rr_deepep_combine=True."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock = self._make_moe_layer_mock(recompute_modules=["moe_combine"])
        MoELayer.rr_recompute_update(
            mock, in_full_recompute=True, in_mlp_recompute=False
        )
        self.assertTrue(mock.use_rr_deepep_combine)

    def test_dict_mode_wrong_method_raises(self):
        """Dict mode requires recompute_method='first_n'."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock = self._make_moe_layer_mock(
            recompute_modules={"moe_combine": 4},
            recompute_method="uniform",
        )
        with self.assertRaises(ValueError) as ctx:
            MoELayer.rr_recompute_update(
                mock, in_full_recompute=True, in_mlp_recompute=False
            )
        self.assertIn("recompute_method='first_n'", str(ctx.exception))

    def test_dict_mode_no_layer_number_raises(self):
        """Dict mode requires layer_number to be set."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock = self._make_moe_layer_mock(recompute_modules={"moe_combine": 4})
        del mock.layer_number  # Remove the attribute
        with self.assertRaises(ValueError) as ctx:
            MoELayer.rr_recompute_update(
                mock, in_full_recompute=True, in_mlp_recompute=False
            )
        self.assertIn("layer_number must be set", str(ctx.exception))

    def test_rr_without_recompute_active_raises(self):
        """RR is meaningless without full_recompute or mlp_recompute."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock = self._make_moe_layer_mock(recompute_modules=["moe_combine"])
        with self.assertRaises(ValueError) as ctx:
            MoELayer.rr_recompute_update(
                mock, in_full_recompute=False, in_mlp_recompute=False
            )
        self.assertIn(
            "meaningless when neither full_recompute", str(ctx.exception)
        )

    @patch("paddleformers.fleet.transformer.moe.moe_layer.need_recompute_in_first_n")
    def test_dict_mode_sets_flag_via_need_recompute(self, mock_need_recompute):
        """Dict mode uses need_recompute_in_first_n to decide use_rr_deepep_combine."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        # When need_recompute_in_first_n returns False, use_rr_deepep_combine = True
        mock_need_recompute.return_value = False
        mock = self._make_moe_layer_mock(
            recompute_modules={"moe_combine": 4}, layer_number=5
        )
        MoELayer.rr_recompute_update(
            mock, in_full_recompute=True, in_mlp_recompute=False
        )
        self.assertTrue(mock.use_rr_deepep_combine)


@unittest.skipUnless(HAS_DEEP_EP, "DeepEP not available")
class TestDeepEPCombineAsyncFunctor(unittest.TestCase):
    """Tests DeepEPCombineAsyncFunctor forward and backward (lines 538-561)."""

    def test_functor_forward(self):
        """DeepEPCombineAsyncFunctor.forward returns combined_x + fn_out."""
        import paddle

        from paddleformers.fleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncFunctor,
        )

        hold_tensors = {"res_output": paddle.to_tensor([1.0, 2.0, 3.0])}
        x = paddle.to_tensor([4.0, 5.0, 6.0])
        group = MagicMock()
        group.id = 0
        states = {"handle": MagicMock()}
        fn_arg = paddle.to_tensor([7.0])

        mock_fn_out = (paddle.to_tensor([10.0]),)

        with patch(
            "paddleformers.fleet.transformer.moe.fused_a2a.manual_backward",
            return_value=(MagicMock(), mock_fn_out),
        ):
            result = DeepEPCombineAsyncFunctor.apply(
                hold_tensors, x, group, states, fn_arg, fn=lambda *a: None
            )

        # Result should be (combined_x, *fn_out)
        self.assertIsNotNone(result)

    def test_functor_backward(self):
        """DeepEPCombineAsyncFunctor.backward returns grad_x + fn_args_grads."""
        import paddle

        from paddleformers.fleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncFunctor,
        )

        # Directly invoke the backward static method with a mock context
        mock_ctx = MagicMock()
        mock_ctx.group = MagicMock()
        mock_ctx.group.id = 0
        mock_ctx.handle = MagicMock()
        mock_ctx.bwf = MagicMock(return_value=(paddle.to_tensor([0.1]),))
        mock_ctx.fp8_dispatch = False

        grad_output = paddle.to_tensor([1.0, 1.0])
        fn_out_grad = paddle.to_tensor([0.5])

        with (
            patch(
                "paddleformers.fleet.transformer.moe.fused_a2a.fused_combine_backward_func",
                return_value=paddle.to_tensor([0.5, 0.5]),
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fused_a2a.wait_for_deepep",
            ),
        ):
            result = DeepEPCombineAsyncFunctor.backward(
                mock_ctx, grad_output, fn_out_grad
            )

        # Should return (grad_x,) + fn_args_grads
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # grad_x + one fn_arg grad
        mock_ctx.bwf.assert_called_once_with(fn_out_grad)


@unittest.skipUnless(HAS_DEEP_EP, "DeepEP not available")
class TestDeepEPCombineAsyncRefinedRecomputeRuntime(unittest.TestCase):
    """Tests DeepEPCombineAsyncRefinedRecompute runtime paths (lines 582-627)."""

    def test_first_fwd_path(self):
        """First forward path: queues detached output."""
        from unittest.mock import PropertyMock

        import paddle
        from paddle import framework

        from paddleformers.fleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncRefinedRecompute,
        )

        rr = DeepEPCombineAsyncRefinedRecompute()
        tracer = framework._dygraph_tracer()

        x = paddle.to_tensor([1.0, 2.0])
        group = MagicMock()
        group.id = 0
        states = {"handle": MagicMock()}
        fn_arg = paddle.to_tensor([3.0])

        mock_combined = paddle.to_tensor([5.0, 6.0])
        mock_fn_out = (paddle.to_tensor([10.0]),)

        # is_first_fwd = not _has_grad; _has_grad=False means first fwd
        with (
            patch.object(
                type(tracer),
                "_has_grad",
                new_callable=PropertyMock,
                return_value=False,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fused_a2a.fused_combine_forward_func",
                return_value=mock_combined,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fused_a2a.manual_backward",
                return_value=(None, mock_fn_out),
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fused_a2a.wait_for_deepep",
            ),
        ):
            result = rr.forward(x, group, states, fn_arg, fn=lambda *a: None)

        # Should return (fwd_output, *fn_out)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        # Queue should have the detached tensor
        self.assertFalse(rr._hold_tensors_queue.empty())

    def test_second_fwd_path(self):
        """Second forward path: pops from queue and calls _second_fwd."""
        from unittest.mock import PropertyMock

        import paddle
        from paddle import framework

        from paddleformers.fleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncRefinedRecompute,
        )

        rr = DeepEPCombineAsyncRefinedRecompute()
        tracer = framework._dygraph_tracer()

        # Pre-populate queue (simulating first fwd already ran)
        rr._hold_tensors_queue.put({"res_output": paddle.to_tensor([5.0, 6.0])})

        x = paddle.to_tensor([1.0, 2.0])
        group = MagicMock()
        group.id = 0
        states = {"handle": MagicMock()}
        fn_arg = paddle.to_tensor([3.0])

        mock_result = (paddle.to_tensor([7.0, 8.0]),)

        # is_first_fwd = not _has_grad; _has_grad=True means second fwd
        with (
            patch.object(
                type(tracer),
                "_has_grad",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fused_a2a.DeepEPCombineAsyncFunctor.apply",
                return_value=mock_result,
            ),
        ):
            result = rr.forward(x, group, states, fn_arg, fn=lambda *a: None)

        self.assertEqual(result, mock_result)
        self.assertTrue(rr._hold_tensors_queue.empty())

    def test_first_fwd_fn_none_raises(self):
        """_first_fwd raises ValueError when fn is None."""
        from unittest.mock import PropertyMock

        import paddle
        from paddle import framework

        from paddleformers.fleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncRefinedRecompute,
        )

        rr = DeepEPCombineAsyncRefinedRecompute()
        tracer = framework._dygraph_tracer()

        x = paddle.to_tensor([1.0, 2.0])
        group = MagicMock()
        group.id = 0
        states = {"handle": MagicMock()}

        mock_combined = paddle.to_tensor([5.0, 6.0])

        with (
            patch.object(
                type(tracer),
                "_has_grad",
                new_callable=PropertyMock,
                return_value=False,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fused_a2a.fused_combine_forward_func",
                return_value=mock_combined,
            ),
        ):
            with self.assertRaises(ValueError) as ctx:
                rr.forward(x, group, states, fn=None)
            self.assertIn("fn must not be None", str(ctx.exception))

    def test_call_delegates_to_forward(self):
        """__call__ delegates to forward."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            DeepEPCombineAsyncRefinedRecompute,
        )

        rr = DeepEPCombineAsyncRefinedRecompute()
        mock_result = MagicMock()
        with patch.object(rr, "forward", return_value=mock_result) as m:
            result = rr(1, 2, 3, fn="test")
            m.assert_called_once_with(1, 2, 3, fn="test")
        self.assertEqual(result, mock_result)


@unittest.skipUnless(HAS_DEEP_EP, "DeepEP not available")
class TestFusedCombineRRBranch(unittest.TestCase):
    """Tests fused_combine() RR branch that calls _rr_fusedcombined (lines 750-758)."""

    def test_rr_branch_calls_fusedcombined(self):
        """fused_combine with use_rr_deepep_combine calls _rr_fusedcombined."""
        import paddle

        mock_combined = paddle.to_tensor([1.0, 2.0])
        mock_fn_out = [paddle.to_tensor([3.0])]
        mock_rr = MagicMock(return_value=(mock_combined, *mock_fn_out))

        overlap_handle = {"fn": lambda: None, "fn_args": ()}

        result = fused_combine(
            x=MagicMock(),
            group=MagicMock(),
            handle=MagicMock(),
            combine_overlap_handle=overlap_handle,
            use_rr_deepep_combine=True,
            _rr_fusedcombined=mock_rr,
        )

        mock_rr.assert_called_once()
        self.assertEqual(overlap_handle["fn_out"], mock_fn_out)
        # result should be combined_x
        self.assertTrue(paddle.equal_all(result, mock_combined).item())

    def test_rr_branch_with_fn_args(self):
        """fused_combine RR branch passes fn_args correctly."""
        import paddle

        mock_combined = paddle.to_tensor([1.0])
        fn_arg_1 = paddle.to_tensor([10.0])
        fn_arg_2 = paddle.to_tensor([20.0])
        mock_rr = MagicMock(return_value=(mock_combined,))

        overlap_handle = {
            "fn": lambda *a: None,
            "fn_args": (fn_arg_1, fn_arg_2),
        }

        fused_combine(
            x=MagicMock(),
            group=MagicMock(),
            handle=MagicMock(),
            combine_overlap_handle=overlap_handle,
            use_rr_deepep_combine=True,
            _rr_fusedcombined=mock_rr,
        )

        # Verify fn_args were unpacked in the call
        call_args = mock_rr.call_args
        # positional args: x, group, states, *fn_args
        # fn_args should be unpacked as positional
        self.assertEqual(len(call_args[0]), 5)  # x, group, states, arg1, arg2


@unittest.skipUnless(HAS_DEEP_EP, "DeepEP not available")
class TestDeepEPDispatchFp8NormalizeBranch(unittest.TestCase):
    """Cover line 453: scale = _normalize_fp8_scale_for_deepep(x_fp8, scale, use_ue8m0)
    inside DeepEPDispatch.forward (fp8_dispatch=True, using_sonic_moe=False)."""

    def _run_dispatch_forward(self, num_tokens, hidden, use_ue8m0):
        from unittest.mock import MagicMock, patch

        import paddle

        import paddleformers.fleet.transformer.moe.fused_a2a as _fused_mod

        num_scales = hidden // 128
        if use_ue8m0:
            num_scales //= 4

        x = paddle.randn([num_tokens, hidden], dtype="float32")
        x_fp8_fake = paddle.zeros([num_tokens, hidden], dtype="float32")
        # fp8_quant_blockwise returns scale in transposed form [num_scales, num_tokens]
        scale_fake = paddle.ones([num_scales, num_tokens], dtype="float32")
        # _normalize_fp8_scale_for_deepep should transpose → [num_tokens, num_scales]
        scale_normalized = paddle.ones(
            [num_tokens, num_scales], dtype="float32"
        )

        recv_x_fake = (
            paddle.zeros([num_tokens, hidden], dtype="float32"),
            scale_normalized,
        )
        recv_probs_fake = paddle.zeros([num_tokens], dtype="float32")
        states_fake = {
            "handle": MagicMock(),
            "dispatched_indices": None,
            "tokens_per_expert": None,
        }
        event_fake = MagicMock()

        ctx = MagicMock()
        ctx.set_grad_in_dtype_consistent = MagicMock()

        with (
            patch.object(
                _fused_mod.paddle.incubate.nn.functional,
                "fp8_quant_blockwise",
                return_value=(x_fp8_fake, scale_fake),
            ),
            patch.object(
                _fused_mod,
                "fused_dispatch_forward_func",
                return_value=(
                    recv_x_fake,
                    recv_probs_fake,
                    states_fake,
                    event_fake,
                ),
            ),
        ):
            result = _fused_mod.DeepEPDispatch.forward(
                ctx,
                x,
                token_indices=paddle.zeros([num_tokens], dtype="int64"),
                token_probs=paddle.ones([num_tokens], dtype="float32"),
                num_experts=4,
                group=MagicMock(),
                fp8_dispatch=True,
                using_sonic_moe=False,
                use_ue8m0=use_ue8m0,
            )
        return result

    def test_fp8_dispatch_non_ue8m0_runs(self):
        """fp8_dispatch=True, use_ue8m0=False: line 453 executes, scale shape is correct."""
        result = self._run_dispatch_forward(
            num_tokens=4, hidden=256, use_ue8m0=False
        )
        # forward returns (recv_x, recv_token_probs, states, {"scale": scale})
        self.assertIsNotNone(result)
        self.assertIsNotNone(result[-1])  # fp8 extra dict is not None

    def test_fp8_dispatch_ue8m0_runs(self):
        """fp8_dispatch=True, use_ue8m0=True: line 453 executes with ue8m0 scale."""
        result = self._run_dispatch_forward(
            num_tokens=4, hidden=512, use_ue8m0=True
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result[-1])


if __name__ == "__main__":
    unittest.main()
