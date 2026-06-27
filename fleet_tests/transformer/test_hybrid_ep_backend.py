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

import unittest
from unittest.mock import patch

import paddle

from paddleformers.fleet.transformer.moe import fused_a2a
from paddleformers.fleet.transformer.moe.fp8_utils import (
    FP8_ALIGN,
    ExpertsGroupGemmContiguousNode,
)
from paddleformers.fleet.transformer.moe.fused_a2a import (
    HYBRIDEP_TOKEN_ALIGNMENT,
    HybridEPCombine,
    HybridEPDispatch,
    _replay_hybrid_ep_dispatch_backward,
)
from paddleformers.fleet.transformer.moe.fusion_layer_utils import (
    HybridEPMoePyLayer,
    _hybrid_ep_prepare_expert_counts,
    _pad_front_rows,
    _restore_hybrid_ep_prob_grad_shape,
)
from paddleformers.fleet.transformer.moe.token_dispatcher import (
    MoEFlexTokenDispatcher,
    _HybridEPManager,
    is_hybrid_ep_backend_selected,
)


class _HybridEPGroup:
    def __init__(self, nranks=2):
        self.nranks = nranks
        self.world_size = nranks


class _HybridEPHandleConfig:
    def __init__(self, token_data_type="BF16", num_experts_per_rank=2):
        self.token_data_type = token_data_type
        self.num_of_experts_per_rank = num_experts_per_rank


class _HybridEPDispatcher:
    def __init__(self, manager):
        self._comm_manager = manager


class _HybridEPCustomMap:
    def __init__(self, manager, experts=None):
        self.token_dispatcher = _HybridEPDispatcher(manager)
        self.experts = [] if experts is None else experts
        self.grouped_gemm_experts = None


class _ExpertProjection:
    def __init__(self, shape):
        self.weight = paddle.create_parameter(shape=shape, dtype="float32")


class _TinyExpert:
    def __init__(self, hidden_size=2, intermediate_size=1):
        self.up_gate_proj = _ExpertProjection(
            [hidden_size, intermediate_size * 2]
        )
        self.down_proj = _ExpertProjection([intermediate_size, hidden_size])


class _ExpertsGroupGemmCustomMap:
    def __init__(self):
        self.experts = []
        self.grouped_gemm_experts = None


def _new_hybrid_manager(**overrides):
    init_kwargs = {
        "group": overrides.pop("group", _HybridEPGroup(nranks=2)),
        "router_topk": overrides.pop("router_topk", 2),
        "num_experts": overrides.pop("num_experts", 4),
        "num_local_experts": overrides.pop("num_local_experts", 2),
        "hybridep_buffer_configs": overrides.pop(
            "hybridep_buffer_configs", None
        ),
    }
    manager = _HybridEPManager(**init_kwargs)
    for key, value in overrides.items():
        setattr(manager, key, value)
    return manager


def _make_hybrid_ep_handle(
    num_dispatched_tokens=2,
    local_expert_routing_map=None,
    tokens_per_rank=8,
    token_data_type="BF16",
    num_experts_per_rank=2,
):
    if local_expert_routing_map is None:
        local_expert_routing_map = paddle.ones(
            [num_dispatched_tokens, num_experts_per_rank],
            dtype="bool",
        )
    return (
        None,
        None,
        None,
        paddle.to_tensor(num_dispatched_tokens, dtype="int64"),
        local_expert_routing_map,
        None,
        tokens_per_rank,
        _HybridEPHandleConfig(
            token_data_type=token_data_type,
            num_experts_per_rank=num_experts_per_rank,
        ),
        None,
    )


class _RecordingHybridEPBuffer:
    def __init__(
        self,
        dispatch_results=(),
        combine_results=(),
        replay_config=None,
    ):
        self.dispatch_results = list(dispatch_results)
        self.combine_results = list(combine_results)
        self.replay_config = replay_config
        self.dispatch_calls = []
        self.combine_calls = []
        self.update_template_config_calls = []

    def dispatch_with_permute(self, **kwargs):
        self.dispatch_calls.append(kwargs)
        return self.dispatch_results.pop(0)

    def combine_with_unpermute(self, **kwargs):
        self.combine_calls.append(kwargs)
        return self.combine_results.pop(0)

    def update_template_config(self, **kwargs):
        self.update_template_config_calls.append(kwargs)
        return self.replay_config


class _ConstructedHybridEPBuffer:
    instances = []

    class Config:
        def __init__(
            self,
            hidden_dim,
            max_num_of_tokens_per_rank,
            num_local_experts,
        ):
            self.hidden_dim = hidden_dim
            self.max_num_of_tokens_per_rank = max_num_of_tokens_per_rank
            self.num_of_experts_per_rank = num_local_experts

    def __init__(
        self,
        group,
        hidden_dim,
        max_num_of_tokens_per_rank,
        num_local_experts,
        use_fp8,
        num_sms_dispatch_api=None,
        num_sms_combine_api=None,
        num_sms_preprocessing_api=None,
        load_cached_kernels=False,
    ):
        self.group = group
        self.config = self.Config(
            hidden_dim,
            max_num_of_tokens_per_rank,
            num_local_experts,
        )
        self.hidden_dim = hidden_dim
        self.max_num_of_tokens_per_rank = max_num_of_tokens_per_rank
        self.num_local_experts = num_local_experts
        self.use_fp8 = use_fp8
        self.num_sms_dispatch_api = num_sms_dispatch_api
        self.num_sms_combine_api = num_sms_combine_api
        self.num_sms_preprocessing_api = num_sms_preprocessing_api
        self.load_cached_kernels = load_cached_kernels
        self.instances.append(self)


class _HybridEPRuntimeModule:
    HybridEPBuffer = _ConstructedHybridEPBuffer


def _bind_buffer(manager, buffer):
    def _get_buffer(*args, **kwargs):
        del args, kwargs
        manager._active_buffer = buffer
        return buffer

    manager._get_buffer = _get_buffer


class TestHybridEPBackendSelection(unittest.TestCase):
    def test_deep_ep_and_hybrid_ep_imports_are_separate(self):
        import paddlefleet_ops as ops
        from paddlefleet_ops import deep_ep, hybrid_ep

        self.assertIn("hybrid_ep", ops.__dict__)
        self.assertTrue(hasattr(deep_ep, "Buffer"))
        self.assertFalse(hasattr(deep_ep, "HybridEPBuffer"))
        self.assertTrue(hasattr(hybrid_ep, "HybridEPBuffer"))
        self.assertIsNot(deep_ep, hybrid_ep)

    def test_dispatcher_type_selects_hybrid_ep_only_when_requested(self):
        self.assertFalse(is_hybrid_ep_backend_selected())
        for dispatcher_type in ("allgather", "alltoall", "deepep"):
            with self.subTest(dispatcher_type=dispatcher_type):
                self.assertFalse(is_hybrid_ep_backend_selected(dispatcher_type))
        self.assertTrue(is_hybrid_ep_backend_selected("hybridep"))

        for dispatcher_type in ("unknown", "hybrid", "hybrid_ep", "deep_ep"):
            with (
                self.subTest(dispatcher_type=dispatcher_type),
                self.assertRaisesRegex(ValueError, "moe_token_dispatcher_type"),
            ):
                is_hybrid_ep_backend_selected(dispatcher_type)

    def test_flex_dispatcher_uses_hybrid_ep_manager(self):
        group = _HybridEPGroup(nranks=2)

        dispatcher = MoEFlexTokenDispatcher(
            num_local_experts=2,
            num_experts_per_tok=2,
            n_routed_experts=4,
            ep_group=group,
            dispatcher_type="hybridep",
        )

        self.assertIsInstance(dispatcher._comm_manager, _HybridEPManager)
        self.assertIs(dispatcher._comm_manager.group, group)
        self.assertEqual(dispatcher._comm_manager.num_local_experts, 2)


class TestHybridEPMetadata(unittest.TestCase):
    def test_topk_indices_can_be_converted_to_dense_metadata(self):
        manager = _new_hybrid_manager(num_experts=4)
        token_indices = paddle.to_tensor(
            [[1, -1], [0, 2], [3, 1]], dtype="int64"
        )
        token_weights = paddle.to_tensor(
            [[0.5, 0.0], [0.25, 0.75], [0.6, 0.4]],
            dtype="float16",
        )

        routing_map, probs = manager._indices_to_dense_metadata(
            token_indices, token_weights
        )

        self.assertEqual(
            routing_map.numpy().tolist(),
            [
                [False, True, False, False],
                [True, False, True, False],
                [False, True, False, True],
            ],
        )
        self.assertEqual(probs.dtype, paddle.float32)
        self.assertTrue(
            paddle.allclose(
                probs,
                paddle.to_tensor(
                    [
                        [0.0, 0.5, 0.0, 0.0],
                        [0.25, 0.0, 0.75, 0.0],
                        [0.0, 0.4, 0.0, 0.6],
                    ],
                    dtype="float32",
                ),
                atol=1e-3,
            ).item()
        )

    def test_setup_metadata_prefers_router_topk_metadata(self):
        manager = _new_hybrid_manager(router_topk=2, num_experts=4)
        routing_map = paddle.to_tensor(
            [[True, False, True, False], [False, True, True, False]],
            dtype="bool",
        )
        probs = paddle.to_tensor(
            [[0.7, 0.0, 0.3, 0.0], [0.0, 0.6, 0.4, 0.0]],
            dtype="float16",
        )
        topk_weights = paddle.to_tensor(
            [[0.7, 0.3], [0.6, 0.4]], dtype="float32"
        )
        topk_indices = paddle.to_tensor([[0, 2], [1, 2]], dtype="int64")

        manager.setup_metadata(
            routing_map,
            probs,
            topk_weights=topk_weights,
            topk_indices=topk_indices,
        )

        self.assertEqual(manager.routing_probs.dtype, paddle.float32)
        self.assertTrue(
            paddle.allclose(manager.token_probs, topk_weights, atol=1e-6).item()
        )
        self.assertEqual(
            manager.token_indices.numpy().tolist(), [[0, 2], [1, 2]]
        )
        self.assertTrue(manager.token_indices.stop_gradient)

    def test_setup_metadata_falls_back_to_dense_topk(self):
        manager = _new_hybrid_manager(router_topk=2, num_experts=4)
        manager.setup_metadata(
            paddle.to_tensor([[True, False, True, False]], dtype="bool"),
            paddle.to_tensor([[0.2, 0.1, 0.7, 0.0]], dtype="float32"),
        )

        self.assertEqual(manager.token_indices.numpy().tolist(), [[2, 0]])
        self.assertTrue(
            paddle.allclose(
                manager.token_probs,
                paddle.to_tensor([[0.7, 0.2]], dtype="float32"),
                atol=1e-6,
            ).item()
        )

    def test_runtime_count_and_layout_helpers(self):
        manager = _new_hybrid_manager(router_topk=2, num_local_experts=3)
        self.assertEqual(
            manager._get_num_permuted_tokens_upper_bound(5),
            10 * manager.group.nranks + 3 * (FP8_ALIGN - 1),
        )

        local_expert_routing_map = paddle.to_tensor(
            [
                [True, False, False],
                [False, True, True],
                [False, True, False],
            ],
            dtype="bool",
        )
        tokens_per_expert = manager._extract_tokens_per_expert(
            2, local_expert_routing_map
        )
        self.assertEqual(tokens_per_expert.numpy().tolist(), [1, 1, 1])

        hidden_states = paddle.ones([2, 3], dtype="float32")
        self.assertIs(
            manager.get_permuted_hidden_states_by_experts(hidden_states),
            hidden_states,
        )

        manager.dispatched_probs = paddle.to_tensor([0.25, 0.5])
        restored = manager.get_restored_hidden_states_by_experts(hidden_states)
        self.assertEqual(
            restored.numpy().tolist(),
            [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
        )

    def test_dispatch_metadata_uses_cached_dense_metadata_when_available(self):
        manager = _new_hybrid_manager(router_topk=2, num_experts=4)
        routing_map = paddle.to_tensor(
            [[True, False, True, False]], dtype="bool"
        )
        routing_probs = paddle.to_tensor(
            [[0.2, 0.0, 0.8, 0.0]], dtype="float32"
        )
        manager.routing_map = routing_map
        manager.routing_probs = routing_probs

        dispatch_map, dispatch_probs = manager._get_dispatch_metadata(
            paddle.to_tensor([[3, 1]], dtype="int64"),
            paddle.ones([1, 2], dtype="float32"),
        )

        self.assertIs(dispatch_map, routing_map)
        self.assertIs(dispatch_probs, routing_probs)

    def test_dispatch_metadata_can_start_from_topk_metadata(self):
        manager = _new_hybrid_manager(router_topk=2, num_experts=4)

        routing_map, routing_probs = manager._get_dispatch_metadata(
            paddle.to_tensor([[3, 1], [0, -1]], dtype="int64"),
            paddle.to_tensor([[0.75, 0.25], [1.0, 0.0]], dtype="float32"),
        )

        self.assertEqual(
            routing_map.numpy().tolist(),
            [
                [False, True, False, True],
                [True, False, False, False],
            ],
        )
        self.assertTrue(
            paddle.allclose(
                routing_probs,
                paddle.to_tensor(
                    [
                        [0.0, 0.25, 0.0, 0.75],
                        [1.0, 0.0, 0.0, 0.0],
                    ],
                    dtype="float32",
                ),
                atol=1e-6,
            ).item()
        )

        with self.assertRaisesRegex(AssertionError, "routing metadata"):
            manager._get_dispatch_metadata(None, None)


class TestHybridEPDispatchBoundary(unittest.TestCase):
    def tearDown(self):
        fused_a2a.reset_hybrid_ep_buffer()

    def test_hybrid_ep_buffer_is_shared_across_managers(self):
        group = _HybridEPGroup(nranks=2)
        manager_a = _new_hybrid_manager(group=group)
        manager_b = _new_hybrid_manager(group=group)
        _ConstructedHybridEPBuffer.instances = []

        with patch.object(
            fused_a2a,
            "hybrid_ep",
            _HybridEPRuntimeModule,
            create=True,
        ):
            first = manager_a._get_buffer(paddle.zeros([4, 8], dtype="float32"))
            second = manager_b._get_buffer(
                paddle.zeros([2, 8], dtype="float32")
            )
            larger = manager_b._get_buffer(
                paddle.zeros([8, 8], dtype="float32")
            )
            reused_larger = manager_a._get_buffer(
                paddle.zeros([6, 8], dtype="float32")
            )

        self.assertIs(first, second)
        self.assertIsNot(first, larger)
        self.assertIs(larger, reused_larger)
        self.assertEqual(
            [
                item.max_num_of_tokens_per_rank
                for item in _ConstructedHybridEPBuffer.instances
            ],
            [4, 8],
        )

    def test_hybrid_ep_buffer_rebuild_check_respects_explicit_sms(self):
        group = _HybridEPGroup(nranks=2)
        fused_a2a._hybrid_ep_buffer = _ConstructedHybridEPBuffer(
            group=group,
            hidden_dim=8,
            max_num_of_tokens_per_rank=4,
            num_local_experts=2,
            use_fp8=False,
            num_sms_dispatch_api=8,
            num_sms_combine_api=16,
            num_sms_preprocessing_api=32,
        )

        self.assertFalse(
            fused_a2a._need_new_hybrid_ep_buffer(group, 8, 4, 2, 8, 16, 32)
        )
        self.assertTrue(
            fused_a2a._need_new_hybrid_ep_buffer(group, 8, 4, 2, 9, 16, 32)
        )
        self.assertTrue(
            fused_a2a._need_new_hybrid_ep_buffer(group, 8, 4, 2, 8, 17, 32)
        )
        self.assertTrue(
            fused_a2a._need_new_hybrid_ep_buffer(group, 8, 4, 2, 8, 16, 33)
        )
        self.assertFalse(
            fused_a2a._need_new_hybrid_ep_buffer(
                group, 8, 4, 2, None, None, None
            )
        )

    def test_rank_max_tokens_use_ep_max_and_chunk_alignment(self):
        manager = _new_hybrid_manager(group=_HybridEPGroup(nranks=2))

        def fake_all_reduce(tensor, op=None, group=None):
            self.assertEqual(int(tensor.item()), 3)
            self.assertEqual(op, paddle.distributed.ReduceOp.MAX)
            self.assertIs(group, manager.group)
            tensor.set_value(paddle.to_tensor([65], dtype="int64"))

        with patch(
            "paddle.distributed.all_reduce", side_effect=fake_all_reduce
        ):
            max_tokens = manager._get_max_num_tokens_per_rank(
                3, paddle.CPUPlace()
            )

        self.assertEqual(max_tokens, HYBRIDEP_TOKEN_ALIGNMENT * 2)

    def test_pad_tokens_to_rank_max_handles_noop_none_and_invalid(self):
        manager = _new_hybrid_manager(group=_HybridEPGroup(nranks=1))

        self.assertIsNone(manager._pad_tokens_to_rank_max(None, 4))

        tensor = paddle.ones([4, 2], dtype="float32")
        self.assertIs(manager._pad_tokens_to_rank_max(tensor, 4), tensor)

        with self.assertRaisesRegex(AssertionError, "local tokens <= EP max"):
            manager._pad_tokens_to_rank_max(
                paddle.ones([5, 2], dtype="float32"), 4
            )

    def test_dispatch_with_permute_uses_hybrid_ep_runtime_contract(self):
        routing_map = paddle.to_tensor(
            [[True, False], [False, True]], dtype="bool"
        )
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
            routing_map=routing_map,
            routing_probs=paddle.to_tensor(
                [[1.0, 0.0], [0.0, 1.0]], dtype="float32"
            ),
        )
        padded_counts = paddle.to_tensor([1, 1], dtype="int64")
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    paddle.zeros([2, 4], dtype="float32"),
                    paddle.ones([2], dtype="float32"),
                    None,
                    padded_counts,
                    _make_hybrid_ep_handle(
                        num_dispatched_tokens=2,
                        local_expert_routing_map=routing_map,
                    ),
                )
            ]
        )
        _bind_buffer(manager, buffer)

        manager._dispatch_with_permute_impl(
            paddle.zeros([2, 4], dtype="float32"),
            paddle.to_tensor([[0], [1]], dtype="int64"),
            paddle.ones([2, 1], dtype="float32"),
            use_fp8=False,
        )

        dispatch_kwargs = buffer.dispatch_calls[-1]
        self.assertEqual(
            dispatch_kwargs["hidden"].shape, [HYBRIDEP_TOKEN_ALIGNMENT, 4]
        )
        self.assertEqual(
            dispatch_kwargs["routing_map"].shape,
            [HYBRIDEP_TOKEN_ALIGNMENT, 2],
        )
        self.assertEqual(
            dispatch_kwargs["routing_map"][:2].numpy().tolist(),
            routing_map.numpy().tolist(),
        )
        self.assertEqual(
            dispatch_kwargs["routing_map"][2:].astype("int64").sum().item(), 0
        )
        self.assertFalse(dispatch_kwargs["use_fp8"])
        self.assertIsNone(dispatch_kwargs["pad_multiple"])
        self.assertTrue(dispatch_kwargs["non_blocking"])
        self.assertIs(manager.padded_tokens_per_expert, padded_counts)
        self.assertEqual(manager.num_permuted_tokens, 2)
        self.assertEqual(manager.tokens_per_expert.numpy().tolist(), [1, 1])

    def test_dispatch_pads_rank_tokens_to_chunk_multiple(self):
        routing_map = paddle.to_tensor(
            [[True, False], [False, True], [True, False]], dtype="bool"
        )
        routing_probs = paddle.to_tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.0]], dtype="float32"
        )
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
            routing_map=routing_map,
            routing_probs=routing_probs,
        )
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    paddle.zeros([3, 4], dtype="float32"),
                    paddle.ones([3], dtype="float32"),
                    None,
                    paddle.to_tensor([2, 1], dtype="int64"),
                    _make_hybrid_ep_handle(
                        num_dispatched_tokens=3,
                        local_expert_routing_map=routing_map,
                    ),
                )
            ]
        )
        _bind_buffer(manager, buffer)

        manager._dispatch_with_permute_impl(
            paddle.ones([3, 4], dtype="float32"),
            paddle.to_tensor([[0], [1], [0]], dtype="int64"),
            paddle.ones([3, 1], dtype="float32"),
            use_fp8=False,
        )

        dispatch_kwargs = buffer.dispatch_calls[-1]
        self.assertEqual(
            dispatch_kwargs["hidden"].shape, [HYBRIDEP_TOKEN_ALIGNMENT, 4]
        )
        self.assertEqual(
            dispatch_kwargs["routing_map"].shape,
            [HYBRIDEP_TOKEN_ALIGNMENT, 2],
        )
        self.assertEqual(
            dispatch_kwargs["probs"].shape, [HYBRIDEP_TOKEN_ALIGNMENT, 2]
        )
        self.assertEqual(
            dispatch_kwargs["hidden"][:3].numpy().tolist(),
            [[1.0] * 4] * 3,
        )
        self.assertEqual(
            dispatch_kwargs["hidden"][3:].astype("int64").sum().item(), 0
        )
        self.assertEqual(manager._num_unpadded_tokens, 3)
        self.assertEqual(manager.num_permuted_tokens, 3)

    def test_dispatch_trims_padded_runtime_outputs_to_active_rows(self):
        routing_map = paddle.to_tensor(
            [[True, False], [False, True], [True, False]], dtype="bool"
        )
        routing_probs = paddle.to_tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.0]], dtype="float32"
        )
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
            routing_map=routing_map,
            routing_probs=routing_probs,
        )
        dispatched = paddle.arange(
            HYBRIDEP_TOKEN_ALIGNMENT * 4, dtype="float32"
        ).reshape([HYBRIDEP_TOKEN_ALIGNMENT, 4])
        dispatched_probs = paddle.arange(
            HYBRIDEP_TOKEN_ALIGNMENT, dtype="float32"
        )
        scale = paddle.arange(
            HYBRIDEP_TOKEN_ALIGNMENT, dtype="float32"
        ).reshape([HYBRIDEP_TOKEN_ALIGNMENT, 1])
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    dispatched,
                    dispatched_probs,
                    scale,
                    paddle.to_tensor([2, 1], dtype="int64"),
                    _make_hybrid_ep_handle(
                        num_dispatched_tokens=3,
                        local_expert_routing_map=routing_map,
                    ),
                )
            ]
        )
        _bind_buffer(manager, buffer)

        hidden_states, probs, scale_handle = (
            manager._dispatch_with_permute_impl(
                paddle.ones([3, 4], dtype="float32"),
                paddle.to_tensor([[0], [1], [0]], dtype="int64"),
                paddle.ones([3, 1], dtype="float32"),
                use_fp8=False,
            )
        )

        self.assertEqual(hidden_states.shape, [3, 4])
        self.assertEqual(
            hidden_states.numpy().tolist(), dispatched[:3].numpy().tolist()
        )
        self.assertEqual(probs.shape, [3])
        self.assertEqual(probs.numpy().tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(scale_handle.shape, [3, 1])
        self.assertEqual(scale_handle.numpy().tolist(), [[0.0], [1.0], [2.0]])
        self.assertEqual(manager.num_permuted_tokens, 3)
        self.assertEqual(manager.tokens_per_expert.numpy().tolist(), [2, 1])

    def test_fp8_dispatch_quantizes_and_aligns_expert_inputs(self):
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
            routing_map=paddle.to_tensor(
                [[True, False], [False, True]], dtype="bool"
            ),
            routing_probs=paddle.to_tensor(
                [[1.0, 0.0], [0.0, 1.0]], dtype="float32"
            ),
        )
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    paddle.zeros([2, 4], dtype="float32"),
                    paddle.ones([2], dtype="float32"),
                    paddle.ones([2, 1], dtype="float32"),
                    paddle.to_tensor([1, 1], dtype="int64"),
                    _make_hybrid_ep_handle(
                        num_dispatched_tokens=2,
                        local_expert_routing_map=manager.routing_map,
                    ),
                )
            ]
        )
        _bind_buffer(manager, buffer)

        manager._dispatch_with_permute_impl(
            paddle.ones([2, 4], dtype="bfloat16"),
            paddle.to_tensor([[0], [1]], dtype="int64"),
            paddle.ones([2, 1], dtype="float32"),
            use_fp8=True,
        )

        dispatch_kwargs = buffer.dispatch_calls[-1]
        self.assertEqual(dispatch_kwargs["hidden"].dtype, paddle.float8_e4m3fn)
        self.assertTrue(dispatch_kwargs["use_fp8"])
        self.assertEqual(dispatch_kwargs["pad_multiple"], FP8_ALIGN)
        self.assertEqual(
            dispatch_kwargs["scaling_factor"].shape,
            [HYBRIDEP_TOKEN_ALIGNMENT, 1],
        )

    def test_public_dispatch_uses_router_metadata_and_records_runtime_state(
        self,
    ):
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
        )
        manager.setup_metadata(
            paddle.to_tensor([[True, False], [False, True]], dtype="bool"),
            paddle.to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
            topk_weights=paddle.ones([2, 1], dtype="float32"),
            topk_indices=paddle.to_tensor([[0], [1]], dtype="int64"),
        )
        dispatched = paddle.full([2, 4], 2.0, dtype="float32")
        dispatched_probs = paddle.to_tensor([1.0, 1.0], dtype="float32")
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    dispatched,
                    dispatched_probs,
                    None,
                    paddle.to_tensor([1, 1], dtype="int64"),
                    _make_hybrid_ep_handle(
                        num_dispatched_tokens=2,
                        local_expert_routing_map=manager.routing_map,
                    ),
                )
            ]
        )
        _bind_buffer(manager, buffer)

        output, fp8_handle = manager.dispatch(
            paddle.zeros([2, 4], dtype="float32"),
            fp8_dispatch=False,
            async_finish=True,
        )

        self.assertIsNone(fp8_handle)
        self.assertTrue(paddle.allclose(output, dispatched).item())
        self.assertTrue(
            paddle.allclose(manager.dispatched_probs, dispatched_probs).item()
        )
        self.assertIsNone(manager.dispatched_indices)
        self.assertEqual(manager.num_permuted_tokens, 2)
        self.assertEqual(manager.tokens_per_expert.numpy().tolist(), [1, 1])
        self.assertFalse(buffer.dispatch_calls[-1]["use_fp8"])

    def test_public_dispatch_returns_fp8_scale_handle(self):
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
        )
        scale = paddle.ones([2, 1], dtype="float32")
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[
                (
                    paddle.zeros([2, 4], dtype="float32"),
                    paddle.ones([2], dtype="float32"),
                    scale,
                    paddle.to_tensor([1, 1], dtype="int64"),
                    _make_hybrid_ep_handle(
                        num_dispatched_tokens=2,
                        local_expert_routing_map=paddle.to_tensor(
                            [[True, False], [False, True]], dtype="bool"
                        ),
                    ),
                )
            ]
        )
        _bind_buffer(manager, buffer)

        token_indices = paddle.to_tensor([[0], [1]], dtype="int64")
        token_weights = paddle.ones([2, 1], dtype="float32")

        _, fp8_handle = manager.dispatch_overlap(
            paddle.ones([2, 4], dtype="bfloat16"),
            token_indices,
            token_weights,
            fp8_dispatch=True,
        )

        self.assertTrue(paddle.allclose(fp8_handle["scale"], scale).item())
        self.assertIs(manager.token_indices, token_indices)
        self.assertIs(manager.token_probs, token_weights)
        self.assertEqual(
            manager.get_number_of_tokens_per_expert().numpy().tolist(), [1, 1]
        )

    def test_public_combine_rejects_overlap_and_clears_runtime_state(self):
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
        )
        buffer = _RecordingHybridEPBuffer(
            combine_results=[
                (
                    paddle.ones([HYBRIDEP_TOKEN_ALIGNMENT, 4], dtype="float32"),
                    None,
                )
            ]
        )
        manager._active_buffer = buffer
        manager.handle = _make_hybrid_ep_handle(token_data_type="BF16")
        manager.dispatched_probs = paddle.ones([2], dtype="float32")
        manager.tokens_per_expert = paddle.to_tensor([1, 1], dtype="int64")
        manager.padded_tokens_per_expert = paddle.to_tensor(
            [1, 1], dtype="int64"
        )
        manager.num_permuted_tokens = 2
        manager._num_unpadded_tokens = 2

        with self.assertRaisesRegex(NotImplementedError, "combine overlap"):
            manager.combine(
                paddle.zeros([2, 4], dtype="float32"),
                combine_overlap_handle={"fn": object()},
            )

        output = manager.combine(
            paddle.zeros([2, 4], dtype="float32"),
            async_finish=True,
        )

        self.assertEqual(output.numpy().tolist(), [[1.0] * 4, [1.0] * 4])
        self.assertIsNone(manager.handle)
        self.assertIsNone(manager.dispatched_probs)
        self.assertIsNone(manager.num_permuted_tokens)
        self.assertIsNone(manager._num_unpadded_tokens)
        self.assertEqual(buffer.combine_calls[-1]["hidden"].shape, [2, 4])
        self.assertIsNone(buffer.combine_calls[-1].get("pad_multiple"))

    def test_public_combine_keeps_matching_unpadded_shape(self):
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
        )
        combined = paddle.full([2, 4], 3.0, dtype="float32")
        buffer = _RecordingHybridEPBuffer(combine_results=[(combined, None)])
        manager._active_buffer = buffer
        manager.handle = _make_hybrid_ep_handle(token_data_type="BF16")
        manager.num_permuted_tokens = 2
        manager._num_unpadded_tokens = 2

        output = manager.combine(paddle.zeros([2, 4], dtype="float32"))

        self.assertEqual(output.numpy().tolist(), [[3.0] * 4, [3.0] * 4])
        self.assertEqual(buffer.combine_calls[-1]["hidden"].shape, [2, 4])
        self.assertIsNone(manager._num_unpadded_tokens)

    def test_dispatched_metadata_is_unavailable_in_hybrid_fused_mode(self):
        manager = _new_hybrid_manager()

        with self.assertRaisesRegex(NotImplementedError, "dispatch metadata"):
            manager.get_dispatched_metadata()

        manager.dispatched_indices = paddle.to_tensor([0, 1], dtype="int64")
        manager.dispatched_probs = paddle.to_tensor(
            [0.5, 0.25], dtype="float32"
        )

        self.assertEqual(
            tuple(
                item.numpy().tolist()
                for item in manager.get_dispatched_metadata()
            ),
            ([0, 1], [0.5, 0.25]),
        )


class TestHybridEPAutogradBridge(unittest.TestCase):
    def test_dispatch_pylayer_maps_dense_prob_grad_back_to_topk(self):
        grad_x = paddle.concat(
            [
                paddle.full([2, 4], 3.0, dtype="float32"),
                paddle.full([2, 4], 9.0, dtype="float32"),
            ],
            axis=0,
        )
        grad_dense_probs = paddle.to_tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ],
            dtype="float32",
        )
        buffer = _RecordingHybridEPBuffer(
            combine_results=[(grad_x, grad_dense_probs)]
        )
        handle = _make_hybrid_ep_handle(token_data_type="UINT8")

        class DispatchingManager:
            def _dispatch_with_permute_impl(
                self, x, token_indices, token_probs, use_fp8
            ):
                self._active_buffer = buffer
                self.handle = handle
                self.use_fp8 = use_fp8
                return (
                    x * 2,
                    token_probs.reshape([-1]),
                    paddle.ones([2, 1], dtype="float32"),
                )

        manager = DispatchingManager()
        x = paddle.zeros([2, 4], dtype="float32")
        x.stop_gradient = False
        token_indices = paddle.to_tensor([[2, 0], [1, 1]], dtype="int64")
        token_probs = paddle.ones([2, 2], dtype="float32")
        token_probs.stop_gradient = False

        recv_x, recv_probs, scale = HybridEPDispatch.apply(
            x, token_indices, token_probs, manager, True
        )
        (recv_x.sum() + recv_probs.sum() + scale.sum()).backward()

        self.assertTrue(manager.use_fp8)
        self.assertEqual(x.grad.numpy().tolist(), [[3.0] * 4, [3.0] * 4])
        self.assertTrue(
            paddle.allclose(
                token_probs.grad,
                paddle.to_tensor([[0.3, 0.1], [0.5, 0.5]], dtype="float32"),
                atol=1e-6,
            ).item()
        )
        self.assertEqual(buffer.combine_calls[-1]["pad_multiple"], FP8_ALIGN)
        self.assertEqual(
            buffer.combine_calls[-1]["hidden"].dtype, paddle.float32
        )
        self.assertEqual(
            buffer.combine_calls[-1]["probs"].dtype, paddle.float32
        )

    def test_dispatch_pylayer_trims_padded_grad_without_dense_probs(self):
        grad_x = paddle.concat(
            [
                paddle.full([2, 4], 5.0, dtype="float32"),
                paddle.full([2, 4], 7.0, dtype="float32"),
            ],
            axis=0,
        )
        buffer = _RecordingHybridEPBuffer(combine_results=[(grad_x, None)])
        handle = _make_hybrid_ep_handle(token_data_type="BF16")

        class DispatchingManager:
            def _dispatch_with_permute_impl(
                self, x, token_indices, token_probs, use_fp8
            ):
                self._active_buffer = buffer
                self.handle = handle
                return x * 2, token_probs.reshape([-1]), None

        manager = DispatchingManager()
        x = paddle.zeros([2, 4], dtype="float32")
        x.stop_gradient = False
        token_indices = paddle.to_tensor([[0], [1]], dtype="int64")
        token_probs = paddle.ones([2, 1], dtype="float32")

        recv_x, recv_probs, _ = HybridEPDispatch.apply(
            x,
            token_indices,
            token_probs,
            manager,
            False,
        )
        (recv_x.sum() + recv_probs.sum()).backward()

        self.assertEqual(x.grad.numpy().tolist(), [[5.0] * 4, [5.0] * 4])
        self.assertEqual(
            buffer.combine_calls[-1]["probs"].dtype, paddle.float32
        )

    def test_combine_pylayer_requires_active_permuted_rows(self):
        manager = _new_hybrid_manager(group=_HybridEPGroup(nranks=1))
        manager.handle = _make_hybrid_ep_handle(token_data_type="BF16")
        manager._active_buffer = _RecordingHybridEPBuffer(
            combine_results=[(paddle.ones([2, 4], dtype="float32"), None)]
        )

        with self.assertRaisesRegex(AssertionError, "active permuted rows"):
            HybridEPCombine.apply(
                paddle.zeros([3, 4], dtype="float32"),
                manager,
                2,
            )

    def test_combine_pylayer_accepts_explicit_active_permuted_rows(self):
        manager = _new_hybrid_manager(group=_HybridEPGroup(nranks=1))
        manager.handle = _make_hybrid_ep_handle(token_data_type="BF16")
        manager._active_buffer = _RecordingHybridEPBuffer(
            combine_results=[(paddle.full([2, 4], 4.0, dtype="float32"), None)]
        )

        result = HybridEPCombine.apply(
            paddle.zeros([2, 4], dtype="float32"),
            manager,
            2,
        )

        self.assertEqual(result.numpy().tolist(), [[4.0] * 4, [4.0] * 4])

    def test_combine_pylayer_replays_dispatch_in_backward(self):
        combined = paddle.ones([2, 4], dtype="float32")
        replay_config = _HybridEPHandleConfig(token_data_type="UINT16")
        buffer = _RecordingHybridEPBuffer(
            combine_results=[(combined, None)],
            dispatch_results=[
                (
                    paddle.full([3, 4], 2.0, dtype="float32"),
                    None,
                    None,
                    None,
                    None,
                )
            ],
            replay_config=replay_config,
        )
        handle = _make_hybrid_ep_handle(
            tokens_per_rank=8,
            token_data_type="UINT8",
            num_experts_per_rank=2,
        )
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            router_topk=1,
            num_experts=2,
            num_local_experts=2,
        )
        manager.handle = handle
        manager._active_buffer = buffer
        manager.tokens_per_expert = paddle.to_tensor([1, 1], dtype="int64")
        manager.padded_tokens_per_expert = paddle.to_tensor(
            [1, 1], dtype="int64"
        )
        x = paddle.zeros([2, 4], dtype="float32")
        x.stop_gradient = False

        HybridEPCombine.apply(x, manager).sum().backward()

        self.assertEqual(x.grad.numpy().tolist(), [[2.0] * 4, [2.0] * 4])
        self.assertEqual(
            buffer.update_template_config_calls,
            [
                {
                    "hidden_dim": 4,
                    "num_of_tokens_per_rank": 8,
                    "num_local_experts": 2,
                    "use_fp8": False,
                }
            ],
        )
        self.assertIs(buffer.dispatch_calls[-1]["handle"][7], replay_config)
        self.assertEqual(buffer.dispatch_calls[-1]["pad_multiple"], FP8_ALIGN)
        self.assertFalse(buffer.dispatch_calls[-1]["non_blocking"])

    def test_replay_dispatch_backward_preserves_bf16_handle_contract(self):
        grad_output = paddle.ones([3, 4], dtype="float32")
        replayed_grad = paddle.arange(12, dtype="float32").reshape([3, 4])
        handle = _make_hybrid_ep_handle(token_data_type="BF16")
        buffer = _RecordingHybridEPBuffer(
            dispatch_results=[(replayed_grad, None, None, None, None)]
        )

        result = _replay_hybrid_ep_dispatch_backward(
            buffer,
            handle,
            grad_output,
            num_permuted_tokens=2,
            use_fp8_dispatch=False,
        )

        self.assertEqual(
            result.numpy().tolist(), replayed_grad[:2].numpy().tolist()
        )
        self.assertEqual(buffer.update_template_config_calls, [])
        self.assertIs(buffer.dispatch_calls[-1]["handle"], handle)
        self.assertIsNone(buffer.dispatch_calls[-1]["pad_multiple"])
        self.assertFalse(buffer.dispatch_calls[-1]["non_blocking"])


class TestHybridEPExpertInputCounts(unittest.TestCase):
    def test_gen_m_indices_accepts_tensor_and_empty_counts(self):
        node = ExpertsGroupGemmContiguousNode(
            _ExpertsGroupGemmCustomMap(),
            use_fp8_mlp=False,
        )

        self.assertEqual(
            node.gen_m_indices(paddle.to_tensor([1, 0, 2], dtype="int64"))
            .numpy()
            .tolist(),
            [0, 2, 2],
        )
        self.assertEqual(
            node.gen_m_indices(paddle.to_tensor([], dtype="int64")).shape,
            [0],
        )

    def test_prepare_expert_counts_matches_expert_compute_contract(self):
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            num_experts=3,
            num_local_experts=3,
        )
        manager.padded_tokens_per_expert = paddle.to_tensor(
            [2, 0, 1], dtype="int32"
        )
        manager.num_permuted_tokens = 3
        custom_map = _HybridEPCustomMap(manager)

        counts, num_tokens = _hybrid_ep_prepare_expert_counts(
            custom_map,
            use_fp8_mlp=False,
            moe_expert_fusion=False,
        )

        self.assertEqual(counts, [2, 0, 1])
        self.assertEqual(num_tokens, 3)

        manager.padded_tokens_per_expert = paddle.to_tensor(
            [4, 2], dtype="int32"
        )
        manager.num_permuted_tokens = 6
        counts, num_tokens = _hybrid_ep_prepare_expert_counts(
            custom_map,
            use_fp8_mlp=True,
            moe_expert_fusion=True,
        )

        self.assertIsInstance(counts, paddle.Tensor)
        self.assertEqual(counts.dtype, paddle.int64)
        self.assertEqual(counts.numpy().tolist(), [4, 2])
        self.assertEqual(num_tokens, 6)

    def test_prepare_expert_counts_requires_hybrid_ep_counts(self):
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            num_experts=3,
            num_local_experts=3,
        )
        custom_map = _HybridEPCustomMap(manager)

        with self.assertRaisesRegex(AssertionError, "padded_tokens_per_expert"):
            _hybrid_ep_prepare_expert_counts(
                custom_map,
                use_fp8_mlp=False,
                moe_expert_fusion=False,
            )
        manager.padded_tokens_per_expert = paddle.to_tensor(
            [1, 1], dtype="int64"
        )
        with self.assertRaisesRegex(AssertionError, "num_permuted_tokens"):
            _hybrid_ep_prepare_expert_counts(
                custom_map,
                use_fp8_mlp=False,
                moe_expert_fusion=False,
            )

    def test_padding_helpers_restore_forward_shapes(self):
        tensor = paddle.ones([2, 3], dtype="float32")
        self.assertIs(_pad_front_rows(tensor, (2, 3)), tensor)

        padded = _pad_front_rows(tensor, (4, 3))
        self.assertEqual(padded.shape, [4, 3])
        self.assertEqual(padded[2:].numpy().tolist(), [[0.0] * 3, [0.0] * 3])

        restored = _restore_hybrid_ep_prob_grad_shape(
            paddle.to_tensor([[0.25], [0.5]], dtype="float32"),
            (4,),
        )
        self.assertEqual(restored.shape, [4])
        self.assertEqual(restored.numpy().tolist(), [0.25, 0.5, 0.0, 0.0])

    def test_hybrid_ep_moe_pylayer_restores_padded_zero_token_grads(self):
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            num_experts=2,
            num_local_experts=2,
        )
        manager.padded_tokens_per_expert = paddle.to_tensor(
            [0, 0], dtype="int64"
        )
        manager.num_permuted_tokens = 0
        custom_map = _HybridEPCustomMap(
            manager,
            experts=[_TinyExpert(), _TinyExpert()],
        )
        hidden_states = paddle.ones([2, 2], dtype="float32")
        hidden_states.stop_gradient = False
        dispatched_probs = paddle.ones([2], dtype="float32")
        dispatched_probs.stop_gradient = False

        output = HybridEPMoePyLayer.apply(
            hidden_states,
            dispatched_probs,
            custom_map,
            use_fp8_mlp=False,
            moe_deep_gemm=False,
            moe_expert_fusion=False,
            is_first_fwd=True,
        )
        output.sum().backward()

        self.assertEqual(output.shape, [0, 2])
        self.assertEqual(hidden_states.grad.numpy().tolist(), [[0.0, 0.0]] * 2)
        self.assertEqual(dispatched_probs.grad.numpy().tolist(), [0.0, 0.0])

    def test_hybrid_ep_moe_pylayer_accepts_fp8_dispatch_scale(self):
        manager = _new_hybrid_manager(
            group=_HybridEPGroup(nranks=1),
            num_experts=2,
            num_local_experts=2,
        )
        manager.padded_tokens_per_expert = paddle.to_tensor(
            [0, 0], dtype="int64"
        )
        manager.num_permuted_tokens = 0
        custom_map = _HybridEPCustomMap(
            manager,
            experts=[_TinyExpert(), _TinyExpert()],
        )

        output = HybridEPMoePyLayer.apply(
            paddle.empty([2, 2], dtype=paddle.float8_e4m3fn),
            paddle.ones([2], dtype="float32"),
            custom_map,
            use_fp8_mlp=True,
            moe_deep_gemm=False,
            moe_expert_fusion=False,
            fp8_dispatched_handle={
                "scale": paddle.ones([2, 1], dtype="float32")
            },
        )

        self.assertEqual(output.shape, [0, 2])


if __name__ == "__main__":
    unittest.main()
