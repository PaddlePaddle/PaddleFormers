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

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import paddle.nn as nn


def _create_no_hcg_fleet():
    """Create a fleet mock that does NOT have _hcg attribute."""

    class NoHCGFleet:
        pass

    return NoHCGFleet()


class TestReshardCombineWeightImport(unittest.TestCase):
    """Tests for importing ReshardCombineWeight."""

    def test_import_reshard_combine_weight(self):
        """Test that ReshardCombineWeight can be imported."""
        from paddleformers.nn.moe.moe_allgather_layer import ReshardCombineWeight

        self.assertIsNotNone(ReshardCombineWeight)

    def test_import_moe_allgather_layer_v2(self):
        """Test that MOEAllGatherLayerV2 can be imported."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        self.assertIsNotNone(MOEAllGatherLayerV2)


class TestReshardCombineWeightForward(unittest.TestCase):
    """Tests for ReshardCombineWeight PyLayer."""

    def test_forward_stores_mask_and_group(self):
        """Test that forward stores mask and group in context."""
        from paddleformers.nn.moe.moe_allgather_layer import ReshardCombineWeight

        ctx = MagicMock()
        # Create input with some zeros (mask positions)
        x = paddle.to_tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], dtype="float32")

        mock_result = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        with patch(
            "paddleformers.nn.moe.moe_allgather_layer.reduce_scatter_group", return_value=mock_result
        ) as mock_rs:
            ReshardCombineWeight.forward(ctx, x, group=None)
            mock_rs.assert_called_once_with(x, group=None)
            # Check mask was computed
            expected_mask = paddle.to_tensor([[False, False], [True, True], [False, False]])
            np.testing.assert_allclose(ctx.mask.numpy(), expected_mask.numpy())
            self.assertIsNone(ctx.group)

    def test_forward_stores_custom_group(self):
        """Test that forward stores custom group in context."""
        from paddleformers.nn.moe.moe_allgather_layer import ReshardCombineWeight

        ctx = MagicMock()
        x = paddle.randn([4, 8], dtype="float32")
        mock_group = MagicMock()
        mock_result = paddle.randn([4, 8], dtype="float32")

        with patch("paddleformers.nn.moe.moe_allgather_layer.reduce_scatter_group", return_value=mock_result):
            ReshardCombineWeight.forward(ctx, x, group=mock_group)
            self.assertIs(ctx.group, mock_group)

    def test_forward_returns_reduce_scatter_result(self):
        """Test that forward returns the result of reduce_scatter_group."""
        from paddleformers.nn.moe.moe_allgather_layer import ReshardCombineWeight

        ctx = MagicMock()
        x = paddle.randn([4, 8], dtype="float32")
        mock_result = paddle.randn([2, 8], dtype="float32")

        with patch("paddleformers.nn.moe.moe_allgather_layer.reduce_scatter_group", return_value=mock_result):
            result = ReshardCombineWeight.forward(ctx, x, group=None)
            np.testing.assert_allclose(result.numpy(), mock_result.numpy())

    def test_forward_mask_computation(self):
        """Test mask computation for different zero patterns."""
        from paddleformers.nn.moe.moe_allgather_layer import ReshardCombineWeight

        ctx = MagicMock()
        # All zeros
        x_all_zeros = paddle.zeros([3, 4], dtype="float32")
        mock_result = paddle.randn([3, 4], dtype="float32")
        with patch("paddleformers.nn.moe.moe_allgather_layer.reduce_scatter_group", return_value=mock_result):
            ReshardCombineWeight.forward(ctx, x_all_zeros, group=None)
            self.assertTrue(ctx.mask.numpy().all())

        # No zeros
        ctx2 = MagicMock()
        x_no_zeros = paddle.ones([3, 4], dtype="float32") * 2.0
        with patch("paddleformers.nn.moe.moe_allgather_layer.reduce_scatter_group", return_value=mock_result):
            ReshardCombineWeight.forward(ctx2, x_no_zeros, group=None)
            self.assertFalse(ctx2.mask.numpy().any())


class TestReshardCombineWeightBackward(unittest.TestCase):
    """Tests for ReshardCombineWeight backward pass."""

    def test_backward_calls_all_gather_group(self):
        """Test that backward calls all_gather_group."""
        from paddleformers.nn.moe.moe_allgather_layer import ReshardCombineWeight

        ctx = MagicMock()
        ctx.group = None
        ctx.mask = paddle.to_tensor([[False, True], [True, False]], dtype="bool")
        grad = paddle.randn([2, 2], dtype="float32")
        # gathered must have same shape as mask for masked_fill to work
        mock_gathered = paddle.randn([2, 2], dtype="float32")

        with patch("paddleformers.nn.moe.moe_allgather_layer.all_gather_group", return_value=mock_gathered) as mock_ag:
            ReshardCombineWeight.backward(ctx, grad)
            mock_ag.assert_called_once_with(grad, group=None)

    def test_backward_applies_mask(self):
        """Test that backward applies masked_fill to zero out non-local expert positions."""
        from paddleformers.nn.moe.moe_allgather_layer import ReshardCombineWeight

        ctx = MagicMock()
        ctx.group = None
        mask = paddle.to_tensor([[False, True], [True, False], [False, False]], dtype="bool")
        ctx.mask = mask

        gathered = paddle.ones([3, 2], dtype="float32")
        gathered.clone()

        with patch("paddleformers.nn.moe.moe_allgather_layer.all_gather_group", return_value=gathered):
            result = ReshardCombineWeight.backward(ctx, paddle.randn([3, 2], dtype="float32"))

        # Masked positions should be zero
        self.assertEqual(result[mask].numpy()[0], 0.0)

    def test_backward_passes_group(self):
        """Test that backward passes group to all_gather_group."""
        from paddleformers.nn.moe.moe_allgather_layer import ReshardCombineWeight

        ctx = MagicMock()
        mock_group = MagicMock()
        ctx.group = mock_group
        ctx.mask = paddle.zeros([2, 2], dtype="bool")
        grad = paddle.randn([2, 2], dtype="float32")

        with patch(
            "paddleformers.nn.moe.moe_allgather_layer.all_gather_group", return_value=paddle.randn([2, 2])
        ) as mock_ag:
            ReshardCombineWeight.backward(ctx, grad)
            mock_ag.assert_called_once_with(grad, group=mock_group)


class TestMOEAllGatherLayerV2Init(unittest.TestCase):
    """Tests for MOEAllGatherLayerV2 initialization."""

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_basic(self, mock_rank, mock_ws):
        """Test basic initialization of MOEAllGatherLayerV2."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.moe_world_size = 1
        mock_gate.config.moe_rank = 0
        mock_gate.config.sequence_parallel = False
        mock_gate.num_experts_tensor = paddle.to_tensor(2, dtype="int64")
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(8, 8), nn.Linear(8, 8)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAllGatherLayerV2(
            gate=mock_gate,
            experts=experts,
            layer_idx=0,
        )
        self.assertFalse(layer.enable_reverse_token_drop)
        self.assertTrue(layer.use_padding)
        self.assertTrue(layer.use_expert_out_alltoall)
        self.assertEqual(layer.dense_token_type, 3)
        self.assertIsNone(layer.capacity_tensor)
        self.assertIsNone(layer.send_rank)
        self.assertIsNone(layer.local_expert_id)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_custom_params(self, mock_rank, mock_ws):
        """Test initialization with custom parameters."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.moe_world_size = 1
        mock_gate.config.moe_rank = 0
        mock_gate.config.sequence_parallel = False
        mock_gate.num_experts_tensor = paddle.to_tensor(2, dtype="int64")
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAllGatherLayerV2(
            gate=mock_gate,
            experts=experts,
            layer_idx=2,
            enable_reverse_token_drop=True,
            use_padding=False,
            use_expert_out_alltoall=False,
            dense_token_type=5,
        )
        self.assertTrue(layer.enable_reverse_token_drop)
        self.assertFalse(layer.use_padding)
        self.assertFalse(layer.use_expert_out_alltoall)
        self.assertEqual(layer.dense_token_type, 5)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_multimodal_experts_tuple(self, mock_rank, mock_ws):
        """Test multimodal_experts detection with tuple of expert counts."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.moe_world_size = 1
        mock_gate.config.moe_rank = 0
        mock_gate.config.sequence_parallel = False
        mock_gate.num_experts_tensor = paddle.to_tensor(6, dtype="int64")
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)] * 6)
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAllGatherLayerV2(
            gate=mock_gate,
            experts=experts,
            layer_idx=0,
            moe_num_experts=[4, 2],
        )
        self.assertTrue(layer.multimodal_experts)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_multimodal_experts_single(self, mock_rank, mock_ws):
        """Test multimodal_experts is False with single-element list."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.moe_world_size = 1
        mock_gate.config.moe_rank = 0
        mock_gate.config.sequence_parallel = False
        mock_gate.num_experts_tensor = paddle.to_tensor(4, dtype="int64")
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)] * 4)
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAllGatherLayerV2(
            gate=mock_gate,
            experts=experts,
            layer_idx=0,
            moe_num_experts=[4],
        )
        self.assertFalse(layer.multimodal_experts)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_zero_tensor_dtype(self, mock_rank, mock_ws):
        """Test that self.zero tensor has correct dtype."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.moe_world_size = 1
        mock_gate.config.moe_rank = 0
        mock_gate.config.sequence_parallel = False
        mock_gate.num_experts_tensor = paddle.to_tensor(2, dtype="int64")
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAllGatherLayerV2(gate=mock_gate, experts=experts, layer_idx=0)
        self.assertEqual(layer.zero.dtype, paddle.float32)


class TestMOEAllGatherLayerV2Inheritance(unittest.TestCase):
    """Tests for class hierarchy."""

    def test_inherits_from_alltoall_layer(self):
        """Test that MOEAllGatherLayerV2 is a subclass of MOEAlltoAllLayer."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        self.assertTrue(issubclass(MOEAllGatherLayerV2, MOEAlltoAllLayer))

    def test_inherits_from_moe_layer_base(self):
        """Test that MOEAllGatherLayerV2 is a subclass of MOELayerBase."""
        from paddleformers.nn.moe.abstract import MOELayerBase
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        self.assertTrue(issubclass(MOEAllGatherLayerV2, MOELayerBase))


class TestMOEAllGatherLayerV2ForwardExperts(unittest.TestCase):
    """Tests for forward_experts method of MOEAllGatherLayerV2."""

    class DummyExpert(nn.Layer):
        """Dummy expert that mimics real MoE expert structure for testing."""

        def __init__(self, dim=8):
            super().__init__()
            self.down_proj = nn.Linear(dim, dim)
            self.training = False

        def forward(self, x):
            return self.down_proj(x)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def _make_layer(self, mock_rank, mock_ws, multimodal=False, moe_num_experts=None):
        """Helper to create an MOEAllGatherLayerV2 for forward_experts tests."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.moe_world_size = 1
        mock_gate.config.moe_rank = 0
        mock_gate.config.sequence_parallel = False
        mock_gate.num_experts_tensor = paddle.to_tensor(2, dtype="int64")
        mock_gate.parameters.return_value = []

        num_experts = 3 if multimodal else 2
        experts = nn.LayerList([self.DummyExpert() for _ in range(num_experts)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAllGatherLayerV2(
            gate=mock_gate,
            experts=experts,
            layer_idx=0,
            moe_num_experts=moe_num_experts,
        )
        return layer

    def test_forward_experts_non_multimodal(self):
        """Test forward_experts with non-multimodal configuration."""
        layer = self._make_layer(multimodal=False)
        # Dispatch 2 experts: one with tokens, one None
        dispatched = [
            paddle.randn([4, 8], dtype="float32"),
            None,
        ]
        results = layer.forward_experts(*dispatched)
        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0])
        # Second expert has no tokens -> should be None
        self.assertIsNone(results[1])

    def test_forward_experts_all_none(self):
        """Test forward_experts handles None inputs gracefully for inactive experts."""
        layer = self._make_layer(multimodal=False)
        # When all dispatched are None, the code expects at least one active expert
        # to accumulate the no-token outputs. Provide one non-None input.
        dispatched = [
            paddle.randn([1, 8], dtype="float32"),
            None,
        ]
        results = layer.forward_experts(*dispatched)
        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])

    def test_forward_experts_all_have_tokens(self):
        """Test forward_experts when all experts have tokens."""
        layer = self._make_layer(multimodal=False)
        dispatched = [
            paddle.randn([4, 8], dtype="float32"),
            paddle.randn([4, 8], dtype="float32"),
        ]
        results = layer.forward_experts(*dispatched)
        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])


class TestMOEAllGatherLayerV2CalcRouterLoss(unittest.TestCase):
    """Tests for calc_router_loss_and_logging of MOEAllGatherLayerV2."""

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_calc_router_loss_llm_path(self, mock_rank, mock_ws):
        """Test calc_router_loss_and_logging in LLM path (no token_type_ids)."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.moe_world_size = 1
        mock_gate.config.moe_rank = 0
        mock_gate.config.sequence_parallel = False
        mock_gate.num_experts_tensor = paddle.to_tensor(2, dtype="int64")
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAllGatherLayerV2(gate=mock_gate, experts=experts, layer_idx=0)

        router_loss = paddle.zeros([1], dtype="float32")
        gate_logits = paddle.randn([4, 4], dtype="float32")
        gate_prob = paddle.abs(paddle.randn([4, 4], dtype="float32"))
        combine_weights = paddle.randn([4, 2], dtype="float32")
        dispatch_mask = paddle.randn([4], dtype="float32")

        with patch.object(layer, "_calc_router_loss", return_value=0.1) as mock_calc:
            result = layer.calc_router_loss_and_logging(
                router_loss,
                gate_logits,
                gate_prob,
                None,
                None,
                combine_weights,
                dispatch_mask,
                None,
                None,
            )
            mock_calc.assert_called_once()
            self.assertIsNotNone(result)


class TestMOEAllGatherLayerV2FusedGateLogitsProcessFused(unittest.TestCase):
    """Tests for fused_gate_logits_process_fused method."""

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def _make_layer(self, mock_rank, mock_ws):
        """Helper to create MOEAllGatherLayerV2 for gate logits tests."""
        from paddleformers.nn.moe.moe_allgather_layer import MOEAllGatherLayerV2

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.moe_world_size = 1
        mock_gate.config.moe_rank = 0
        mock_gate.config.sequence_parallel = False
        mock_gate.num_experts_tensor = paddle.to_tensor(2, dtype="int64")
        mock_gate.act = paddle.nn.functional.softmax
        mock_gate.parameters.return_value = []
        mock_gate.experts_type_mask = None

        experts = nn.LayerList([nn.Linear(4, 4), nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAllGatherLayerV2(
            gate=mock_gate,
            experts=experts,
            layer_idx=0,
            k=2,
            group_experts=False,
        )
        return layer

    def test_fused_gate_logits_process_fused_no_token_type(self):
        """Test fused_gate_logits_process_fused without token_type_ids."""
        layer = self._make_layer()
        gate_logits_lm = paddle.randn([4, 8], dtype="float32")

        result = layer.fused_gate_logits_process_fused(gate_logits_lm)
        self.assertEqual(len(result), 3)  # (weight_and_expert_id, prob_flat, None)
        self.assertIsNone(result[2])  # gate_prob_mm should be None

    def test_fused_gate_logits_process_fused_with_mm_logits(self):
        """Test fused_gate_logits_process_fused with multimodal logits."""
        layer = self._make_layer()
        gate_logits_lm = paddle.randn([4, 8], dtype="float32")
        gate_logits_mm = paddle.randn([4, 8], dtype="float32")
        token_type_ids = paddle.to_tensor([0, 0, 1, 1], dtype="int64")

        with patch(
            "paddleformers.nn.moe.moe_allgather_layer.expand_modality_expert_id",
            return_value=paddle.randint(0, 4, [4, 2]),
        ):
            result = layer.fused_gate_logits_process_fused(gate_logits_lm, gate_logits_mm, token_type_ids)
            self.assertEqual(len(result), 3)
            self.assertIsNotNone(result[2])  # gate_prob_mm should not be None

    def test_fused_gate_logits_process_fused_group_experts(self):
        """Test fused_gate_logits_process_fused with group_experts=True."""
        layer = self._make_layer()
        layer.group_experts = True
        layer.use_correction_bias = False
        gate_logits_lm = paddle.randn([4, 8], dtype="float32")

        with patch(
            "paddleformers.nn.moe.moe_allgather_layer.expand_modality_expert_id",
            return_value=paddle.randint(0, 4, [4, 2]),
        ):
            result = layer.fused_gate_logits_process_fused(gate_logits_lm)
            self.assertEqual(len(result), 3)
            self.assertIsNotNone(result[0])  # weight_and_expert_id


if __name__ == "__main__":
    unittest.main()
