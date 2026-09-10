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


class TestGateCombineImport(unittest.TestCase):
    """Tests for importing GateCombine from moe_alltoall_layer."""

    def test_import_gate_combine(self):
        """Test that GateCombine class can be imported."""
        from paddleformers.nn.moe.moe_alltoall_layer import GateCombine

        self.assertIsNotNone(GateCombine)

    def test_import_combining(self):
        """Test that combining function can be imported."""
        from paddleformers.nn.moe.moe_alltoall_layer import combining

        self.assertTrue(callable(combining))

    def test_import_moe_alltoall_layer(self):
        """Test that MOEAlltoAllLayer class can be imported."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        self.assertIsNotNone(MOEAlltoAllLayer)


class TestCombiningFunction(unittest.TestCase):
    """Tests for the combining standalone function."""

    def test_combining_hard_gate_true(self):
        """Test combining with hard_gate=True uses embedding path."""
        from paddleformers.nn.moe.moe_alltoall_layer import combining

        seq_len, dim = 4, 8
        x = paddle.randn([seq_len, dim], dtype="float32")
        k = 1
        combine_weights = paddle.randn([seq_len, k], dtype="float32")
        scatter_index = paddle.randint(0, seq_len, [k, seq_len], dtype="int64")

        result = combining(x, combine_weights, scatter_index, hard_gate=True)
        # F.embedding with scatter_index [k, s] and x [s, dim] gives [k, s, dim]
        # squeeze(-2) removes the second-to-last dim only if it is 1
        # With k=1 and s=4, result is [1, 4, 8]
        self.assertEqual(result.shape[0], k)
        self.assertEqual(result.shape[1], seq_len)
        self.assertEqual(result.shape[2], dim)

    def test_combining_hard_gate_false(self):
        """Test combining with hard_gate=False uses GateCombine path."""
        from paddleformers.nn.moe.moe_alltoall_layer import combining

        seq_len, dim = 4, 8
        x = paddle.randn([seq_len, dim], dtype="float32")
        k = 2
        combine_weights = paddle.abs(paddle.randn([seq_len, k], dtype="float32"))
        scatter_index = paddle.randint(0, seq_len, [k, seq_len], dtype="int64")

        with patch("paddleformers.nn.moe.moe_alltoall_layer.GateCombine") as mock_gc:
            mock_result = paddle.randn([seq_len, dim], dtype="float32")
            mock_gc.apply.return_value = mock_result
            result = combining(x, combine_weights, scatter_index, hard_gate=False)
            self.assertEqual(result.shape, [seq_len, dim])
            mock_gc.apply.assert_called_once()

    def test_combining_result_not_stop_gradient(self):
        """Test that combining result via GateCombine path sets stop_gradient=False."""
        from paddleformers.nn.moe.moe_alltoall_layer import combining

        seq_len, dim = 4, 8
        x = paddle.randn([seq_len, dim], dtype="float32")
        k = 2
        combine_weights = paddle.abs(paddle.randn([seq_len, k], dtype="float32"))
        scatter_index = paddle.randint(0, seq_len, [k, seq_len], dtype="int64")

        # Use mocked GateCombine path (hard_gate=False) to test stop_gradient behavior
        with patch("paddleformers.nn.moe.moe_alltoall_layer.GateCombine") as mock_gc:
            mock_result = paddle.randn([seq_len, dim], dtype="float32")
            mock_gc.apply.return_value = mock_result
            result = combining(x, combine_weights, scatter_index, hard_gate=False)
            # combining sets ret.stop_gradient = False after GateCombine.apply
            self.assertFalse(result.stop_gradient)


class TestGateCombinePyLayer(unittest.TestCase):
    """Tests for GateCombine PyLayer forward method."""

    def test_gate_combine_forward_stores_context(self):
        """Test that GateCombine.forward stores inputs in context."""
        from paddleformers.nn.moe.moe_alltoall_layer import GateCombine

        ctx = MagicMock()
        x = paddle.randn([4, 8], dtype="float32")
        combine_weights = paddle.randn([4, 2], dtype="float32")
        scatter_index = paddle.randint(0, 4, [2, 4], dtype="int64")

        mock_result = paddle.randn([4, 8], dtype="float32")
        with patch("paddleformers.nn.moe.moe_alltoall_layer.moe_combine", return_value=mock_result):
            GateCombine.forward(ctx, x, combine_weights, scatter_index)

        np.testing.assert_allclose(ctx.x.numpy(), x.numpy())
        np.testing.assert_allclose(ctx.combine_weights.numpy(), combine_weights.numpy())
        np.testing.assert_allclose(ctx.scatter_index.numpy(), scatter_index.numpy())


def _create_no_hcg_fleet():
    """Create a fleet mock that does NOT have _hcg attribute."""

    class NoHCGFleet:
        pass

    return NoHCGFleet()


class TestMOEAlltoAllLayerInit(unittest.TestCase):
    """Tests for MOEAlltoAllLayer initialization logic."""

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_basic(self, mock_rank, mock_ws):
        """Test basic initialization of MOEAlltoAllLayer."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.01
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.norm_gate_logits = False
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(8, 8), nn.Linear(8, 8)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(
            gate=mock_gate,
            experts=experts,
            layer_idx=0,
            shared_experts=None,
            group=None,
            recompute=False,
            k=2,
            all_to_all_dropout=0,
            group_experts=False,
            moe_statics=None,
            moe_num_experts=None,
        )
        self.assertEqual(layer.k, 2)
        self.assertEqual(layer.layer_idx, 0)
        self.assertEqual(layer.all_to_all_dropout, 0)
        self.assertFalse(layer.group_experts)
        self.assertIsNone(layer.shared_experts)
        self.assertFalse(layer.use_correction_bias)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_with_correction_bias(self, mock_rank, mock_ws):
        """Test initialization with moe_statics (correction bias enabled)."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.01
        mock_gate.config.moe_use_aux_free = True
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4), nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        moe_statics = MagicMock()

        layer = MOEAlltoAllLayer(
            gate=mock_gate,
            experts=experts,
            layer_idx=1,
            moe_statics=moe_statics,
        )
        self.assertTrue(layer.use_correction_bias)
        self.assertIs(layer.moe_statics, moe_statics)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_multimodal_experts(self, mock_rank, mock_ws):
        """Test initialization with multimodal expert configuration."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(
            gate=mock_gate,
            experts=experts,
            layer_idx=0,
            moe_num_experts=[2, 2],
        )
        self.assertTrue(layer.multimodal_experts)
        self.assertEqual(layer.num_local_multimodal_experts, [2, 2])

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_gate_params_marked_is_gate(self, mock_rank, mock_ws):
        """Test that gate parameters have is_gate attribute set to True."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        gate = nn.Linear(8, 4)
        gate.config = MagicMock()
        gate.config.router_aux_loss_coef = 0.0
        gate.config.moe_use_aux_free = True

        experts = nn.LayerList([nn.Linear(4, 4), nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        MOEAlltoAllLayer(gate=gate, experts=experts, layer_idx=0)
        for p in gate.parameters():
            self.assertTrue(p.is_gate)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_world_size_and_rank_defaults(self, mock_rank, mock_ws):
        """Test world_size and rank default values."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(gate=mock_gate, experts=experts, layer_idx=0)
        self.assertEqual(layer.world_size, 1)
        self.assertEqual(layer.rank, 0)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_config_assigned(self, mock_rank, mock_ws):
        """Test that self.config is set from gate.config."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(gate=mock_gate, experts=experts, layer_idx=0)
        self.assertIs(layer.config, mock_gate.config)

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_zero_tensor_created(self, mock_rank, mock_ws):
        """Test that self.zero is a float32 tensor."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(gate=mock_gate, experts=experts, layer_idx=0)
        self.assertEqual(layer.zero.dtype, paddle.float32)
        self.assertEqual(layer.zero.shape, [])

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_init_num_local_experts(self, mock_rank, mock_ws):
        """Test num_local_experts is correctly computed."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4), nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(gate=mock_gate, experts=experts, layer_idx=0)
        # With world_size=1 and 2 experts, num_local_experts should be 2
        self.assertEqual(layer.num_local_experts, 2)


class TestMOEAlltoAllLayerCalcRouterLoss(unittest.TestCase):
    """Tests for _calc_router_loss method."""

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def _make_layer(self, mock_rank, mock_ws, aux_loss_coef=0.01):
        """Helper to create a MOEAlltoAllLayer with mocked gate config."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = aux_loss_coef
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_orthogonal_loss_lambda = 0.0
        mock_gate.config.router_z_loss_coef = 0.0
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.router_aux_loss_coef = {0: aux_loss_coef, 1: 0.01}
        mock_gate.moe_orthogonal_loss_lambda = {0: 0.0, 1: 0.0}
        mock_gate.router_z_loss_coef = {0: 0.0, 1: 0.0}
        mock_gate.num_experts_tensor = paddle.to_tensor(4, dtype="int64")
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4), nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(
            gate=mock_gate,
            experts=experts,
            layer_idx=0,
            group_experts=False,
        )
        return layer

    def test_calc_router_loss_with_aux_loss(self):
        """Test _calc_router_loss when router_aux_loss_coef is nonzero."""
        layer = self._make_layer(aux_loss_coef=0.01)

        dispatch_mask = paddle.ones([4], dtype="float32")
        gate_logits = paddle.randn([4, 4], dtype="float32")
        gate_prob = paddle.abs(paddle.randn([4, 4], dtype="float32"))
        tokens_type_mask = paddle.ones([4], dtype="bool")
        dispatch_tokens_mask = paddle.ones([4], dtype="bool")

        with patch.object(layer.gate, "_cal_aux_loss", return_value=paddle.to_tensor(0.1)):
            result = layer._calc_router_loss(
                dispatch_mask,
                gate_logits,
                gate_prob,
                4,
                False,
                0,
                0,
                tokens_type_mask,
                dispatch_tokens_mask,
            )
        self.assertIsInstance(result, (int, float, paddle.Tensor))

    def test_calc_router_loss_without_aux_loss(self):
        """Test _calc_router_loss when router_aux_loss_coef is zero."""
        layer = self._make_layer(aux_loss_coef=0.0)
        layer.gate.config.router_aux_loss_coef = 0.0

        dispatch_mask = paddle.ones([4], dtype="float32")
        gate_logits = paddle.randn([4, 4], dtype="float32")
        gate_prob = paddle.abs(paddle.randn([4, 4], dtype="float32"))

        result = layer._calc_router_loss(
            dispatch_mask,
            gate_logits,
            gate_prob,
            4,
            False,
            0,
        )
        # When router_aux_loss_coef is 0, should still return a numeric result
        self.assertIsInstance(result, (int, float, paddle.Tensor))


class TestMOEAlltoAllLayerCombineExpertOutput(unittest.TestCase):
    """Tests for combine_expert_output method."""

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_combine_expert_output_with_postprocess(self, mock_rank, mock_ws):
        """Test combine_expert_output with output_postprocess set."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(gate=mock_gate, experts=experts, layer_idx=0)
        post_fn = MagicMock(return_value=paddle.randn([4, 4]))
        layer.output_postprocess = post_fn

        expert_output = paddle.randn([4, 4], dtype="float32")
        combine_weights = paddle.abs(paddle.randn([4, 2], dtype="float32"))
        scatter_index = paddle.randint(0, 4, [2, 4], dtype="int64")

        with patch("paddleformers.nn.moe.moe_alltoall_layer.combining", return_value=expert_output):
            layer.combine_expert_output(expert_output, combine_weights, scatter_index)
            post_fn.assert_called_once()

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_combine_expert_output_no_postprocess(self, mock_rank, mock_ws):
        """Test combine_expert_output without output_postprocess."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(gate=mock_gate, experts=experts, layer_idx=0)
        self.assertIsNone(layer.output_postprocess)

        mock_combined = paddle.randn([4, 4], dtype="float32")
        with patch("paddleformers.nn.moe.moe_alltoall_layer.combining", return_value=mock_combined):
            result = layer.combine_expert_output(
                paddle.randn([4, 4]),
                paddle.abs(paddle.randn([4, 2])),
                paddle.randint(0, 4, [2, 4], dtype="int64"),
            )
            np.testing.assert_allclose(result.numpy(), mock_combined.numpy())


class TestMOEAlltoAllLayerForwardExperts(unittest.TestCase):
    """Tests for forward_experts method."""

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_forward_experts_layerlist(self, mock_rank, mock_ws):
        """Test forward_experts with LayerList experts."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.parameters.return_value = []

        expert1 = nn.Linear(8, 8)
        expert2 = nn.Linear(8, 8)
        experts = nn.LayerList([expert1, expert2])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(gate=mock_gate, experts=experts, layer_idx=0)
        dispatched = paddle.randn([2, 1, 4, 8], dtype="float32")

        result = layer.forward_experts(dispatched)
        # dispatched [2, 1, 4, 8] reshaped to [world_size=1, num_local=2, 4, 8]
        # after transpose [2, 1, 4, 8], unbind(0) gives 2 chunks of [1, 4, 8]
        # stack(axis=1) gives [1, 2, 4, 8]
        self.assertEqual(result.shape[0], 1)  # world_size
        self.assertEqual(result.shape[1], 2)  # num_local_experts

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_forward_experts_fused_experts(self, mock_rank, mock_ws):
        """Test forward_experts with fused (non-LayerList) experts."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.parameters.return_value = []

        # Fused experts must support len(), subscripting, and be callable
        class FusedExperts(nn.Layer):
            def __init__(self):
                super().__init__()
                self._len = 2
                self._experts = nn.LayerList([nn.Linear(8, 8), nn.Linear(8, 8)])

            def __len__(self):
                return self._len

            def __getitem__(self, idx):
                return self._experts[idx]

            def forward(self, x):
                return x

        fused_expert = FusedExperts()
        for p in fused_expert.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(gate=mock_gate, experts=fused_expert, layer_idx=0)
        dispatched = paddle.randn([2, 1, 4, 8], dtype="float32")

        result = layer.forward_experts(dispatched)
        self.assertIsNotNone(result)


class TestMOEAlltoAllLayerFusedGateLogitsProcess(unittest.TestCase):
    """Tests for fused_gate_logits_process method."""

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def _make_layer(self, mock_rank, mock_ws):
        """Helper to create a layer for fused_gate_logits_process tests."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.config.moe_use_hard_gate = False
        mock_gate.config.norm_gate_logits = False
        mock_gate.act = paddle.nn.functional.softmax
        mock_gate.experts_type_ids = paddle.zeros([8], dtype="int64")
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4), nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(
            gate=mock_gate,
            experts=experts,
            layer_idx=0,
            k=2,
            group_experts=False,
        )
        return layer

    def test_fused_gate_logits_process_no_token_type(self):
        """Test fused_gate_logits_process without token_type_ids."""
        layer = self._make_layer()
        gate_logits = paddle.randn([4, 8], dtype="float32")
        prob, max_prob = layer.fused_gate_logits_process(gate_logits)
        self.assertEqual(prob.shape, [4, 8])
        self.assertIsNone(max_prob)

    def test_fused_gate_logits_process_group_experts(self):
        """Test fused_gate_logits_process with group_experts=True."""
        layer = self._make_layer()
        layer.group_experts = True
        gate_logits = paddle.randn([4, 8], dtype="float32")
        prob, max_prob = layer.fused_gate_logits_process(gate_logits)
        self.assertEqual(prob.shape[0], 4)
        self.assertIsNotNone(max_prob)


class TestMOEAlltoAllLayerIsSubclass(unittest.TestCase):
    """Tests for inheritance chain."""

    def test_moe_alltoall_layer_inherits_from_base(self):
        """Test that MOEAlltoAllLayer is a subclass of MOELayerBase."""
        from paddleformers.nn.moe.abstract import MOELayerBase
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        self.assertTrue(issubclass(MOEAlltoAllLayer, MOELayerBase))


class TestMOEAlltoAllLayerForwardSingleStage(unittest.TestCase):
    """Tests for forward_single_stage method."""

    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_world_size", return_value=1)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.dist.get_rank", return_value=0)
    @patch("paddleformers.nn.moe.moe_alltoall_layer.fleet.fleet", _create_no_hcg_fleet())
    def test_forward_single_stage(self, mock_rank, mock_ws):
        """Test forward_single_stage calls the right expert."""
        from paddleformers.nn.moe.moe_alltoall_layer import MOEAlltoAllLayer

        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.router_aux_loss_coef = 0.0
        mock_gate.config.moe_use_aux_free = True
        mock_gate.parameters.return_value = []

        experts = nn.LayerList([nn.Linear(4, 4), nn.Linear(4, 4)])
        for p in experts.parameters():
            p.expert = False
            p.no_sync = False

        layer = MOEAlltoAllLayer(gate=mock_gate, experts=experts, layer_idx=0)
        dispatched = paddle.randn([4, 4], dtype="float32")  # Linear(4, 4) expects last dim = 4
        result = layer.forward_single_stage(dispatched, stage_id=0)
        self.assertEqual(result.shape, [4, 4])


if __name__ == "__main__":
    unittest.main()
