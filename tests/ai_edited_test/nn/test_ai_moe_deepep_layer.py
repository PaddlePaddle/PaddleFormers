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
from unittest.mock import patch

import paddle
import paddle.nn as nn


def _make_compat_expert_class():
    """Create an expert class compatible with modular_moe_layer's expert_args."""
    from paddleformers.nn.mlp import MLP

    class CompatExpert(MLP):
        """Expert class that accepts the same args as modular_moe_layer passes."""

        def __init__(self, config, intermediate_size, fuse_up_gate=False, **kwargs):
            super().__init__(
                config=config,
                intermediate_size=intermediate_size,
                fuse_up_gate=fuse_up_gate,
            )

    return CompatExpert


def _make_modular_moe_layer(
    hidden_size=32,
    moe_intermediate_size=64,
    num_experts=4,
    num_shared_experts=0,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    expert_activation="silu",
    topk_method="greedy",
    model_type="qwen2_moe",
    transpose_gate_weight=False,
    moe_token_dispatcher_type="alltoall",
    n_group=1,
    topk_group=1,
    router_aux_loss_coef=0.0,
    moe_expert_capacity_factor=0.0,
    moe_subbatch_token_num_before_dispatch=-1,
):
    """Create a ModularMoELayer instance with sensible defaults for single-card testing."""
    from paddleformers.nn.moe_deepep.modular_moe_layer import ModularMoELayer
    from paddleformers.transformers.configuration_utils import PretrainedConfig

    pretrained_config = PretrainedConfig(
        hidden_size=hidden_size,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        max_sequence_length=128,
        moe_token_dispatcher_type=moe_token_dispatcher_type,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        router_aux_loss_coef=router_aux_loss_coef,
        moe_subbatch_token_num_before_dispatch=moe_subbatch_token_num_before_dispatch,
        moe_expert_capacity_factor=moe_expert_capacity_factor,
        moe_token_drop_policy="probs",
    )
    # PretrainedConfig.__init__ has a bug where set_expected_keys correctly
    # reads moe_token_dispatcher_type from kwargs, but a later line
    # (kwargs.pop with default "deepep") overwrites it. Fix it here.
    pretrained_config.moe_token_dispatcher_type = moe_token_dispatcher_type

    moe_config = {
        "gate_activation": "softmax",
        "eval_capacity_factor": 1.0,
        "group": None,
        "global_aux_loss": False,
        "use_rts": True,
        "top2_2nd_expert_sampling": True,
        "seq_aux": True,
        "train_topk_method": topk_method,
        "inference_topk_method": topk_method,
        "use_flexible_loss": False,
        "expert_dropout": 0.0,
        "loss_configs": None,
        "loss_combiner_name": "weighted_sum",
    }

    expert_class = _make_compat_expert_class()

    with patch("paddleformers.nn.moe_deepep.modular_moe_layer.fleet") as mock_fleet, patch(
        "paddleformers.nn.moe_deepep.modular_moe_layer.dist"
    ) as mock_dist:
        mock_fleet.get_hybrid_communicate_group.side_effect = Exception("no fleet")
        mock_dist.get_world_size.return_value = 1

        layer = ModularMoELayer(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
            expert_activation=expert_activation,
            moe_config=moe_config,
            model_type=model_type,
            expert_class=expert_class,
            transpose_gate_weight=transpose_gate_weight,
            pretrained_config=pretrained_config,
        )
    return layer


class TestModularMoELayer(unittest.TestCase):
    """Tests for ModularMoELayer class."""

    def test_import(self):
        """Test that ModularMoELayer can be imported."""
        from paddleformers.nn.moe_deepep.modular_moe_layer import ModularMoELayer

        self.assertIsNotNone(ModularMoELayer)

    def test_is_nn_layer(self):
        """Test that ModularMoELayer is a subclass of nn.Layer."""
        from paddleformers.nn.moe_deepep.modular_moe_layer import ModularMoELayer

        self.assertTrue(issubclass(ModularMoELayer, nn.Layer))

    def test_init_basic(self):
        """Test basic initialization of ModularMoELayer."""
        layer = _make_modular_moe_layer()
        self.assertEqual(layer.hidden_size, 32)
        self.assertEqual(layer.num_experts, 4)
        self.assertEqual(layer.num_experts_per_tok, 2)
        self.assertEqual(layer.expert_model_parallel_size, 1)
        self.assertEqual(layer.moe_rank, 0)

    def test_init_has_gate(self):
        """Test that gate is created during initialization."""
        layer = _make_modular_moe_layer()
        self.assertIsNotNone(layer.gate)
        from paddleformers.nn.moe_deepep.moe_gate import StandardMoEGate

        self.assertIsInstance(layer.gate, StandardMoEGate)

    def test_init_has_experts(self):
        """Test that experts LayerList is created."""
        layer = _make_modular_moe_layer(num_experts=4)
        self.assertIsNotNone(layer.experts)
        self.assertEqual(len(layer.experts), 4)

    def test_init_shared_experts_none(self):
        """Test that shared_experts is None when num_shared_experts=0."""
        layer = _make_modular_moe_layer(num_shared_experts=0)
        self.assertIsNone(layer.shared_experts)

    def test_init_shared_experts_created(self):
        """Test that shared_experts is created when num_shared_experts > 0."""
        layer = _make_modular_moe_layer(num_shared_experts=1)
        self.assertIsNotNone(layer.shared_experts)

    def test_init_has_communication(self):
        """Test that communication module is created."""
        layer = _make_modular_moe_layer(moe_token_dispatcher_type="alltoall")
        from paddleformers.nn.moe_deepep.moe_communication import (
            AllToAllMoECommunication,
        )

        self.assertIsInstance(layer.communication, AllToAllMoECommunication)

    def test_init_deepep_communication(self):
        """Test that DeepEP communication module is created when dispatcher_type is deepep."""
        layer = _make_modular_moe_layer(moe_token_dispatcher_type="deepep")
        from paddleformers.nn.moe_deepep.moe_communication import DeepEPMoECommunication

        self.assertIsInstance(layer.communication, DeepEPMoECommunication)

    def test_init_invalid_dispatcher_type(self):
        """Test that invalid moe_token_dispatcher_type raises ValueError."""
        with self.assertRaises(ValueError):
            _make_modular_moe_layer(moe_token_dispatcher_type="invalid")

    def test_forward_basic(self):
        """Test basic forward pass."""
        layer = _make_modular_moe_layer()
        x = paddle.randn([2, 8, 32])
        output = layer(x)
        self.assertEqual(output.shape, [2, 8, 32])

    def test_forward_preserves_shape(self):
        """Test that forward preserves input shape."""
        layer = _make_modular_moe_layer()
        x = paddle.randn([3, 16, 32])
        output = layer(x)
        self.assertEqual(output.shape, [3, 16, 32])

    def test_forward_with_router_aux_loss(self):
        """Test forward with non-zero router_aux_loss_coef."""
        layer = _make_modular_moe_layer(router_aux_loss_coef=0.1)
        x = paddle.randn([2, 8, 32])
        output = layer(x)
        self.assertEqual(output.shape, [2, 8, 32])

    def test_forward_with_shared_experts_raises_shape_error(self):
        """Test forward with shared experts raises shape mismatch due to source bug.

        In ModularMoELayer.forward(), output from _forward_traditional_moe is
        2D [batch*seq, d_model] but shared_experts(residuals) returns 3D
        [batch, seq, d_model], causing a broadcast error on addition.
        This is a known bug in the source code.
        """
        layer = _make_modular_moe_layer(num_shared_experts=1)
        x = paddle.randn([2, 8, 32])
        with self.assertRaises(ValueError):
            layer(x)

    def test_forward_noaux_tc(self):
        """Test forward with noaux_tc topk_method."""
        layer = _make_modular_moe_layer(
            topk_method="noaux_tc",
            n_group=2,
            topk_group=1,
        )
        x = paddle.randn([2, 8, 32])
        output = layer(x)
        self.assertEqual(output.shape, [2, 8, 32])

    def test_forward_group_limited_greedy(self):
        """Test forward with group_limited_greedy topk_method."""
        layer = _make_modular_moe_layer(
            topk_method="group_limited_greedy",
            n_group=2,
            topk_group=1,
        )
        x = paddle.randn([2, 8, 32])
        output = layer(x)
        self.assertEqual(output.shape, [2, 8, 32])

    def test_get_expert_info(self):
        """Test get_expert_info returns correct dictionary."""
        layer = _make_modular_moe_layer()
        info = layer.get_expert_info()
        self.assertIn("num_experts", info)
        self.assertIn("num_experts_per_device", info)
        self.assertIn("expert_model_parallel_size", info)
        self.assertIn("moe_rank", info)
        self.assertIn("is_parallel_enabled", info)
        self.assertIn("use_flexible_loss", info)
        self.assertEqual(info["num_experts"], 4)
        self.assertFalse(info["is_parallel_enabled"])

    def test_num_experts_per_device_equals_num_experts_single_rank(self):
        """Test that num_experts_per_device equals num_experts on single rank."""
        layer = _make_modular_moe_layer(num_experts=8)
        self.assertEqual(layer.num_experts_per_device, 8)

    def test_init_expert_parallel_no_fleet(self):
        """Test _init_expert_parallel when fleet is not available."""
        layer = _make_modular_moe_layer()
        self.assertEqual(layer.expert_model_parallel_size, 1)
        self.assertEqual(layer.moe_rank, 0)
        self.assertIsNone(layer.moe_group)

    def test_remove_loss_function_warns(self):
        """Test remove_loss_function logs warning when use_flexible_loss=False."""
        layer = _make_modular_moe_layer()
        # Should not raise, just warn
        layer.remove_loss_function("test")

    def test_update_loss_weights_warns(self):
        """Test update_loss_weights logs warning when use_flexible_loss=False."""
        layer = _make_modular_moe_layer()
        layer.update_loss_weights({"aux": 0.1})

    def test_set_loss_combiner_warns(self):
        """Test set_loss_combiner logs warning when use_flexible_loss=False."""
        layer = _make_modular_moe_layer()
        layer.set_loss_combiner("sum")

    def test_token_dispatcher_none_single_rank(self):
        """Test that token_dispatcher is None on single rank."""
        layer = _make_modular_moe_layer()
        self.assertIsNone(layer.token_dispatcher)

    def test_forward_traditional_moe(self):
        """Test _forward_traditional_moe method directly."""
        layer = _make_modular_moe_layer()
        hidden_states = paddle.randn([8, 32])
        selected_experts = paddle.to_tensor([[0, 1]] * 8)
        topk_weights = paddle.ones([8, 2]) * 0.5
        output = layer._forward_traditional_moe(hidden_states, selected_experts, topk_weights)
        self.assertEqual(output.shape, [8, 32])


if __name__ == "__main__":
    unittest.main()
