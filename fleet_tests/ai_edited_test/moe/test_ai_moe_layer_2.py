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


import unittest
from unittest.mock import MagicMock, patch

import paddle


def _make_moe_config(**overrides):
    """Helper to create a TransformerConfig for MoE testing."""
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "gated_linear_unit": True,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "moe_token_dispatcher_type": "alltoall",
        "moe_use_fusion_node": False,
        "moe_expert_fusion": False,
        "moe_deep_gemm": False,
        "moe_ep_barrier": True,
        "fp8": None,
        "fp8_wgrad": True,
        "using_sonic_moe": False,
        "router_aux_loss_coef": 0.01,
        "router_z_loss_coef": None,
        "topk_method": "greedy",
        "norm_topk_prob": True,
        "scoring_func": "softmax",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "moe_router_force_load_balancing": False,
        "moe_router_load_balancing_type": "aux_loss",
        "n_shared_experts": 0,
        "moe_shared_expert_overlap": False,
        "recompute_granularity": None,
        "recompute_modules": [],
        "use_bias": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_pg_collection(moe_world_size=1):
    """Helper to create a mock ProcessGroupCollection."""
    pg = MagicMock()
    pg.ep = MagicMock()
    pg.ep.world_size = moe_world_size
    pg.expt_dp = MagicMock()
    pg.tp = MagicMock()
    pg.tp.size.return_value = 1
    pg.cp = MagicMock()
    pg.cp.rank.return_value = 0
    pg.cp.size.return_value = 1
    return pg


def _make_moe_sublayers():
    """Helper to create MoESublayers with a valid mlp_spec."""
    from paddleformers.fleet.tensor_parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
    from paddleformers.fleet.transformer.moe.moe_layer import MoESublayers

    return MoESublayers(
        mlp_spec=MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            hidden_act=None,
            down_proj=RowParallelLinear,
        )
    )


def _setup_moe_mocks(
    mock_utils,
    mock_paddlefleet_ops,
    mock_version,
    mock_expert,
    mock_shared,
    ep_size=1,
    ep_rank=0,
):
    """Common mock setup for MoELayer tests."""
    mock_utils.get_pg_size.return_value = ep_size
    mock_utils.get_pg_rank.return_value = ep_rank
    mock_paddlefleet_ops.is_sonic_moe_available.return_value = False
    mock_version.cuda.return_value = "12.2"
    mock_expert.return_value = paddle.nn.Layer()
    mock_shared.return_value = paddle.nn.Layer()


class TestMoELayerInitExpertParallel(unittest.TestCase):
    """Extra tests for MoELayer expert parallel initialization."""


class TestMoELayerInitSharedExperts(unittest.TestCase):
    """Extra tests for MoELayer shared experts."""

    def test_shared_expert_gate_weight_uses_config_init_method(self):
        """Test shared expert gate weight is initialized by config.init_method."""
        from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
        from paddleformers.fleet.transformer.moe.moe_shared_expert import (
            StandardMLPSharedExpert,
        )

        def init_gate_weight(tensor):
            tensor.set_value(paddle.ones(tensor.shape, dtype=tensor.dtype))

        init_method = MagicMock(side_effect=init_gate_weight)
        config = _make_moe_config(
            hidden_size=4,
            intermediate_size=8,
            moe_intermediate_size=8,
            moe_shared_expert_gate=True,
            init_method=init_method,
        )

        def fake_mlp_init(
            self,
            config,
            sublayers_spec,
            is_expert=False,
            input_size=None,
            intermediate_size=None,
            hidden_size=None,
            tp_group=None,
        ):
            paddle.nn.Layer.__init__(self)
            self.config = config

        with patch(
            "paddleformers.fleet.transformer.moe.moe_shared_expert.MLP.__init__",
            new=fake_mlp_init,
        ):
            shared_expert = StandardMLPSharedExpert(
                config,
                moe_intermediate_size=config.moe_intermediate_size,
                is_expert=True,
                mlp_spec=MLPSublayersSpec(),
            )

        self.assertEqual(init_method.call_count, 1)
        self.assertIs(init_method.call_args[0][0], shared_expert.gate_weight)
        self.assertTrue(
            paddle.allclose(
                shared_expert.gate_weight,
                paddle.ones(shared_expert.gate_weight.shape),
            )
        )


class TestMoELayerInitExpertParallelParse(unittest.TestCase):
    """Tests for _init_expert_parallel internals."""

    @patch("paddleformers.fleet.transformer.moe.moe_layer.paddle.version")
    @patch("paddleformers.fleet.transformer.moe.moe_layer.paddlefleet_ops")
    @patch("paddleformers.fleet.transformer.moe.moe_layer.utils")
    def test_num_experts_less_than_ep_raises(
        self, mock_utils, mock_paddlefleet_ops, mock_version
    ):
        """Test that num_experts < expert_model_parallel_size raises."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock_utils.get_pg_size.return_value = 4
        mock_utils.get_pg_rank.return_value = 0
        mock_paddlefleet_ops.is_sonic_moe_available.return_value = False
        mock_version.cuda.return_value = "12.2"

        config = _make_moe_config(
            n_routed_experts=2,
            moe_intermediate_size=64,
        )
        pg_collection = _make_pg_collection(moe_world_size=4)

        sublayers = _make_moe_sublayers()
        with self.assertRaises(AssertionError):
            MoELayer(config, sublayers=sublayers, pg_collection=pg_collection)

    @patch("paddleformers.fleet.transformer.moe.moe_layer.paddle.version")
    @patch("paddleformers.fleet.transformer.moe.moe_layer.paddlefleet_ops")
    @patch("paddleformers.fleet.transformer.moe.moe_layer.utils")
    def test_num_experts_not_divisible_by_ep_raises(
        self, mock_utils, mock_paddlefleet_ops, mock_version
    ):
        """Test that num_experts % expert_model_parallel_size != 0 raises."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        mock_utils.get_pg_size.return_value = 3
        mock_utils.get_pg_rank.return_value = 0
        mock_paddlefleet_ops.is_sonic_moe_available.return_value = False
        mock_version.cuda.return_value = "12.2"

        config = _make_moe_config(
            n_routed_experts=4,
            moe_intermediate_size=64,
        )
        pg_collection = _make_pg_collection(moe_world_size=3)

        sublayers = _make_moe_sublayers()
        with self.assertRaises(AssertionError):
            MoELayer(config, sublayers=sublayers, pg_collection=pg_collection)


class TestMoELayerRecomputeFlags(unittest.TestCase):
    pass
    """Tests for recompute flags in MoELayer."""


class TestMoELayerComputeGate(unittest.TestCase):
    pass
    """Tests for compute_gate method."""


class TestMoELayerAuxLossCompute(unittest.TestCase):
    pass
    """Tests for aux_loss_compute method."""
