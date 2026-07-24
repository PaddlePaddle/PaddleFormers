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
from unittest.mock import MagicMock

import paddle

try:
    from paddle.distributed.communication.group import Group
except ImportError:
    Group = None


def _make_moe_config(**overrides):
    """Helper to create a TransformerConfig for MoE testing."""
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )

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


def _mock_moe_deps(
    mock_utils,
    mock_paddlefleet_ops,
    mock_version,
    mock_expert,
    mock_shared_expert=None,
):
    """Common mock setup for MoELayer tests."""
    mock_utils.get_pg_size.return_value = 1
    mock_utils.get_pg_rank.return_value = 0
    mock_paddlefleet_ops.is_sonic_moe_available.return_value = False
    mock_version.cuda.return_value = "12.2"
    # Return a real paddle.nn.Layer so nn.LayerList accepts it
    mock_expert.return_value = paddle.nn.Layer()
    if mock_shared_expert is not None:
        mock_shared_expert.return_value = paddle.nn.Layer()


class TestMoELayer(unittest.TestCase):
    """Unit tests for moe_layer module."""

    def test_grad_dtype_guard_forward(self):
        """Test GradDtypeGuard PyLayer forward."""
        from paddleformers.fleet.transformer.moe.moe_layer import GradDtypeGuard

        x = paddle.randn([4, 64], dtype=paddle.float32)
        out, status = GradDtypeGuard.apply(x, paddle.float32)
        self.assertEqual(out.shape[0], 0)
        self.assertIn("x", status)

    def test_grad_dtype_unguard_forward(self):
        """Test GradDtypeUnguard PyLayer forward."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            GradDtypeUnguard,
        )

        x = paddle.randn([4, 64], dtype=paddle.float32)
        status = {"x": x}
        result = GradDtypeUnguard.apply(x, status)
        self.assertTrue(paddle.allclose(result, x))

    def test_moe_sublayers_dataclass(self):
        """Test MoESublayers dataclass defaults."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoESublayers

        sublayers = MoESublayers()
        self.assertIsNone(sublayers.mlp_spec)


if __name__ == "__main__":
    unittest.main()
