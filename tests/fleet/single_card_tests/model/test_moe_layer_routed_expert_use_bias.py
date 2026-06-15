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
from types import SimpleNamespace

import paddle.nn.functional as F

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer, MoESublayers
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_layer(use_bias=False, moe_routed_expert_use_bias=None):
    config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=32,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        moe_token_dispatcher_type="alltoall",
        moe_expert_fusion=False,
        fp8=None,
        gated_linear_unit=True,
        hidden_act=F.silu,
        use_bias=use_bias,
        moe_routed_expert_use_bias=moe_routed_expert_use_bias,
        tensor_model_parallel_size=1,
    )
    mlp_spec = MLPSublayersSpec(
        up_gate_proj=ColumnParallelLinear,
        down_proj=RowParallelLinear,
    )
    return MoELayer(
        config,
        sublayers=MoESublayers(mlp_spec=mlp_spec),
        pg_collection=SimpleNamespace(ep=None, expt_dp=None),
    )


class TestMoELayerRoutedExpertUseBias(unittest.TestCase):
    def test_none_follows_global_use_bias(self):
        layer = _make_layer(use_bias=True, moe_routed_expert_use_bias=None)

        self.assertTrue(layer.experts[0].config.use_bias)
        self.assertTrue(layer.shared_experts.config.use_bias)

    def test_true_overrides_global_use_bias(self):
        layer = _make_layer(use_bias=False, moe_routed_expert_use_bias=True)

        self.assertTrue(layer.experts[0].config.use_bias)
        self.assertFalse(layer.shared_experts.config.use_bias)

    def test_false_overrides_global_use_bias(self):
        layer = _make_layer(use_bias=True, moe_routed_expert_use_bias=False)

        self.assertFalse(layer.experts[0].config.use_bias)
        self.assertTrue(layer.shared_experts.config.use_bias)


if __name__ == "__main__":
    unittest.main()
