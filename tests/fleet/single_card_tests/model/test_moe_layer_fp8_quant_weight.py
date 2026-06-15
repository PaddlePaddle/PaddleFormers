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

import paddle
import paddle.nn.functional as F
from paddle.base import core

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer, MoESublayers
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

HIDDEN = 128
INTERMEDIATE = 128
NUM_EXPERTS = 2


class TestMoELayerFp8QuantWeight(unittest.TestCase):
    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        try:
            from paddlefleet_ops import (
                fuse_stack_fp8_quant,
                fuse_stack_transpose_fp8_quant,
            )

            self.assertTrue(callable(fuse_stack_fp8_quant))
            self.assertTrue(callable(fuse_stack_transpose_fp8_quant))
        except (ImportError, RuntimeError):
            self.skipTest("paddlefleet_ops not available")
        model_parallel_cuda_manual_seed(1234, tp_rank=0, ep_rank=0, etp_rank=0)

    def _make_layer(self):
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=1,
            intermediate_size=INTERMEDIATE,
            n_routed_experts=NUM_EXPERTS,
            n_shared_experts=0,
            num_experts_per_tok=1,
            moe_intermediate_size=INTERMEDIATE,
            moe_token_dispatcher_type="alltoall",
            moe_expert_fusion=False,
            moe_use_fusion_node=True,
            fp8=None,
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=False,
            tensor_model_parallel_size=1,
            params_dtype=paddle.bfloat16,
        )
        mlp_spec = MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            down_proj=RowParallelLinear,
            hidden_act=None,
        )
        layer = MoELayer(
            config,
            sublayers=MoESublayers(mlp_spec=mlp_spec),
            pg_collection=SimpleNamespace(ep=None, expt_dp=None),
        )
        layer.moe_use_fusion_node = True
        layer.fp8 = "e4m3"
        return layer

    def _expert_weights(self, layer):
        return [
            layer.experts[0].up_gate_proj.weight,
            layer.experts[0].down_proj.weight,
        ]

    def test_quant_transpose_none_stores_both_layouts(self):
        layer = self._make_layer()
        layer.fp8_quant_weight(batch_mode=False, quant_transpose=None)

        for weight in self._expert_weights(layer):
            self.assertEqual(weight.fp8_weight_stacked.dtype, paddle.float8_e4m3fn)
            self.assertEqual(weight.fp8_scale_stacked.dtype, paddle.float32)
            self.assertEqual(
                weight.fp8_weight_stacked_transpose.dtype,
                paddle.float8_e4m3fn,
            )
            self.assertEqual(
                weight.fp8_scale_stacked_transpose.dtype,
                paddle.float32,
            )

    def test_quant_transpose_false_stores_nontranspose_and_nulls_transpose(
        self,
    ):
        layer = self._make_layer()
        layer.fp8_quant_weight(batch_mode=False, quant_transpose=False)

        for weight in self._expert_weights(layer):
            self.assertEqual(weight.fp8_weight_stacked.dtype, paddle.float8_e4m3fn)
            self.assertEqual(weight.fp8_scale_stacked.dtype, paddle.float32)
            # fp8_weight_stacked_transpose is explicitly set to None
            self.assertIsNone(weight.fp8_weight_stacked_transpose)
            self.assertIsNone(weight.fp8_scale_stacked_transpose)

    def test_quant_transpose_true_stores_both_layouts(self):
        layer = self._make_layer()
        layer.fp8_quant_weight(batch_mode=False, quant_transpose=True)

        for weight in self._expert_weights(layer):
            self.assertEqual(weight.fp8_weight_stacked.dtype, paddle.float8_e4m3fn)
            self.assertEqual(weight.fp8_scale_stacked.dtype, paddle.float32)
            self.assertEqual(
                weight.fp8_weight_stacked_transpose.dtype,
                paddle.float8_e4m3fn,
            )
            self.assertEqual(
                weight.fp8_scale_stacked_transpose.dtype,
                paddle.float32,
            )


if __name__ == "__main__":
    unittest.main()
