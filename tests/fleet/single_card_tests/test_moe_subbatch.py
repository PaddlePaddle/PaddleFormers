#!/usr/bin/env python3
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
Single-card unit tests for ordinary (non-auto) subbatch functionality.

Tests:
  1. test_subbatch_vs_ref: Compare ordinary subbatch results against group_gemm reference.
     - split_gemm + selective_recompute (moe_expert_fusion=False,
       use_auto_subbatch=False).

Run with:
  python tests/fleet/single_card_tests/test_moe_subbatch.py
"""

import os
import unittest

import numpy as np

# os.environ["FLAGS_use_virtual_memory_auto_growth"] = "True"
os.environ["FLAGS_cudnn_deterministic"] = "True"

from types import SimpleNamespace

import paddle
from paddle import nn

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
from paddleformers.fleet.transformer.moe.fp8_utils import tilewise_quant
from paddleformers.fleet.transformer.moe.fusion_layer_utils import (
    FusionMoePyLayer,
)
from paddleformers.fleet.transformer.moe.moe_expert import StandardMLPExpert
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class FakeMOELayer(nn.Layer):
    """
    A mock MoE layer that provides the interface expected by FusionMoePyLayer.

    Uses StandardMLPExpert (native PaddleFleet expert) for realistic testing.

    Required attributes:
      - self.experts: nn.LayerList of expert modules
      - self.token_dispatcher._comm_manager.tokens_per_expert: list[int]
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        n_routed_experts,
        tokens_per_expert,
    ):
        super().__init__()
        config = TransformerConfig(
            hidden_size=hidden_size,
            gated_linear_unit=True,
        )
        mlp_spec = MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            down_proj=RowParallelLinear,
        )
        self.experts = nn.LayerList(
            [
                StandardMLPExpert(
                    config,
                    moe_intermediate_size=intermediate_size,
                    is_expert=True,
                    mlp_spec=mlp_spec,
                )
                for _ in range(n_routed_experts)
            ]
        )
        self.token_dispatcher = SimpleNamespace(
            _comm_manager=SimpleNamespace(
                tokens_per_expert=tokens_per_expert,
            ),
        )

    def clear_main_grad(self):
        for expert in self.experts:
            expert.up_gate_proj.weight.main_grad = None
            expert.down_proj.weight.main_grad = None


class TestSubbatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置默认配置"""
        model_parallel_cuda_manual_seed(1234)
        cls.seq_len = 1000
        cls.topk = 4
        cls.hidden_size = 4096
        cls.intermediate_size = 1536
        cls.n_routed_experts = 8

    def setUp(self):
        """创建测试层和输入数据"""
        paddle.seed(2026)
        np.random.seed(2026)

        hidden_states = paddle.randn(
            [self.seq_len, self.hidden_size], "bfloat16"
        )
        hidden_states_out_grad = paddle.randn_like(hidden_states)
        hidden_states, scale = tilewise_quant(hidden_states)
        probs = paddle.randn([self.seq_len, self.topk])
        hidden_states.stop_gradient = False
        probs.stop_gradient = False

        self.hidden_states = hidden_states
        self.hidden_states_out_grad = hidden_states_out_grad
        self.scale = scale
        self.probs = probs

        # 每个 token 随机分配 1 到 topk 个专家，但总是包括专家0，给专家0增加压力
        indices_np = np.full([self.seq_len, self.topk], -1, dtype=np.int64)
        tokens_per_expert = [0] * self.n_routed_experts
        for i in range(self.seq_len):
            chosen = np.array([0])
            n_active = np.random.randint(self.topk)
            if n_active > 0:
                chosen = np.append(
                    chosen,
                    np.random.choice(
                        self.n_routed_experts - 1,
                        size=n_active,
                        replace=False,
                    )
                    + 1,
                )
            indices_np[i, : n_active + 1] = np.sort(chosen)
            for expert_id in chosen:
                tokens_per_expert[expert_id] += 1
        self.indices = paddle.to_tensor(indices_np)

        moe_layer = FakeMOELayer(
            self.hidden_size,
            self.intermediate_size,
            self.n_routed_experts,
            tokens_per_expert,
        )
        moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
        moe_layer.clear_main_grad()
        self.moe_layer = moe_layer

    def run_moe_layer(
        self, is_ref=False, tight_forward=False, tight_backward=False, **kwargs
    ):
        params = {
            "use_fp8_mlp": True,
            # "moe_deep_gemm": True,
            "recompute_moe_gate_up": True,
            "dequant_input": True,
            "moe_expert_fusion": True,
            "recompute_moe_premute": False,
            "use_bf16_gemm_weight_grad": True,
            "fp8_dispatched_handle": {"scale": self.scale},
            "use_auto_subbatch": False,
        }
        params.update(kwargs)

        hidden_states = FusionMoePyLayer.apply(
            self.hidden_states,
            self.probs,
            self.indices.clone(),
            self.moe_layer,
            self.topk,
            **params,
        )

        paddle.autograd.backward(hidden_states, self.hidden_states_out_grad)

        hidden_states_grad = self.hidden_states.grad
        probs_grad = self.probs.grad
        self.hidden_states.clear_grad()
        self.probs.clear_grad()

        # 专家0最大，只要检查专家0的 weight_grad 即可
        weight_grad = self.moe_layer.experts[0].down_proj.weight.main_grad
        self.moe_layer.clear_main_grad()

        return hidden_states, hidden_states_grad, probs_grad, weight_grad

    def compare_results(self, ref_out, tgt_out, loose_weight=False):
        for i, name in enumerate(
            ["hidden_states", "hidden_states_grad", "probs_grad"]
        ):
            np.testing.assert_equal(
                ref_out[i].float().numpy(),
                tgt_out[i].float().numpy(),
                name,
            )
        i += 1
        if loose_weight:
            np.testing.assert_allclose(
                ref_out[i].numpy(),
                tgt_out[i].numpy(),
                atol=1.0,
                rtol=1e-5,
            )
        else:
            np.testing.assert_equal(ref_out[i].numpy(), tgt_out[i].numpy())

    def test_subbatch_vs_ref(self):
        """测试普通 subbatch (非 auto_subbatch) 的多种情况与 group_gemm 是否对齐"""
        # group_gemm (reference, moe_expert_fusion=True, no subbatch)
        ref_out = self.run_moe_layer(is_ref=True)

        # --- split_gemm + selective_recompute (普通 subbatch，非 auto_subbatch) ---
        kwargs = {
            "moe_expert_fusion": False,
            "recompute_moe_premute": True,
            "recompute_moe_gate_up": True,
            "moe_subbatch_token_num_after_dispatch": 512,
        }
        # case: 显存充裕
        out = self.run_moe_layer(**kwargs)

        # split_gemm vs group_gemm 路径有固有精度差，weight_grad 允许有小误差
        self.compare_results(ref_out, out, loose_weight=True)


if __name__ == "__main__":
    unittest.main()
