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
Single-card unit tests for auto_subbatch functionality.

Tests:
  1. test_auto_subbatch_vs_ref: Compare auto_subbatch results against group_gemm reference.
     - case1-4: split_gemm + selective_recompute (moe_expert_fusion=False),
                varying tight_forward/tight_backward combinations.
     - case5-7: group_gemm + selective_recompute (Level 0, moe_expert_fusion=True),
                varying tight_forward/tight_backward combinations.
     - case8-10: group_gemm + full_recompute (Level 0, fp8_quant_weight 预量化),
                 varying tight_forward/tight_backward combinations.
  2. test_vmm_utils: Unit tests for VMM utility functions.

Note: Ordinary (non-auto) subbatch tests are in test_moe_subbatch.py.

Run with:
  python tests/single_card_tests/test_moe_auto_subbatch.py
"""

import contextlib
import logging
import os
import unittest

import numpy as np

os.environ["FLAGS_use_virtual_memory_auto_growth"] = "True"
os.environ["FLAGS_cudnn_deterministic"] = "True"

from types import SimpleNamespace

import paddle
from paddle import nn
from paddle.device.cuda.memory_analyzer import MemoryAnalysisTool

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
from paddleformers.fleet.transformer.moe.fp8_utils import tilewise_quant
from paddleformers.fleet.transformer.moe.fusion_layer_utils import FusionMoePyLayer
from paddleformers.fleet.transformer.moe.moe_expert import StandardMLPExpert
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer
from paddleformers.fleet.transformer.moe.vmm_utils import (
    find_max_concurrent_subbatch_size,
    find_max_sequence_subbatch_size,
    vmm_free_and_growable_block_info,
)
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

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        """借用 MoELayer 的实现，对 expert 权重做 FP8 预量化。"""
        # MoELayer.fp8_quant_weight 检查 self.moe_use_fusion_node and self.fp8
        self.moe_use_fusion_node = True
        self.fp8 = True
        self.use_ue8m0 = False
        MoELayer.fp8_quant_weight(self, batch_mode=batch_mode, quant_transpose=quant_transpose)


@contextlib.contextmanager
def vmm_no_free_space():
    """占用所有的 free block 并且禁止 growable 空间，假装没有多余显存"""
    (old_value,) = paddle.framework.get_flags("FLAGS_max_reserved_threshold_in_gb").values()
    paddle.set_flags({"FLAGS_max_reserved_threshold_in_gb": 0})
    buffers = []
    for size, _ in MemoryAnalysisTool.vmm_free_block_info()[-1]:
        buffers.append(paddle.empty([size], dtype="uint8"))
    try:
        yield
    finally:
        paddle.set_flags({"FLAGS_max_reserved_threshold_in_gb": old_value})
        del buffers


class TestAutoSubbatch(unittest.TestCase):
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

        hidden_states = paddle.randn([self.seq_len, self.hidden_size], "bfloat16")
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

    def run_moe_layer(self, is_ref=False, tight_forward=False, tight_backward=False, **kwargs):
        params = {
            "use_fp8_mlp": True,
            "moe_deep_gemm": False,
            "recompute_moe_gate_up": True,
            "dequant_input": True,
            "moe_expert_fusion": is_ref,
            "recompute_moe_premute": False,
            "use_bf16_gemm_weight_grad": True,
            "fp8_dispatched_handle": {"scale": self.scale},
            "use_auto_subbatch": not is_ref,
            "moe_subbatch_diag": True,
        }
        params.update(kwargs)

        with vmm_no_free_space() if tight_forward else contextlib.nullcontext():
            hidden_states = FusionMoePyLayer.apply(
                self.hidden_states,
                self.probs,
                self.indices.clone(),
                self.moe_layer,
                self.topk,
                **params,
            )

        with (vmm_no_free_space() if tight_backward else contextlib.nullcontext()):
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
        for i, name in enumerate(["hidden_states", "hidden_states_grad", "probs_grad"]):
            np.testing.assert_equal(
                ref_out[i].float().numpy(),
                tgt_out[i].float().numpy(),
                name,
            )
        i = 3  # weight_grad
        if loose_weight:
            np.testing.assert_allclose(
                ref_out[i].numpy(),
                tgt_out[i].numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        else:
            np.testing.assert_equal(ref_out[i].numpy(), tgt_out[i].numpy())

    def test_auto_subbatch_vs_ref(self):
        """测试 auto_subbatch 的多种情况与 group_gemm 是否对齐"""
        logging.info("baseline group_gemm")
        # group_gemm (reference, no auto_subbatch)
        ref_out = self.run_moe_layer(is_ref=True)

        cases = {}

        # --- group_gemm + no_recompute (moe_expert_fusion=False) ---
        kwargs = {
            "moe_expert_fusion": True,
            "recompute_moe_premute": False,
            "recompute_moe_gate_up": False,
        }

        logging.info("case1 (group, plenty)")
        # case1: 显存充裕
        cases["case1 (group, plenty)"] = self.run_moe_layer(**kwargs)
        # case2: 前向紧张
        logging.info("case2 (group, tight_fwd)")
        cases["case2 (group, tight_fwd)"] = self.run_moe_layer(tight_forward=True, **kwargs)
        # case3: 反向紧张
        logging.info("case3 (group, tight_bwd)")
        cases["case3 (group, tight_bwd)"] = self.run_moe_layer(tight_backward=True, **kwargs)
        # case4: 前向+反向都紧张
        logging.info("case4 (group, tight_both)")
        cases["case4 (group, tight_both)"] = self.run_moe_layer(tight_forward=True, tight_backward=True, **kwargs)

        kwargs = {
            "moe_expert_fusion": True,
            "recompute_moe_premute": False,
            "recompute_moe_gate_up": True,
            # "recompute_unzipped": True,
        }
        # --- group_gemm + selective_recompute (recompute_moe_gate_up)---
        # case5: 显存充裕 → 走 3a group_gemm
        logging.info("case5 (recompute_moe_gate_up, group, plenty)")
        cases["case5 (group, plenty)"] = self.run_moe_layer(**kwargs)
        # case6: 前向紧张 → fallback 到逐专家
        logging.info("case6 (recompute_moe_gate_up, group, tight_fwd)")
        cases["case6 (group, tight_fwd)"] = self.run_moe_layer(tight_forward=True, **kwargs)
        # case7: 反向紧张 → 前向 3a, 反向 fallback
        logging.info("case7 (recompute_moe_gate_up, group, tight_bwd)")
        cases["case7 (group, tight_bwd)"] = self.run_moe_layer(tight_backward=True, **kwargs)
        # case8: 向+反向都紧张, fallback
        logging.info("case8 (recompute_moe_gate_up, group, tight_both)")
        cases["case8 (group, tight_both)"] = self.run_moe_layer(tight_forward=True, tight_backward=True, **kwargs)

        # --- group_gemm + selective_recompute (recompute_moe_gate_up + recompute_moe_premute) ---
        # when recompute_moe_premute is True. recompute_moe_gate_up must be true
        kwargs = {
            "moe_expert_fusion": True,
            "recompute_moe_premute": False,
            "recompute_moe_gate_up": False,
        }
        # case9: 显存充裕 → 走 3a group_gemm
        logging.info("case9 (recompute_moe_gate_up,recompute_moe_premute, group, plenty)")
        cases["case9 (group, plenty)"] = self.run_moe_layer(**kwargs)
        # case10: 前向紧张 → fallback 到逐专家
        cases["case10 (group, tight_fwd)"] = self.run_moe_layer(tight_forward=True, **kwargs)
        # case11: 反向紧张 → 前向 3a, 反向 fallback
        logging.info("case11 (recompute_moe_gate_up,recompute_moe_premute, group, tight_bwd)")
        cases["case11 (group, tight_bwd)"] = self.run_moe_layer(tight_backward=True, **kwargs)
        # case12: 向+反向都紧张, fallback
        logging.info("case12 (recompute_moe_gate_up,recompute_moe_premute, group, tight_both)")
        cases["case12 (group, tight_both)"] = self.run_moe_layer(tight_forward=True, tight_backward=True, **kwargs)

        # split gemm + no moe_subbatch_token_num_after_dispatch ---
        kwargs = {
            "moe_expert_fusion": False,
            "recompute_moe_premute": False,
            "recompute_moe_gate_up": False,
        }
        # case13: 显存充裕 → 走 3a group_gemm
        logging.info("case13 (split, plenty)")
        cases["case13 (split, plenty)"] = self.run_moe_layer(**kwargs)
        # case14: 前向紧张 → fallback 到逐专家
        logging.info("case14 (split, tight_fwd)")
        cases["case14 (split, tight_fwd)"] = self.run_moe_layer(tight_forward=True, **kwargs)
        # case15: 反向紧张 → 前向 3a, 反向 fallback
        logging.info("case15 (split, tight_bwd)")
        cases["case15 (split, tight_bwd)"] = self.run_moe_layer(tight_backward=True, **kwargs)
        # case16: 向+反向都紧张, fallback
        logging.info("case16 (split, tight_both)")
        cases["case16 (split, tight_both)"] = self.run_moe_layer(tight_forward=True, tight_backward=True, **kwargs)

        # split gemm + moe_subbatch_token_num_after_dispatch 512 ---
        kwargs = {
            "moe_expert_fusion": False,
            "recompute_moe_premute": False,
            "recompute_moe_gate_up": True,
            "moe_subbatch_token_num_after_dispatch": 512,
        }
        # case13: 显存充裕 → 走 3a group_gemm
        logging.info("case13 (split, plenty)")
        cases["case13 (split, plenty)"] = self.run_moe_layer(**kwargs)
        # case14: 前向紧张 → fallback 到逐专家
        logging.info("case14 (split, tight_fwd)")
        cases["case14 (split, tight_fwd)"] = self.run_moe_layer(tight_forward=True, **kwargs)
        # case15: 反向紧张 → 前向 3a, 反向 fallback
        logging.info("case15 (split, tight_bwd)")
        cases["case15 (split, tight_bwd)"] = self.run_moe_layer(tight_backward=True, **kwargs)
        # case16: 向+反向都紧张, fallback
        logging.info("case16 (split, tight_both)")
        cases["case16 (split, tight_both)"] = self.run_moe_layer(tight_forward=True, tight_backward=True, **kwargs)

        # group gemm + 预量化
        kwargs = {
            "moe_expert_fusion": True,
            "recompute_moe_premute": False,
            "recompute_moe_gate_up": False,
        }
        self.moe_layer.fp8_quant_weight(batch_mode=True, quant_transpose=False)
        logging.info("case17 (group, plenty) pre_quant")
        cases["case17 (group, plenty) pre_quant"] = self.run_moe_layer(**kwargs)
        logging.info("case18 (group, tight_fwd) pre_quant")
        cases["case18 (group, tight_fwd) pre_quant"] = self.run_moe_layer(tight_forward=True, **kwargs)
        logging.info("case19 (group, tight_bwd) pre_quant")
        cases["case19 (group, tight_bwd) pre_quant"] = self.run_moe_layer(tight_backward=True, **kwargs)
        logging.info("case20 (group, tight_both) pre_quant")
        cases["case20 (group, tight_both) pre_quant"] = self.run_moe_layer(
            tight_forward=True, tight_backward=True, **kwargs
        )

        # split gemm + recompute + moe_subbatch_token_num_after_dispatch 512 ---
        kwargs = {
            "moe_expert_fusion": False,
            "recompute_moe_premute": True,
            "recompute_moe_gate_up": True,
            "moe_subbatch_token_num_after_dispatch": 512,
        }
        # case21: 显存充裕 → 走 3a group_gemm
        logging.info("case21 (split, plenty)")
        cases["case21 (split, plenty)"] = self.run_moe_layer(**kwargs)
        # case22: 前向紧张 → fallback 到逐专家
        logging.info("case22 (split, tight_fwd)")
        cases["case22 (split, tight_fwd)"] = self.run_moe_layer(tight_forward=True, **kwargs)
        # case23: 反向紧张 → 前向 3a, 反向 fallback
        logging.info("case23 (split, tight_bwd)")
        cases["case23 (split, tight_bwd)"] = self.run_moe_layer(tight_backward=True, **kwargs)
        # case24: 向+反向都紧张, fallback
        logging.info("case24 (split, tight_both)")
        cases["case24 (split, tight_both)"] = self.run_moe_layer(tight_forward=True, tight_backward=True, **kwargs)

        loose_cases = {}
        # --- 数值对比 ---
        loose_cases = {
            "case15 (split, tight_bwd)",
            "case16 (split, tight_both)",
            "case23 (split, tight_bwd)",
            "case24 (split, tight_both)",
        }
        for name, result in cases.items():
            with self.subTest(case=name):
                self.compare_results(ref_out, result, loose_weight=(name in loose_cases))


class TestVMMUtils(unittest.TestCase):
    def test_vmm_utils(self):
        """测试 vmm 相关搜索函数"""
        old_func = MemoryAnalysisTool.vmm_all_block_info

        # test empty heap
        MemoryAnalysisTool.vmm_all_block_info = lambda: [[]]
        info = vmm_free_and_growable_block_info()
        self.assertEqual(len(info), 1)

        # test heap with separate free blocks
        MemoryAnalysisTool.vmm_all_block_info = lambda: [[(1024, 0, True), (1024, 1024, False)]]
        info = vmm_free_and_growable_block_info()
        self.assertEqual(len(info), 2)

        # test corner cases
        self.assertEqual(find_max_concurrent_subbatch_size([]), 1)
        self.assertEqual(find_max_sequence_subbatch_size(1024, 1), 1)

        MemoryAnalysisTool.vmm_all_block_info = old_func


if __name__ == "__main__":
    unittest.main()
