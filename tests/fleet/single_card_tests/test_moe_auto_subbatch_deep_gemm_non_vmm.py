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
Single-card unit tests for auto_subbatch with moe_deep_gemm=True and VMM disabled.

Tests auto_subbatch fallback correctness with the legacy allocator when using
GroupedMLPExpert (stacked weights) instead of per-expert weight lists.

Run with:
  python tests/single_card_tests/test_moe_auto_subbatch_deep_gemm_non_vmm.py
"""

import contextlib
import logging
import os
import unittest

import numpy as np

os.environ["FLAGS_use_virtual_memory_auto_growth"] = "False"
os.environ["FLAGS_cudnn_deterministic"] = "True"

from types import SimpleNamespace

import paddle
from paddle import nn
from paddle.device.cuda.memory_analyzer import MemoryAnalysisTool

from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.transformer.moe.fp8_utils import (
    fused_stack_quant_without_cache,
    tilewise_quant,
)
from paddleformers.fleet.transformer.moe.fusion_layer_utils import (
    FusionMoePyLayer,
    MlpNode,
)
from paddleformers.fleet.transformer.moe.moe_expert import GroupedMLPExpert
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# hack: lower min_auto_subbatch_rows so small seq_len can trigger multi-chunk subbatch
_orig_mlpnode_init = MlpNode.__init__


def _patched_mlpnode_init(self, *args, **kwargs):
    _orig_mlpnode_init(self, *args, **kwargs)
    self.min_auto_subbatch_rows = 256


MlpNode.__init__ = _patched_mlpnode_init


class FakeDeepGemmMOELayer(nn.Layer):
    """
    A mock MoE layer for deep_gemm mode using GroupedMLPExpert (stacked weights).

    In deep_gemm mode, MoELayer creates grouped_gemm_experts instead of per-expert list.
    This fake layer mimics that structure.
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
            moe_intermediate_size=intermediate_size,
            gated_linear_unit=True,
        )
        self.grouped_gemm_experts = GroupedMLPExpert(
            num_local_experts=n_routed_experts,
            config=config,
            moe_deep_gemm=True,
        )
        # Initialize weights with small random values for numerical stability
        with paddle.no_grad():
            self.grouped_gemm_experts.weight1.set_value(
                paddle.randn(
                    self.grouped_gemm_experts.weight1.shape, dtype="bfloat16"
                )
                * 0.01
            )
            self.grouped_gemm_experts.weight2.set_value(
                paddle.randn(
                    self.grouped_gemm_experts.weight2.shape, dtype="bfloat16"
                )
                * 0.01
            )
        self.token_dispatcher = SimpleNamespace(
            _comm_manager=SimpleNamespace(
                tokens_per_expert=tokens_per_expert,
            ),
        )
        # deep_gemm mode has no per-expert list
        self.experts = None

    def clear_main_grad(self):
        self.grouped_gemm_experts.weight1.main_grad = None
        self.grouped_gemm_experts.weight2.main_grad = None

    def fp8_quant_weight(self, quant_transpose=True):
        """Simulate FP8QuantWeightCallback: quant bf16 weights to fp8 stacked tensors."""
        w1 = self.grouped_gemm_experts.weight1
        w2 = self.grouped_gemm_experts.weight2
        local_expert_num = w1.shape[0]
        w1_list = [w1[i, :, :] for i in range(local_expert_num)]
        w2_list = [w2[i, :, :] for i in range(local_expert_num)]

        # Non-transpose version
        fp8_w1, fp8_s1 = fused_stack_quant_without_cache(
            w1_list, transpose=False
        )
        w1.fp8_weight_stacked = fp8_w1
        w1.fp8_scale_stacked = fp8_s1

        fp8_w2, fp8_s2 = fused_stack_quant_without_cache(
            w2_list, transpose=False
        )
        w2.fp8_weight_stacked = fp8_w2
        w2.fp8_scale_stacked = fp8_s2

        # Transpose version
        if quant_transpose:
            fp8_w1_t, fp8_s1_t = fused_stack_quant_without_cache(
                w1_list, transpose=True
            )
            w1.fp8_weight_stacked_transpose = fp8_w1_t
            w1.fp8_scale_stacked_transpose = fp8_s1_t

            fp8_w2_t, fp8_s2_t = fused_stack_quant_without_cache(
                w2_list, transpose=True
            )
            w2.fp8_weight_stacked_transpose = fp8_w2_t
            w2.fp8_scale_stacked_transpose = fp8_s2_t
        else:
            w1.fp8_weight_stacked_transpose = None
            w1.fp8_scale_stacked_transpose = None
            w2.fp8_weight_stacked_transpose = None
            w2.fp8_scale_stacked_transpose = None

    def clear_weight_storage(self):
        """Simulate optimizer.clear_param_storage: clear bf16 weight memory."""
        self.grouped_gemm_experts.weight1._clear_to_zero_allocation()
        self.grouped_gemm_experts.weight2._clear_to_zero_allocation()


@contextlib.contextmanager
def legacy_no_free_space():
    """Mock legacy allocator free blocks as empty to simulate tight memory."""
    old_all_block_info = MemoryAnalysisTool.all_block_info
    MemoryAnalysisTool.all_block_info = lambda: [[]]
    try:
        yield
    finally:
        MemoryAnalysisTool.all_block_info = old_all_block_info


class TestAutoSubbatchDeepGemm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (vmm_enabled,) = paddle.framework.get_flags(
            "FLAGS_use_virtual_memory_auto_growth"
        ).values()
        print(
            "[test_moe_auto_subbatch_deep_gemm_non_vmm] "
            f"FLAGS_use_virtual_memory_auto_growth={vmm_enabled}"
        )
        model_parallel_cuda_manual_seed(1234)
        cls.seq_len = 1024
        cls.topk = 4
        cls.hidden_size = 4096
        cls.intermediate_size = 1536
        cls.n_routed_experts = 8

    def setUp(self):
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

        # Each token is assigned 1 to topk experts, always including expert 0
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

        moe_layer = FakeDeepGemmMOELayer(
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
            "moe_deep_gemm": True,
            "recompute_moe_gate_up": False,
            "dequant_input": True,
            "moe_expert_fusion": True,
            "recompute_moe_premute": False,
            "use_bf16_gemm_weight_grad": True,
            "fp8_dispatched_handle": {"scale": self.scale},
            "use_auto_subbatch": not is_ref,
            "moe_subbatch_diag": True,
        }
        params.update(kwargs)

        with (
            legacy_no_free_space()
            if tight_forward
            else contextlib.nullcontext()
        ):
            hidden_states = FusionMoePyLayer.apply(
                self.hidden_states,
                self.probs,
                self.indices.clone(),
                self.moe_layer,
                self.topk,
                **params,
            )

        with (
            legacy_no_free_space()
            if tight_backward
            else contextlib.nullcontext()
        ):
            paddle.autograd.backward(hidden_states, self.hidden_states_out_grad)

        hidden_states_grad = self.hidden_states.grad
        probs_grad = self.probs.grad
        self.hidden_states.clear_grad()
        self.probs.clear_grad()

        weight_grad = self.moe_layer.grouped_gemm_experts.weight2.main_grad
        self.moe_layer.clear_main_grad()

        return hidden_states, hidden_states_grad, probs_grad, weight_grad

    def compare_results(self, ref_out, tgt_out):
        names = [
            "hidden_states",
            "hidden_states_grad",
            "probs_grad",
            "weight_grad",
        ]
        # First three must be bitwise equal; weight_grad allows FP8 accumulation order diff
        tolerances = {
            "weight_grad": {"atol": 1e-4, "rtol": 1e-5},
        }
        for i, name in enumerate(names):
            ref_np = ref_out[i].float().numpy()
            tgt_np = tgt_out[i].float().numpy()
            tol = tolerances.get(name)
            if tol is None:
                np.testing.assert_equal(ref_np, tgt_np, err_msg=name)
            else:
                diff = np.abs(ref_np - tgt_np)
                max_diff = diff.max()
                if max_diff > tol["atol"]:
                    # Print top diffs on failure for debugging
                    flat = diff.flatten()
                    top_idx = np.argsort(flat)[-10:][::-1]
                    lines = [
                        f"\n=== {name} FAILED: max_abs_diff={max_diff:.6g} ==="
                    ]
                    for rank, idx in enumerate(top_idx):
                        coords = np.unravel_index(idx, ref_np.shape)
                        lines.append(
                            f"  #{rank} {coords} ref={ref_np.flat[idx]:.6g}, "
                            f"tgt={tgt_np.flat[idx]:.6g}, diff={flat[idx]:.6g}"
                        )
                    self.fail("\n".join(lines))
                np.testing.assert_allclose(ref_np, tgt_np, **tol, err_msg=name)

    def test_auto_subbatch_deep_gemm_no_recompute(self):
        """deep_gemm + no_recompute: auto_subbatch vs group_gemm reference"""
        ref_out = self.run_moe_layer(is_ref=True)

        cases = {}
        kwargs = {"recompute_moe_gate_up": False}

        logging.info("case1 (deep_gemm, plenty)")
        cases["case1 (plenty)"] = self.run_moe_layer(**kwargs)
        logging.info("case2 (deep_gemm, tight_fwd)")
        cases["case2 (tight_fwd)"] = self.run_moe_layer(
            tight_forward=True, **kwargs
        )

        logging.info("case3 (deep_gemm, tight_bwd)")
        cases["case3 (tight_bwd)"] = self.run_moe_layer(
            tight_backward=True, **kwargs
        )
        logging.info("case4 (deep_gemm, tight_both)")
        cases["case4 (tight_both)"] = self.run_moe_layer(
            tight_forward=True, tight_backward=True, **kwargs
        )

        for name, result in cases.items():
            with self.subTest(case=name):
                self.compare_results(ref_out, result)

    def test_auto_subbatch_deep_gemm_recompute(self):
        """deep_gemm + recompute_moe_gate_up: auto_subbatch vs group_gemm reference"""
        ref_out = self.run_moe_layer(is_ref=True, recompute_moe_gate_up=True)

        cases = {}
        kwargs = {"recompute_moe_gate_up": True}

        logging.info("case5 (deep_gemm recompute, plenty)")
        cases["case5 (plenty)"] = self.run_moe_layer(**kwargs)
        logging.info("case6 (deep_gemm recompute, tight_fwd)")
        cases["case6 (tight_fwd)"] = self.run_moe_layer(
            tight_forward=True, **kwargs
        )

        logging.info("case7 (deep_gemm recompute, tight_bwd)")
        cases["case7 (tight_bwd)"] = self.run_moe_layer(
            tight_backward=True, **kwargs
        )
        logging.info("case8 (deep_gemm recompute, tight_both)")
        cases["case8 (tight_both)"] = self.run_moe_layer(
            tight_forward=True, tight_backward=True, **kwargs
        )

        for name, result in cases.items():
            with self.subTest(case=name):
                self.compare_results(ref_out, result)

    def test_auto_subbatch_deep_gemm_offline_quant(self):
        """deep_gemm + offline fp8 quant + clear bf16 weight: simulates FP8QuantWeightCallback.

        After offline quant, bf16 weight memory is cleared (memory_size=0).
        Forward/backward must work using only fp8_weight_stacked attributes.
        """
        # First get reference output BEFORE clearing weight
        ref_out = self.run_moe_layer(is_ref=True)

        # Now simulate FP8QuantWeightCallback: quant + clear bf16
        self.moe_layer.fp8_quant_weight(quant_transpose=False)
        self.moe_layer.clear_weight_storage()

        cases = {}
        # no recompute
        kwargs = {"recompute_moe_gate_up": False}

        logging.info("case9 (offline_quant, plenty)")
        cases["case9 (plenty)"] = self.run_moe_layer(**kwargs)
        logging.info("case10 (offline_quant, tight_fwd)")
        cases["case10 (tight_fwd)"] = self.run_moe_layer(
            tight_forward=True, **kwargs
        )
        logging.info("case11 (offline_quant, tight_bwd)")
        cases["case11 (tight_bwd)"] = self.run_moe_layer(
            tight_backward=True, **kwargs
        )
        logging.info("case12 (offline_quant, tight_both)")
        cases["case12 (tight_both)"] = self.run_moe_layer(
            tight_forward=True, tight_backward=True, **kwargs
        )

        # with recompute
        kwargs_rc = {"recompute_moe_gate_up": True}

        logging.info("case9r (offline_quant+recompute, plenty)")
        cases["case9r (plenty)"] = self.run_moe_layer(**kwargs_rc)
        logging.info("case10r (offline_quant+recompute, tight_fwd)")
        cases["case10r (tight_fwd)"] = self.run_moe_layer(
            tight_forward=True, **kwargs_rc
        )
        logging.info("case11r (offline_quant+recompute, tight_bwd)")
        cases["case11r (tight_bwd)"] = self.run_moe_layer(
            tight_backward=True, **kwargs_rc
        )
        logging.info("case12r (offline_quant+recompute, tight_both)")
        cases["case12r (tight_both)"] = self.run_moe_layer(
            tight_forward=True, tight_backward=True, **kwargs_rc
        )

        for name, result in cases.items():
            with self.subTest(case=name):
                self.compare_results(ref_out, result)

    def test_auto_subbatch_deep_gemm_offline_quant_transpose(self):
        """deep_gemm + offline fp8 quant with quant_transpose=True + clear bf16 weight.

        When quant_transpose=True, fp8_weight_stacked_transpose is pre-computed.
        The per-expert fallback should use the pre-computed transpose directly.
        """
        # First get reference output BEFORE clearing weight
        ref_out = self.run_moe_layer(is_ref=True)

        # Simulate FP8QuantWeightCallback with quant_transpose=True
        self.moe_layer.fp8_quant_weight(quant_transpose=True)
        self.moe_layer.clear_weight_storage()

        cases = {}
        # no recompute
        kwargs = {"recompute_moe_gate_up": False}

        logging.info("case13 (offline_quant_transpose, plenty)")
        cases["case13 (plenty)"] = self.run_moe_layer(**kwargs)
        logging.info("case14 (offline_quant_transpose, tight_fwd)")
        cases["case14 (tight_fwd)"] = self.run_moe_layer(
            tight_forward=True, **kwargs
        )
        logging.info("case15 (offline_quant_transpose, tight_bwd)")
        cases["case15 (tight_bwd)"] = self.run_moe_layer(
            tight_backward=True, **kwargs
        )
        logging.info("case16 (offline_quant_transpose, tight_both)")
        cases["case16 (tight_both)"] = self.run_moe_layer(
            tight_forward=True, tight_backward=True, **kwargs
        )

        # with recompute
        kwargs_rc = {"recompute_moe_gate_up": True}

        logging.info("case13r (offline_quant_transpose+recompute, plenty)")
        cases["case13r (plenty)"] = self.run_moe_layer(**kwargs_rc)
        logging.info("case14r (offline_quant_transpose+recompute, tight_fwd)")
        cases["case14r (tight_fwd)"] = self.run_moe_layer(
            tight_forward=True, **kwargs_rc
        )
        logging.info("case15r (offline_quant_transpose+recompute, tight_bwd)")
        cases["case15r (tight_bwd)"] = self.run_moe_layer(
            tight_backward=True, **kwargs_rc
        )
        logging.info("case16r (offline_quant_transpose+recompute, tight_both)")
        cases["case16r (tight_both)"] = self.run_moe_layer(
            tight_forward=True, tight_backward=True, **kwargs_rc
        )

        for name, result in cases.items():
            with self.subTest(case=name):
                self.compare_results(ref_out, result)


if __name__ == "__main__":
    unittest.main()
