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
Supplementary coverage tests for auto_subbatch_mode="pre_permute".

Covers branches missed by the main test_moe_auto_subbatch_pre_permute.py:
  1. BF16 gemm path (use_fp8_mlp=False) - forward and backward
  2. Non-FP8 dispatch a2a input (no fp8_dispatched_handle, use_fp8_mlp=True)
  3. recompute_moe_premute=True path
  4. S=0 empty input backward early return
  5. Invalid auto_subbatch_mode ValueError
  6. BF16 path + recompute backward combination

Run with:
  python tests/fleet/single_card_tests/ai_edited_test/moe/test_ai_pre_permute_coverage.py
"""

import logging
import os
import unittest

import numpy as np

os.environ["FLAGS_use_virtual_memory_auto_growth"] = "True"
os.environ["FLAGS_cudnn_deterministic"] = "True"

from types import SimpleNamespace

import paddle
from paddle import nn

from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.transformer.moe.fp8_utils import (
    tilewise_quant,
)
from paddleformers.fleet.transformer.moe.fusion_layer_utils import (
    FusionMoePyLayer,
    MlpNode,
)
from paddleformers.fleet.transformer.moe.moe_expert import GroupedMLPExpert
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# Patch MlpNode to force multi-chunk for small test sizes
_orig_mlpnode_init = MlpNode.__init__


def _patched_mlpnode_init(self, *args, **kwargs):
    _orig_mlpnode_init(self, *args, **kwargs)
    self.min_auto_subbatch_rows = 256
    self.max_pre_permute_chunk_size_fwd = 512
    self.max_pre_permute_chunk_size_bwd = 384


MlpNode.__init__ = _patched_mlpnode_init


class FakeDeepGemmMOELayer(nn.Layer):
    """Mock MoE layer using GroupedMLPExpert (stacked weights)."""

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
        self.experts = None

    def clear_main_grad(self):
        self.grouped_gemm_experts.weight1.main_grad = None
        self.grouped_gemm_experts.weight2.main_grad = None


class TestPrePermuteCoverage(unittest.TestCase):
    """Supplementary coverage tests for pre_permute auto subbatch."""

    @classmethod
    def setUpClass(cls):
        model_parallel_cuda_manual_seed(1234)
        cls.seq_len = 1024
        cls.topk = 4
        cls.hidden_size = 4096
        cls.intermediate_size = 1536
        cls.n_routed_experts = 8

    def _make_inputs_and_layer(self, seq_len=None, topk=None):
        """Create fresh inputs and moe_layer for each run."""
        seq_len = seq_len or self.seq_len
        topk = topk or self.topk

        paddle.seed(2026)
        np.random.seed(2026)

        hidden_states = paddle.randn([seq_len, self.hidden_size], "bfloat16")
        hidden_states_out_grad = paddle.randn_like(hidden_states)
        probs = paddle.randn([seq_len, topk])
        hidden_states.stop_gradient = False
        probs.stop_gradient = False

        # Routing
        indices_np = np.full([seq_len, topk], -1, dtype=np.int64)
        tokens_per_expert = [0] * self.n_routed_experts
        for i in range(seq_len):
            chosen = np.array([0])
            n_active = np.random.randint(topk)
            if n_active > 0:
                chosen = np.append(
                    chosen,
                    np.random.choice(
                        self.n_routed_experts - 1, size=n_active, replace=False
                    )
                    + 1,
                )
            indices_np[i, : n_active + 1] = np.sort(chosen)
            for expert_id in chosen:
                tokens_per_expert[expert_id] += 1
        indices = paddle.to_tensor(indices_np)

        moe_layer = FakeDeepGemmMOELayer(
            self.hidden_size,
            self.intermediate_size,
            self.n_routed_experts,
            tokens_per_expert,
        )
        moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
        moe_layer.clear_main_grad()

        return (
            hidden_states,
            hidden_states_out_grad,
            probs,
            indices,
            moe_layer,
            topk,
        )

    def _run_once(
        self,
        hidden_states,
        hidden_states_out_grad,
        probs,
        indices,
        moe_layer,
        topk,
        **kwargs,
    ):
        """Run forward + backward once. Returns (output, hs_grad, probs_grad, wgrad)."""
        params = {
            "use_fp8_mlp": True,
            "moe_deep_gemm": True,
            "recompute_moe_gate_up": False,
            "dequant_input": True,
            "moe_expert_fusion": True,
            "recompute_moe_premute": False,
            "use_bf16_gemm_weight_grad": True,
            "use_auto_subbatch": False,
            "auto_subbatch_mode": None,
            "moe_subbatch_diag": False,
        }
        params.update(kwargs)

        hs_out = FusionMoePyLayer.apply(
            hidden_states,
            probs,
            indices.clone(),
            moe_layer,
            topk,
            **params,
        )
        paddle.autograd.backward(hs_out, hidden_states_out_grad)

        hs_grad = hidden_states.grad
        probs_grad = probs.grad
        hidden_states.clear_grad()
        probs.clear_grad()
        wgrad = moe_layer.grouped_gemm_experts.weight2.main_grad
        moe_layer.clear_main_grad()
        return hs_out, hs_grad, probs_grad, wgrad

    def test_bf16_path_forward_backward(self):
        """Cover BF16 gemm path (use_fp8_mlp=False) forward + backward.

        Covers: 1429-1431, 1456-1458, 1554-1556, 1674-1725 (fwd BF16),
                2251-2253, 2476-2506 (bwd BF16).
        """
        logging.info("=== BF16 path (use_fp8_mlp=False) ref ===")
        hs, grad, probs, indices, layer, topk = self._make_inputs_and_layer()
        ref_out = self._run_once(
            hs,
            grad,
            probs,
            indices,
            layer,
            topk,
            use_fp8_mlp=False,
        )

        logging.info("=== BF16 path (use_fp8_mlp=False) pre_permute ===")
        hs2, grad2, probs2, indices2, layer2, topk2 = (
            self._make_inputs_and_layer()
        )
        pre_out = self._run_once(
            hs2,
            grad2,
            probs2,
            indices2,
            layer2,
            topk2,
            use_fp8_mlp=False,
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
        )

        for i, name in enumerate(
            ["hidden_states", "hidden_states_grad", "probs_grad"]
        ):
            np.testing.assert_equal(
                ref_out[i].float().numpy(),
                pre_out[i].float().numpy(),
                err_msg=f"{name} mismatch in BF16 path",
            )
        np.testing.assert_allclose(
            ref_out[3].float().numpy(),
            pre_out[3].float().numpy(),
            atol=1e-4,
            rtol=1e-5,
            err_msg="weight_grad mismatch in BF16 path",
        )

    def test_non_fp8_dispatch_input(self):
        """Cover non-FP8 dispatch a2a input (elif use_fp8_path branch, lines 1450-1453).

        No fp8_dispatched_handle → tilewise_quant called inside pre_permute fwd.
        """
        logging.info("=== Non-FP8 dispatch a2a ref ===")
        hs, grad, probs, indices, layer, topk = self._make_inputs_and_layer()
        ref_out = self._run_once(
            hs,
            grad,
            probs,
            indices,
            layer,
            topk,
            dequant_input=True,
            fp8_dispatched_handle=None,
        )

        logging.info("=== Non-FP8 dispatch a2a pre_permute ===")
        hs2, grad2, probs2, indices2, layer2, topk2 = (
            self._make_inputs_and_layer()
        )
        pre_out = self._run_once(
            hs2,
            grad2,
            probs2,
            indices2,
            layer2,
            topk2,
            dequant_input=True,
            fp8_dispatched_handle=None,
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
        )

        for i, name in enumerate(
            ["hidden_states", "hidden_states_grad", "probs_grad"]
        ):
            np.testing.assert_allclose(
                ref_out[i].float().numpy(),
                pre_out[i].float().numpy(),
                atol=1e-3,
                rtol=1e-4,
                err_msg=f"{name} mismatch in non-a2a path",
            )

    def test_recompute_moe_premute_path(self):
        """Cover recompute_moe_premute=True (line 1472, backward recompute path).

        Requires recompute_moe_gate_up=True and dequant_input=True.
        """
        logging.info("=== recompute_moe_premute=True ref ===")
        hs, grad, probs, indices, layer, topk = self._make_inputs_and_layer()
        hs_fp8, scale = tilewise_quant(hs)
        hs_fp8.stop_gradient = False
        ref_out = self._run_once(
            hs_fp8,
            grad,
            probs,
            indices,
            layer,
            topk,
            fp8_dispatched_handle={"scale": scale},
            recompute_moe_gate_up=True,
        )

        logging.info("=== recompute_moe_premute=True pre_permute ===")
        hs2, grad2, probs2, indices2, layer2, topk2 = (
            self._make_inputs_and_layer()
        )
        hs2_fp8, scale2 = tilewise_quant(hs2)
        hs2_fp8.stop_gradient = False
        pre_out = self._run_once(
            hs2_fp8,
            grad2,
            probs2,
            indices2,
            layer2,
            topk2,
            fp8_dispatched_handle={"scale": scale2},
            recompute_moe_gate_up=True,
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            recompute_moe_premute=True,
        )

        for i, name in enumerate(
            ["hidden_states", "hidden_states_grad", "probs_grad"]
        ):
            np.testing.assert_equal(
                ref_out[i].float().numpy(),
                pre_out[i].float().numpy(),
                err_msg=f"{name} mismatch with recompute_moe_premute=True",
            )
        np.testing.assert_allclose(
            ref_out[3].float().numpy(),
            pre_out[3].float().numpy(),
            atol=1e-4,
            rtol=1e-5,
            err_msg="weight_grad mismatch with recompute_moe_premute=True",
        )

    def test_empty_input_backward(self):
        """Cover S=0 early return in backward (lines 2110, 2119-2120)."""
        logging.info("=== S=0 empty input backward ===")
        tokens_per_expert = [0] * self.n_routed_experts
        moe_layer = FakeDeepGemmMOELayer(
            self.hidden_size,
            self.intermediate_size,
            self.n_routed_experts,
            tokens_per_expert,
        )
        moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
        moe_layer.clear_main_grad()

        hidden_states = paddle.empty([0, self.hidden_size], dtype="bfloat16")
        hidden_states_out_grad = paddle.empty(
            [0, self.hidden_size], dtype="bfloat16"
        )
        probs = paddle.empty([0, self.topk], dtype="float32")
        indices = paddle.empty([0, self.topk], dtype="int64")
        hidden_states.stop_gradient = False
        probs.stop_gradient = False

        hs_out = FusionMoePyLayer.apply(
            hidden_states,
            probs,
            indices.clone(),
            moe_layer,
            self.topk,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            recompute_moe_gate_up=False,
            dequant_input=True,
            moe_expert_fusion=True,
            recompute_moe_premute=False,
            use_bf16_gemm_weight_grad=True,
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            moe_subbatch_diag=False,
        )
        paddle.autograd.backward(hs_out, hidden_states_out_grad)

        self.assertEqual(hidden_states.grad.shape, [0, self.hidden_size])
        self.assertEqual(probs.grad.shape, [0, self.topk])

    def test_invalid_auto_subbatch_mode(self):
        """Cover ValueError for invalid auto_subbatch_mode (line 382)."""
        logging.info("=== Invalid auto_subbatch_mode ===")
        hs, grad, probs, indices, layer, topk = self._make_inputs_and_layer()
        hs_fp8, scale = tilewise_quant(hs)
        hs_fp8.stop_gradient = False

        with self.assertRaisesRegex(
            ValueError, "auto_subbatch_mode must be one of"
        ):
            self._run_once(
                hs_fp8,
                grad,
                probs,
                indices,
                layer,
                topk,
                fp8_dispatched_handle={"scale": scale},
                use_auto_subbatch=True,
                auto_subbatch_mode="invalid_mode",
            )

    def test_bf16_path_with_recompute(self):
        """Cover BF16 + recompute backward path (lines 2476-2506, 2318)."""
        logging.info("=== BF16 + recompute ref ===")
        hs, grad, probs, indices, layer, topk = self._make_inputs_and_layer()
        ref_out = self._run_once(
            hs,
            grad,
            probs,
            indices,
            layer,
            topk,
            use_fp8_mlp=False,
            recompute_moe_gate_up=True,
        )

        logging.info("=== BF16 + recompute pre_permute ===")
        hs2, grad2, probs2, indices2, layer2, topk2 = (
            self._make_inputs_and_layer()
        )
        pre_out = self._run_once(
            hs2,
            grad2,
            probs2,
            indices2,
            layer2,
            topk2,
            use_fp8_mlp=False,
            recompute_moe_gate_up=True,
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            recompute_moe_premute=True,
        )

        for i, name in enumerate(
            ["hidden_states", "hidden_states_grad", "probs_grad"]
        ):
            np.testing.assert_equal(
                ref_out[i].float().numpy(),
                pre_out[i].float().numpy(),
                err_msg=f"{name} mismatch in BF16+recompute path",
            )
        np.testing.assert_allclose(
            ref_out[3].float().numpy(),
            pre_out[3].float().numpy(),
            atol=1e-4,
            rtol=1e-5,
            err_msg="weight_grad mismatch in BF16+recompute path",
        )

    def test_bf16_cached_backward_path(self):
        """Cover BF16 cached backward path (lines 2251-2253)."""
        logging.info("=== BF16 cached backward ref ===")
        hs, grad, probs, indices, layer, topk = self._make_inputs_and_layer()
        ref_out = self._run_once(
            hs,
            grad,
            probs,
            indices,
            layer,
            topk,
            use_fp8_mlp=False,
        )

        logging.info("=== BF16 cached backward pre_permute ===")
        hs2, grad2, probs2, indices2, layer2, topk2 = (
            self._make_inputs_and_layer()
        )
        pre_out = self._run_once(
            hs2,
            grad2,
            probs2,
            indices2,
            layer2,
            topk2,
            use_fp8_mlp=False,
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            recompute_moe_premute=False,
        )

        for i, name in enumerate(
            ["hidden_states", "hidden_states_grad", "probs_grad"]
        ):
            np.testing.assert_equal(
                ref_out[i].float().numpy(),
                pre_out[i].float().numpy(),
                err_msg=f"{name} mismatch in BF16 cached backward",
            )
        np.testing.assert_allclose(
            ref_out[3].float().numpy(),
            pre_out[3].float().numpy(),
            atol=1e-4,
            rtol=1e-5,
            err_msg="weight_grad mismatch in BF16 cached backward",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
