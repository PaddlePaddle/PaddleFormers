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
Single-card unit tests for auto_subbatch_mode="pre_permute" with deep_gemm stacked weights.

Uses FakeDeepGemmMOELayer (GroupedMLPExpert with stacked [E, K, N] weights) to match
real production deployment.

Tests:
  1. test_pre_permute_vs_ref: Compare pre_permute results against group_gemm reference.
     - case1: plenty memory (no shrink)
     - case2: tight forward (force chunk shrink)
     - case3: tight backward
     - case4: tight both
  2. test_pre_permute_empty_experts: Some experts get 0 tokens in certain chunks.
  3. test_pre_permute_offline_quant: With offline FP8 weight quantization.
  4. test_pre_permute_offline_quant_transpose: With offline FP8 quant + transpose.

Run with:
  python tests/fleet/single_card_tests/test_moe_auto_subbatch_pre_permute.py
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

from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
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
    # Force multi-chunk with asymmetric fwd/bwd:
    # fwd: 512 tokens → 2 chunks, bwd: 384 tokens → 3 chunks
    self.max_pre_permute_chunk_size_fwd = 512
    self.max_pre_permute_chunk_size_bwd = 384


MlpNode.__init__ = _patched_mlpnode_init


class FakeDeepGemmMOELayer(nn.Layer):
    """
    A mock MoE layer for deep_gemm mode using GroupedMLPExpert (stacked weights).
    Matches real production deployment structure.
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
def vmm_no_free_space():
    """Occupy all free blocks and disable growable space to simulate tight memory."""
    (old_value,) = paddle.framework.get_flags(
        "FLAGS_max_reserved_threshold_in_gb"
    ).values()
    paddle.set_flags({"FLAGS_max_reserved_threshold_in_gb": 0})
    buffers = []
    for size, _ in MemoryAnalysisTool.vmm_free_block_info()[-1]:
        buffers.append(paddle.empty([size], dtype="uint8"))
    try:
        yield
    finally:
        paddle.set_flags({"FLAGS_max_reserved_threshold_in_gb": old_value})
        del buffers


class TestAutoSubbatchPrePermute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        self, tight_forward=False, tight_backward=False, **kwargs
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
            "use_auto_subbatch": False,
            "auto_subbatch_mode": None,
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

        with (
            vmm_no_free_space() if tight_backward else contextlib.nullcontext()
        ):
            paddle.autograd.backward(hidden_states, self.hidden_states_out_grad)

        hidden_states_grad = self.hidden_states.grad
        probs_grad = self.probs.grad
        self.hidden_states.clear_grad()
        self.probs.clear_grad()

        weight_grad = self.moe_layer.grouped_gemm_experts.weight2.main_grad
        self.moe_layer.clear_main_grad()

        return hidden_states, hidden_states_grad, probs_grad, weight_grad

    def compare_results(self, ref_out, tgt_out, loose_weight=False):
        names = [
            "hidden_states",
            "hidden_states_grad",
            "probs_grad",
            "weight_grad",
        ]
        tolerances = (
            {
                "weight_grad": {"atol": 1e-4, "rtol": 1e-5},
            }
            if loose_weight
            else {}
        )
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

    def test_pre_permute_vs_ref(self):
        """Test pre_permute mode against group_gemm reference (no subbatch)."""
        logging.info("=== Reference: group_gemm (no subbatch) ===")
        ref_out = self.run_moe_layer()

        cases = {}

        logging.info("case1: pre_permute, plenty memory")
        cases["case1 (plenty)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
        )

        logging.info("case2: pre_permute, tight forward")
        cases["case2 (tight_fwd)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            tight_forward=True,
        )

        logging.info("case3: pre_permute, tight backward")
        cases["case3 (tight_bwd)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            tight_backward=True,
        )

        logging.info("case4: pre_permute, tight both")
        cases["case4 (tight_both)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            tight_forward=True,
            tight_backward=True,
        )

        loose_cases = {
            "case1 (plenty)",
            "case2 (tight_fwd)",
            "case3 (tight_bwd)",
            "case4 (tight_both)",
        }
        for name, result in cases.items():
            with self.subTest(case=name):
                self.compare_results(
                    ref_out, result, loose_weight=(name in loose_cases)
                )

    def test_pre_permute_empty_experts(self):
        """Test pre_permute when some experts get 0 tokens in a chunk.

        Create routing where only expert 0 and 1 are used, others get 0.
        """
        logging.info("=== empty experts test ===")
        indices_np = np.full([self.seq_len, self.topk], -1, dtype=np.int64)
        tokens_per_expert = [0] * self.n_routed_experts
        for i in range(self.seq_len):
            expert_id = i % 2
            indices_np[i, 0] = expert_id
            tokens_per_expert[expert_id] += 1
        self.indices = paddle.to_tensor(indices_np)

        # Recreate layer with new tokens_per_expert
        moe_layer = FakeDeepGemmMOELayer(
            self.hidden_size,
            self.intermediate_size,
            self.n_routed_experts,
            tokens_per_expert,
        )
        moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
        moe_layer.clear_main_grad()
        self.moe_layer = moe_layer

        ref_out = self.run_moe_layer()

        pre_out = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
        )

        self.compare_results(ref_out, pre_out, loose_weight=True)

    def test_pre_permute_requires_expert_fusion(self):
        """pre_permute currently uses fused group_gemm per chunk."""
        with self.assertRaisesRegex(
            AssertionError,
            "auto_subbatch_mode='pre_permute' requires moe_expert_fusion=True",
        ):
            self.run_moe_layer(
                use_auto_subbatch=True,
                auto_subbatch_mode="pre_permute",
                moe_expert_fusion=False,
            )

    def test_pre_permute_cached_backward_empty_chunk(self):
        """Test cached backward when one chunk has no valid routed tokens."""
        logging.info("=== cached backward empty chunk test ===")
        indices_np = np.full([self.seq_len, self.topk], -1, dtype=np.int64)
        tokens_per_expert = [0] * self.n_routed_experts
        split = self.seq_len // 2
        for i in range(split):
            expert_id = i % 2
            indices_np[i, 0] = expert_id
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

        ref_out = self.run_moe_layer()
        pre_out = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
        )

        self.compare_results(ref_out, pre_out, loose_weight=True)
        np.testing.assert_equal(
            pre_out[1][split:].float().numpy(),
            np.zeros_like(pre_out[1][split:].float().numpy()),
        )
        np.testing.assert_equal(
            pre_out[2][split:].float().numpy(),
            np.zeros_like(pre_out[2][split:].float().numpy()),
        )

    def test_pre_permute_offline_quant(self):
        """Test pre_permute with offline FP8 weight quantization (no transpose)."""
        logging.info("=== pre_permute + offline quant ===")

        self.moe_layer.fp8_quant_weight(quant_transpose=False)

        # Reference: group_gemm without subbatch
        ref_out = self.run_moe_layer()

        cases = {}

        logging.info("case1: offline_quant, plenty")
        cases["case1 (offline_quant, plenty)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
        )

        logging.info("case2: offline_quant, tight_fwd")
        cases["case2 (offline_quant, tight_fwd)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            tight_forward=True,
        )

        logging.info("case3: offline_quant, tight_bwd")
        cases["case3 (offline_quant, tight_bwd)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            tight_backward=True,
        )

        logging.info("case4: offline_quant, tight_both")
        cases["case4 (offline_quant, tight_both)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            tight_forward=True,
            tight_backward=True,
        )

        loose_cases = {
            "case1 (offline_quant, plenty)",
            "case2 (offline_quant, tight_fwd)",
            "case3 (offline_quant, tight_bwd)",
            "case4 (offline_quant, tight_both)",
        }
        for name, result in cases.items():
            with self.subTest(case=name):
                self.compare_results(
                    ref_out, result, loose_weight=(name in loose_cases)
                )

    def test_pre_permute_offline_quant_transpose(self):
        """Test pre_permute with offline FP8 quant + quant_transpose=True."""
        logging.info("=== pre_permute + offline quant transpose ===")

        self.moe_layer.fp8_quant_weight(quant_transpose=True)

        # Reference: group_gemm without subbatch
        ref_out = self.run_moe_layer()

        cases = {}

        logging.info("case5: offline_quant_transpose, plenty")
        cases["case5 (offline_quant_t, plenty)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
        )

        logging.info("case6: offline_quant_transpose, tight_fwd")
        cases["case6 (offline_quant_t, tight_fwd)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            tight_forward=True,
        )

        logging.info("case7: offline_quant_transpose, tight_bwd")
        cases["case7 (offline_quant_t, tight_bwd)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            tight_backward=True,
        )

        logging.info("case8: offline_quant_transpose, tight_both")
        cases["case8 (offline_quant_t, tight_both)"] = self.run_moe_layer(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            tight_forward=True,
            tight_backward=True,
        )

        loose_cases = {
            "case5 (offline_quant_t, plenty)",
            "case6 (offline_quant_t, tight_fwd)",
            "case7 (offline_quant_t, tight_bwd)",
            "case8 (offline_quant_t, tight_both)",
        }
        for name, result in cases.items():
            with self.subTest(case=name):
                self.compare_results(
                    ref_out, result, loose_weight=(name in loose_cases)
                )

    def test_weight_grad_error_scaling(self):
        """Verify weight_grad error is bounded under multi-chunk splitting.

        Weight grad error comes from FP accumulation order difference (random walk):
          - Different chunk counts → different reduction trees → different rounding
          - Error magnitude is NOT guaranteed monotonic (can cancel out)
          - But it IS bounded by O(eps_bf16 * scale * sqrt(N))

        Verification:
          1. 1 chunk must be bitwise identical to reference (same accumulation path)
          2. All chunk counts must produce error < max_allowed bound
          3. forward output / input_grad / probs_grad must always be bitwise identical
             (per-token computation, independent of chunking)
        """
        logging.info("=== weight_grad error scaling test ===")

        self.moe_layer.fp8_quant_weight(quant_transpose=True)

        # Reference: no subbatch
        ref_out = self.run_moe_layer()
        ref_wgrad = ref_out[3].float().numpy()

        # Test with different chunk sizes → different chunk counts
        # S=1024: chunk_size 1024→1chunk, 512→2, 384→3, 256→4
        chunk_sizes = [1024, 512, 384, 256]
        errors = []

        for cs in chunk_sizes:
            # Temporarily override chunk sizes
            from paddleformers.fleet.transformer.moe.fusion_layer_utils import MlpNode

            orig_init = MlpNode.__init__

            def make_patched(chunk_size_val):
                def _patched(self_node, *args, **kwargs):
                    orig_init(self_node, *args, **kwargs)
                    self_node.min_auto_subbatch_rows = 256
                    self_node.max_pre_permute_chunk_size_fwd = chunk_size_val
                    self_node.max_pre_permute_chunk_size_bwd = chunk_size_val

                return _patched

            MlpNode.__init__ = make_patched(cs)
            try:
                result = self.run_moe_layer(
                    use_auto_subbatch=True,
                    auto_subbatch_mode="pre_permute",
                )
            finally:
                MlpNode.__init__ = orig_init

            tgt_wgrad = result[3].float().numpy()
            max_abs_diff = np.abs(ref_wgrad - tgt_wgrad).max()
            num_chunks = (1024 + cs - 1) // cs
            errors.append(max_abs_diff)
            logging.info(
                "  chunk_size=%d, num_chunks=%d, max_abs_diff=%.6e",
                cs,
                num_chunks,
                max_abs_diff,
            )

            # Verify forward output and input_grad are bitwise identical
            # (per-token computation, independent of chunking)
            for i, name in enumerate(
                ["hidden_states", "hidden_states_grad", "probs_grad"]
            ):
                np.testing.assert_equal(
                    ref_out[i].float().numpy(),
                    result[i].float().numpy(),
                    err_msg=f"{name} must be bitwise identical at chunk_size={cs}",
                )

        # Assertions:
        # 1. chunk_size=S (1 chunk) must be bitwise identical to reference
        self.assertEqual(
            errors[0],
            0.0,
            "1-chunk (full batch) must produce identical weight_grad to reference",
        )

        # 2. All errors must be bounded (FP accumulation order diff is random walk,
        #    bounded by eps_bf16 * scale * sqrt(N) ≈ 1e-4 for our test dimensions)
        max_allowed = 1e-3
        for i, (cs, err) in enumerate(zip(chunk_sizes, errors)):
            self.assertLess(
                err,
                max_allowed,
                f"Error too large at chunk_size={cs}: {err:.6e} >= {max_allowed}",
            )

        logging.info(
            "  PASS: errors=[%s], all bounded < %.0e",
            ", ".join(f"{e:.2e}" for e in errors),
            max_allowed,
        )

    def test_weight_grad_single_token_per_expert(self):
        """Absolute correctness proof: 1 token per expert per chunk → no reduction.

        When each expert in each chunk receives exactly 1 token, weight grad
        is a pure outer product (no accumulation over multiple tokens).
        This eliminates FP reduction order as a variable:
          dW[e] = x_e^T @ dy_e   (single rank-1 outer product, exact)

        Therefore chunked and non-chunked results MUST be bitwise identical,
        including weight grad. Any difference proves a logic bug.

        Construction:
          - seq_len = n_routed_experts (8 tokens, each goes to exactly 1 expert)
          - topk = 1
          - chunk_size = 1 (each chunk has exactly 1 token → 1 expert gets 1 token)
        """
        logging.info(
            "=== single token per expert per chunk (bitwise proof) ==="
        )

        # Override class-level params for this test
        n_experts = self.n_routed_experts  # 8
        seq_len = n_experts  # 8 tokens, one per expert
        topk = 1

        # Create input: 8 tokens, each assigned to a unique expert
        paddle.seed(42)
        hidden_states = paddle.randn([seq_len, self.hidden_size], "bfloat16")
        hidden_states_out_grad = paddle.randn(
            [seq_len, self.hidden_size], "bfloat16"
        )
        hidden_states, scale = tilewise_quant(hidden_states)
        probs = paddle.ones([seq_len, topk])  # uniform probs
        hidden_states.stop_gradient = False
        probs.stop_gradient = False

        # Routing: token i → expert i (round-robin, each expert gets exactly 1)
        indices_np = np.arange(n_experts, dtype=np.int64).reshape(seq_len, topk)
        tokens_per_expert = [1] * n_experts

        indices = paddle.to_tensor(indices_np)

        # Create layer
        moe_layer = FakeDeepGemmMOELayer(
            self.hidden_size,
            self.intermediate_size,
            n_experts,
            tokens_per_expert,
        )
        moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
        moe_layer.clear_main_grad()
        moe_layer.fp8_quant_weight(quant_transpose=True)

        # Helper to run
        def run_once(
            use_auto_subbatch=False, auto_subbatch_mode=None, chunk_size=None
        ):
            from paddleformers.fleet.transformer.moe.fusion_layer_utils import MlpNode

            orig_init = MlpNode.__init__

            def make_patched(cs):
                def _patched(self_node, *args, **kwargs):
                    orig_init(self_node, *args, **kwargs)
                    self_node.min_auto_subbatch_rows = 1
                    if cs is not None:
                        self_node.max_pre_permute_chunk_size_fwd = cs
                        self_node.max_pre_permute_chunk_size_bwd = cs

                return _patched

            MlpNode.__init__ = make_patched(chunk_size)
            try:
                params = {
                    "use_fp8_mlp": True,
                    "moe_deep_gemm": True,
                    "recompute_moe_gate_up": False,
                    "dequant_input": True,
                    "moe_expert_fusion": True,
                    "recompute_moe_premute": False,
                    "use_bf16_gemm_weight_grad": True,
                    "fp8_dispatched_handle": {"scale": scale},
                    "use_auto_subbatch": use_auto_subbatch,
                    "auto_subbatch_mode": auto_subbatch_mode,
                    "moe_subbatch_diag": True,
                }
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
                wgrad = moe_layer.grouped_gemm_experts.weight2.main_grad.clone()
                moe_layer.clear_main_grad()
                return hs_out, hs_grad, probs_grad, wgrad
            finally:
                MlpNode.__init__ = orig_init

        # Reference: no subbatch (1 chunk covering all 8 tokens)
        ref_out = run_once(auto_subbatch_mode=None)

        print("ref_out: end==========================")

        # Chunked: chunk_size=1 → 8 chunks, each with 1 token → 1 expert
        print("chunked_out: start==========================")
        chunked_out = run_once(
            use_auto_subbatch=True,
            auto_subbatch_mode="pre_permute",
            chunk_size=1,
        )
        print("chunked_out: end==========================")

        # ALL outputs must be bitwise identical (including weight grad!)
        names = [
            "hidden_states",
            "hidden_states_grad",
            "probs_grad",
            "weight_grad",
        ]
        for i, name in enumerate(names):
            ref_np = ref_out[i].float().numpy()
            tgt_np = chunked_out[i].float().numpy()
            np.testing.assert_equal(
                ref_np,
                tgt_np,
                err_msg=(
                    f"{name} must be bitwise identical when each expert has "
                    f"exactly 1 token per chunk (no accumulation)"
                ),
            )
            logging.info("  %s: bitwise identical ✓", name)

        logging.info(
            "  PASS: single-token-per-expert proves chunking correctness"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
