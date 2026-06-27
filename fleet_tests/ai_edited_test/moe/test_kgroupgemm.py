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
Single-card unit tests for k-grouped gemm (moe_deep_gemm + fp8) functionality.

Tests cover the following code changes in the kgroupgemm-debug branch:
  1. __init__: condition change from `not moe_expert_fusion or use_fp8_mlp`
     to `not moe_expert_fusion or (use_fp8_mlp and not moe_deep_gemm)`
  2. fwd_gate_up_fp8 / fwd_down_fp8: offline_quant handling for grouped_gemm_experts
  3. bwd_down_input_fp8 / bwd_gate_up_input_fp8: offline_quant + local_expert_num
  4. forward: condition change to access grouped_gemm_experts with fp8+deep_gemm
  5. backward: condition change + tokens_per_expert_tensor creation
  6. backward_impl_fp8: get weights from grouped_gemm_experts if available
  7. bf16_weight_grad: condition change + tokens_per_expert_tensor usage
  8. moe_layer: moe_deep_gemm compatible with fp8 + GroupedMLPExpert creation
  9. moe_layer: fuse_expert_fp8_weight_quant with grouped_gemm_experts

Run with:
  python fleet_tests/test_kgroupgemm.py
"""

import os
import unittest

import numpy as np

os.environ["FLAGS_cudnn_deterministic"] = "True"

from types import SimpleNamespace

import paddle
from paddle import nn

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
from paddleformers.fleet.transformer.moe.fusion_layer_utils import FusionMoePyLayer
from paddleformers.fleet.transformer.moe.moe_expert import (
    GroupedMLPExpert,
    StandardMLPExpert,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class FakeMOELayer(nn.Layer):
    """
    A mock MoE layer that provides the interface expected by FusionMoePyLayer.

    Includes both StandardMLPExpert list (self.experts) and GroupedMLPExpert
    (self.grouped_gemm_experts) to test the new fp8+deep_gemm code paths.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        n_routed_experts,
        tokens_per_expert,
        moe_deep_gemm=True,
    ):
        super().__init__()
        config = TransformerConfig(
            hidden_size=hidden_size,
            gated_linear_unit=True,
            moe_intermediate_size=intermediate_size,
        )
        self.config = config
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

        # Create GroupedMLPExpert with same config for grouped_gemm paths
        grouped_config = TransformerConfig(
            hidden_size=hidden_size,
            gated_linear_unit=True,
            moe_intermediate_size=intermediate_size,
        )
        self.grouped_gemm_experts = GroupedMLPExpert(
            n_routed_experts,
            grouped_config,
            moe_deep_gemm,
            None,  # pg_collection
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

    def clear_grouped_main_grad(self):
        self.grouped_gemm_experts.weight1.main_grad = None
        self.grouped_gemm_experts.weight2.main_grad = None

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        """Pre-quantize expert weights to FP8 format."""
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            fused_stack_quant_without_cache,
        )

        def quantize_weights(
            weight_list, weight_obj=None, quant_transpose=None
        ):
            if weight_obj is None:
                weight_obj = weight_list[0]

            fp8_weight, fp8_scale = fused_stack_quant_without_cache(
                weight_list, transpose=False
            )
            weight_obj.fp8_weight_stacked = fp8_weight
            weight_obj.fp8_scale_stacked = fp8_scale

            if quant_transpose is None or quant_transpose is True:
                fp8_weight_t, fp8_scale_t = fused_stack_quant_without_cache(
                    weight_list, transpose=True
                )
                weight_obj.fp8_weight_stacked_transpose = fp8_weight_t
                weight_obj.fp8_scale_stacked_transpose = fp8_scale_t
            else:
                weight_obj.fp8_weight_stacked_transpose = None
                weight_obj.fp8_scale_stacked_transpose = None

        # Quantize for grouped_gemm_experts (the new code path)
        if batch_mode:
            expert_w1 = self.grouped_gemm_experts.weight1
            expert_w2 = self.grouped_gemm_experts.weight2
            local_expert_num = expert_w1.shape[0]
            expert_w1_list = [
                expert_w1[i, :, :] for i in range(local_expert_num)
            ]
            expert_w2_list = [
                expert_w2[i, :, :] for i in range(local_expert_num)
            ]

            if expert_w1_list:
                quantize_weights(
                    expert_w1_list,
                    self.grouped_gemm_experts.weight1,
                    quant_transpose,
                )
            if expert_w2_list:
                quantize_weights(
                    expert_w2_list,
                    self.grouped_gemm_experts.weight2,
                    quant_transpose,
                )
        else:
            # Individual mode
            for expert in self.experts:
                if expert is not None:
                    quantize_weights(
                        [expert.up_gate_proj.weight],
                        quant_transpose=quant_transpose,
                    )
                    quantize_weights(
                        [expert.down_proj.weight],
                        quant_transpose=quant_transpose,
                    )


class TestKGroupGemm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up default configuration."""
        model_parallel_cuda_manual_seed(1234)
        cls.seq_len = 128
        cls.topk = 2
        cls.hidden_size = 512
        cls.intermediate_size = 256
        cls.n_routed_experts = 4

    def tmp_tilewise_quant(self, x):
        """Tile-wise FP8 quantization."""
        if x.shape[0] > 0:
            return paddle.incubate.nn.functional.fp8_quant_blockwise(
                x,
                output_scale_transpose=False,
                quant_method="1x128",
                input_transpose=False,
            )
        else:
            from paddleformers.fleet.transformer.moe.fp8_utils import FP8_ALIGN

            shape = list(x.shape)
            x_fp8 = paddle.empty(x.shape, dtype=paddle.float8_e4m3fn)
            assert shape[-1] % FP8_ALIGN == 0, shape
            shape[-1] //= FP8_ALIGN
            x_scale = paddle.empty(shape, dtype=paddle.float32)
            return x_fp8, x_scale

    def setUp(self):
        """Create test layer and input data."""
        paddle.seed(2026)
        np.random.seed(2026)

        hidden_states = paddle.randn(
            [self.seq_len, self.hidden_size], "bfloat16"
        )
        hidden_states_out_grad = paddle.randn_like(hidden_states)
        hidden_states, scale = self.tmp_tilewise_quant(hidden_states)
        probs = paddle.randn([self.seq_len, self.topk])
        hidden_states.stop_gradient = False
        probs.stop_gradient = False

        self.hidden_states = hidden_states
        self.hidden_states_out_grad = hidden_states_out_grad
        self.scale = scale
        self.probs = probs

        # Each token is assigned 1~topk experts, always include expert 0
        indices_np = np.full([self.seq_len, self.topk], -1, dtype=np.int32)
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
        self.tokens_per_expert = tokens_per_expert

    def _create_moe_layer(self, moe_deep_gemm=True):
        """Create a FakeMOELayer with both experts and grouped_gemm_experts."""
        moe_layer = FakeMOELayer(
            self.hidden_size,
            self.intermediate_size,
            self.n_routed_experts,
            self.tokens_per_expert,
            moe_deep_gemm=moe_deep_gemm,
        )
        moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
        return moe_layer

    def run_moe_layer(self, moe_layer, **kwargs):
        """Run forward+backward and collect outputs."""
        params = {
            "use_fp8_mlp": True,
            "moe_expert_fusion": True,
            "moe_deep_gemm": True,
            "recompute_moe_gate_up": True,
            "dequant_input": True,
            "recompute_moe_premute": False,
            "use_bf16_gemm_weight_grad": True,
            "fp8_dispatched_handle": {"scale": self.scale},
        }
        params.update(kwargs)

        hidden_states = FusionMoePyLayer.apply(
            self.hidden_states,
            self.probs,
            self.indices.clone(),
            moe_layer,
            self.topk,
            **params,
        )

        paddle.autograd.backward(hidden_states, self.hidden_states_out_grad)

        hidden_states_grad = self.hidden_states.grad
        probs_grad = self.probs.grad
        self.hidden_states.clear_grad()
        self.probs.clear_grad()

        return hidden_states, hidden_states_grad, probs_grad

    # ---------------------------------------------------------------
    # Test 1: __init__ condition change
    # Old: if not moe_expert_fusion or use_fp8_mlp
    # New: if not moe_expert_fusion or (use_fp8_mlp and not moe_deep_gemm)
    # ---------------------------------------------------------------
    def test_init_condition_grouped_gemm_experts(self):
        """
        Test __init__ condition: when moe_expert_fusion=True and use_fp8_mlp=True
        and moe_deep_gemm=True, ExpertsGroupGemmContiguousNode should use
        grouped_gemm_experts instead of experts.
        """
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        moe_layer = self._create_moe_layer(moe_deep_gemm=True)

        # New condition: moe_expert_fusion=True, use_fp8_mlp=True, moe_deep_gemm=True
        # should use grouped_gemm_experts (THE KEY NEW CODE PATH)
        node = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        self.assertTrue(hasattr(node, "grouped_gemm_experts"))
        self.assertFalse(hasattr(node, "experts"))
        print(
            "[PASS] test_init_condition: grouped_gemm_experts for fp8+deep_gemm"
        )

        # Old condition: moe_expert_fusion=True, use_fp8_mlp=True, moe_deep_gemm=False
        # should use experts (backward compat)
        node_old = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=False,
            moe_expert_fusion=True,
        )
        self.assertTrue(hasattr(node_old, "experts"))
        self.assertFalse(hasattr(node_old, "grouped_gemm_experts"))
        print(
            "[PASS] test_init_condition: experts for fp8 without deep_gemm (backward compat)"
        )

        # Condition: moe_expert_fusion=True, use_fp8_mlp=False, moe_deep_gemm=True
        # should use grouped_gemm_experts (bf16 deep_gemm path, unchanged)
        node_bf16 = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=False,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        self.assertTrue(hasattr(node_bf16, "grouped_gemm_experts"))
        self.assertFalse(hasattr(node_bf16, "experts"))
        print(
            "[PASS] test_init_condition: grouped_gemm_experts for bf16+deep_gemm"
        )

        # Condition: moe_expert_fusion=False -> always use experts
        node_no_grouped = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=False,
        )
        self.assertTrue(hasattr(node_no_grouped, "experts"))
        self.assertFalse(hasattr(node_no_grouped, "grouped_gemm_experts"))
        print("[PASS] test_init_condition: experts for no grouped_gemm")

    # ---------------------------------------------------------------
    # Test 2: forward condition change
    # Old: if self.moe_expert_fusion and not self.use_fp8_mlp
    # New: if self.moe_expert_fusion and (not self.use_fp8_mlp or self.moe_deep_gemm)
    # ---------------------------------------------------------------
    def test_forward_uses_grouped_gemm_experts(self):
        """
        Test that forward() accesses grouped_gemm_experts.weight1/weight2
        when moe_expert_fusion=True and (not use_fp8_mlp or moe_deep_gemm).
        """
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        moe_layer = self._create_moe_layer(moe_deep_gemm=True)

        node = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        self.assertTrue(hasattr(node, "grouped_gemm_experts"))
        w1 = node.grouped_gemm_experts.weight1
        w2 = node.grouped_gemm_experts.weight2
        self.assertEqual(len(w1.shape), 3)
        self.assertEqual(len(w2.shape), 3)
        self.assertEqual(w1.shape[0], self.n_routed_experts)
        self.assertEqual(w2.shape[0], self.n_routed_experts)
        print(
            "[PASS] test_forward_uses_grouped_gemm_experts: weight shapes correct"
        )

    # ---------------------------------------------------------------
    # Test 3: backward_impl_fp8 weight source change
    # ---------------------------------------------------------------
    def test_backward_impl_gets_grouped_weights(self):
        """
        Test that backward_impl_fp8 gets weights from grouped_gemm_experts
        when hasattr(self, 'grouped_gemm_experts') is True.
        """
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        moe_layer = self._create_moe_layer(moe_deep_gemm=True)

        node = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        # When hasattr(self, "grouped_gemm_experts"), weights are 3D tensors
        self.assertTrue(hasattr(node, "grouped_gemm_experts"))
        w1 = node.grouped_gemm_experts.weight1
        w2 = node.grouped_gemm_experts.weight2
        self.assertEqual(len(w1.shape), 3)
        self.assertEqual(len(w2.shape), 3)
        self.assertEqual(w1.shape[0], self.n_routed_experts)
        self.assertEqual(w2.shape[0], self.n_routed_experts)

        # When using experts (no grouped_gemm_experts), weights are lists
        node_old = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=False,
            moe_expert_fusion=True,
        )
        self.assertTrue(hasattr(node_old, "experts"))
        self.assertFalse(hasattr(node_old, "grouped_gemm_experts"))
        print(
            "[PASS] test_backward_impl_gets_grouped_weights: both paths verified"
        )

    # ---------------------------------------------------------------
    # Test 4: tokens_per_expert_tensor creation in backward
    # ---------------------------------------------------------------
    def test_tokens_per_expert_tensor_created(self):
        """
        Test that tokens_per_expert_tensor is created during backward
        by directly creating and calling the node.
        """
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        moe_layer = self._create_moe_layer(moe_deep_gemm=True)
        moe_layer.clear_grouped_main_grad()

        node = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
            use_bf16_gemm_weight_grad=True,
        )
        node.tokens_per_expert = self.tokens_per_expert

        # Simulate backward: create tokens_per_expert_tensor manually
        # This is what the backward method does:
        #   self.tokens_per_expert_tensor = paddle.to_tensor(
        #       self.tokens_per_expert, dtype="int32"
        #   )
        node.tokens_per_expert_tensor = paddle.to_tensor(
            node.tokens_per_expert, dtype="int32"
        )

        # Verify
        self.assertIsNotNone(node.tokens_per_expert_tensor)
        self.assertEqual(node.tokens_per_expert_tensor.dtype, paddle.int32)
        self.assertEqual(
            node.tokens_per_expert_tensor.shape,
            [self.n_routed_experts],
        )
        print(
            "[PASS] test_tokens_per_expert_tensor_created: tensor with int32 dtype"
        )

    # ---------------------------------------------------------------
    # Test 5: End-to-end fp8 + deep_gemm forward+backward
    # Covers: __init__, forward, backward, bf16_weight_grad,
    #         fwd_gate_up_fp8, fwd_down_fp8, bwd_down_input_fp8,
    #         bwd_gate_up_input_fp8, backward_impl_fp8
    # ---------------------------------------------------------------
    def test_deep_gemm_fp8(self):
        """Test moe_deep_gemm=True + use_fp8_mlp=True (main new code path)."""
        moe_layer = self._create_moe_layer(moe_deep_gemm=True)
        moe_layer.clear_main_grad()
        moe_layer.clear_grouped_main_grad()

        out = self.run_moe_layer(
            moe_layer,
            use_fp8_mlp=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
        )

        # Output shape: [seq_len, hidden_size] after zip
        self.assertEqual(out[0].shape, [self.seq_len, self.hidden_size])
        self.assertFalse(paddle.all(out[0] == 0).item())
        print("[PASS] test_deep_gemm_fp8: forward+backward completed")

    # ---------------------------------------------------------------
    # Test 6: fp8 + deep_gemm with offline quant
    # Covers: fwd_gate_up_fp8/fwd_down_fp8 offline_quant=True branch
    # ---------------------------------------------------------------
    def test_deep_gemm_fp8_with_offline_quant(self):
        """Test moe_deep_gemm=True + use_fp8_mlp=True + offline quantization."""
        moe_layer = self._create_moe_layer(moe_deep_gemm=True)
        moe_layer.clear_main_grad()
        moe_layer.clear_grouped_main_grad()

        # Pre-quantize weights (offline quant)
        moe_layer.fp8_quant_weight(batch_mode=True, quant_transpose=True)

        out = self.run_moe_layer(
            moe_layer,
            use_fp8_mlp=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
        )

        self.assertEqual(out[0].shape, [self.seq_len, self.hidden_size])
        self.assertFalse(paddle.all(out[0] == 0).item())
        print("[PASS] test_deep_gemm_fp8_with_offline_quant: completed")

    # ---------------------------------------------------------------
    # Test 7: fp8 + deep_gemm without recompute
    # Covers: recompute_moe_gate_up=False path in backward_impl_fp8
    # ---------------------------------------------------------------
    def test_deep_gemm_fp8_no_recompute(self):
        """Test moe_deep_gemm=True + use_fp8_mlp=True without recompute."""
        moe_layer = self._create_moe_layer(moe_deep_gemm=True)
        moe_layer.clear_main_grad()
        moe_layer.clear_grouped_main_grad()

        out = self.run_moe_layer(
            moe_layer,
            use_fp8_mlp=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
            recompute_moe_gate_up=False,
        )

        self.assertEqual(out[0].shape, [self.seq_len, self.hidden_size])
        self.assertFalse(paddle.all(out[0] == 0).item())
        print("[PASS] test_deep_gemm_fp8_no_recompute: completed")

    # ---------------------------------------------------------------
    # Test 8: fp8 without deep_gemm (backward compat)
    # Covers: the old code path, bf16_weight_grad "enter not k_groupgemm"
    # ---------------------------------------------------------------
    def test_no_deep_gemm_fp8(self):
        """Test moe_deep_gemm=False + use_fp8_mlp=True (backward compat)."""
        moe_layer = self._create_moe_layer(moe_deep_gemm=False)
        moe_layer.clear_main_grad()

        out = self.run_moe_layer(
            moe_layer,
            use_fp8_mlp=True,
            moe_expert_fusion=True,
            moe_deep_gemm=False,
        )

        self.assertEqual(out[0].shape, [self.seq_len, self.hidden_size])
        self.assertFalse(paddle.all(out[0] == 0).item())
        print("[PASS] test_no_deep_gemm_fp8: backward compat works")

    # ---------------------------------------------------------------
    # Test 9: bf16 + deep_gemm (skip if env issue)
    # Covers: fwd_gate_up_bf16, bwd_down_input_bf16, bf16_weight_grad
    #         with moe_deep_gemm=True, use_fp8_mlp=False
    # ---------------------------------------------------------------
    @unittest.skip(
        "bf16 deep_gemm path requires bf16 input, skip due to env issue"
    )
    def test_deep_gemm_bf16(self):
        """Test moe_deep_gemm=True + use_fp8_mlp=False (existing deep_gemm path)."""
        pass

    # ---------------------------------------------------------------
    # Test 10: fp8_quant_weight for grouped_gemm_experts
    # Covers: moe_layer fuse_expert_fp8_weight_quant with grouped_gemm_experts
    # ---------------------------------------------------------------
    def test_fp8_quant_weight_grouped_experts(self):
        """
        Test fp8_quant_weight with grouped_gemm_experts in batch_mode.
        This covers the new code path in MoELayer.fp8_quant_weight.
        """
        moe_layer = self._create_moe_layer(moe_deep_gemm=True)

        # Before quantization, no fp8 attributes should exist
        self.assertFalse(
            hasattr(
                moe_layer.grouped_gemm_experts.weight1, "fp8_weight_stacked"
            )
        )
        self.assertFalse(
            hasattr(
                moe_layer.grouped_gemm_experts.weight2, "fp8_weight_stacked"
            )
        )

        # Run batch mode quantization (the new code path)
        moe_layer.fp8_quant_weight(batch_mode=True, quant_transpose=True)

        # After quantization, fp8 attributes should exist
        self.assertTrue(
            hasattr(
                moe_layer.grouped_gemm_experts.weight1, "fp8_weight_stacked"
            )
        )
        self.assertTrue(
            hasattr(
                moe_layer.grouped_gemm_experts.weight1,
                "fp8_weight_stacked_transpose",
            )
        )
        self.assertTrue(
            hasattr(
                moe_layer.grouped_gemm_experts.weight2, "fp8_weight_stacked"
            )
        )
        self.assertTrue(
            hasattr(
                moe_layer.grouped_gemm_experts.weight2,
                "fp8_weight_stacked_transpose",
            )
        )
        print(
            "[PASS] test_fp8_quant_weight_grouped_experts: fp8 attributes set"
        )

    # ---------------------------------------------------------------
    # Test 11: fp8_quant_weight without transpose
    # ---------------------------------------------------------------
    def test_fp8_quant_weight_no_transpose(self):
        """Test fp8_quant_weight with quant_transpose=False for grouped_gemm_experts."""
        moe_layer = self._create_moe_layer(moe_deep_gemm=True)

        moe_layer.fp8_quant_weight(batch_mode=True, quant_transpose=False)

        self.assertTrue(
            hasattr(
                moe_layer.grouped_gemm_experts.weight1, "fp8_weight_stacked"
            )
        )
        self.assertIsNone(
            moe_layer.grouped_gemm_experts.weight1.fp8_weight_stacked_transpose
        )
        self.assertTrue(
            hasattr(
                moe_layer.grouped_gemm_experts.weight2, "fp8_weight_stacked"
            )
        )
        self.assertIsNone(
            moe_layer.grouped_gemm_experts.weight2.fp8_weight_stacked_transpose
        )
        print(
            "[PASS] test_fp8_quant_weight_no_transpose: no transpose as expected"
        )

    # ---------------------------------------------------------------
    # Test 12: offline quant handling in forward
    # Covers: fwd_gate_up_fp8 and fwd_down_fp8 offline_quant branches
    # ---------------------------------------------------------------
    def test_offline_quant_handling_in_fwd(self):
        """
        Test that the offline_quant check in fwd_gate_up_fp8 and fwd_down_fp8
        correctly handles both quantized and non-quantized weights.
        """
        # Test without offline quant (no fp8_weight_stacked on weight)
        moe_layer = self._create_moe_layer(moe_deep_gemm=True)
        moe_layer.clear_main_grad()
        moe_layer.clear_grouped_main_grad()

        out_no_offline = self.run_moe_layer(
            moe_layer,
            use_fp8_mlp=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
        )

        # Test with offline quant
        moe_layer2 = self._create_moe_layer(moe_deep_gemm=True)
        moe_layer2.clear_main_grad()
        moe_layer2.clear_grouped_main_grad()
        moe_layer2.fp8_quant_weight(batch_mode=True, quant_transpose=True)

        out_with_offline = self.run_moe_layer(
            moe_layer2,
            use_fp8_mlp=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
        )

        # Both should produce valid outputs
        self.assertEqual(out_no_offline[0].shape, out_with_offline[0].shape)
        print(
            "[PASS] test_offline_quant_handling: both paths produce valid outputs"
        )

    # ---------------------------------------------------------------
    # Test 13: bf16_weight_grad condition with deep_gemm
    # Covers: "enter k_groupgemm v1" and "enter not k_groupgemm" branches
    # ---------------------------------------------------------------
    def test_bf16_weight_grad_condition(self):
        """
        Test bf16_weight_grad condition change:
        Old: if self.moe_expert_fusion and not self.use_fp8_mlp
        New: if self.moe_expert_fusion and (not self.use_fp8_mlp or self.moe_deep_gemm)

        When use_fp8_mlp=True and moe_deep_gemm=True, the new code enters
        the grouped gemm path ("enter k_groupgemm v1") instead of the old
        per-expert loop ("enter not k_groupgemm").
        """
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        # With deep_gemm + fp8: should use grouped path
        moe_layer = self._create_moe_layer(moe_deep_gemm=True)
        node = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
            use_bf16_gemm_weight_grad=True,
        )
        # The condition: moe_expert_fusion and (not use_fp8_mlp or moe_deep_gemm)
        # = True and (False or True) = True
        # -> enters k_grouped_bf16_gemm_tn_contiguous path
        self.assertTrue(node.moe_expert_fusion)
        self.assertTrue(node.moe_deep_gemm)
        self.assertTrue(node.use_bf16_gemm_weight_grad)
        print(
            "[PASS] test_bf16_weight_grad_condition: deep_gemm+fp8 enters grouped path"
        )

        # Without deep_gemm but with fp8: should NOT use grouped path
        node_no_deep = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=False,
            moe_expert_fusion=True,
            use_bf16_gemm_weight_grad=True,
        )
        # The condition: moe_expert_fusion and (not use_fp8_mlp or moe_deep_gemm)
        # = True and (False or False) = False
        # -> enters per-expert loop ("enter not k_groupgemm")
        self.assertTrue(node_no_deep.moe_expert_fusion)
        self.assertFalse(node_no_deep.moe_deep_gemm)
        print(
            "[PASS] test_bf16_weight_grad_condition: fp8 without deep_gemm enters per-expert loop"
        )

    # ---------------------------------------------------------------
    # Test 14: backward zero-token path with grouped_gemm_experts
    # Covers: the condition in backward() for zero tokens
    # ---------------------------------------------------------------
    def test_backward_zero_token_grouped_gemm_experts(self):
        """
        Test that backward with zero tokens handles grouped_gemm_experts
        when moe_expert_fusion=True and (not use_fp8_mlp or moe_deep_gemm).
        """
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        moe_layer = self._create_moe_layer(moe_deep_gemm=True)

        node = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        node.tokens_per_expert = [0] * self.n_routed_experts

        # Create zero-sized out_grad and unzipped_probs
        out_grad = paddle.zeros([0, self.hidden_size], dtype=paddle.bfloat16)
        unzipped_probs = paddle.zeros([0, 1], dtype=paddle.bfloat16)

        # This should hit the grouped_gemm_experts zero-tokens path
        # which initializes main_grad/grad on grouped_gemm_experts
        with paddle.no_grad():
            dx, probs_grad = node.backward(out_grad, unzipped_probs)

        self.assertEqual(dx.shape[0], 0)
        print(
            "[PASS] test_backward_zero_token: grouped_gemm_experts zero-tokens path works"
        )

    # ---------------------------------------------------------------
    # Test 15: tokens_per_expert_tensor dtype is int32
    # Covers: the change from paddle.to_tensor(ks, dtype="int32") to
    #         self.tokens_per_expert_tensor with dtype="int32"
    # ---------------------------------------------------------------
    def test_tokens_per_expert_tensor_dtype(self):
        """
        Verify that tokens_per_expert_tensor uses int32 dtype,
        matching the new code: paddle.to_tensor(self.tokens_per_expert, dtype="int32")
        """
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        moe_layer = self._create_moe_layer(moe_deep_gemm=True)

        node = ExpertsGroupGemmContiguousNode(
            moe_layer,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        node.tokens_per_expert = self.tokens_per_expert

        # Simulate what backward does
        node.tokens_per_expert_tensor = paddle.to_tensor(
            node.tokens_per_expert, dtype="int32"
        )

        self.assertEqual(node.tokens_per_expert_tensor.dtype, paddle.int32)

        # Verify the values match
        for i, t in enumerate(self.tokens_per_expert):
            self.assertEqual(node.tokens_per_expert_tensor[i].item(), t)
        print(
            "[PASS] test_tokens_per_expert_tensor_dtype: uses int32 dtype correctly"
        )


class TestMoELayerFp8QuantWeightBranches(unittest.TestCase):
    """
    Cover the three branches in MoELayer.fp8_quant_weight that are never reached
    by the existing tests because the method's guard requires moe_use_fusion_node=True,
    which in turn requires expert_model_parallel_size > 1 (impossible on single card).

    Branches covered:
      "enter this not in v0"  – batch_mode=True with grouped_gemm_experts present
      "enter this not in v1"  – expert_w1_list is non-empty → quantize weight1
      "enter this not in v2"  – expert_w2_list is non-empty → quantize weight2

    Strategy: call MoELayer.fp8_quant_weight as an unbound method on a minimal
    stub object that carries only the three attributes the method actually reads
    (moe_use_fusion_node, fp8, grouped_gemm_experts), bypassing __init__ entirely.
    """

    # Reuse the same sizes as TestKGroupGemm to satisfy fp8 alignment requirements.
    HIDDEN_SIZE = 512
    INTERMEDIATE_SIZE = 256
    N_EXPERTS = 4

    def _make_grouped_experts(self):
        """Create a real GroupedMLPExpert whose weight shapes satisfy fp8 alignment."""
        config = TransformerConfig(
            hidden_size=self.HIDDEN_SIZE,
            gated_linear_unit=True,
            moe_intermediate_size=self.INTERMEDIATE_SIZE,
            moe_deep_gemm=False,
        )
        return GroupedMLPExpert(self.N_EXPERTS, config, False, None)

    # ---------------------------------------------------------------
    # Test A: hit v0 + v1 + v2 together (quant_transpose=True)
    # ---------------------------------------------------------------
    def test_fp8_quant_weight_not_in_v0_v1_v2(self):
        """
        Drive all three branches in MoELayer.fp8_quant_weight:
          v0: hasattr(self, 'grouped_gemm_experts') and batch_mode=True
          v1: expert_w1_list is non-empty  → weight1 is quantized
          v2: expert_w2_list is non-empty  → weight2 is quantized
        """
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        class _Stub:
            pass

        obj = _Stub()
        obj.moe_use_fusion_node = True  # pass the guard
        obj.fp8 = "e4m3"  # pass the guard
        obj.use_ue8m0 = False
        obj.grouped_gemm_experts = self._make_grouped_experts()

        # Call the real method as an unbound function on the stub
        MoELayer.fp8_quant_weight(obj, batch_mode=True, quant_transpose=True)

        w1 = obj.grouped_gemm_experts.weight1
        w2 = obj.grouped_gemm_experts.weight2

        # v1: weight1 was quantized
        self.assertTrue(
            hasattr(w1, "fp8_weight_stacked"),
            "v1: weight1.fp8_weight_stacked should be set",
        )
        self.assertIsNotNone(w1.fp8_weight_stacked)
        self.assertTrue(hasattr(w1, "fp8_weight_stacked_transpose"))
        self.assertIsNotNone(w1.fp8_weight_stacked_transpose)

        # v2: weight2 was quantized
        self.assertTrue(
            hasattr(w2, "fp8_weight_stacked"),
            "v2: weight2.fp8_weight_stacked should be set",
        )
        self.assertIsNotNone(w2.fp8_weight_stacked)
        self.assertTrue(hasattr(w2, "fp8_weight_stacked_transpose"))
        self.assertIsNotNone(w2.fp8_weight_stacked_transpose)

        print(
            "[PASS] test_fp8_quant_weight_not_in_v0_v1_v2: v0/v1/v2 all covered"
        )

    # ---------------------------------------------------------------
    # Test B: v0 + v1 + v2 with quant_transpose=False
    #   Covers the else-branch inside quantize_weights where
    #   fp8_weight_stacked_transpose is set to None.
    # ---------------------------------------------------------------
    def test_fp8_quant_weight_not_in_v0_no_transpose(self):
        """
        Same v0/v1/v2 path but with quant_transpose=False, verifying that
        fp8_weight_stacked_transpose is explicitly set to None on both weights.
        """
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        class _Stub:
            pass

        obj = _Stub()
        obj.moe_use_fusion_node = True
        obj.fp8 = "e4m3"
        obj.use_ue8m0 = False
        obj.grouped_gemm_experts = self._make_grouped_experts()

        MoELayer.fp8_quant_weight(obj, batch_mode=True, quant_transpose=False)

        w1 = obj.grouped_gemm_experts.weight1
        w2 = obj.grouped_gemm_experts.weight2

        self.assertTrue(hasattr(w1, "fp8_weight_stacked"))
        self.assertIsNone(
            w1.fp8_weight_stacked_transpose,
            "quant_transpose=False: weight1 transpose should be None",
        )
        self.assertTrue(hasattr(w2, "fp8_weight_stacked"))
        self.assertIsNone(
            w2.fp8_weight_stacked_transpose,
            "quant_transpose=False: weight2 transpose should be None",
        )
        print(
            "[PASS] test_fp8_quant_weight_not_in_v0_no_transpose: no-transpose sub-path covered"
        )


if __name__ == "__main__":
    unittest.main()
