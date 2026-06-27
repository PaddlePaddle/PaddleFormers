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
Single-card unit tests for use_ue8m0 code coverage.

Tests cover the following use_ue8m0 code paths:
  1. fused_stack_quant_without_cache with use_ue8m0=True (transpose=False/True)
  2. FusionMoePyLayer with use_ue8m0=True + moe_expert_fusion=True + deepep gemm path
  3. MoELayer.fp8_quant_weight with use_ue8m0=True (stub-based, transpose True/False)

Run with:
  cd /path/to/PaddleFleet && python fleet_tests/ai_edited_test/fp8/test_ue8m0.py
"""

import os

os.environ["FLAGS_cudnn_deterministic"] = "True"

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
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
    (self.grouped_gemm_experts) to test the use_ue8m0 code paths.
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
                weight_list, transpose=False, use_ue8m0=False
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

        # Quantize for grouped_gemm_experts
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


class TestUe8m0CodePaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up default configuration."""
        model_parallel_cuda_manual_seed(1234)
        cls.seq_len = 128
        cls.topk = 2
        # Must be divisible by 1024 for use_ue8m0 TMA alignment
        cls.hidden_size = 2048
        cls.intermediate_size = 1024
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
        self.tokens_per_expert = tokens_per_expert

    def _make_node(self, use_ue8m0=True, moe_expert_fusion=True):
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        moe_layer = self._create_moe_layer(moe_deep_gemm=True)
        node = ExpertsGroupGemmContiguousNode(
            moe_layer,
            recompute_moe_gate_up=True,
            dequant_input=True,
            use_bf16_gemm_weight_grad=True,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=moe_expert_fusion,
            use_ue8m0=use_ue8m0,
        )
        node.tokens_per_expert = [self.seq_len]
        node.tokens_per_expert_indices = paddle.zeros(
            [self.seq_len], dtype="int32"
        )
        node.m_indices = node.gen_m_indices(node.tokens_per_expert)
        node.input_fp8 = self.hidden_states
        node.input_scale = self.scale
        return node, moe_layer

    def _make_fp8_tensor_and_scale(self, shape):
        x = paddle.randn(shape, dtype=paddle.bfloat16)
        x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x,
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=False,
            using_ue8m0_scale=True,
        )
        return x_fp8, x_scale.T

    def _fake_fp8_gemm_nt(self, lhs, rhs, out, *extra_args, **kwargs):
        out.zero_()
        return out

    def _fake_grouped_fp8_gemm(self, lhs, rhs, out, *extra_args, **kwargs):
        out.zero_()
        return out

    def _fake_fused_weighted_swiglu_fp8_quant(self, o1, probs, **kwargs):
        return self._make_fp8_tensor_and_scale([o1.shape[0], o1.shape[1] // 2])

    def _fake_transpose_split_quant(
        self, x, scale, tokens_per_expert, pow_2_scales
    ):
        rows = len(tokens_per_expert)
        cols = x.shape[-1]
        x_fp8, x_scale = self._make_fp8_tensor_and_scale([rows * 128, cols])
        return x_fp8.reshape([rows, 128, cols]), x_scale.reshape(
            [rows, 128, -1]
        )

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
            "moe_deep_gemm": True,
            "recompute_moe_gate_up": True,
            "dequant_input": True,
            "moe_expert_fusion": True,
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
    # Test 1: fused_stack_quant_without_cache with use_ue8m0=True
    # Covers: fp8_utils.py line 159 (scale transpose)
    # ---------------------------------------------------------------
    def test_fused_stack_quant_with_ue8m0(self):
        """Test fused_stack_quant_without_cache with use_ue8m0=True, both transpose modes."""
        from paddleformers.fleet.transformer.moe.fp8_utils import (
            fused_stack_quant_without_cache,
        )

        moe_layer = self._create_moe_layer(moe_deep_gemm=True)

        # Test with transpose=False, use_ue8m0=True
        w1_list = [
            moe_layer.grouped_gemm_experts.weight1[i, :, :]
            for i in range(self.n_routed_experts)
        ]
        w_fp8_not, w_scale_not = fused_stack_quant_without_cache(
            w1_list, transpose=False, use_ue8m0=True
        )
        self.assertIsNotNone(w_fp8_not)
        self.assertIsNotNone(w_scale_not)
        print(
            "[PASS] test_fused_stack_quant_with_ue8m0: transpose=False, use_ue8m0=True"
        )

        # Test with transpose=True, use_ue8m0=True
        w_fp8_t, w_scale_t = fused_stack_quant_without_cache(
            w1_list, transpose=True, use_ue8m0=True
        )
        self.assertIsNotNone(w_fp8_t)
        self.assertIsNotNone(w_scale_t)
        print(
            "[PASS] test_fused_stack_quant_with_ue8m0: transpose=True, use_ue8m0=True"
        )

    # ---------------------------------------------------------------
    # Test 2: FusionMoePyLayer with use_ue8m0=True + grouped_gemm + deep_gemm
    # Covers:
    #   fp8_utils - fwd_gate_up_fp8 deepep gemm use_ue8m0=True (line 571)
    #   fp8_utils - fwd_down_fp8 deepep gemm use_ue8m0=True (line 720)
    #   fp8_utils - bwd_down_input_fp8 use_ue8m0=True + moe_expert_fusion (line 841)
    #   fp8_utils - bwd_down_input_fp8 deepep gemm use_ue8m0=True (line 879)
    #   fp8_utils - bwd_gate_up_input_fp8 use_ue8m0=True + moe_expert_fusion (1006)
    #   fp8_utils - bwd_gate_up_input_fp8 deepep gemm use_ue8m0=True (1048)
    #   fusion_layer_utils - FusionMoePyLayer.forward with use_ue8m0=True
    # ---------------------------------------------------------------
    def test_moe_fusion_with_ue8m0_grouped_gemm(self):
        if (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.cuda.get_device_capability()[0] != 10
        ):
            raise unittest.SkipTest("use_ue8m0 requires Blackwell GPU (SM100)")
        """Test FusionMoePyLayer with use_ue8m0=True, grouped_gemm + deep_gemm."""
        moe_layer = self._create_moe_layer(moe_deep_gemm=True)
        moe_layer.clear_main_grad()
        moe_layer.clear_grouped_main_grad()

        out = self.run_moe_layer(
            moe_layer,
            use_fp8_mlp=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
            use_ue8m0=True,
        )

        self.assertEqual(out[0].shape, [self.seq_len, self.hidden_size])
        self.assertFalse(paddle.all(out[0] == 0).item())
        print(
            "[PASS] test_moe_fusion_with_ue8m0_grouped_gemm: forward+backward completed"
        )

    # ---------------------------------------------------------------
    # Test 3: MoELayer.fp8_quant_weight with use_ue8m0=True
    # Covers: moe_layer.py fp8_quant_weight use_ue8m0=True (lines 1097, 1105)
    # ---------------------------------------------------------------
    def test_moe_layer_fp8_quant_weight_with_ue8m0(self):
        """
        Test MoELayer.fp8_quant_weight with use_ue8m0=True via stub.
        Covers the moe_layer.py fp8_quant_weight quantize_weights helper's
        use_ue8m0=True path for both transpose=False and transpose=True.
        """
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        class _Stub:
            pass

        obj = _Stub()
        obj.moe_use_fusion_node = True
        obj.fp8 = "e4m3"
        obj.use_ue8m0 = True
        obj.grouped_gemm_experts = self._create_moe_layer(
            moe_deep_gemm=True
        ).grouped_gemm_experts

        # Call the real method as an unbound function on the stub
        MoELayer.fp8_quant_weight(obj, batch_mode=True, quant_transpose=True)

        w1 = obj.grouped_gemm_experts.weight1
        w2 = obj.grouped_gemm_experts.weight2

        self.assertTrue(hasattr(w1, "fp8_weight_stacked"))
        self.assertIsNotNone(w1.fp8_weight_stacked)
        self.assertTrue(hasattr(w1, "fp8_weight_stacked_transpose"))
        self.assertIsNotNone(w1.fp8_weight_stacked_transpose)

        self.assertTrue(hasattr(w2, "fp8_weight_stacked"))
        self.assertIsNotNone(w2.fp8_weight_stacked)
        self.assertTrue(hasattr(w2, "fp8_weight_stacked_transpose"))
        self.assertIsNotNone(w2.fp8_weight_stacked_transpose)

        print(
            "[PASS] test_moe_layer_fp8_quant_weight_with_ue8m0: fp8_quant_weight with use_ue8m0=True"
        )

    def test_triton_scale_transpose_zero_and_launch_branches(self):
        from paddleformers.fleet.triton_ops import fuse_stack_ue8m0_scale_transpose

        zero_scale = paddle.empty([0, 1], dtype=paddle.int32)
        zero_out = fuse_stack_ue8m0_scale_transpose(zero_scale, 0, 512, 512)
        self.assertEqual(list(zero_out.shape), [0, 1])

        scale = paddle.arange(512, dtype=paddle.int32).reshape([512, 1])
        out = fuse_stack_ue8m0_scale_transpose(scale, 1, 512, 512)
        self.assertEqual(list(out.shape), [512, 1])

    def test_split_group_gemm_ue8m0_branch(self):
        from paddleformers.fleet.transformer.moe.fp8_utils import split_group_gemm

        x_fp8, x_scale = self._make_fp8_tensor_and_scale(
            [128, self.hidden_size]
        )
        w_fp8, w_scale = self._make_fp8_tensor_and_scale(
            [self.hidden_size, self.hidden_size]
        )
        gemm_out = paddle.empty([128, self.hidden_size], dtype=paddle.bfloat16)
        with patch(
            "paddleformers.fleet.transformer.moe.fp8_utils.deep_gemm.fp8_gemm_nt",
            side_effect=self._fake_fp8_gemm_nt,
        ):
            split_group_gemm(
                x_fp8,
                x_scale,
                w_fp8.reshape([1, self.hidden_size, self.hidden_size]),
                w_scale.reshape([1, self.hidden_size, -1]),
                [128],
                gemm_out,
                use_ue8m0=True,
            )
        self.assertEqual(list(gemm_out.shape), [128, self.hidden_size])

    def test_fp8_grouped_ue8m0_manual_branches(self):
        if (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.cuda.get_device_capability()[0] != 10
        ):
            raise unittest.SkipTest("use_ue8m0 requires Blackwell GPU (SM100)")

        node, moe_layer = self._make_node(
            use_ue8m0=True, moe_expert_fusion=True
        )
        o1 = paddle.randn(
            [self.seq_len, self.intermediate_size * 2], dtype=paddle.bfloat16
        )
        probs = paddle.randn([self.seq_len, 1], dtype=paddle.float32)
        do3 = paddle.randn(
            [self.seq_len, self.hidden_size], dtype=paddle.bfloat16
        )
        do1 = paddle.randn(
            [self.seq_len, self.intermediate_size * 2], dtype=paddle.bfloat16
        )

        with (
            patch(
                "paddleformers.fleet.transformer.moe.fp8_utils.fuse_weighted_swiglu_fp8_quant",
                side_effect=self._fake_fused_weighted_swiglu_fp8_quant,
            ),
            patch.object(
                node,
                "fused_transpose_split_quant",
                side_effect=self._fake_transpose_split_quant,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fp8_utils.paddlefleet_deep_gemm.m_grouped_fp8_gemm_nt_contiguous",
                side_effect=self._fake_grouped_fp8_gemm,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fp8_utils.deep_gemm.fp8_gemm_nt",
                side_effect=self._fake_fp8_gemm_nt,
            ),
        ):
            o3 = node.fwd_down_fp8(
                o1,
                probs,
                moe_layer.grouped_gemm_experts.weight2,
                1,
            )
            do1_out, o2_s, probs_grad = node.bwd_down_input_fp8(
                moe_layer.grouped_gemm_experts.weight2,
                do3,
                o1,
                probs,
            )
            dx = node.bwd_gate_up_input_fp8(
                do1,
                moe_layer.grouped_gemm_experts.weight1,
            )
            node.bwd_down_weight(
                do3,
                paddle.randn(
                    [self.seq_len, self.intermediate_size],
                    dtype=paddle.bfloat16,
                ),
                [moe_layer.experts[0].down_proj.weight],
            )
            node.bwd_gate_up_weight(
                do1,
                self.hidden_states,
                [moe_layer.experts[0].up_gate_proj.weight],
            )

        self.assertEqual(
            list(o3.shape),
            [self.seq_len, self.hidden_size * self.n_routed_experts],
        )
        self.assertEqual(
            list(do1_out.shape), [self.seq_len, self.intermediate_size * 2]
        )
        self.assertEqual(
            list(o2_s.shape), [self.seq_len, self.intermediate_size]
        )
        print("probs_grad:", probs_grad)
        print("dx:", dx)
        self.assertEqual(list(probs_grad.shape), [self.seq_len])
        self.assertEqual(list(dx.shape), [self.seq_len, self.hidden_size])

    def test_fp8_weight_grad_without_main_grad_branches(self):
        node, _ = self._make_node(use_ue8m0=True, moe_expert_fusion=True)
        do3 = paddle.randn(
            [self.seq_len, self.hidden_size], dtype=paddle.bfloat16
        )
        o2 = paddle.randn(
            [self.seq_len, self.intermediate_size], dtype=paddle.bfloat16
        )
        do1 = paddle.randn(
            [self.seq_len, self.intermediate_size * 2], dtype=paddle.bfloat16
        )
        input_x = paddle.randn(
            [self.seq_len, self.hidden_size], dtype=paddle.bfloat16
        )
        w2 = paddle.create_parameter(
            shape=[self.hidden_size, self.intermediate_size],
            dtype=paddle.bfloat16,
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        w1 = paddle.create_parameter(
            shape=[self.hidden_size, self.intermediate_size * 2],
            dtype=paddle.bfloat16,
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

        with (
            patch.object(
                node,
                "fused_transpose_split_quant",
                side_effect=self._fake_transpose_split_quant,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.fp8_utils.deep_gemm.fp8_gemm_nt",
                side_effect=self._fake_fp8_gemm_nt,
            ),
        ):
            node.bwd_down_weight(do3, o2, [w2])
            node.bwd_gate_up_weight(do1, input_x, [w1])

        self.assertIsNotNone(w2.grad)
        self.assertIsNotNone(w1.grad)

    def test_moe_layer_init_ue8m0_assert_branch(self):
        if (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.cuda.get_device_capability()[0] != 10
        ):
            raise unittest.SkipTest("use_ue8m0 requires Blackwell GPU (SM100)")
        from types import SimpleNamespace

        import paddle.nn.functional as F

        from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer, MoESublayers

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=512,
            num_attention_heads=1,
            intermediate_size=512,
            n_routed_experts=1,
            n_shared_experts=0,
            num_experts_per_tok=1,
            moe_intermediate_size=512,
            moe_token_dispatcher_type="deepep",
            moe_expert_fusion=False,
            moe_deep_gemm=False,
            moe_use_fusion_node=True,
            use_ue8m0=True,
            fp8=None,
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=False,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=2,
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
        self.assertTrue(layer.use_ue8m0)

    # ---------------------------------------------------------------
    # Test 6: MoELayer.fp8_quant_weight with use_ue8m0=True + no transpose
    # Covers: moe_layer.py fp8_quant_weight use_ue8m0=True, quant_transpose=False
    # ---------------------------------------------------------------
    def test_moe_layer_fp8_quant_weight_ue8m0_no_transpose(self):
        """
        Test MoELayer.fp8_quant_weight with use_ue8m0=True, quant_transpose=False.
        Covers the else-branch where fp8_weight_stacked_transpose is None.
        """
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        class _Stub:
            pass

        obj = _Stub()
        obj.moe_use_fusion_node = True
        obj.fp8 = "e4m3"
        obj.use_ue8m0 = True
        obj.grouped_gemm_experts = self._create_moe_layer(
            moe_deep_gemm=True
        ).grouped_gemm_experts

        MoELayer.fp8_quant_weight(obj, batch_mode=True, quant_transpose=False)

        w1 = obj.grouped_gemm_experts.weight1
        w2 = obj.grouped_gemm_experts.weight2

        self.assertTrue(hasattr(w1, "fp8_weight_stacked"))
        self.assertIsNotNone(w1.fp8_weight_stacked)
        self.assertIsNone(w1.fp8_weight_stacked_transpose)
        self.assertTrue(hasattr(w2, "fp8_weight_stacked"))
        self.assertIsNotNone(w2.fp8_weight_stacked)
        self.assertIsNone(w2.fp8_weight_stacked_transpose)

        print(
            "[PASS] test_moe_layer_fp8_quant_weight_ue8m0_no_transpose: completed"
        )


if __name__ == "__main__":
    unittest.main()
