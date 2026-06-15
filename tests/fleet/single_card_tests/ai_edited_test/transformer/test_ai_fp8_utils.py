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
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import paddle

from paddleformers.fleet.transformer.moe.fp8_utils import (
    FP8_ALIGN,
    ExpertsGroupGemmContiguousNode,
    _get_fp8_weight_and_scale,
    fused_stack_quant,
    has_config,
    kitchen_gemm,
    tilewise_quant,
)


class CachedWeight:
    def __init__(self):
        self.shape = [2, 3]
        self.fp8_weight_stacked = paddle.arange(12, dtype="float32").reshape([4, 3])
        self.fp8_scale_stacked = paddle.arange(8, dtype="float32").reshape([4, 2])
        self.fp8_weight_stacked_transpose = None
        self.fp8_scale_stacked_transpose = None


class CustomMap:
    def __init__(self):
        self.experts = ["expert0", "expert1"]
        self.grouped_gemm_experts = "grouped"


class TestFP8WeightAndScaleCachedPaths(unittest.TestCase):
    def test_returns_cached_weight_and_scale_without_transpose(self):
        weight = CachedWeight()

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(weight)

        self.assertIs(fp8_weight, weight.fp8_weight_stacked)
        self.assertIs(fp8_scale, weight.fp8_scale_stacked)

    def test_returns_precomputed_transpose_when_present(self):
        weight = CachedWeight()
        weight.fp8_weight_stacked_transpose = paddle.ones([3, 4])
        weight.fp8_scale_stacked_transpose = paddle.ones([2, 4])

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(weight, transpose=True)

        self.assertIs(fp8_weight, weight.fp8_weight_stacked_transpose)
        self.assertIs(fp8_scale, weight.fp8_scale_stacked_transpose)

    def test_builds_transpose_from_cached_non_transposed_tensors(self):
        weight = CachedWeight()

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(weight, transpose=True)

        self.assertEqual(fp8_weight.shape, [6, 2])
        self.assertEqual(fp8_scale.shape, [4, 2])
        self.assertEqual(fp8_weight[0, 0].item(), 0.0)
        self.assertEqual(fp8_weight[0, 1].item(), 3.0)

    def test_fused_stack_quant_uses_cached_weight(self):
        weight = CachedWeight()

        fp8_weight, fp8_scale = fused_stack_quant([weight])

        self.assertIs(fp8_weight, weight.fp8_weight_stacked)
        self.assertIs(fp8_scale, weight.fp8_scale_stacked)


class TestFP8UtilityBranches(unittest.TestCase):
    def test_has_config_truth_table(self):
        self.assertFalse(has_config(None, "enabled"))
        self.assertFalse(has_config({}, "enabled"))
        self.assertFalse(has_config({"enabled": 0}, "enabled"))
        self.assertTrue(has_config({"enabled": 1}, "enabled"))

    def test_kitchen_gemm_zero_input_without_out(self):
        x_fp8 = paddle.empty([0, FP8_ALIGN], dtype="float32")
        x_scale = paddle.empty([0, 1], dtype="float32")
        w_fp8 = paddle.empty([3, FP8_ALIGN], dtype="float32")
        w_scale = paddle.empty([3, 1], dtype="float32")

        result = kitchen_gemm(
            x_fp8,
            x_scale,
            w_fp8,
            w_scale,
            is_a_1d_scaled=False,
            is_b_1d_scaled=False,
            rtn_dtype=paddle.float32,
        )

        self.assertEqual(result.shape, [0, 3])
        self.assertEqual(result.dtype, paddle.float32)

    def test_kitchen_gemm_zero_input_accumulates_into_out(self):
        x_fp8 = paddle.empty([0, FP8_ALIGN], dtype="float32")
        x_scale = paddle.empty([0, 1], dtype="float32")
        w_fp8 = paddle.empty([3, FP8_ALIGN], dtype="float32")
        w_scale = paddle.empty([3, 1], dtype="float32")
        out = paddle.empty([0, 3], dtype="float32")

        result = kitchen_gemm(
            x_fp8,
            x_scale,
            w_fp8,
            w_scale,
            is_a_1d_scaled=False,
            is_b_1d_scaled=False,
            out=out,
        )

        self.assertEqual(result.shape, [0, 3])
        self.assertEqual(result.dtype, paddle.float32)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA is required for FP8 tensors")
    def test_tilewise_quant_empty_input(self):
        x = paddle.empty([0, FP8_ALIGN], dtype="bfloat16")

        x_fp8, x_scale = tilewise_quant(x)

        self.assertEqual(x_fp8.shape, [0, FP8_ALIGN])
        self.assertEqual(x_scale.shape, [0, 1])
        self.assertEqual(x_scale.dtype, paddle.float32)


class TestExpertsGroupGemmContiguousNodeState(unittest.TestCase):
    def test_selects_single_expert_and_manages_cached_tensors(self):
        node = ExpertsGroupGemmContiguousNode(
            CustomMap(),
            expert_id=1,
            moe_subbatch_token_num_after_dispatch=FP8_ALIGN,
        )

        self.assertEqual(node.experts, ["expert1"])
        self.assertEqual(node.cached_tensors(), [None] * 6)

        tensors = [1, 2, 3, 4, 5, 6]
        node.set_cached_tensors(tensors)
        self.assertEqual(node.cached_tensors(), tensors)

        node.clear_cached_tensors()
        self.assertEqual(node.cached_tensors(), [None] * 6)

    def test_reset_state_preserves_input_and_scale_clear(self):
        node = ExpertsGroupGemmContiguousNode(CustomMap())
        node.tokens_per_expert = paddle.to_tensor([2, 0, 1], dtype="int32")
        node.m_indices = node.gen_m_indices(node.tokens_per_expert)
        node.input = paddle.ones([1, 2])
        node.input_fp8 = paddle.ones([1, 2])
        node.input_scale = paddle.ones([1, 1])
        node.o1 = paddle.ones([1, 2])

        self.assertEqual(node.m_indices.numpy().tolist(), [0, 0, 2])
        node.reset_state()

        self.assertIsNone(node.tokens_per_expert)
        self.assertIsNone(node.m_indices)
        self.assertEqual(node.cached_tensors(), [None] * 6)

    def test_grouped_expert_fusion_path(self):
        node = ExpertsGroupGemmContiguousNode(
            CustomMap(),
            use_fp8_mlp=False,
            moe_expert_fusion=True,
        )

        self.assertEqual(node.grouped_gemm_experts, "grouped")
        self.assertFalse(hasattr(node, "experts"))


if __name__ == "__main__":
    unittest.main()
