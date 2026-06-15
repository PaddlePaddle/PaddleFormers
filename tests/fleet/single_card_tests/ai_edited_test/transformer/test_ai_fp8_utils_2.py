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
    ExpertsGroupGemmContiguousNode,
    _get_fp8_weight_and_scale,
)


class CachedWeight:
    def __init__(self):
        self.shape = [2, 2]
        self.fp8_weight_stacked = paddle.arange(12, dtype="float32").reshape([6, 2])
        self.fp8_scale_stacked = paddle.arange(18, dtype="float32").reshape([6, 3])
        self.fp8_weight_stacked_transpose = None
        self.fp8_scale_stacked_transpose = paddle.full([9, 2], 5.0, dtype="float32")


class InvalidCachedWeight:
    shape = [4, 2]
    fp8_weight_stacked = paddle.ones([6, 2], dtype="float32")
    fp8_scale_stacked = paddle.ones([6, 2], dtype="float32")
    fp8_weight_stacked_transpose = None
    fp8_scale_stacked_transpose = None


class CustomMap:
    def __init__(self):
        self.experts = ["expert0", "expert1"]
        self.grouped_gemm_experts = "grouped"


class TestFP8WeightScaleBranches(unittest.TestCase):
    def test_transpose_uses_explicit_num_experts_and_ue8m0_scale(self):
        weight = CachedWeight()

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(weight, transpose=True, num_expert=3, use_ue8m0=True)

        self.assertEqual(fp8_weight.shape, [6, 2])
        self.assertIs(fp8_scale, weight.fp8_scale_stacked_transpose)
        self.assertEqual(
            fp8_weight.numpy().tolist(),
            [
                [0.0, 2.0],
                [1.0, 3.0],
                [4.0, 6.0],
                [5.0, 7.0],
                [8.0, 10.0],
                [9.0, 11.0],
            ],
        )

    def test_transpose_rejects_incompatible_stacked_shape(self):
        with self.assertRaises(AssertionError):
            _get_fp8_weight_and_scale(InvalidCachedWeight(), transpose=True)


class TestExpertsGroupGemmContiguousNodeMore(unittest.TestCase):
    def test_constructor_selects_single_expert_or_grouped_experts(self):
        custom_map = CustomMap()
        node = ExpertsGroupGemmContiguousNode(custom_map, expert_id=1)
        self.assertEqual(node.experts, ["expert1"])
        self.assertEqual(node.expert_id, 1)
        self.assertTrue(node.use_fp8_mlp)

        grouped = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_expert_fusion=True,
        )
        self.assertEqual(grouped.grouped_gemm_experts, "grouped")
        self.assertFalse(grouped.is_split_group_gemm)

    def test_constructor_validates_subbatch_size(self):
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(CustomMap(), moe_subbatch_token_num_after_dispatch=127)

    def test_gen_m_indices_accepts_lists_tensors_and_empty_counts(self):
        node = ExpertsGroupGemmContiguousNode(CustomMap())

        self.assertEqual(node.gen_m_indices([2, 0, 1]).numpy().tolist(), [0, 0, 2])
        self.assertEqual(
            node.gen_m_indices(paddle.to_tensor([1, 2], dtype="int64")).numpy().tolist(),
            [0, 1, 1],
        )
        self.assertEqual(node.gen_m_indices(paddle.to_tensor([], dtype="int64")).shape, [0])

    def test_fwd_gate_up_bf16_empty_split_and_grouped_outputs(self):
        node = ExpertsGroupGemmContiguousNode(CustomMap(), use_fp8_mlp=False)
        node.tokens_per_expert = [0, 0]
        x = paddle.empty([0, 4], dtype="float32")
        split_weight = [
            paddle.ones([4, 6], dtype="float32"),
            paddle.ones([4, 6], dtype="float32"),
        ]

        split_output = node.fwd_gate_up_bf16(x, split_weight)
        self.assertEqual(split_output.shape, [0, 6])
        self.assertIs(node.input, x)

        grouped = ExpertsGroupGemmContiguousNode(CustomMap(), use_fp8_mlp=False, moe_expert_fusion=True)
        grouped.tokens_per_expert = paddle.to_tensor([0, 0], dtype="int64")
        grouped_weight = paddle.ones([2, 4, 8], dtype="float32")

        grouped_output = grouped.fwd_gate_up_bf16(x, grouped_weight)
        self.assertEqual(grouped_output.shape, [0, 8])

    def test_fwd_gate_up_uses_bf16_path_and_records_deep_gemm_indices(self):
        node = ExpertsGroupGemmContiguousNode(CustomMap(), use_fp8_mlp=False, moe_deep_gemm=True)
        x = paddle.empty([0, 4], dtype="float32")
        weight = [paddle.ones([4, 6], dtype="float32")]

        output = node.fwd_gate_up(x, weight, 1, [0])

        self.assertEqual(output.shape, [0, 6])
        self.assertEqual(node.m_indices.shape, [0])


if __name__ == "__main__":
    unittest.main()
