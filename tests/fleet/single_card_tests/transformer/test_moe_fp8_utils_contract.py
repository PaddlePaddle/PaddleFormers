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

import paddle

from paddleformers.fleet.transformer.moe.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
    _get_fp8_weight_and_scale,
)


class _ExpertsGroupGemmCustomMap:
    def __init__(self):
        self.experts = []
        self.grouped_gemm_experts = None


def _new_bf16_node(**kwargs):
    return ExpertsGroupGemmContiguousNode(
        _ExpertsGroupGemmCustomMap(),
        use_fp8_mlp=False,
        **kwargs,
    )


class TestExpertsGroupGemmContiguousNodeCounts(unittest.TestCase):
    def test_get_fp8_weight_and_scale_returns_cached_nontranspose(self):
        weight = paddle.create_parameter([2, 3], dtype="float32")
        weight.fp8_weight_stacked = paddle.ones([2, 3], dtype="float32")
        weight.fp8_scale_stacked = paddle.full([2, 1], 0.5, dtype="float32")

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(weight)

        self.assertIs(fp8_weight, weight.fp8_weight_stacked)
        self.assertIs(fp8_scale, weight.fp8_scale_stacked)

    def test_gen_m_indices_accepts_list_tensor_and_empty_counts(self):
        node = _new_bf16_node()

        self.assertEqual(
            node.gen_m_indices([2, 0, 1]).numpy().tolist(),
            [0, 0, 2],
        )
        self.assertEqual(
            node.gen_m_indices(paddle.to_tensor([1, 0, 2], dtype="int64"))
            .numpy()
            .tolist(),
            [0, 2, 2],
        )
        self.assertEqual(
            node.gen_m_indices(paddle.to_tensor([], dtype="int64")).shape,
            [0],
        )

    def test_fwd_gate_up_runs_bf16_path_with_tensor_counts(self):
        node = _new_bf16_node(moe_deep_gemm=True)
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
        )
        expert_w1 = [
            paddle.eye(2, dtype="float32"),
            paddle.full([2, 2], 2.0, dtype="float32"),
        ]
        token_counts = paddle.to_tensor([1, 2], dtype="int64")

        out = node.fwd_gate_up(
            x,
            expert_w1=expert_w1,
            num_expert=2,
            tokens_per_expert=token_counts,
        )

        self.assertEqual(
            out.numpy().tolist(),
            [[1.0, 2.0], [14.0, 14.0], [22.0, 22.0]],
        )
        self.assertIs(node.tokens_per_expert, token_counts)
        self.assertEqual(node.m_indices.numpy().tolist(), [0, 1, 1])

    def test_fwd_gate_up_runs_bf16_path_with_list_counts(self):
        node = _new_bf16_node(moe_deep_gemm=True)
        out = node.fwd_gate_up(
            paddle.to_tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
            ),
            expert_w1=[
                paddle.full([2, 2], 1.0, dtype="float32"),
                paddle.eye(2, dtype="float32"),
            ],
            num_expert=2,
            tokens_per_expert=[2, 1],
        )

        self.assertEqual(
            out.numpy().tolist(),
            [[3.0, 3.0], [7.0, 7.0], [5.0, 6.0]],
        )
        self.assertEqual(node.m_indices.numpy().tolist(), [0, 0, 1])

    def test_fwd_gate_up_can_reuse_cached_input(self):
        node = _new_bf16_node(moe_deep_gemm=False)
        node.input = paddle.to_tensor([[2.0, 3.0]], dtype="float32")

        out = node.fwd_gate_up(
            None,
            expert_w1=[paddle.eye(2, dtype="float32")],
            num_expert=1,
            tokens_per_expert=[1],
        )

        self.assertEqual(out.numpy().tolist(), [[2.0, 3.0]])

    def test_fwd_gate_up_preserves_empty_bf16_shape(self):
        node = _new_bf16_node(moe_deep_gemm=False)

        out = node.fwd_gate_up(
            paddle.empty([0, 2], dtype="float32"),
            expert_w1=[paddle.empty([2, 3], dtype="float32")],
            num_expert=1,
            tokens_per_expert=[0],
        )

        self.assertEqual(out.shape, [0, 3])


if __name__ == "__main__":
    unittest.main()
