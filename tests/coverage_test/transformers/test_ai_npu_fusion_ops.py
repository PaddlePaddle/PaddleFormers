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

import unittest

import paddle

from paddleformers.transformers.ernie4_5_moe_vl.model.fusion_ops.npu_fusion_ops import (
    npu_cal_aux_loss_func,
    npu_combining,
)


class TestNpuCombining(unittest.TestCase):
    """Tests for npu_combining function."""

    def test_basic_combining(self):
        x = paddle.randn([6, 4])
        combine_weights = paddle.ones([6, 2]) * 0.5
        scatter_index = paddle.randint(0, 6, [6, 2])

        result = npu_combining(x, combine_weights, scatter_index)
        self.assertEqual(len(result.shape), 2)

    def test_hard_gate(self):
        x = paddle.randn([6, 4])
        combine_weights = paddle.ones([6, 2]) * 0.5
        scatter_index = paddle.randint(0, 6, [6, 1])

        result = npu_combining(x, combine_weights, scatter_index, hard_gate=True)
        # With hard_gate, it does squeeze(-2) so result should be [6, 4]
        self.assertEqual(result.shape[0], 6)
        self.assertEqual(result.shape[-1], 4)

    def test_output_shape(self):
        x = paddle.randn([8, 16])
        combine_weights = paddle.ones([8, 3]) / 3
        scatter_index = paddle.randint(0, 8, [8, 3])

        result = npu_combining(x, combine_weights, scatter_index)
        self.assertEqual(result.shape, (8, 16))


class TestNpuCalAuxLossFunc(unittest.TestCase):
    """Tests for npu_cal_aux_loss_func."""

    def test_basic(self):
        gate_prob = paddle.rand([8, 4])
        dispatch_mask = paddle.ones([8, 4])
        tokens_mask = paddle.ones([8])
        dispatch_tokens_mask = paddle.ones([8, 4])
        num_experts = 4

        l_aux, _, _ = npu_cal_aux_loss_func(
            gate_prob, dispatch_mask, tokens_mask, dispatch_tokens_mask, num_experts, use_group=False, moe_k=2
        )
        self.assertEqual(l_aux.shape, [])

    def test_with_group(self):
        gate_prob = paddle.rand([8, 4])
        dispatch_mask = paddle.ones([8, 4])
        tokens_mask = paddle.ones([8])
        dispatch_tokens_mask = paddle.ones([8, 4])
        num_experts = 4

        l_aux, _, _ = npu_cal_aux_loss_func(
            gate_prob, dispatch_mask, tokens_mask, dispatch_tokens_mask, num_experts, use_group=True, moe_k=2
        )
        self.assertEqual(l_aux.shape, [])

    def test_without_tokens_mask(self):
        gate_prob = paddle.rand([8, 4])
        dispatch_mask = paddle.ones([8, 4])
        tokens_mask = None
        dispatch_tokens_mask = None
        num_experts = 4

        l_aux, _, _ = npu_cal_aux_loss_func(
            gate_prob, dispatch_mask, tokens_mask, dispatch_tokens_mask, num_experts, use_group=False, moe_k=2
        )
        self.assertEqual(l_aux.shape, [])

    def test_2d_dispatch_mask(self):
        gate_prob = paddle.rand([8, 4])
        dispatch_mask = paddle.ones([8, 4])
        tokens_mask = paddle.ones([8])
        dispatch_tokens_mask = paddle.ones([8, 4])
        num_experts = 4

        l_aux, _, _ = npu_cal_aux_loss_func(
            gate_prob, dispatch_mask, tokens_mask, dispatch_tokens_mask, num_experts, use_group=False, moe_k=2
        )
        self.assertEqual(l_aux.shape, [])

    def test_dtype_mismatch_tokens_mask(self):
        """Test that tokens_mask is cast to match gate_prob dtype."""
        gate_prob = paddle.rand([8, 4])
        dispatch_mask = paddle.ones([8, 4])
        tokens_mask = paddle.ones([8], dtype=paddle.int32)
        dispatch_tokens_mask = None
        num_experts = 4

        l_aux, _, _ = npu_cal_aux_loss_func(
            gate_prob, dispatch_mask, tokens_mask, dispatch_tokens_mask, num_experts, use_group=False, moe_k=2
        )
        self.assertEqual(l_aux.shape, [])

    def test_scale_factor(self):
        """Test with mismatched gate_prob and dispatch_tokens_mask shapes."""
        gate_prob = paddle.rand([10, 4])
        dispatch_mask = paddle.ones([10, 4])
        tokens_mask = paddle.ones([10])
        dispatch_tokens_mask = paddle.ones([6, 4])
        num_experts = 4

        l_aux, _, _ = npu_cal_aux_loss_func(
            gate_prob, dispatch_mask, tokens_mask, dispatch_tokens_mask, num_experts, use_group=False, moe_k=2
        )
        self.assertEqual(l_aux.shape, [])


if __name__ == "__main__":
    unittest.main()
