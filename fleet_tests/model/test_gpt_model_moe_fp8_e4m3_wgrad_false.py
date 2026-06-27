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

"""Tests for fp8='e4m3' + fp8_wgrad=False with moe_expert_fusion=True.

Covers the bug fix in commit 3352089: when moe_expert_fusion=True and
use_fp8_mlp=True, the four backward branch guards must use the per-expert
list path instead of grouped_gemm_experts.

Only tests the bf16 backward path so that it can run on any GPU.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.transformer.moe.fp8_utils import ExpertsGroupGemmContiguousNode

# All dimensions must be multiples of FP8_ALIGN (128) for fused_linear_param_grad_add
HIDDEN = 128
INTER = 128
NUM_EXPERTS = 4
TOKENS_PER_EXPERT = 128


def _make_expert_weight(hidden, inter):
    """One expert's weights as trainable parameters (bfloat16).

    Paddle stores weights as [in_features, out_features]:
      up_gate_proj: [hidden, inter*2]  (gate+up merged)
      down_proj:    [inter,  hidden]
    """
    up_gate = paddle.create_parameter(
        shape=[hidden, inter * 2],
        dtype="bfloat16",
        default_initializer=paddle.nn.initializer.Normal(),
    )
    down = paddle.create_parameter(
        shape=[inter, hidden],
        dtype="bfloat16",
        default_initializer=paddle.nn.initializer.Normal(),
    )
    w = MagicMock()
    w.up_gate_proj = MagicMock()
    w.up_gate_proj.weight = up_gate
    w.down_proj = MagicMock()
    w.down_proj.weight = down
    return w


def _make_node(use_fp8_mlp, use_bf16_gemm_weight_grad):
    experts = [_make_expert_weight(HIDDEN, INTER) for _ in range(NUM_EXPERTS)]
    cm = MagicMock()
    cm.experts = experts
    node = ExpertsGroupGemmContiguousNode(
        custom_map=cm,
        use_fp8_mlp=use_fp8_mlp,
        use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
        moe_expert_fusion=True,
        moe_deep_gemm=False,
    )
    return node, experts


class TestFp8E4m3WgradFalseBf16Backward(unittest.TestCase):
    """ExpertsGroupGemmContiguousNode: fp8=e4m3 (use_fp8_mlp=True) +
    fp8_wgrad=False (use_bf16_gemm_weight_grad=True) + moe_expert_fusion=True.

    Before the fix, backward_impl_bf16 / bwd_down_input_bf16 /
    bwd_gate_up_input_bf16 / bf16_weight_grad all incorrectly entered the
    grouped_gemm branch (guarded by `moe_expert_fusion` alone) and tried to
    access self.grouped_gemm_experts, which does not exist when use_fp8_mlp=True.

    After the fix the guard is `moe_expert_fusion and not use_fp8_mlp`, so these
    functions fall through to the per-expert list path.
    """

    def setUp(self):
        total = TOKENS_PER_EXPERT * NUM_EXPERTS
        self.total_tokens = total
        self.node, self.experts = _make_node(
            use_fp8_mlp=True, use_bf16_gemm_weight_grad=True
        )
        self.node.tokens_per_expert = [TOKENS_PER_EXPERT] * NUM_EXPERTS
        self.node.input = paddle.randn([total, HIDDEN], dtype=paddle.bfloat16)
        self.node.input_fp8 = None
        self.node.input_scale = None
        self.node.o1 = paddle.randn([total, INTER * 2], dtype=paddle.bfloat16)
        self.out_grad = paddle.randn([total, HIDDEN], dtype=paddle.bfloat16)
        self.unzipped_probs = paddle.ones([total], dtype=paddle.bfloat16)

    def test_node_uses_per_expert_list(self):
        """With use_fp8_mlp=True + moe_expert_fusion=True the node must hold
        self.experts, not self.grouped_gemm_experts."""
        self.assertTrue(hasattr(self.node, "experts"))
        self.assertFalse(hasattr(self.node, "grouped_gemm_experts"))

    def test_backward_impl_bf16_does_not_raise(self):
        """backward_impl_bf16 must not raise AttributeError.
        Pre-fix it crashed accessing grouped_gemm_experts."""
        try:
            dx, probs_grad = self.node.backward_impl_bf16(
                self.out_grad, self.unzipped_probs
            )
        except AttributeError as exc:
            self.fail(
                f"backward_impl_bf16 raised AttributeError (bug: accessed "
                f"grouped_gemm_experts instead of per-expert list): {exc}"
            )

    def test_backward_impl_bf16_output_shapes(self):
        """dx shape must be [total_tokens, HIDDEN]; probs_grad length must match."""
        dx, probs_grad = self.node.backward_impl_bf16(
            self.out_grad, self.unzipped_probs
        )
        self.assertEqual(list(dx.shape), [self.total_tokens, HIDDEN])
        self.assertEqual(probs_grad.shape[0], self.total_tokens)

    def test_bwd_gate_up_input_bf16_does_not_raise(self):
        """bwd_gate_up_input_bf16 with use_fp8_mlp=True + moe_expert_fusion=True
        must not raise (pre-fix: tried to access grouped_gemm_experts weight tensor)."""
        expert_w1 = [e.up_gate_proj.weight for e in self.node.experts]
        do1 = paddle.randn(
            [self.total_tokens, INTER * 2], dtype=paddle.bfloat16
        )
        try:
            dx = self.node.bwd_gate_up_input_bf16(do1, expert_w1)
        except AttributeError as exc:
            self.fail(
                f"bwd_gate_up_input_bf16 raised AttributeError (bug: accessed "
                f"grouped_gemm_experts instead of per-expert list): {exc}"
            )
        self.assertEqual(list(dx.shape), [self.total_tokens, HIDDEN])

    def test_bwd_gate_up_input_bf16_zero_tokens_does_not_raise(self):
        """bwd_gate_up_input_bf16 else branch (numpy.prod == 0): must use
        expert_w1[0].shape[0] for dx_shape, not expert_w1.shape[1]."""
        expert_w1 = [e.up_gate_proj.weight for e in self.node.experts]
        do1 = paddle.empty([0, INTER * 2], dtype=paddle.bfloat16)
        self.node.tokens_per_expert = [0] * NUM_EXPERTS
        try:
            dx = self.node.bwd_gate_up_input_bf16(do1, expert_w1)
        except AttributeError as exc:
            self.fail(
                f"bwd_gate_up_input_bf16 zero-token branch raised AttributeError: {exc}"
            )
        self.assertEqual(dx.shape[0], 0)
        self.assertEqual(dx.shape[1], HIDDEN)


if __name__ == "__main__":
    unittest.main()
