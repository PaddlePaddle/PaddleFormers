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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle

from paddleformers.fleet.triton_ops import moe_topk_fusion
from paddleformers.fleet.triton_ops.moe_topk_fusion import (
    MoETopkFusion,
    routing_map_fusion_forward,
)


class Context:
    def save_for_backward(self, *values):
        self.saved = values

    def saved_tensor(self):
        return self.saved


class KernelRecorder:
    def __init__(self):
        self.grid = None
        self.args = None
        self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TestMoETopkFusionWrappers(unittest.TestCase):
    def test_forward_launches_kernel_and_saves_context_without_norm(self):
        old_kernel = moe_topk_fusion._fwd_kernel
        recorder = KernelRecorder()
        moe_topk_fusion._fwd_kernel = recorder
        ctx = Context()
        gate_probs = paddle.arange(10, dtype="float32").reshape([2, 5])
        probs_for_choice = gate_probs + 0.5
        try:
            topk_probs, topk_indices = MoETopkFusion.forward(
                ctx,
                gate_probs,
                probs_for_choice,
                moe_k=2,
                use_node_limit=False,
                n_group=4,
                topk_group=2,
                norm_gate_logits=False,
            )
        finally:
            moe_topk_fusion._fwd_kernel = old_kernel

        self.assertEqual(recorder.grid, (2,))
        self.assertIs(recorder.args[0], gate_probs)
        self.assertIs(recorder.args[1], probs_for_choice)
        self.assertIs(recorder.args[4], topk_probs)
        self.assertEqual(recorder.args[12:16], (2, False, 1, 1))
        self.assertEqual(recorder.args[16], False)
        self.assertEqual(recorder.args[17], 32)
        self.assertEqual(topk_probs.shape, [2, 2])
        self.assertEqual(topk_indices.dtype, paddle.int64)
        saved_indices, saved_probs, saved_sum = ctx.saved
        self.assertEqual(saved_indices.shape, [2, 2])
        self.assertIs(saved_probs, topk_probs)
        self.assertIsNone(saved_sum)
        self.assertEqual(ctx.input_shape, [2, 5])
        self.assertFalse(ctx.norm_gate_logits)
        self.assertEqual(ctx.moe_k, 2)

    def test_forward_launches_kernel_with_node_limit_and_norm_sum(self):
        old_kernel = moe_topk_fusion._fwd_kernel
        recorder = KernelRecorder()
        moe_topk_fusion._fwd_kernel = recorder
        ctx = Context()
        gate_probs = paddle.ones([1, 64], dtype="float32")
        try:
            topk_probs, topk_indices = MoETopkFusion.forward(
                ctx,
                gate_probs,
                gate_probs,
                moe_k=3,
                use_node_limit=True,
                n_group=8,
                topk_group=2,
                norm_gate_logits=True,
            )
        finally:
            moe_topk_fusion._fwd_kernel = old_kernel

        self.assertEqual(recorder.args[12:16], (3, True, 8, 2))
        self.assertTrue(recorder.args[16])
        self.assertEqual(recorder.args[17], 64)
        self.assertEqual(ctx.saved[2].shape, [1])
        self.assertEqual(topk_probs.shape, [1, 3])
        self.assertEqual(topk_indices.shape, [1, 3])

    def test_backward_launches_kernel_for_saved_tensors(self):
        old_kernel = moe_topk_fusion._bwd_kernel
        recorder = KernelRecorder()
        moe_topk_fusion._bwd_kernel = recorder
        ctx = Context()
        ctx.save_for_backward(
            paddle.to_tensor([[1, 3]], dtype="int32"),
            paddle.ones([1, 2], dtype="float32"),
            paddle.ones([1], dtype="float32"),
        )
        ctx.input_shape = [1, 5]
        ctx.norm_gate_logits = True
        ctx.moe_k = 2
        grad_output_probs = paddle.ones([1, 2], dtype="float32")
        try:
            grad_gate_probs, none_value = MoETopkFusion.backward(
                ctx, grad_output_probs, None
            )
        finally:
            moe_topk_fusion._bwd_kernel = old_kernel

        self.assertEqual(recorder.grid, (1,))
        self.assertIs(recorder.args[0], grad_output_probs)
        self.assertEqual(recorder.args[13:16], (2, True, 2))
        self.assertEqual(grad_gate_probs.shape, [1, 5])
        self.assertIsNone(none_value)

    def test_routing_map_wrapper_launches_kernel_with_masks(self):
        old_kernel = moe_topk_fusion._routing_map_fwd_kernel
        recorder = KernelRecorder()
        moe_topk_fusion._routing_map_fwd_kernel = recorder
        gate_probs = paddle.ones([3, 5], dtype="float32")
        topk_indices = paddle.to_tensor([[1, 2], [3, 4], [0, 1]], dtype="int64")
        input_ids = paddle.to_tensor([1, 0, 2], dtype="int64")
        pure_text = paddle.to_tensor([1, 1, 0], dtype="int64")
        try:
            routing_map, topk_out, dispatch_mask = routing_map_fusion_forward(
                gate_probs, topk_indices, input_ids, pure_text
            )
        finally:
            moe_topk_fusion._routing_map_fwd_kernel = old_kernel

        self.assertEqual(recorder.grid, (1, 1))
        self.assertIs(recorder.kwargs["topk_indices_ptr"], topk_indices)
        self.assertIs(recorder.kwargs["input_ids_ptr"], input_ids)
        self.assertIs(recorder.kwargs["is_pure_text_line_ptr"], pure_text)
        self.assertTrue(recorder.kwargs["has_input_ids"])
        self.assertTrue(recorder.kwargs["has_pure_text_mask"])
        self.assertEqual(recorder.kwargs["BLOCK_K"], 2)
        self.assertEqual(routing_map.shape, [3, 5])
        self.assertEqual(topk_out.shape, [3, 2])
        self.assertEqual(dispatch_mask.shape, [5])

    def test_routing_map_wrapper_uses_placeholders_without_masks(self):
        old_kernel = moe_topk_fusion._routing_map_fwd_kernel
        recorder = KernelRecorder()
        moe_topk_fusion._routing_map_fwd_kernel = recorder
        gate_probs = paddle.ones([2, 4], dtype="float32")
        topk_indices = paddle.ones([2, 3], dtype="int64")
        try:
            routing_map_fusion_forward(gate_probs, topk_indices)
        finally:
            moe_topk_fusion._routing_map_fwd_kernel = old_kernel

        self.assertIs(recorder.kwargs["input_ids_ptr"], topk_indices)
        self.assertIs(recorder.kwargs["is_pure_text_line_ptr"], topk_indices)
        self.assertFalse(recorder.kwargs["has_input_ids"])
        self.assertFalse(recorder.kwargs["has_pure_text_mask"])
        self.assertEqual(recorder.kwargs["BLOCK_K"], 4)


if __name__ == "__main__":
    unittest.main()
