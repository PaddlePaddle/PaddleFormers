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

try:
    import triton

    triton.runtime.driver.active.get_current_device()

    from paddleformers.fleet.triton_ops.moe_topk_fusion import (
        MoETopkFusion,
        routing_map_fusion_forward,
    )

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or not _TRITON_AVAILABLE,
    "CUDA and Triton are required for fused MoE topk kernels",
)
class TestMoETopkFusionCuda(unittest.TestCase):
    def setUp(self):
        paddle.device.set_device("gpu:0")

    def test_forward_selects_choice_topk_and_gate_probs(self):
        gate_probs = paddle.to_tensor(
            [[0.1, 0.9, 0.3, 0.7], [0.5, 0.2, 0.8, 0.1]],
            dtype="float32",
        )
        probs_for_choice = gate_probs.clone()

        topk_probs, topk_indices = MoETopkFusion.apply(gate_probs, probs_for_choice, 2, False, 1, 1, False)

        self.assertEqual(topk_probs.shape, [2, 2])
        self.assertEqual(topk_indices.dtype, paddle.int64)
        self.assertEqual(topk_indices.numpy().tolist(), [[1, 3], [2, 0]])
        self.assertEqual(topk_probs.numpy().tolist(), [[0.9, 0.7], [0.8, 0.5]])

    def test_forward_normalizes_selected_gate_probs(self):
        gate_probs = paddle.to_tensor(
            [[0.1, 0.9, 0.3, 0.7], [0.5, 0.2, 0.8, 0.1]],
            dtype="float32",
        )

        topk_probs, topk_indices = MoETopkFusion.apply(gate_probs, gate_probs, 2, False, 1, 1, True)

        self.assertEqual(topk_indices.numpy().tolist(), [[1, 3], [2, 0]])
        sums = topk_probs.sum(axis=-1).numpy().tolist()
        self.assertAlmostEqual(sums[0], 1.0, places=6)
        self.assertAlmostEqual(sums[1], 1.0, places=6)

    def test_node_limit_restricts_selected_group(self):
        gate_probs = paddle.arange(8, dtype="float32").reshape([1, 8]) / 10
        probs_for_choice = paddle.to_tensor(
            [[0.9, 0.8, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3]],
            dtype="float32",
        )

        topk_probs, topk_indices = MoETopkFusion.apply(gate_probs, probs_for_choice, 2, True, 4, 1, False)

        self.assertEqual(topk_indices.numpy().tolist(), [[0, 1]])
        self.assertEqual(topk_probs.numpy().tolist(), [[0.0, 0.1]])

    def test_backward_scatter_gradients_to_selected_experts(self):
        gate_probs = paddle.to_tensor(
            [[0.1, 0.9, 0.3, 0.7], [0.5, 0.2, 0.8, 0.1]],
            dtype="float32",
        )
        gate_probs.stop_gradient = False

        topk_probs, _ = MoETopkFusion.apply(gate_probs, gate_probs, 2, False, 1, 1, False)
        topk_probs.sum().backward()

        self.assertEqual(
            gate_probs.grad.numpy().tolist(),
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
        )

    def test_routing_map_forward_without_masks(self):
        gate_probs = paddle.ones([3, 4], dtype="float32")
        topk_indices = paddle.to_tensor([[0, 2], [1, 3], [0, 1]], dtype="int64")

        routing_map, topk_indices_out, dispatch_mask = routing_map_fusion_forward(gate_probs, topk_indices)

        self.assertEqual(
            routing_map.numpy().tolist(),
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
        )
        self.assertEqual(topk_indices_out.numpy().tolist(), [[0, 2], [1, 3], [0, 1]])
        self.assertEqual(dispatch_mask.numpy().tolist(), [2, 2, 1, 1])

    def test_routing_map_forward_with_padding_and_text_masks(self):
        gate_probs = paddle.ones([4, 4], dtype="float32")
        topk_indices = paddle.to_tensor([[0, 2], [1, 3], [0, 1], [2, 3]], dtype="int64")
        input_ids = paddle.to_tensor([5, 0, 7, 8], dtype="int64")
        is_pure_text_line = paddle.to_tensor([1, 1, 0, 1], dtype="int32")

        routing_map, topk_indices_out, dispatch_mask = routing_map_fusion_forward(
            gate_probs, topk_indices, input_ids, is_pure_text_line
        )

        self.assertEqual(
            routing_map.numpy().tolist(),
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
        )
        self.assertEqual(
            topk_indices_out.numpy().tolist(),
            [[0, 2], [-1, -1], [-1, -1], [2, 3]],
        )
        self.assertEqual(dispatch_mask.numpy().tolist(), [1, 0, 2, 1])


if __name__ == "__main__":
    unittest.main()
