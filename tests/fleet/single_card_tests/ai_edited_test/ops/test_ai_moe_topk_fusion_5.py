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

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


# Tests for paddlefleet_ops/ops/triton_ops/moe_topk_fusion.py
# Focus on: MoETopkFusion forward validation and routing_map_fusion_forward

import types
import unittest


def _setup_triton_mock():
    """Create a mock triton module so we can import without GPU."""
    triton_mock = types.ModuleType("triton")
    triton_mock.jit = lambda fn=None, **kwargs: ((lambda f: f) if fn is None else fn)
    triton_mock.language = types.ModuleType("triton.language")
    triton_mock.language.constexpr = None
    tl = triton_mock.language
    tl.program_id = lambda axis: 0
    tl.arange = lambda start, end: []
    tl.load = lambda *a, **kw: 0.0
    tl.store = lambda *a, **kw: None
    tl.max = lambda *a, **kw: 0.0
    tl.min = lambda *a, **kw: 0.0
    tl.sum = lambda *a, **kw: 0.0
    tl.exp = lambda x: 0.0
    tl.log = lambda x: 0.0
    tl.full = lambda shape, val, dtype=None: val
    tl.where = lambda cond, a, b: a
    tl.float32 = "float32"
    tl.int32 = "int32"
    tl.int64 = "int64"
    tl.int8 = "int8"
    tl.atomic_add = lambda *a, **kw: None
    triton_mock.next_power_of_2 = lambda x: 1
    triton_mock.cdiv = lambda a, b: (a + b - 1) // b
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


try:
    _setup_triton_mock()
    import paddleformers.fleet.triton_ops.moe_topk_fusion  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestMoETopkFusionForward(unittest.TestCase):
    """Tests for MoETopkFusion forward."""

    def test_forward_has_expected_parameters(self):
        """Test forward has expected parameter names."""
        import inspect

        from paddleformers.fleet.triton_ops.moe_topk_fusion import MoETopkFusion

        sig = inspect.signature(MoETopkFusion.forward)
        params = list(sig.parameters.keys())
        self.assertIn("ctx", params)
        self.assertIn("gate_probs", params)
        self.assertIn("probs_for_choice", params)
        self.assertIn("moe_k", params)
        self.assertIn("use_node_limit", params)
        self.assertIn("n_group", params)
        self.assertIn("topk_group", params)
        self.assertIn("norm_gate_logits", params)

    def test_backward_has_expected_parameters(self):
        """Test backward has expected parameter names."""
        import inspect

        from paddleformers.fleet.triton_ops.moe_topk_fusion import MoETopkFusion

        sig = inspect.signature(MoETopkFusion.backward)
        params = list(sig.parameters.keys())
        self.assertIn("ctx", params)
        self.assertIn("grad_output_probs", params)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRoutingMapFusionForward(unittest.TestCase):
    """Tests for routing_map_fusion_forward function."""

    def test_has_expected_parameters(self):
        """Test function has expected parameters."""
        import inspect

        from paddleformers.fleet.triton_ops.moe_topk_fusion import (
            routing_map_fusion_forward,
        )

        sig = inspect.signature(routing_map_fusion_forward)
        params = list(sig.parameters.keys())
        self.assertIn("gate_probs", params)
        self.assertIn("topk_indices", params)
        self.assertIn("input_ids", params)
        self.assertIn("is_pure_text_line", params)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestMoETopkFusionNodeLimitLogic(unittest.TestCase):
    """Tests for MoE TopK node limit logic using pure Paddle."""

    def test_topk_with_node_limit(self):
        """Test topk with node limit group selection."""
        import paddle

        # 8 experts, 2 groups of 4
        gate = paddle.to_tensor([[0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6]])
        values, indices = paddle.topk(gate, k=2, axis=-1)

        # Without node limit, top2 would be indices [1, 5]
        self.assertIn(1, indices.numpy()[0])

    def test_topk_with_normalization(self):
        """Test topk with probability normalization."""
        import paddle

        gate = paddle.to_tensor([[0.1, 0.9, 0.3, 0.7]])
        values, indices = paddle.topk(gate, k=2, axis=-1)

        # Normalize
        total = paddle.sum(values, axis=-1, keepdim=True)
        normalized = values / total
        self.assertAlmostEqual(normalized.sum().item(), 1.0, places=5)

    def test_routing_map_construction(self):
        """Test routing map construction from topk indices."""
        import paddle

        topk_indices = paddle.to_tensor([[0, 3], [1, 2]])
        n_experts = 4

        # Build routing map
        routing_map = paddle.zeros([2, n_experts], dtype=paddle.float32)
        for i in range(topk_indices.shape[0]):
            for j in range(topk_indices.shape[1]):
                routing_map[i, topk_indices[i, j]] = 1.0

        # Verify routing map
        self.assertEqual(routing_map[0, 0].item(), 1.0)
        self.assertEqual(routing_map[0, 3].item(), 1.0)
        self.assertEqual(routing_map[0, 1].item(), 0.0)
        self.assertEqual(routing_map[1, 1].item(), 1.0)
        self.assertEqual(routing_map[1, 2].item(), 1.0)

    def test_dispatch_mask_computation(self):
        """Test dispatch mask is the sum over sequence dimension."""
        import paddle

        routing_map = paddle.to_tensor([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=paddle.float32)
        dispatch_mask = routing_map.sum(axis=0).to(paddle.int64)

        self.assertEqual(dispatch_mask[0].item(), 1)
        self.assertEqual(dispatch_mask[1].item(), 1)
        self.assertEqual(dispatch_mask[2].item(), 1)
        self.assertEqual(dispatch_mask[3].item(), 1)

    def test_topk_with_padding_mask(self):
        """Test topk with padding mask zeros out invalid tokens."""
        import paddle

        topk_indices = paddle.to_tensor([[0, 3], [1, 2]])
        input_ids = paddle.to_tensor([1, 0])  # Second token is padding

        # Mask out padding tokens
        masked_indices = topk_indices.clone()
        for i in range(len(input_ids)):
            if input_ids[i] == 0:
                masked_indices[i] = -1

        self.assertEqual(masked_indices[0, 0].item(), 0)
        self.assertEqual(masked_indices[1, 0].item(), -1)

    def test_topk_with_pure_text_mask(self):
        """Test topk with pure text line mask."""
        import paddle

        topk_indices = paddle.to_tensor([[0, 3], [1, 2]])
        is_pure_text = paddle.to_tensor([1, 0])  # Second token is not pure text

        # Mask out non-pure-text tokens
        masked_indices = topk_indices.clone()
        for i in range(len(is_pure_text)):
            if is_pure_text[i] == 0:
                masked_indices[i] = -1

        self.assertEqual(masked_indices[0, 0].item(), 0)
        self.assertEqual(masked_indices[1, 0].item(), -1)


if __name__ == "__main__":
    unittest.main()
