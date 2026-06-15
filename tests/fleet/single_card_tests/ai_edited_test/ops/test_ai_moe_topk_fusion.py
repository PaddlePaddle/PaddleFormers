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
# Triton kernels are mocked since they require GPU.

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
class TestMoETopkFusionDefinition(unittest.TestCase):
    """Tests for MoETopkFusion PyLayer class definition."""

    def test_moe_topk_fusion_class_exists(self):
        """Test that MoETopkFusion class can be imported."""
        from paddleformers.fleet.triton_ops.moe_topk_fusion import MoETopkFusion

        self.assertTrue(callable(MoETopkFusion))

    def test_moe_topk_fusion_has_forward(self):
        """Test that MoETopkFusion has forward method."""
        from paddleformers.fleet.triton_ops.moe_topk_fusion import MoETopkFusion

        self.assertTrue(hasattr(MoETopkFusion, "forward"))

    def test_moe_topk_fusion_has_backward(self):
        """Test that MoETopkFusion has backward method."""
        from paddleformers.fleet.triton_ops.moe_topk_fusion import MoETopkFusion

        self.assertTrue(hasattr(MoETopkFusion, "backward"))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRoutingMapFusionForward(unittest.TestCase):
    """Tests for routing_map_fusion_forward function."""

    def test_function_is_callable(self):
        """Test that routing_map_fusion_forward is callable."""
        from paddleformers.fleet.triton_ops.moe_topk_fusion import (
            routing_map_fusion_forward,
        )

        self.assertTrue(callable(routing_map_fusion_forward))

    def test_routing_map_forward_signature(self):
        """Test routing_map_fusion_forward has expected parameters."""
        import inspect

        from paddleformers.fleet.triton_ops.moe_topk_fusion import (
            routing_map_fusion_forward,
        )

        sig = inspect.signature(routing_map_fusion_forward)
        params = list(sig.parameters.keys())
        self.assertIn("gate_probs", params)
        self.assertIn("topk_indices", params)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestTopkLogicPurePaddle(unittest.TestCase):
    """Tests for TopK logic using pure Paddle operations."""

    def test_simple_topk(self):
        """Test simple topk selection."""
        import paddle

        gate = paddle.to_tensor([[0.1, 0.9, 0.3, 0.7]])
        values, indices = paddle.topk(gate, k=2, axis=-1)
        self.assertEqual(indices[0, 0].item(), 1)
        self.assertEqual(indices[0, 1].item(), 3)

    def test_topk_with_normalization(self):
        """Test topk with probability normalization."""
        import paddle

        gate = paddle.to_tensor([[0.1, 0.3, 0.2, 0.4]])
        values, indices = paddle.topk(gate, k=2, axis=-1)
        # Normalize selected probabilities
        total = paddle.sum(values, axis=-1, keepdim=True)
        normalized = values / total
        self.assertAlmostEqual(normalized.sum().item(), 1.0, places=5)

    def test_topk_indices_shape(self):
        """Test topk indices have correct shape."""
        import paddle

        gate = paddle.to_tensor([[0.1, 0.3, 0.2, 0.4], [0.4, 0.1, 0.3, 0.2]])
        values, indices = paddle.topk(gate, k=2, axis=-1)
        self.assertEqual(indices.shape, [2, 2])


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFwdKernelDefinition(unittest.TestCase):
    """Tests for _fwd_kernel and _bwd_kernel definitions."""

    def test_fwd_kernel_callable(self):
        """Test _fwd_kernel is callable."""
        from paddleformers.fleet.triton_ops.moe_topk_fusion import _fwd_kernel

        self.assertTrue(callable(_fwd_kernel))

    def test_bwd_kernel_callable(self):
        """Test _bwd_kernel is callable."""
        from paddleformers.fleet.triton_ops.moe_topk_fusion import _bwd_kernel

        self.assertTrue(callable(_bwd_kernel))

    def test_routing_map_fwd_kernel_callable(self):
        """Test _routing_map_fwd_kernel is callable."""
        from paddleformers.fleet.triton_ops.moe_topk_fusion import (
            _routing_map_fwd_kernel,
        )

        self.assertTrue(callable(_routing_map_fwd_kernel))


if __name__ == "__main__":
    unittest.main()
