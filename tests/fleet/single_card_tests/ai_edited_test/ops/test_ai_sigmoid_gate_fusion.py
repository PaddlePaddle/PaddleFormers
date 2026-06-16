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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


# Tests for paddlefleet_ops/ops/triton_ops/sigmoid_gate_fusion.py

import types
import unittest
from unittest.mock import MagicMock


def _setup_triton_mock():
    """Create a mock triton module so we can import without GPU."""
    triton_mock = types.ModuleType("triton")
    triton_mock.jit = lambda fn=None, **kwargs: (
        (lambda f: f) if fn is None else fn
    )
    triton_mock.language = types.ModuleType("triton.language")
    triton_mock.language.constexpr = None
    tl = triton_mock.language
    tl.program_id = lambda axis: 0
    tl.arange = lambda start, end: []
    tl.load = lambda *a, **kw: 0.0
    tl.store = lambda *a, **kw: None
    tl.max = lambda *a, **kw: 0.0
    tl.sum = lambda *a, **kw: 0.0
    tl.full = lambda shape, val, dtype=None: val
    tl.where = lambda cond, a, b: a
    tl.float32 = "float32"
    tl.int64 = "int64"
    triton_mock.next_power_of_2 = lambda x: 1
    triton_mock.cdiv = lambda a, b: (a + b - 1) // b

    # Mock libdevice for sigmoid
    libdevice = types.ModuleType("triton.language.extra")
    libdevice2 = types.ModuleType("triton.language.extra.cuda")
    libdevice2.exp = lambda x: x
    libdevice2.div_rn = lambda a, b: a
    sys.modules.setdefault("triton.language.extra", libdevice)
    sys.modules.setdefault("triton.language.extra.cuda", libdevice2)
    sys.modules.setdefault("triton.language.extra.cuda.libdevice", libdevice2)
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


try:
    _setup_triton_mock()
    import paddleformers.fleet.triton_ops.sigmoid_gate_fusion  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestSigmoidGateFusionTritonDefinition(unittest.TestCase):
    """Tests for SigmoidGateFusionTriton PyLayer class definition."""

    def test_class_exists(self):
        """Test that SigmoidGateFusionTriton class can be imported."""
        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )

        self.assertTrue(callable(SigmoidGateFusionTriton))

    def test_has_forward(self):
        """Test that SigmoidGateFusionTriton has forward method."""
        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )

        self.assertTrue(hasattr(SigmoidGateFusionTriton, "forward"))

    def test_has_backward(self):
        """Test that SigmoidGateFusionTriton has backward method."""
        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )

        self.assertTrue(hasattr(SigmoidGateFusionTriton, "backward"))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestSigmoidGateKernels(unittest.TestCase):
    """Tests for sigmoid gate kernel definitions."""

    def test_fwd_kernel_callable(self):
        """Test fused_sigmoid_gate_fwd_kernel is callable."""
        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            fused_sigmoid_gate_fwd_kernel,
        )

        self.assertTrue(callable(fused_sigmoid_gate_fwd_kernel))

    def test_bwd_kernel_callable(self):
        """Test fused_sigmoid_gate_bwd_kernel is callable."""
        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            fused_sigmoid_gate_bwd_kernel,
        )

        self.assertTrue(callable(fused_sigmoid_gate_bwd_kernel))

    def test_sigmoid_precise_callable(self):
        """Test _sigmoid_precise is callable."""
        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            _sigmoid_precise,
        )

        self.assertTrue(callable(_sigmoid_precise))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestSigmoidGatePurePaddle(unittest.TestCase):
    """Tests for sigmoid gate computation using pure Paddle."""

    def test_sigmoid_gate_forward(self):
        """Test sigmoid gate forward: out = attn_out * sigmoid(gate)."""
        import paddle

        attn_out = paddle.to_tensor([1.0, 2.0, 3.0])
        gate = paddle.to_tensor([0.0, 1.0, -1.0])

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate

        # sigmoid(0) = 0.5, sigmoid(1) ~ 0.7311, sigmoid(-1) ~ 0.2689
        self.assertAlmostEqual(out[0].item(), 0.5, places=4)
        self.assertAlmostEqual(out[1].item(), 2.0 * 0.7311, places=3)
        self.assertAlmostEqual(out[2].item(), 3.0 * 0.2689, places=3)

    def test_sigmoid_gate_backward(self):
        """Test sigmoid gate backward gradients."""
        import paddle

        attn_out = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        gate = paddle.to_tensor([0.0, 1.0, -1.0], stop_gradient=False)

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(attn_out.grad)
        self.assertIsNotNone(gate.grad)

    def test_sigmoid_gate_shape_preserved(self):
        """Test that sigmoid gate preserves shape."""
        import paddle

        attn_out = paddle.randn([2, 4, 8])
        gate = paddle.randn([2, 4, 8])

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate

        self.assertEqual(out.shape, attn_out.shape)

    def test_sigmoid_gate_output_range(self):
        """Test that sigmoid gate output is bounded by attn_out magnitude."""
        import paddle

        attn_out = paddle.to_tensor([1.0, -1.0, 2.0, -2.0])
        gate = paddle.to_tensor([0.0, 0.0, 100.0, -100.0])

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate

        # sigmoid is in (0, 1), so |out| <= |attn_out|
        self.assertTrue(paddle.all(out.abs() <= attn_out.abs() + 1e-6))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestSigmoidGateFusionTritonForward(unittest.TestCase):
    """Tests for SigmoidGateFusionTriton forward validation."""

    def test_shape_mismatch_asserts(self):
        """Test that forward asserts when shapes don't match."""
        import paddle

        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )

        attn_out = paddle.randn([2, 4])
        gate = paddle.randn([2, 8])

        with self.assertRaises(AssertionError):
            SigmoidGateFusionTriton.forward(MagicMock(), attn_out, gate)

    def test_dtype_mismatch_asserts(self):
        """Test that forward asserts when dtypes don't match."""
        import paddle

        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )

        attn_out = paddle.randn([2, 4], dtype=paddle.float32)
        gate = paddle.randn([2, 4], dtype=paddle.float16)

        with self.assertRaises(AssertionError):
            SigmoidGateFusionTriton.forward(MagicMock(), attn_out, gate)

    def test_unsupported_dtype_asserts(self):
        """Test that forward asserts for unsupported dtype."""
        import paddle

        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )

        attn_out = paddle.randn([2, 4], dtype=paddle.int64)
        gate = paddle.randn([2, 4], dtype=paddle.int64)

        with self.assertRaises(AssertionError):
            SigmoidGateFusionTriton.forward(MagicMock(), attn_out, gate)


if __name__ == "__main__":
    unittest.main()
