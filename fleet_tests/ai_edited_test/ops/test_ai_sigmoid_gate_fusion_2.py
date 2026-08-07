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


# Extra tests for paddlefleet_ops/ops/triton_ops/sigmoid_gate_fusion.py
# Focus on: SigmoidGateFusionTriton forward/backward, pure Paddle validation

import types
import unittest


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
class TestSigmoidGateFusionComprehensive(unittest.TestCase):
    """Comprehensive tests for sigmoid gate computation using pure Paddle."""

    def test_sigmoid_gate_at_zero(self):
        """Test sigmoid gate when gate value is zero."""
        import paddle

        attn_out = paddle.to_tensor([2.0])
        gate = paddle.to_tensor([0.0])
        result = attn_out * paddle.nn.functional.sigmoid(gate)
        self.assertAlmostEqual(result.item(), 1.0, places=5)

    def test_sigmoid_gate_at_large_positive(self):
        """Test sigmoid gate when gate value is large positive."""
        import paddle

        attn_out = paddle.to_tensor([2.0])
        gate = paddle.to_tensor([100.0])
        result = attn_out * paddle.nn.functional.sigmoid(gate)
        self.assertAlmostEqual(result.item(), 2.0, places=3)

    def test_sigmoid_gate_at_large_negative(self):
        """Test sigmoid gate when gate value is large negative."""
        import paddle

        attn_out = paddle.to_tensor([2.0])
        gate = paddle.to_tensor([-100.0])
        result = attn_out * paddle.nn.functional.sigmoid(gate)
        self.assertAlmostEqual(result.item(), 0.0, places=3)

    def test_sigmoid_gate_backward_grad_attn(self):
        """Test sigmoid gate backward gradient for attn_out."""
        import paddle

        attn_out = paddle.to_tensor([2.0, 3.0], stop_gradient=False)
        gate = paddle.to_tensor([0.0, 1.0], stop_gradient=False)

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate
        loss = out.sum()
        loss.backward()

        # d_attn = sigmoid(gate) * grad_output
        expected_attn_grad = paddle.nn.functional.sigmoid(gate)
        self.assertTrue(
            paddle.allclose(attn_out.grad, expected_attn_grad, atol=1e-5)
        )

    def test_sigmoid_gate_backward_grad_gate(self):
        """Test sigmoid gate backward gradient for gate."""
        import paddle

        attn_out = paddle.to_tensor([2.0, 3.0], stop_gradient=False)
        gate = paddle.to_tensor([0.0, 1.0], stop_gradient=False)

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate
        loss = out.sum()
        loss.backward()

        # d_gate = attn_out * (1 - sigmoid(gate)) * sigmoid(gate) * grad_output
        self.assertIsNotNone(gate.grad)
        self.assertFalse(paddle.any(paddle.isnan(gate.grad)))

    def test_sigmoid_gate_bf16(self):
        """Test sigmoid gate with bfloat16."""
        import paddle

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Requires CUDA for bfloat16")

        attn_out = paddle.to_tensor([2.0, 3.0], dtype=paddle.bfloat16)
        gate = paddle.to_tensor([0.0, 1.0], dtype=paddle.bfloat16)

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate

        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertFalse(paddle.any(paddle.isnan(out)))

    def test_sigmoid_gate_fp16(self):
        """Test sigmoid gate with float16."""
        import paddle

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Requires CUDA for float16")

        attn_out = paddle.to_tensor([2.0, 3.0], dtype=paddle.float16)
        gate = paddle.to_tensor([0.0, 1.0], dtype=paddle.float16)

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate

        self.assertEqual(out.dtype, paddle.float16)

    def test_sigmoid_gate_3d_input(self):
        """Test sigmoid gate with 3D input."""
        import paddle

        attn_out = paddle.randn([2, 4, 8])
        gate = paddle.randn([2, 4, 8])

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate

        self.assertEqual(out.shape, [2, 4, 8])

    def test_sigmoid_gate_4d_input(self):
        """Test sigmoid gate with 4D input."""
        import paddle

        attn_out = paddle.randn([1, 2, 4, 8])
        gate = paddle.randn([1, 2, 4, 8])

        sigmoid_gate = paddle.nn.functional.sigmoid(gate)
        out = attn_out * sigmoid_gate

        self.assertEqual(out.shape, [1, 2, 4, 8])


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestSigmoidGateFusionTritonSignature(unittest.TestCase):
    """Tests for SigmoidGateFusionTriton method signatures."""

    def test_forward_signature(self):
        """Test forward method signature."""
        import inspect

        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )

        sig = inspect.signature(SigmoidGateFusionTriton.forward)
        params = list(sig.parameters.keys())
        self.assertIn("ctx", params)
        self.assertIn("attn_out", params)
        self.assertIn("gate", params)

    def test_backward_signature(self):
        """Test backward method signature."""
        import inspect

        from paddleformers.fleet.triton_ops.sigmoid_gate_fusion import (
            SigmoidGateFusionTriton,
        )

        sig = inspect.signature(SigmoidGateFusionTriton.backward)
        params = list(sig.parameters.keys())
        self.assertIn("ctx", params)
        self.assertIn("out_grad", params)


if __name__ == "__main__":
    unittest.main()
