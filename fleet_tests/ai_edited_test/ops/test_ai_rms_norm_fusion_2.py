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


# Extra tests for paddlefleet_ops/ops/triton_ops/rms_norm_fusion.py
# Focus on: RMSNormFusionTriton forward/backward parameter handling

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
    tl.num_programs = lambda axis: 1
    tl.arange = lambda start, end: []
    tl.load = lambda *a, **kw: 0.0
    tl.store = lambda *a, **kw: None
    tl.max = lambda *a, **kw: 0.0
    tl.min = lambda *a, **kw: 0.0
    tl.sum = lambda *a, **kw: 0.0
    tl.exp = lambda x: 0.0
    tl.log = lambda x: 0.0
    tl.sqrt = lambda x: 0.0
    tl.full = lambda shape, val, dtype=None: val
    tl.zeros = lambda shape, dtype=None: 0.0
    tl.where = lambda cond, a, b: a
    tl.float32 = "float32"
    tl.int64 = "int64"
    triton_mock.next_power_of_2 = lambda x: 1
    triton_mock.cdiv = lambda a, b: (a + b - 1) // b
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


try:
    _setup_triton_mock()
    import paddleformers.fleet.triton_ops.rms_norm_fusion  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRMSNormFusionTritonForward(unittest.TestCase):
    """Tests for RMSNormFusionTriton forward parameter handling."""

    def test_forward_has_expected_parameters(self):
        """Test forward has expected parameters."""
        import inspect

        from paddleformers.fleet.triton_ops.rms_norm_fusion import (
            RMSNormFusionTriton,
        )

        sig = inspect.signature(RMSNormFusionTriton.forward)
        params = list(sig.parameters.keys())
        self.assertIn("ctx", params)
        self.assertIn("x", params)
        self.assertIn("weight", params)
        self.assertIn("epsilon", params)

    def test_backward_has_expected_parameters(self):
        """Test backward has expected parameters."""
        import inspect

        from paddleformers.fleet.triton_ops.rms_norm_fusion import (
            RMSNormFusionTriton,
        )

        sig = inspect.signature(RMSNormFusionTriton.backward)
        params = list(sig.parameters.keys())
        self.assertIn("ctx", params)
        self.assertIn("dy", params)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRMSNormComprehensive(unittest.TestCase):
    """Comprehensive tests for RMS norm computation using pure Paddle."""

    def test_rms_norm_with_different_epsilons(self):
        """Test RMS norm with different epsilon values."""
        import paddle

        x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]])
        weight = paddle.ones([4])

        for eps in [1e-5, 1e-6, 1e-8]:
            variance = paddle.mean(x * x, axis=-1, keepdim=True)
            invvar = 1.0 / paddle.sqrt(variance + eps)
            y = x * invvar * weight
            self.assertFalse(paddle.any(paddle.isnan(y)))

    def test_rms_norm_zero_input(self):
        """Test RMS norm with zero input."""
        import paddle

        x = paddle.zeros([1, 4])
        weight = paddle.ones([4])
        eps = 1e-6

        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + eps)
        y = x * invvar * weight

        # With zero input, output should be zero (0 * invvar = 0)
        self.assertTrue(paddle.allclose(y, paddle.zeros_like(y), atol=1e-5))

    def test_rms_norm_large_input(self):
        """Test RMS norm with large input values."""
        import paddle

        x = paddle.to_tensor([[1000.0, 2000.0, 3000.0, 4000.0]])
        weight = paddle.ones([4])
        eps = 1e-6

        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + eps)
        y = x * invvar * weight

        self.assertFalse(paddle.any(paddle.isnan(y)))
        self.assertFalse(paddle.any(paddle.isinf(y)))

    def test_rms_norm_batch(self):
        """Test RMS norm with batched input."""
        import paddle

        x = paddle.randn([4, 8, 16])
        weight = paddle.ones([16])
        eps = 1e-6

        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + eps)
        y = x * invvar * weight

        self.assertEqual(y.shape, x.shape)
        self.assertFalse(paddle.any(paddle.isnan(y)))

    def test_rms_norm_1d_input(self):
        """Test RMS norm with 1D input (ndim=1)."""
        import paddle

        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        weight = paddle.ones([4])
        eps = 1e-6

        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + eps)
        y = x * invvar * weight

        self.assertEqual(y.shape, x.shape)
        self.assertFalse(paddle.any(paddle.isnan(y)))

    def test_rms_norm_backward_dx(self):
        """Test RMS norm backward gradient computation."""
        import paddle

        x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], stop_gradient=False)
        weight = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
        eps = 1e-6

        # Forward
        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + eps)
        y = x * invvar * weight
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_rms_norm_strided_input(self):
        """Test RMS norm handles strided (non-contiguous) input."""
        import paddle

        x = paddle.randn([4, 8, 16])
        # Create a strided view
        x_strided = x[:, :, ::2]  # shape [4, 8, 8]
        weight = paddle.ones([8])
        eps = 1e-6

        variance = paddle.mean(x_strided * x_strided, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + eps)
        y = x_strided * invvar * weight

        self.assertEqual(y.shape, x_strided.shape)


if __name__ == "__main__":
    unittest.main()
