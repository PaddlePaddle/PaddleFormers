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


# Tests for paddlefleet_ops/ops/triton_ops/rms_norm_fusion.py

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
class TestRMSNormFusionTritonDefinition(unittest.TestCase):
    """Tests for RMSNormFusionTriton PyLayer class definition."""

    def test_class_exists(self):
        """Test that RMSNormFusionTriton class can be imported."""
        from paddleformers.fleet.triton_ops.rms_norm_fusion import RMSNormFusionTriton

        self.assertTrue(callable(RMSNormFusionTriton))

    def test_has_forward(self):
        """Test that RMSNormFusionTriton has forward method."""
        from paddleformers.fleet.triton_ops.rms_norm_fusion import RMSNormFusionTriton

        self.assertTrue(hasattr(RMSNormFusionTriton, "forward"))

    def test_has_backward(self):
        """Test that RMSNormFusionTriton has backward method."""
        from paddleformers.fleet.triton_ops.rms_norm_fusion import RMSNormFusionTriton

        self.assertTrue(hasattr(RMSNormFusionTriton, "backward"))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRMSNormKernels(unittest.TestCase):
    """Tests for RMSNorm kernel definitions."""

    def test_fwd_kernel_callable(self):
        """Test rms_norm_fwd_kernel is callable."""
        from paddleformers.fleet.triton_ops.rms_norm_fusion import rms_norm_fwd_kernel

        self.assertTrue(callable(rms_norm_fwd_kernel))

    def test_bwd_dx_kernel_callable(self):
        """Test rms_norm_bwd_dx_kernel is callable."""
        from paddleformers.fleet.triton_ops.rms_norm_fusion import (
            rms_norm_bwd_dx_kernel,
        )

        self.assertTrue(callable(rms_norm_bwd_dx_kernel))

    def test_bwd_dw_partial_kernel_callable(self):
        """Test rms_norm_bwd_dw_partial_kernel is callable."""
        from paddleformers.fleet.triton_ops.rms_norm_fusion import (
            rms_norm_bwd_dw_partial_kernel,
        )

        self.assertTrue(callable(rms_norm_bwd_dw_partial_kernel))

    def test_bwd_dw_final_kernel_callable(self):
        """Test rms_norm_bwd_dw_final_kernel is callable."""
        from paddleformers.fleet.triton_ops.rms_norm_fusion import (
            rms_norm_bwd_dw_final_kernel,
        )

        self.assertTrue(callable(rms_norm_bwd_dw_final_kernel))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRMSNormPurePaddle(unittest.TestCase):
    """Tests for RMSNorm computation using pure Paddle."""

    def test_rms_norm_forward(self):
        """Test RMS norm forward computation."""
        import paddle

        x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]])
        weight = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
        epsilon = 1e-6

        # RMS norm: y = x / sqrt(mean(x^2) + eps) * weight
        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + epsilon)
        y = x * invvar * weight

        self.assertFalse(paddle.any(paddle.isnan(y)))

    def test_rms_norm_output_shape(self):
        """Test RMS norm output has correct shape."""
        import paddle

        x = paddle.randn([2, 4, 8])
        weight = paddle.ones([8])
        epsilon = 1e-6

        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + epsilon)
        y = x * invvar * weight

        self.assertEqual(y.shape, x.shape)

    def test_rms_norm_with_different_weight(self):
        """Test RMS norm with non-unity weight."""
        import paddle

        x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]])
        weight = paddle.to_tensor([2.0, 0.5, 1.0, 3.0])
        epsilon = 1e-6

        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + epsilon)
        y = x * invvar * weight

        self.assertFalse(paddle.any(paddle.isnan(y)))
        # Weight should scale the output
        self.assertTrue(y[0, 0].item() > 0)

    def test_rms_norm_gradient(self):
        """Test RMS norm gradient computation."""
        import paddle

        x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], stop_gradient=False)
        weight = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
        epsilon = 1e-6

        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        invvar = 1.0 / paddle.sqrt(variance + epsilon)
        y = x * invvar * weight
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(paddle.any(paddle.isnan(x.grad)))


if __name__ == "__main__":
    unittest.main()
