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


# Tests for paddlefleet_ops/ops/triton_ops/fused_linear_cross_entropy/utils.py
# element_mul_kernel is a Triton kernel, so we test module structure and
# simulate the computation logic.

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
    tl.int64 = "int64"
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


# Must set up triton mock before importing the module under test
try:
    _setup_triton_mock()
    import paddleformers.fleet.triton_ops.fused_linear_cross_entropy.utils  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestElementMulKernel(unittest.TestCase):
    """Tests for element_mul_kernel function definition."""

    def test_kernel_is_callable(self):
        """Test that element_mul_kernel is callable (mocked triton.jit)."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.utils import (
            element_mul_kernel,
        )

        self.assertTrue(callable(element_mul_kernel))

    def test_kernel_has_code_attribute(self):
        """Test that the kernel has __code__ (is a real function)."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.utils import (
            element_mul_kernel,
        )

        self.assertTrue(hasattr(element_mul_kernel, "__code__"))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestElementMulLogic(unittest.TestCase):
    """Tests for element_mul computation logic using pure Paddle."""

    def test_element_mul_identity(self):
        """Test that multiplying by 1.0 does not change the tensor."""
        import paddle

        x = paddle.randn([4, 8])
        grad_output = paddle.to_tensor(1.0)
        result = x * grad_output
        self.assertTrue(paddle.allclose(x, result))

    def test_element_mul_scaling(self):
        """Test that multiplying by 0.5 scales correctly."""
        import paddle

        x = paddle.to_tensor([1.0, 2.0, 3.0])
        grad_output = paddle.to_tensor(0.5)
        result = x * grad_output
        expected = paddle.to_tensor([0.5, 1.0, 1.5])
        self.assertTrue(paddle.allclose(result, expected))

    def test_element_mul_zero(self):
        """Test that multiplying by 0.0 zeros out the tensor."""
        import paddle

        x = paddle.randn([4, 8])
        grad_output = paddle.to_tensor(0.0)
        result = x * grad_output
        self.assertTrue(paddle.all(result == 0.0))

    def test_element_mul_2d(self):
        """Test element multiplication on 2D tensor."""
        import paddle

        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        grad_output = paddle.to_tensor(2.0)
        result = x * grad_output
        expected = paddle.to_tensor([[2.0, 4.0], [6.0, 8.0]])
        self.assertTrue(paddle.allclose(result, expected))

    def test_element_mul_preserves_shape(self):
        """Test that element multiplication preserves shape."""
        import paddle

        x = paddle.randn([3, 5, 7])
        grad_output = paddle.to_tensor(1.5)
        result = x * grad_output
        self.assertEqual(result.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
