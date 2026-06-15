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


import unittest

import numpy as np
import paddle


class TestGeGLUCorrectnessExtra(unittest.TestCase):
    """Additional correctness tests for GEGLU functions."""

    def setUp(self):
        paddle.seed(42)
        self.y = paddle.randn([4, 16], dtype=paddle.float32)
        self.g = paddle.randn([4, 8], dtype=paddle.float32)

    def test_geglu_zero_input(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu

        y_zero = paddle.zeros([4, 16], dtype=paddle.float32)
        out = geglu(y_zero)
        np.testing.assert_allclose(out.numpy(), 0.0, atol=1e-6)

    def test_geglu_back_correctness(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu_back

        out = geglu_back(self.g, self.y)
        y_1, y_2 = paddle.chunk(self.y, 2, -1)
        tanh_out = paddle.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1))
        ff = 0.5 * y_1 * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * y_1 * y_1)) + 0.5 * (1 + tanh_out)
        expected = paddle.concat(((self.g * y_2) * ff, self.g * (y_1 * 0.5 * (1.0 + tanh_out))), -1)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_bias_geglu_back_correctness(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_back

        bias = paddle.randn([16], dtype=paddle.float32)
        out = bias_geglu_back(self.g, self.y, bias)
        y_shifted = self.y + bias
        y_1, y_2 = paddle.chunk(y_shifted, 2, -1)
        tanh_out = paddle.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1))
        ff = 0.5 * y_1 * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * y_1 * y_1)) + 0.5 * (1 + tanh_out)
        expected = paddle.concat(((self.g * y_2) * ff, self.g * (y_1 * 0.5 * (1.0 + tanh_out))), -1)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_geglu_back_zero_grad(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu_back

        g_zero = paddle.zeros([4, 8], dtype=paddle.float32)
        out = geglu_back(g_zero, self.y)
        np.testing.assert_allclose(out.numpy(), 0.0, atol=1e-6)


class TestBiasGeGLUFunctionAutograd(unittest.TestCase):
    """Tests for BiasGeGLUFunction and GeGLUFunction autograd classes."""

    def setUp(self):
        paddle.seed(42)

    def test_bias_geglu_function_forward_backward(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import BiasGeGLUFunction

        input_t = paddle.randn([4, 16], dtype=paddle.float32)
        bias = paddle.randn([16], dtype=paddle.float32)
        input_t.stop_gradient = False
        bias.stop_gradient = False

        out = BiasGeGLUFunction.apply(input_t, bias)
        self.assertEqual(out.shape, [4, 8])

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(input_t.grad)
        self.assertIsNotNone(bias.grad)

    def test_geglu_function_forward_backward(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import GeGLUFunction

        input_t = paddle.randn([4, 16], dtype=paddle.float32)
        input_t.stop_gradient = False

        out = GeGLUFunction.apply(input_t)
        self.assertEqual(out.shape, [4, 8])

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(input_t.grad)
        self.assertEqual(input_t.grad.shape, [4, 16])

    def test_bias_geglu_impl_assert_invalid_dim(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        input_1d = paddle.randn([16], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            bias_geglu_impl(input_1d, None)

    def test_bias_geglu_impl_3d_with_bias(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        input_3d = paddle.randn([2, 4, 16], dtype=paddle.float32)
        bias = paddle.randn([16], dtype=paddle.float32)
        out = bias_geglu_impl(input_3d, bias)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_bias_geglu_impl_3d_no_bias(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        input_3d = paddle.randn([2, 4, 16], dtype=paddle.float32)
        out = bias_geglu_impl(input_3d, None)
        self.assertEqual(out.shape, [2, 4, 8])


class TestQuickGeGLUCorrectnessExtra(unittest.TestCase):
    """Additional tests for Quick-GEGLU functions."""

    def setUp(self):
        paddle.seed(42)
        self.y = paddle.randn([4, 16], dtype=paddle.float32)

    def test_quick_gelu_zero_input(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_gelu

        y_zero = paddle.zeros([4, 16], dtype=paddle.float32)
        out = quick_gelu(y_zero)
        np.testing.assert_allclose(out.numpy(), 0.0, atol=1e-6)

    def test_quick_geglu_negative_offset(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_geglu, quick_gelu

        out = quick_geglu(self.y, linear_offset=-0.5)
        y_1, y_2 = paddle.chunk(self.y, 2, -1)
        expected = quick_gelu(y_1) * (y_2 - 0.5)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_quick_geglu_back_correctness(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_geglu_back

        g = paddle.randn([4, 8], dtype=paddle.float32)
        out = quick_geglu_back(g, self.y, linear_offset=0.0)
        y_1, y_2 = paddle.chunk(self.y, 2, -1)
        sigmoid_out = paddle.sigmoid(1.702 * y_1)
        dy_1 = g * sigmoid_out * (1 + 1.702 * y_1 * (1 - sigmoid_out)) * (y_2 + 0.0)
        dy_2 = g * y_1 * sigmoid_out
        expected = paddle.concat((dy_1, dy_2), -1)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_quick_geglu_back_negative_offset(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_geglu_back

        g = paddle.randn([4, 8], dtype=paddle.float32)
        out = quick_geglu_back(g, self.y, linear_offset=-1.0)
        self.assertEqual(out.shape, [4, 16])


class TestWeightedQuickGeGLUExtra(unittest.TestCase):
    """Additional tests for weighted Quick-GEGLU functions."""

    def setUp(self):
        paddle.seed(42)
        self.y = paddle.randn([4, 16], dtype=paddle.float32)
        self.weights = paddle.randn([4, 1], dtype=paddle.float32)

    def test_weighted_quick_geglu_correctness(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            quick_geglu,
            weighted_quick_geglu,
        )

        out = weighted_quick_geglu(self.y, self.weights, linear_offset=0.0)
        expected = quick_geglu(self.y, 0.0) * self.weights
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_weighted_quick_geglu_back_correctness(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            quick_geglu_back,
            weighted_quick_geglu_back,
        )

        g = paddle.randn([4, 8], dtype=paddle.float32)
        input_grad, weights_grad = weighted_quick_geglu_back(g, self.y, self.weights, linear_offset=0.0)
        # Check input_grad matches chain rule
        manual_input_grad = quick_geglu_back(g * self.weights, self.y, 0.0)
        np.testing.assert_allclose(input_grad.numpy(), manual_input_grad.numpy(), rtol=1e-5)


class TestWeightedBiasQuickGeGLUExtra(unittest.TestCase):
    """Additional tests for weighted bias Quick-GEGLU functions."""

    def setUp(self):
        paddle.seed(42)
        self.y = paddle.randn([4, 16], dtype=paddle.float32)
        self.bias = paddle.randn([4, 16], dtype=paddle.float32)
        self.weights = paddle.randn([4, 1], dtype=paddle.float32)

    def test_weighted_bias_quick_geglu_back_correctness(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            weighted_bias_quick_geglu_back,
        )

        g = paddle.randn([4, 8], dtype=paddle.float32)
        input_grad, bias_grad, weights_grad = weighted_bias_quick_geglu_back(
            g, self.y, self.bias, self.weights, linear_offset=0.0
        )
        # bias_grad should equal input_grad
        np.testing.assert_allclose(input_grad.numpy(), bias_grad.numpy(), rtol=1e-5)

    def test_weighted_bias_quick_geglu_impl_assert_invalid_dim(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            weighted_bias_quick_geglu_impl,
        )

        input_1d = paddle.randn([16], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            weighted_bias_quick_geglu_impl(input_1d, None, self.weights)


_SKIP_GEGLU_AUTOGRAD = not paddle.is_compiled_with_cuda()


@unittest.skipIf(_SKIP_GEGLU_AUTOGRAD, "WeightedQuickGeGLU autograd tests require CUDA")
class TestWeightedQuickGeGLUFunctionAutograd(unittest.TestCase):
    """Tests for WeightedQuickGeGLUFunction and WeightedBiasQuickGeGLUFunction."""

    def setUp(self):
        paddle.seed(42)

    def test_weighted_bias_quick_geglu_impl_3d_with_bias(self):
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            weighted_bias_quick_geglu_impl,
        )

        y_3d = paddle.randn([2, 4, 16], dtype=paddle.float32)
        bias = paddle.randn([8, 16], dtype=paddle.float32)
        weights = paddle.randn([8, 1], dtype=paddle.float32)
        out = weighted_bias_quick_geglu_impl(y_3d, bias, weights)
        self.assertEqual(out.shape, [2, 4, 8])
