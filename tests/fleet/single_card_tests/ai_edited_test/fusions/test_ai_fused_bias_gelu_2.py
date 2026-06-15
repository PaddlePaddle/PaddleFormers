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


class TestBiasGeluCorrectnessExtra(unittest.TestCase):
    """Additional tests for bias_gelu and bias_gelu_back direct functions."""

    def setUp(self):
        paddle.seed(42)

    def test_bias_gelu_large_values(self):
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu

        bias = paddle.full([4], 100.0, dtype=paddle.float32)
        y = paddle.full([2, 4], 50.0, dtype=paddle.float32)
        out = bias_gelu(bias, y)
        # For large positive inputs, gelu(x) ~ x
        expected_sum = 150.0 * 8
        np.testing.assert_allclose(out.numpy().sum(), expected_sum, rtol=0.1)

    def test_bias_gelu_back_ones_grad(self):
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu_back

        g = paddle.ones([4, 8], dtype=paddle.float32)
        bias = paddle.randn([8], dtype=paddle.float32)
        y = paddle.randn([4, 8], dtype=paddle.float32)
        out = bias_gelu_back(g, bias, y)
        # Gradient with ones should give the gelu derivative values
        x = bias + y
        tanh_out = paddle.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
        expected = ff
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_bias_gelu_single_element(self):
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu

        bias = paddle.randn([1], dtype=paddle.float32)
        y = paddle.randn([1, 1], dtype=paddle.float32)
        out = bias_gelu(bias, y)
        self.assertEqual(out.shape, [1, 1])


class TestGeLUFunctionAutograd(unittest.TestCase):
    """Tests for GeLUFunction autograd backward path."""

    def setUp(self):
        paddle.seed(42)

    def test_gelu_function_forward_backward(self):
        from paddleformers.fleet.fusions.fused_bias_gelu import GeLUFunction

        input_t = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([4, 8], dtype=paddle.float32)
        input_t.stop_gradient = False
        bias.stop_gradient = False

        out = GeLUFunction.apply(input_t, bias)
        self.assertEqual(out.shape, [4, 8])

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(input_t.grad)
        self.assertIsNotNone(bias.grad)
        self.assertEqual(input_t.grad.shape, [4, 8])
        self.assertEqual(bias.grad.shape, [4, 8])

    def test_gelu_function_backward_matches_manual(self):
        from paddleformers.fleet.fusions.fused_bias_gelu import (
            GeLUFunction,
            bias_gelu_back,
        )

        input_t = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([4, 8], dtype=paddle.float32)
        input_t.stop_gradient = False
        bias.stop_gradient = False

        out = GeLUFunction.apply(input_t, bias)
        loss = out.sum()
        loss.backward()

        # Manual gradient computation
        grad_output = paddle.ones([4, 8], dtype=paddle.float32)
        manual_grad = bias_gelu_back(grad_output, bias, input_t)
        np.testing.assert_allclose(input_t.grad.numpy(), manual_grad.numpy(), rtol=1e-5)

    def test_gelu_function_backward_returns_two(self):
        """Test that backward returns gradients for both input and bias."""
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu_back

        g = paddle.ones([4, 8], dtype=paddle.float32)
        bias = paddle.randn([4, 8], dtype=paddle.float32)
        y = paddle.randn([4, 8], dtype=paddle.float32)
        out = bias_gelu_back(g, bias, y)
        # bias_gelu_back returns ff * g, which should equal the gradient for both
        self.assertEqual(out.shape, [4, 8])

    def test_gelu_function_apply_is_classmethod(self):
        from paddleformers.fleet.fusions.fused_bias_gelu import GeLUFunction

        input_t = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([4, 8], dtype=paddle.float32)
        # Calling apply as a classmethod should work
        out = GeLUFunction.apply(input_t, bias)
        self.assertEqual(out.shape, [4, 8])

    def test_gelu_function_1d_bias_broadcast(self):
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu

        # bias is 1D, y is 2D - broadcasting
        bias = paddle.randn([16], dtype=paddle.float32)
        y = paddle.randn([8, 16], dtype=paddle.float32)
        out = bias_gelu(bias, y)
        self.assertEqual(out.shape, [8, 16])

    def test_gelu_function_different_shapes(self):
        """Test GeLUFunction with various input shapes."""
        from paddleformers.fleet.fusions.fused_bias_gelu import GeLUFunction

        # 2D input
        input_2d = paddle.randn([8, 16], dtype=paddle.float32)
        bias_2d = paddle.randn([8, 16], dtype=paddle.float32)
        out = GeLUFunction.apply(input_2d, bias_2d)
        self.assertEqual(out.shape, [8, 16])

        # Larger batch
        input_large = paddle.randn([32, 64], dtype=paddle.float32)
        bias_large = paddle.randn([32, 64], dtype=paddle.float32)
        out = GeLUFunction.apply(input_large, bias_large)
        self.assertEqual(out.shape, [32, 64])


if __name__ == "__main__":
    unittest.main()
