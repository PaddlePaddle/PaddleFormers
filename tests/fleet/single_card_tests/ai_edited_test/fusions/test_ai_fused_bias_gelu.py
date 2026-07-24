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

import unittest

import paddle


class TestBiasGeluFunction(unittest.TestCase):
    """Tests for bias_gelu and GeLUFunction."""

    def test_bias_gelu_forward(self):
        """Test bias_gelu forward computation."""
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu

        bias = paddle.randn([8])
        y = paddle.randn([4, 8])
        result = bias_gelu(bias, y)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_gelu_approximation(self):
        """Test bias_gelu approximates GELU."""
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu

        bias = paddle.zeros([4])
        y = paddle.zeros([4])
        result = bias_gelu(bias, y)
        # GELU(0) ~ 0
        self.assertTrue(
            paddle.allclose(result, paddle.zeros_like(result), atol=1e-5)
        )

    def test_bias_gelu_back_exists(self):
        """Test bias_gelu_back function exists."""
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu_back

        self.assertTrue(callable(bias_gelu_back))

    def test_gelu_function_forward(self):
        """Test GeLUFunction forward."""
        from paddleformers.fleet.fusions.fused_bias_gelu import GeLUFunction

        bias = paddle.randn([8])
        y = paddle.randn([4, 8])
        result = GeLUFunction.apply(y, bias)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_gelu_impl_exists(self):
        """Test bias_gelu_impl exists."""
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu_impl

        self.assertTrue(callable(bias_gelu_impl))

    def test_bias_gelu_impl_2d(self):
        """Test bias_gelu_impl with 2D input."""
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu_impl

        y = paddle.randn([4, 8])
        bias = paddle.randn([8])
        result = bias_gelu_impl(y, bias)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_gelu_impl_3d(self):
        """Test bias_gelu_impl with 3D input."""
        from paddleformers.fleet.fusions.fused_bias_gelu import bias_gelu_impl

        y = paddle.randn([2, 4, 8])
        bias = paddle.randn([8])
        result = bias_gelu_impl(y, bias)
        self.assertEqual(result.shape, [2, 4, 8])


if __name__ == "__main__":
    unittest.main()
