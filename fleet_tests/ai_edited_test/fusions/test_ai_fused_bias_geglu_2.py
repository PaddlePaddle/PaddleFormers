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


class TestBiasGeGLUFunction(unittest.TestCase):
    """Tests for BiasGeGLUFunction and GeGLUFunction PyLayers."""

    def test_bias_geglu_function_forward(self):
        """Test BiasGeGLUFunction forward."""
        from paddleformers.fleet.fusions.fused_bias_geglu import BiasGeGLUFunction

        inp = paddle.randn([4, 16])
        bias = paddle.randn([16])
        result = BiasGeGLUFunction.apply(inp, bias)
        self.assertEqual(result.shape, [4, 8])

    def test_geglu_function_forward(self):
        """Test GeGLUFunction forward without bias."""
        from paddleformers.fleet.fusions.fused_bias_geglu import GeGLUFunction

        inp = paddle.randn([4, 16])
        result = GeGLUFunction.apply(inp)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_geglu_function_3d_input(self):
        """Test BiasGeGLUFunction with reshaped 3D input."""
        from paddleformers.fleet.fusions.fused_bias_geglu import BiasGeGLUFunction

        inp = paddle.randn([2, 4, 16]).view(-1, 16)
        bias = paddle.randn([16])
        result = BiasGeGLUFunction.apply(inp, bias)
        self.assertEqual(result.shape, [8, 8])


class TestWeightedQuickGeGLUFunction(unittest.TestCase):
    """Tests for WeightedQuickGeGLUFunction PyLayer."""

    def test_forward(self):
        """Test WeightedQuickGeGLUFunction forward."""
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            WeightedQuickGeGLUFunction,
        )

        inp = paddle.randn([4, 16])
        weights = paddle.randn([4, 1])
        linear_offset = paddle.to_tensor(0.0)
        result = WeightedQuickGeGLUFunction.apply(
            inp, weights, False, linear_offset
        )
        self.assertEqual(result.shape, [4, 8])

    def test_forward_with_offset(self):
        """Test WeightedQuickGeGLUFunction forward with linear offset."""
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            WeightedQuickGeGLUFunction,
        )

        inp = paddle.randn([4, 16])
        weights = paddle.randn([4, 1])
        linear_offset = paddle.to_tensor(0.5)
        result = WeightedQuickGeGLUFunction.apply(
            inp, weights, False, linear_offset
        )
        self.assertEqual(result.shape, [4, 8])


class TestWeightedBiasQuickGeGLUFunction(unittest.TestCase):
    """Tests for WeightedBiasQuickGeGLUFunction PyLayer."""

    def test_forward(self):
        """Test WeightedBiasQuickGeGLUFunction forward."""
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            WeightedBiasQuickGeGLUFunction,
        )

        inp = paddle.randn([4, 16])
        bias = paddle.randn([16])
        weights = paddle.randn([4, 1])
        linear_offset = paddle.to_tensor(0.0)
        result = WeightedBiasQuickGeGLUFunction.apply(
            inp, bias, weights, False, linear_offset
        )
        self.assertEqual(result.shape, [4, 8])


class TestQuickGelu(unittest.TestCase):
    """Tests for quick_gelu function."""

    def test_quick_gelu_zero(self):
        """Test quick_gelu at zero."""
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_gelu

        y = paddle.zeros([4])
        result = quick_gelu(y)
        # sigmoid(0) = 0.5, so quick_gelu(0) = 0 * 0.5 = 0
        self.assertTrue(
            paddle.allclose(result, paddle.zeros_like(result), atol=1e-6)
        )

    def test_quick_gelu_positive(self):
        """Test quick_gelu with positive values."""
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_gelu

        y = paddle.ones([4]) * 2.0
        result = quick_gelu(y)
        # Should be positive
        self.assertTrue(paddle.all(result > 0))


if __name__ == "__main__":
    unittest.main()
