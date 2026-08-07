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


class TestBiasSwiGLUFunction(unittest.TestCase):
    """Tests for BiasSwiGLUFunction PyLayer."""

    def test_bias_swiglu_function_forward(self):
        """Test BiasSwiGLUFunction forward."""
        from paddleformers.fleet.fusions.fused_bias_swiglu import BiasSwiGLUFunction

        inp = paddle.randn([4, 16])
        bias = paddle.randn([16])
        result = BiasSwiGLUFunction.apply(inp, bias, False, False)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_swiglu_function_forward_2d(self):
        """Test BiasSwiGLUFunction forward with 2D input."""
        from paddleformers.fleet.fusions.fused_bias_swiglu import BiasSwiGLUFunction

        inp = paddle.randn([8, 32])
        bias = paddle.randn([32])
        result = BiasSwiGLUFunction.apply(inp, bias, False, False)
        self.assertEqual(result.shape, [8, 16])


class TestSwiGLUFunction(unittest.TestCase):
    """Tests for SwiGLUFunction PyLayer."""

    def test_swiglu_function_forward(self):
        """Test SwiGLUFunction forward."""
        from paddleformers.fleet.fusions.fused_bias_swiglu import SwiGLUFunction

        inp = paddle.randn([4, 16])
        result = SwiGLUFunction.apply(inp, False, False)
        self.assertEqual(result.shape, [4, 8])

    def test_swiglu_function_forward_different_sizes(self):
        """Test SwiGLUFunction with different input sizes."""
        from paddleformers.fleet.fusions.fused_bias_swiglu import SwiGLUFunction

        for hidden in [16, 32, 64]:
            inp = paddle.randn([4, hidden])
            result = SwiGLUFunction.apply(inp, False, False)
            self.assertEqual(result.shape, [4, hidden // 2])


class TestWeightedSwiGLUFunction(unittest.TestCase):
    """Tests for WeightedSwiGLUFunction PyLayer."""

    def test_weighted_swiglu_function_forward(self):
        """Test WeightedSwiGLUFunction forward."""
        from paddleformers.fleet.fusions.fused_bias_swiglu import (
            WeightedSwiGLUFunction,
        )

        inp = paddle.randn([4, 16])
        weights = paddle.randn([4, 1])
        result = WeightedSwiGLUFunction.apply(inp, weights, False)
        self.assertEqual(result.shape, [4, 8])


if __name__ == "__main__":
    unittest.main()
