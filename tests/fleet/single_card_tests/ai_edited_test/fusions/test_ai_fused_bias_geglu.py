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


class TestGeGLUFunctions(unittest.TestCase):
    """Tests for GEGLU activation functions."""

    def test_geglu_forward(self):
        """Test geglu forward computation."""
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu

        y = paddle.randn([4, 16])
        result = geglu(y)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_geglu_forward(self):
        """Test bias_geglu forward computation."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu

        y = paddle.randn([4, 16])
        bias = paddle.randn([16])
        result = bias_geglu(bias, y)
        self.assertEqual(result.shape, [4, 8])

    def test_geglu_back_exists(self):
        """Test geglu_back function exists."""
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu_back

        self.assertTrue(callable(geglu_back))

    def test_bias_geglu_back_exists(self):
        """Test bias_geglu_back function exists."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_back

        self.assertTrue(callable(bias_geglu_back))

    def test_bias_geglu_impl_2d_with_bias(self):
        """Test bias_geglu_impl with 2D input and bias."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        y = paddle.randn([4, 16])
        bias = paddle.randn([16])
        result = bias_geglu_impl(y, bias)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_geglu_impl_2d_without_bias(self):
        """Test bias_geglu_impl with 2D input without bias."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        y = paddle.randn([4, 16])
        result = bias_geglu_impl(y, None)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_geglu_impl_3d_with_bias(self):
        """Test bias_geglu_impl with 3D input."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        y = paddle.randn([2, 4, 16])
        bias = paddle.randn([16])
        result = bias_geglu_impl(y, bias)
        self.assertEqual(result.shape, [2, 4, 8])


class TestQuickGeGLUFunctions(unittest.TestCase):
    """Tests for Quick GEGLU activation functions."""

    def test_quick_gelu(self):
        """Test quick_gelu function."""
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_gelu

        y = paddle.randn([4, 8])
        result = quick_gelu(y)
        self.assertEqual(result.shape, [4, 8])

    def test_quick_geglu(self):
        """Test quick_geglu function."""
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_geglu

        y = paddle.randn([4, 16])
        result = quick_geglu(y)
        self.assertEqual(result.shape, [4, 8])

    def test_quick_geglu_with_offset(self):
        """Test quick_geglu with linear offset."""
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_geglu

        y = paddle.randn([4, 16])
        result = quick_geglu(y, linear_offset=0.5)
        self.assertEqual(result.shape, [4, 8])

    def test_weighted_quick_geglu(self):
        """Test weighted_quick_geglu function."""
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            weighted_quick_geglu,
        )

        y = paddle.randn([4, 16])
        weights = paddle.randn([4, 1])
        result = weighted_quick_geglu(y, weights)
        self.assertEqual(result.shape, [4, 8])

    def test_quick_geglu_back_exists(self):
        """Test quick_geglu_back function exists."""
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            quick_geglu_back,
        )

        self.assertTrue(callable(quick_geglu_back))

    def test_weighted_bias_quick_geglu_impl_no_bias(self):
        """Test weighted_bias_quick_geglu_impl without bias."""
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            weighted_bias_quick_geglu_impl,
        )

        y = paddle.randn([4, 16])
        weights = paddle.randn([4, 1])
        result = weighted_bias_quick_geglu_impl(y, None, weights)
        self.assertEqual(result.shape, [4, 8])


if __name__ == "__main__":
    unittest.main()
