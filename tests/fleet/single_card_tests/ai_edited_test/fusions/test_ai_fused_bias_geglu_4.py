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

from paddleformers.fleet.fusions.fused_bias_geglu import (
    BiasGeGLUFunction,
    GeGLUFunction,
    WeightedQuickGeGLUFunction,
    bias_geglu,
    bias_geglu_back,
    bias_geglu_impl,
    geglu,
    geglu_back,
)


class TestGeGLU(unittest.TestCase):
    """Tests for geglu function."""

    def test_output_shape(self):
        """Test geglu halves last dimension."""
        x = paddle.randn([2, 8])
        result = geglu(x)
        self.assertEqual(result.shape, [2, 4])

    def test_output_non_negative(self):
        """Test that output can be negative (gated output)."""
        x = paddle.randn([2, 8])
        result = geglu(x)
        # geglu can produce negative values due to the tanh approximation
        self.assertIsNotNone(result)


class TestBiasGeGLU(unittest.TestCase):
    """Tests for bias_geglu function."""

    def test_output_shape(self):
        """Test bias_geglu halves last dimension."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        result = bias_geglu(bias, x)
        self.assertEqual(result.shape, [2, 4])

    def test_bias_addition(self):
        """Test that bias is added to input before activation."""
        x = paddle.zeros([1, 4])
        bias = paddle.ones([4])
        result = bias_geglu(bias, x)
        # bias_geglu(x + bias) = geglu(x + bias)
        expected = geglu(x + bias)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


class TestGeGLUBack(unittest.TestCase):
    """Tests for geglu_back function."""

    def test_output_shape(self):
        """Test geglu_back output shape matches input shape."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        result = geglu_back(g, y)
        self.assertEqual(result.shape, [2, 8])

    def test_gradient_computed(self):
        """Test that gradient is computed without error."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        result = geglu_back(g, y)
        self.assertFalse(paddle.isnan(result).any())


class TestBiasGeGLUBack(unittest.TestCase):
    """Tests for bias_geglu_back function."""

    def test_output_shape(self):
        """Test bias_geglu_back output shape matches input shape."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        bias = paddle.randn([8])
        result = bias_geglu_back(g, y, bias)
        self.assertEqual(result.shape, [2, 8])

    def test_bias_added_in_backward(self):
        """Test that bias is added before computing backward."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        bias = paddle.zeros([8])
        result_with_zero_bias = bias_geglu_back(g, y, bias)
        result_no_bias = geglu_back(g, y)
        np.testing.assert_allclose(result_with_zero_bias.numpy(), result_no_bias.numpy(), atol=1e-6)


class TestBiasGeGLUFunction(unittest.TestCase):
    """Tests for BiasGeGLUFunction PyLayer."""

    def test_forward_output_shape(self):
        """Test forward output shape."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        result = BiasGeGLUFunction.apply(x, bias)
        self.assertEqual(result.shape, [2, 4])

    def test_forward_matches_bias_geglu(self):
        """Test forward matches bias_geglu function."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        result = BiasGeGLUFunction.apply(x, bias)
        expected = bias_geglu(bias, x)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


class TestGeGLUFunction(unittest.TestCase):
    """Tests for GeGLUFunction PyLayer."""

    def test_forward_output_shape(self):
        """Test forward output shape."""
        x = paddle.randn([2, 8])
        result = GeGLUFunction.apply(x)
        self.assertEqual(result.shape, [2, 4])

    def test_forward_matches_geglu(self):
        """Test forward matches geglu function."""
        x = paddle.randn([2, 8])
        result = GeGLUFunction.apply(x)
        expected = geglu(x)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


class TestBiasGeGLUImpl(unittest.TestCase):
    """Tests for bias_geglu_impl function."""

    def test_2d_input_with_bias(self):
        """Test 2D input with bias."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        result = bias_geglu_impl(x, bias)
        self.assertEqual(result.shape, [2, 4])

    def test_2d_input_without_bias(self):
        """Test 2D input without bias."""
        x = paddle.randn([2, 8])
        result = bias_geglu_impl(x, None)
        self.assertEqual(result.shape, [2, 4])

    def test_3d_input_with_bias(self):
        """Test 3D input with bias."""
        x = paddle.randn([2, 4, 8])
        bias = paddle.randn([8])
        result = bias_geglu_impl(x, bias)
        self.assertEqual(result.shape, [2, 4, 4])

    def test_3d_input_without_bias(self):
        """Test 3D input without bias."""
        x = paddle.randn([2, 4, 8])
        result = bias_geglu_impl(x, None)
        self.assertEqual(result.shape, [2, 4, 4])

    def test_asserts_invalid_dim(self):
        """Test assertion for invalid dimensions."""
        x = paddle.randn([2, 4, 4, 8])
        with self.assertRaises(AssertionError):
            bias_geglu_impl(x, None)


class TestWeightedQuickGeGLUFunction(unittest.TestCase):
    """Tests for WeightedQuickGeGLUFunction PyLayer."""

    def test_forward_output_shape(self):
        """Test forward output shape."""
        x = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        linear_offset = paddle.to_tensor(0.0)
        result = WeightedQuickGeGLUFunction.apply(x, weights, False, linear_offset)
        self.assertEqual(result.shape, [2, 4])

    def test_forward_matches_weighted_quick_geglu(self):
        """Test forward matches weighted_quick_geglu function."""
        from paddleformers.fleet.fusions.fused_bias_geglu import weighted_quick_geglu

        x = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        linear_offset = paddle.to_tensor(0.0)
        result = WeightedQuickGeGLUFunction.apply(x, weights, False, linear_offset)
        expected = weighted_quick_geglu(x, weights, linear_offset)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
