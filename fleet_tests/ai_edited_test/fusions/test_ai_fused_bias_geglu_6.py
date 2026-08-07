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
from unittest.mock import patch

import numpy as np
import paddle

from paddleformers.fleet.fusions.fused_bias_geglu import (
    WeightedBiasQuickGeGLUFunction,
    quick_geglu,
    quick_geglu_back,
    quick_gelu,
    weighted_bias_quick_geglu,
    weighted_bias_quick_geglu_back,
    weighted_bias_quick_geglu_impl,
    weighted_quick_geglu,
    weighted_quick_geglu_back,
)


class TestQuickGelu(unittest.TestCase):
    """Tests for quick_gelu function."""

    def test_output_shape(self):
        """Test quick_gelu preserves shape."""
        x = paddle.randn([2, 8])
        result = quick_gelu(x)
        self.assertEqual(result.shape, [2, 8])

    def test_output_positive_for_positive_input(self):
        """Test quick_gelu is positive for positive input."""
        x = paddle.ones([4])
        result = quick_gelu(x)
        self.assertTrue((result > 0).all())

    def test_output_range(self):
        """Test quick_gelu output is finite."""
        x = paddle.randn([100]) * 5
        result = quick_gelu(x)
        # quick_gelu = x * sigmoid(1.702 * x) can be negative for negative x
        self.assertTrue(paddle.isfinite(result).all())

    def test_at_zero(self):
        """Test quick_gelu at zero."""
        x = paddle.zeros([4])
        result = quick_gelu(x)
        np.testing.assert_allclose(result.numpy(), np.zeros([4]), atol=1e-6)

    def test_dtype_preserved(self):
        """Test dtype is preserved."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        result = quick_gelu(x)
        self.assertEqual(result.dtype, paddle.float32)


class TestQuickGeGLU(unittest.TestCase):
    """Tests for quick_geglu function."""

    def test_output_shape(self):
        """Test quick_geglu halves last dimension."""
        x = paddle.randn([2, 8])
        result = quick_geglu(x)
        self.assertEqual(result.shape, [2, 4])

    def test_default_offset(self):
        """Test quick_geglu with default offset."""
        x = paddle.randn([2, 8])
        result = quick_geglu(x)
        self.assertEqual(result.shape, [2, 4])

    def test_with_linear_offset(self):
        """Test quick_geglu with linear offset."""
        x = paddle.randn([2, 8])
        result = quick_geglu(x, linear_offset=1.0)
        self.assertEqual(result.shape, [2, 4])

    def test_matches_manual_computation(self):
        """Test quick_geglu matches manual quick_gelu computation."""
        x = paddle.randn([2, 8])
        result = quick_geglu(x, linear_offset=0.0)
        y_1, y_2 = paddle.chunk(x, 2, axis=-1)
        expected = quick_gelu(y_1) * (y_2 + 0.0)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


class TestWeightedQuickGeGLU(unittest.TestCase):
    """Tests for weighted_quick_geglu function."""

    def test_output_shape(self):
        """Test weighted_quick_geglu output shape."""
        x = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        result = weighted_quick_geglu(x, weights)
        self.assertEqual(result.shape, [2, 4])

    def test_with_linear_offset(self):
        """Test weighted_quick_geglu with offset."""
        x = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        result = weighted_quick_geglu(x, weights, linear_offset=0.5)
        self.assertEqual(result.shape, [2, 4])

    def test_dtype_preserved(self):
        """Test dtype is preserved."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        weights = paddle.randn([2, 1], dtype=paddle.float32)
        result = weighted_quick_geglu(x, weights)
        self.assertEqual(result.dtype, paddle.float32)


class TestQuickGeGLUBack(unittest.TestCase):
    """Tests for quick_geglu_back function."""

    def test_output_shape(self):
        """Test quick_geglu_back output shape matches input."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        result = quick_geglu_back(g, y)
        self.assertEqual(result.shape, [2, 8])

    def test_with_linear_offset(self):
        """Test quick_geglu_back with offset."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        result = quick_geglu_back(g, y, linear_offset=0.5)
        self.assertEqual(result.shape, [2, 8])

    def test_gradient_finite(self):
        """Test that gradients are finite."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        result = quick_geglu_back(g, y)
        self.assertFalse(paddle.isnan(result).any())
        self.assertFalse(paddle.isinf(result).any())


class TestWeightedQuickGeGLUBack(unittest.TestCase):
    """Tests for weighted_quick_geglu_back function."""

    def test_output_shapes(self):
        """Test weighted_quick_geglu_back returns correct shapes."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        input_grad, weights_grad = weighted_quick_geglu_back(g, y, weights)
        self.assertEqual(input_grad.shape, [2, 8])
        self.assertEqual(weights_grad.shape, [2, 1])

    def test_with_linear_offset(self):
        """Test weighted_quick_geglu_back with offset."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        input_grad, weights_grad = weighted_quick_geglu_back(
            g, y, weights, linear_offset=0.5
        )
        self.assertEqual(input_grad.shape, [2, 8])
        self.assertEqual(weights_grad.shape, [2, 1])

    def test_dtypes_match_inputs(self):
        """Test gradient dtypes match input dtypes."""
        g = paddle.randn([2, 4], dtype=paddle.float32)
        y = paddle.randn([2, 8], dtype=paddle.float32)
        weights = paddle.randn([2, 1], dtype=paddle.float32)
        input_grad, weights_grad = weighted_quick_geglu_back(g, y, weights)
        self.assertEqual(input_grad.dtype, paddle.float32)
        self.assertEqual(weights_grad.dtype, paddle.float32)


class TestWeightedBiasQuickGeGLU(unittest.TestCase):
    """Tests for weighted_bias_quick_geglu function."""

    def test_output_shape(self):
        """Test weighted_bias_quick_geglu output shape."""
        y = paddle.randn([2, 8])
        bias = paddle.randn([8])
        weights = paddle.randn([2, 1])
        result = weighted_bias_quick_geglu(y, bias, weights)
        self.assertEqual(result.shape, [2, 4])

    def test_with_linear_offset(self):
        """Test weighted_bias_quick_geglu with offset."""
        y = paddle.randn([2, 8])
        bias = paddle.randn([8])
        weights = paddle.randn([2, 1])
        result = weighted_bias_quick_geglu(y, bias, weights, linear_offset=1.0)
        self.assertEqual(result.shape, [2, 4])


class TestWeightedBiasQuickGeGLUBack(unittest.TestCase):
    """Tests for weighted_bias_quick_geglu_back function."""

    def test_output_shapes(self):
        """Test weighted_bias_quick_geglu_back returns correct shapes."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        bias = paddle.randn([8])
        weights = paddle.randn([2, 1])
        input_grad, bias_grad, weights_grad = weighted_bias_quick_geglu_back(
            g, y, bias, weights
        )
        self.assertEqual(input_grad.shape, [2, 8])
        self.assertEqual(bias_grad.shape, [2, 8])
        self.assertEqual(weights_grad.shape, [2, 1])

    def test_bias_grad_equals_input_grad(self):
        """Test that bias_grad equals input_grad."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        bias = paddle.randn([8])
        weights = paddle.randn([2, 1])
        input_grad, bias_grad, weights_grad = weighted_bias_quick_geglu_back(
            g, y, bias, weights
        )
        np.testing.assert_allclose(
            input_grad.numpy(), bias_grad.numpy(), atol=1e-6
        )


class TestWeightedBiasQuickGeGLUImpl(unittest.TestCase):
    """Tests for weighted_bias_quick_geglu_impl function."""

    def test_2d_input_no_bias(self):
        """Test 2D input without bias."""
        x = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        result = weighted_bias_quick_geglu_impl(x, None, weights)
        self.assertEqual(result.shape, [2, 4])

    def test_3d_input_no_bias(self):
        """Test 3D input without bias."""
        x = paddle.randn([2, 4, 8])
        weights = paddle.randn([8, 1])
        result = weighted_bias_quick_geglu_impl(x, None, weights)
        self.assertEqual(result.shape, [2, 4, 4])

    def test_asserts_invalid_dim(self):
        """Test assertion for invalid dimensions."""
        x = paddle.randn([2, 4, 4, 8])
        with self.assertRaises(AssertionError):
            weighted_bias_quick_geglu_impl(x, None, paddle.randn([2, 1]))

    @patch(
        "paddleformers.fleet.fusions.fused_bias_geglu.WeightedBiasQuickGeGLUFunction.apply"
    )
    def test_2d_input_with_bias(self, mock_apply):
        """Test 2D input with bias."""
        mock_apply.return_value = paddle.randn([2, 4])
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        weights = paddle.randn([2, 1])
        result = weighted_bias_quick_geglu_impl(x, bias, weights)
        mock_apply.assert_called_once()

    @patch(
        "paddleformers.fleet.fusions.fused_bias_geglu.WeightedQuickGeGLUFunction.apply"
    )
    def test_with_clamp_value(self, mock_apply):
        """Test with clamp_value parameter."""
        mock_apply.return_value = paddle.randn([2, 4])
        x = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        result = weighted_bias_quick_geglu_impl(
            x, None, weights, clamp_value=6.0
        )
        mock_apply.assert_called_once()


class TestWeightedBiasQuickGeGLUFunction(unittest.TestCase):
    """Tests for WeightedBiasQuickGeGLUFunction PyLayer."""

    def test_forward_output_shape(self):
        """Test forward output shape."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        weights = paddle.randn([2, 1])
        linear_offset = paddle.to_tensor(0.0)
        result = WeightedBiasQuickGeGLUFunction.apply(
            x, bias, weights, False, linear_offset
        )
        self.assertEqual(result.shape, [2, 4])


if __name__ == "__main__":
    unittest.main()
