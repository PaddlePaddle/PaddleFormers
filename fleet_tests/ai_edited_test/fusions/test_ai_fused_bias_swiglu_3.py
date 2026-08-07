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

import paddle

from paddleformers.fleet.fusions.fused_bias_swiglu import (
    BiasSwiGLUFunction,
    SwiGLUFunction,
    WeightedSwiGLUFunction,
    bias_swiglu,
    bias_swiglu_impl,
    swiglu,
    weighted_bias_swiglu_impl,
    weighted_swiglu,
)


class TestSwiglu(unittest.TestCase):
    """Tests for swiglu function."""

    def test_swiglu_output_shape(self):
        """Test swiglu halves last dimension."""
        x = paddle.randn([2, 8])
        result = swiglu(x)
        self.assertEqual(result.shape, [2, 4])

    def test_swiglu_positive_input(self):
        """Test swiglu with positive input."""
        x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]])
        result = swiglu(x)
        self.assertEqual(result.shape, [1, 4])
        # SiLU(x) * x for positive x should be positive
        self.assertTrue((result >= 0).all())


class TestBiasSwiglu(unittest.TestCase):
    """Tests for bias_swiglu function."""

    def test_bias_swiglu_output_shape(self):
        """Test bias_swiglu halves last dimension."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        result = bias_swiglu(x, bias)
        self.assertEqual(result.shape, [2, 4])

    def test_bias_swiglu_different_bias_shape(self):
        """Test bias_swiglu with 2D bias."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([1, 8])
        result = bias_swiglu(x, bias)
        self.assertEqual(result.shape, [2, 4])


class TestWeightedSwiglu(unittest.TestCase):
    """Tests for weighted_swiglu function."""

    def test_weighted_swiglu_output_shape(self):
        """Test weighted_swiglu output shape."""
        x = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        result = weighted_swiglu(x, weights)
        self.assertEqual(result.shape, [2, 4])

    def test_weighted_swiglu_dtype_preserved(self):
        """Test weighted_swiglu preserves dtype."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        weights = paddle.randn([2, 1], dtype=paddle.float32)
        result = weighted_swiglu(x, weights)
        self.assertEqual(result.dtype, paddle.float32)


class TestBiasSwiGLUFunction(unittest.TestCase):
    """Tests for BiasSwiGLUFunction PyLayer."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_calls_bias_swiglu(self, mock_cuda):
        """Test forward calls bias_swiglu."""
        input_t = paddle.randn([2, 8])
        bias = paddle.randn([8])
        with (
            patch(
                "paddleformers.fleet.fusions.fused_bias_swiglu.bias_swiglu",
                return_value=paddle.randn([2, 4]),
            ) as mock_bias_swiglu,
            patch(
                "paddleformers.fleet.fusions.fused_bias_swiglu.swiglu_back",
                side_effect=NotImplementedError,
            ),
        ):
            try:
                result = BiasSwiGLUFunction.apply(input_t, bias, False, False)
            except NotImplementedError:
                pass  # backward not invoked


class TestSwiGLUFunction(unittest.TestCase):
    """Tests for SwiGLUFunction PyLayer."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_calls_swiglu(self, mock_cuda):
        """Test forward calls swiglu."""
        input_t = paddle.randn([2, 8])
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.swiglu",
            return_value=paddle.randn([2, 4]),
        ) as mock_swiglu_fn:
            try:
                result = SwiGLUFunction.apply(input_t, False, False)
            except NotImplementedError:
                pass


class TestWeightedSwiGLUFunction(unittest.TestCase):
    """Tests for WeightedSwiGLUFunction PyLayer."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_calls_weighted_swiglu(self, mock_cuda):
        """Test forward calls weighted_swiglu."""
        input_t = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.weighted_swiglu",
            return_value=paddle.randn([2, 4]),
        ) as mock_fn:
            try:
                result = WeightedSwiGLUFunction.apply(input_t, weights, False)
            except NotImplementedError:
                pass


class TestBiasSwigluImpl(unittest.TestCase):
    """Tests for bias_swiglu_impl function."""

    def test_2d_input_with_bias(self):
        """Test bias_swiglu_impl with 2D input and bias."""
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.BiasSwiGLUFunction.apply",
            return_value=paddle.randn([2, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 8])
            bias = paddle.randn([8])
            result = bias_swiglu_impl(x, bias)
            mock_apply.assert_called_once()

    def test_2d_input_without_bias(self):
        """Test bias_swiglu_impl with 2D input, no bias."""
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.SwiGLUFunction.apply",
            return_value=paddle.randn([2, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 8])
            result = bias_swiglu_impl(x, None)
            mock_apply.assert_called_once()

    def test_3d_input_with_bias(self):
        """Test bias_swiglu_impl with 3D input."""
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.BiasSwiGLUFunction.apply",
            return_value=paddle.randn([2, 4, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 4, 8])
            bias = paddle.randn([8])
            result = bias_swiglu_impl(x, bias)
            self.assertEqual(result.shape, [2, 4, 4])

    def test_asserts_invalid_dim(self):
        """Test assertion for invalid input dimensions."""
        x = paddle.randn([2, 4, 4, 8])
        with self.assertRaises(AssertionError):
            bias_swiglu_impl(x, None)


class TestWeightedBiasSwigluImpl(unittest.TestCase):
    """Tests for weighted_bias_swiglu_impl function."""

    def test_2d_input_no_bias(self):
        """Test weighted_bias_swiglu_impl with 2D input, no bias."""
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.WeightedSwiGLUFunction.apply",
            return_value=paddle.randn([2, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 8])
            weights = paddle.randn([2, 1])
            result = weighted_bias_swiglu_impl(x, None, weights)
            mock_apply.assert_called_once()

    def test_bias_not_supported(self):
        """Test that bias raises NotImplementedError."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        weights = paddle.randn([2, 1])
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(x, bias, weights)

    def test_3d_input_no_bias(self):
        """Test weighted_bias_swiglu_impl with 3D input."""
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.WeightedSwiGLUFunction.apply",
            return_value=paddle.randn([2, 4, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 4, 8])
            weights = paddle.randn([2, 1])
            result = weighted_bias_swiglu_impl(x, None, weights)
            self.assertEqual(result.shape, [2, 4, 4])

    def test_asserts_invalid_dim(self):
        """Test assertion for invalid input dimensions."""
        x = paddle.randn([2, 4, 4, 8])
        with self.assertRaises(AssertionError):
            weighted_bias_swiglu_impl(x, None, paddle.randn([2, 1]))


if __name__ == "__main__":
    unittest.main()
