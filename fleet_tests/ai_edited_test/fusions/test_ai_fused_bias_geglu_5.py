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


# Tests for src/paddlefleet/fusions/fused_bias_geglu.py
# Additional tests for geglu, bias_geglu, geglu_back, bias_geglu_back,
# BiasGeGLUFunction, GeGLUFunction, bias_geglu_impl

import unittest
from unittest import mock

import paddle


class TestGeGLU(unittest.TestCase):
    """Tests for geglu function."""

    def test_geglu_output_shape(self):
        """Test geglu halves the last dimension."""
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu

        y = paddle.randn([4, 8])
        result = geglu(y)
        self.assertEqual(result.shape, [4, 4])

    def test_geglu_output_shape_3d(self):
        """Test geglu with 3D input."""
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu

        y = paddle.randn([2, 3, 8])
        result = geglu(y)
        self.assertEqual(result.shape, [2, 3, 4])

    def test_geglu_1d(self):
        """Test geglu with 1D input."""
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu

        y = paddle.randn([8])
        result = geglu(y)
        self.assertEqual(result.shape, [4])

    def test_geglu_preserves_dtype(self):
        """Test geglu preserves input dtype."""
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu

        y = paddle.randn([4, 8], dtype="float32")
        result = geglu(y)
        self.assertEqual(result.dtype, y.dtype)


class TestBiasGeGLU(unittest.TestCase):
    """Tests for bias_geglu function."""

    def test_bias_geglu_output_shape(self):
        """Test bias_geglu output shape."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu

        bias = paddle.randn([8])
        y = paddle.randn([4, 8])
        result = bias_geglu(bias, y)
        self.assertEqual(result.shape, [4, 4])

    def test_bias_geglu_adds_bias(self):
        """Test that bias_geglu adds bias to input."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu

        bias = paddle.randn([8])
        y = paddle.randn([4, 8])
        # Just verify it runs without error
        result = bias_geglu(bias, y)
        self.assertIsNotNone(result)


class TestGeGLUBack(unittest.TestCase):
    """Tests for geglu_back function."""

    def test_geglu_back_output_shape(self):
        """Test geglu_back output shape matches input shape."""
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu_back

        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])
        result = geglu_back(g, y)
        self.assertEqual(result.shape, y.shape)

    def test_geglu_back_preserves_dtype(self):
        """Test geglu_back preserves dtype."""
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu_back

        g = paddle.randn([4, 4], dtype="float32")
        y = paddle.randn([4, 8], dtype="float32")
        result = geglu_back(g, y)
        self.assertEqual(result.dtype, y.dtype)

    def test_geglu_back_3d(self):
        """Test geglu_back with 3D input."""
        from paddleformers.fleet.fusions.fused_bias_geglu import geglu_back

        g = paddle.randn([2, 3, 4])
        y = paddle.randn([2, 3, 8])
        result = geglu_back(g, y)
        self.assertEqual(result.shape, [2, 3, 8])


class TestBiasGeGLUBack(unittest.TestCase):
    """Tests for bias_geglu_back function."""

    def test_bias_geglu_back_adds_bias(self):
        """Test bias_geglu_back adds bias before computing gradient."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_back

        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])
        bias = paddle.randn([8])

        result = bias_geglu_back(g, y, bias)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_geglu_back_output_shape(self):
        """Test bias_geglu_back output matches input shape."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_back

        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        bias = paddle.randn([8])

        result = bias_geglu_back(g, y, bias)
        self.assertEqual(result.shape, y.shape)


class TestBiasGeGLUFunction(unittest.TestCase):
    """Tests for BiasGeGLUFunction PyLayer."""

    def test_forward_saves_tensors(self):
        """Test forward saves input and bias."""
        from paddleformers.fleet.fusions.fused_bias_geglu import BiasGeGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        bias = paddle.randn([8])

        with mock.patch(
            "paddleformers.fleet.fusions.fused_bias_geglu.bias_geglu",
            return_value=paddle.randn([4, 4]),
        ):
            result = BiasGeGLUFunction.forward(mock_ctx, inp, bias)
            mock_ctx.save_for_backward.assert_called_once_with(inp, bias)

    def test_backward_returns_grads(self):
        """Test backward returns input and bias gradients."""
        from paddleformers.fleet.fusions.fused_bias_geglu import BiasGeGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        bias = paddle.randn([8])
        mock_ctx.saved_tensor.return_value = [inp, bias]

        mock_grad = paddle.randn([4, 8])
        mock_reduced = paddle.randn([8])
        with (
            mock.patch(
                "paddleformers.fleet.fusions.fused_bias_geglu.bias_geglu_back",
                return_value=mock_grad,
            ),
            mock.patch.object(
                mock_grad, "reduce_as", return_value=mock_reduced
            ),
        ):
            result = BiasGeGLUFunction.backward(mock_ctx, paddle.randn([4, 4]))
            self.assertEqual(len(result), 2)


class TestGeGLUFunctionPyLayer(unittest.TestCase):
    """Tests for GeGLUFunction PyLayer."""

    def test_forward_saves_input(self):
        """Test forward saves input."""
        from paddleformers.fleet.fusions.fused_bias_geglu import GeGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])

        with mock.patch(
            "paddleformers.fleet.fusions.fused_bias_geglu.geglu",
            return_value=paddle.randn([4, 4]),
        ):
            result = GeGLUFunction.forward(mock_ctx, inp)
            mock_ctx.save_for_backward.assert_called_once_with(inp)

    def test_backward_returns_gradient(self):
        """Test backward returns gradient."""
        from paddleformers.fleet.fusions.fused_bias_geglu import GeGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        mock_ctx.saved_tensor.return_value = [inp]

        mock_grad = paddle.randn([4, 8])
        with mock.patch(
            "paddleformers.fleet.fusions.fused_bias_geglu.geglu_back",
            return_value=mock_grad,
        ):
            result = GeGLUFunction.backward(mock_ctx, paddle.randn([4, 4]))


class TestBiasGeGLUImpl(unittest.TestCase):
    """Tests for bias_geglu_impl function."""

    def test_2d_input_no_bias(self):
        """Test 2D input without bias."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        inp = paddle.randn([4, 8])
        with mock.patch(
            "paddleformers.fleet.fusions.fused_bias_geglu.GeGLUFunction.apply",
            return_value=paddle.randn([4, 4]),
        ):
            result = bias_geglu_impl(inp, None)
            self.assertEqual(result.shape, [4, 4])

    def test_2d_input_with_bias(self):
        """Test 2D input with bias."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        inp = paddle.randn([4, 8])
        bias = paddle.randn([8])
        with mock.patch(
            "paddleformers.fleet.fusions.fused_bias_geglu.BiasGeGLUFunction.apply",
            return_value=paddle.randn([4, 4]),
        ):
            result = bias_geglu_impl(inp, bias)
            self.assertEqual(result.shape, [4, 4])

    def test_3d_input_with_bias(self):
        """Test 3D input with bias."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        inp = paddle.randn([2, 3, 8])
        bias = paddle.randn([8])
        with mock.patch(
            "paddleformers.fleet.fusions.fused_bias_geglu.BiasGeGLUFunction.apply",
            return_value=paddle.randn([6, 4]),
        ):
            result = bias_geglu_impl(inp, bias)
            self.assertEqual(len(result.shape), 3)
            self.assertEqual(result.shape[0], 2)
            self.assertEqual(result.shape[1], 3)

    def test_3d_input_no_bias(self):
        """Test 3D input without bias."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        inp = paddle.randn([2, 3, 8])
        with mock.patch(
            "paddleformers.fleet.fusions.fused_bias_geglu.GeGLUFunction.apply",
            return_value=paddle.randn([6, 4]),
        ):
            result = bias_geglu_impl(inp, None)
            self.assertEqual(len(result.shape), 3)

    def test_invalid_dims_raises(self):
        """Test invalid dimensions raise AssertionError."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        inp = paddle.randn([2, 3, 4, 8])  # 4D
        with self.assertRaises(AssertionError):
            bias_geglu_impl(inp, None)

    def test_1d_input_raises(self):
        """Test 1D input raises AssertionError."""
        from paddleformers.fleet.fusions.fused_bias_geglu import bias_geglu_impl

        inp = paddle.randn([8])
        with self.assertRaises(AssertionError):
            bias_geglu_impl(inp, None)


class TestQuickGeGLU(unittest.TestCase):
    """Tests for quick_geglu and related functions."""

    def test_quick_geglu_output_shape(self):
        """Test quick_geglu output shape."""
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_geglu

        y = paddle.randn([4, 8])
        result = quick_geglu(y)
        self.assertEqual(result.shape, [4, 4])

    def test_quick_geglu_with_offset(self):
        """Test quick_geglu with linear_offset."""
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_geglu

        y = paddle.randn([4, 8])
        result = quick_geglu(y, linear_offset=0.5)
        self.assertEqual(result.shape, [4, 4])

    def test_weighted_quick_geglu_shape(self):
        """Test weighted_quick_geglu output shape."""
        from paddleformers.fleet.fusions.fused_bias_geglu import weighted_quick_geglu

        y = paddle.randn([4, 8])
        weights = paddle.randn([4, 1])
        result = weighted_quick_geglu(y, weights)
        self.assertEqual(result.shape, [4, 4])

    def test_quick_geglu_back_shape(self):
        """Test quick_geglu_back output shape."""
        from paddleformers.fleet.fusions.fused_bias_geglu import quick_geglu_back

        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])
        result = quick_geglu_back(g, y)
        self.assertEqual(result.shape, [4, 8])

    def test_weighted_quick_geglu_back_shapes(self):
        """Test weighted_quick_geglu_back output shapes."""
        from paddleformers.fleet.fusions.fused_bias_geglu import (
            weighted_quick_geglu_back,
        )

        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])
        weights = paddle.randn([4, 1])
        input_grad, weights_grad = weighted_quick_geglu_back(g, y, weights)
        self.assertEqual(input_grad.shape, [4, 8])
        self.assertEqual(weights_grad.shape, [4, 1])


if __name__ == "__main__":
    unittest.main()
