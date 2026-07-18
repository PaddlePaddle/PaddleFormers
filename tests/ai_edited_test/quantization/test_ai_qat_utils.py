# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for quantization/qat_utils.py"""

import unittest

import paddle

from paddleformers.quantization.qat_utils import (
    QMIN_QMAX_MAPPING,
    dequantize,
    int8_forward,
    quantize,
)


class MockQuantizationConfig:
    """Mock quantization config for QAT testing."""

    def __init__(
        self,
        weight_quantize_algo="a8w8linear",
        fp8_format=None,
        scale_epsilon=1e-5,
        apply_hadamard=False,
        hadamard_block_size=1,
        apply_online_actscale_step=0,
        actscale_moving_rate=0.1,
        quant_input_grad=False,
        quant_weight_grad=False,
    ):
        self.weight_quantize_algo = weight_quantize_algo
        self.fp8_format = fp8_format or {
            "activation": paddle.float8_e4m3fn,
            "weight": paddle.float8_e4m3fn,
            "grad_output": paddle.float8_e5m2,
        }
        self.scale_epsilon = scale_epsilon
        self.apply_hadamard = apply_hadamard
        self.hadamard_block_size = hadamard_block_size
        self.apply_online_actscale_step = apply_online_actscale_step
        self.actscale_moving_rate = actscale_moving_rate
        self.quant_input_grad = quant_input_grad
        self.quant_weight_grad = quant_weight_grad


class TestQminQmaxMapping(unittest.TestCase):
    """Tests for QMIN_QMAX_MAPPING."""

    def test_all_keys_present(self):
        """Test that all expected keys are present."""
        expected_keys = [
            "a8w8linear_activation",
            "a8w4linear_activation",
            "a8w8linear_weight",
            "a8w4linear_weight",
        ]
        for key in expected_keys:
            self.assertIn(key, QMIN_QMAX_MAPPING)

    def test_int8_ranges(self):
        """Test int8 quantization ranges."""
        self.assertEqual(QMIN_QMAX_MAPPING["a8w8linear_activation"], (-128, 127))
        self.assertEqual(QMIN_QMAX_MAPPING["a8w8linear_weight"], (-128, 127))

    def test_int4_weight_range(self):
        """Test int4 weight range."""
        self.assertEqual(QMIN_QMAX_MAPPING["a8w4linear_weight"], (-8, 7))


class TestQuantize(unittest.TestCase):
    """Tests for quantize function."""

    def test_quantize_activation_a8w8(self):
        """Test a8w8 activation quantization."""
        config = MockQuantizationConfig()
        x = paddle.randn([2, 4])
        quant_x, scale = quantize(x, "a8w8linear", "activation", config)
        self.assertEqual(quant_x.dtype, paddle.int8)
        self.assertEqual(scale.shape, [1])

    def test_quantize_weight_a8w8(self):
        """Test a8w8 weight quantization."""
        config = MockQuantizationConfig()
        x = paddle.randn([4, 8])
        quant_x, scale = quantize(x, "a8w8linear", "weight", config)
        self.assertEqual(quant_x.dtype, paddle.int8)

    def test_quantize_weight_a8w4(self):
        """Test a8w4 weight quantization."""
        config = MockQuantizationConfig()
        x = paddle.randn([4, 8])
        quant_x, scale = quantize(x, "a8w4linear", "weight", config)
        self.assertEqual(quant_x.dtype, paddle.int8)

    def test_quantize_activation_with_hadamard(self):
        """Test activation quantization with hadamard."""
        config = MockQuantizationConfig(apply_hadamard=True, hadamard_block_size=4)
        # Hadamard requires last dim to be power of 2
        x = paddle.randn([2, 4])
        quant_x, scale = quantize(x, "a8w8linear", "activation", config, side="right", apply_hadamard=True)
        self.assertIsNotNone(quant_x)

    def test_quantize_with_activation_scale(self):
        """Test quantization with pre-computed activation scale."""
        config = MockQuantizationConfig()
        x = paddle.randn([2, 4])
        activation_scale = paddle.ones([1]) * 2.0
        activation_scale.stop_gradient = True
        quant_x, scale = quantize(x, "a8w8linear", "activation", config, activation_scale=activation_scale)
        self.assertEqual(quant_x.dtype, paddle.int8)

    def test_quantize_unknown_algo_raises(self):
        """Test that unknown algorithm raises KeyError."""
        config = MockQuantizationConfig()
        x = paddle.randn([2, 4])
        with self.assertRaises(KeyError):
            quantize(x, "unknown_algo", "activation", config)

    def test_quantize_unknown_tensor_type_raises(self):
        """Test that unknown tensor type raises KeyError."""
        config = MockQuantizationConfig()
        x = paddle.randn([2, 4])
        with self.assertRaises(KeyError):
            quantize(x, "a8w8linear", "unknown_type", config)


class TestDequantize(unittest.TestCase):
    """Tests for dequantize function."""

    def test_dequantize_weight_a8w8(self):
        """Test a8w8 weight dequantization."""
        config = MockQuantizationConfig()
        quant_x = paddle.randint(-127, 127, [4, 8]).astype("int8")
        scale = paddle.ones([8])
        result = dequantize(quant_x, scale, "weight", "a8w8linear", config)
        self.assertEqual(result.dtype, scale.dtype)

    def test_dequantize_unknown_algo_raises(self):
        """Test that unknown algo raises NotImplementedError."""
        config = MockQuantizationConfig()
        quant_x = paddle.randint(-127, 127, [4, 8]).astype("int8")
        scale = paddle.ones([8])
        with self.assertRaises(NotImplementedError):
            dequantize(quant_x, scale, "weight", "unknown_algo", config)

    def test_dequantize_unknown_type_raises(self):
        """Test that unknown tensor type raises NotImplementedError."""
        config = MockQuantizationConfig()
        quant_x = paddle.randint(-127, 127, [4, 8]).astype("int8")
        scale = paddle.ones([8])
        with self.assertRaises(NotImplementedError):
            dequantize(quant_x, scale, "unknown_type", "a8w8linear", config)


class TestInt8Forward(unittest.TestCase):
    """Tests for int8_forward function."""

    def test_int8_forward_output_shape(self):
        """Test that int8_forward produces correct output shape."""
        config = MockQuantizationConfig()
        x = paddle.randn([2, 4])
        quant_w = paddle.randint(-127, 127, [4, 8]).astype("int8")
        scale_w = paddle.ones([8])

        output, quant_x, scale_x = int8_forward(
            x=x,
            quant_w=quant_w,
            scale_w=scale_w,
            weight_quantize_algo="a8w8linear",
            quantization_config=config,
        )
        self.assertEqual(output.shape, [2, 8])

    def test_int8_forward_with_bias(self):
        """Test int8_forward with bias."""
        config = MockQuantizationConfig()
        x = paddle.randn([2, 4])
        quant_w = paddle.randint(-127, 127, [4, 8]).astype("int8")
        scale_w = paddle.ones([8])
        bias = paddle.randn([8])

        output, quant_x, scale_x = int8_forward(
            x=x,
            quant_w=quant_w,
            scale_w=scale_w,
            weight_quantize_algo="a8w8linear",
            bias=bias,
            quantization_config=config,
        )
        self.assertEqual(output.shape, [2, 8])


if __name__ == "__main__":
    unittest.main()
