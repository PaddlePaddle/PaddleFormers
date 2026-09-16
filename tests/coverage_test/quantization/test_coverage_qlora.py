# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for quantization/qlora.py"""

import unittest

import paddle

# paddleslim is required for qlora functions
try:
    import paddleslim  # noqa: F401

    HAS_PADDLESLIM = True
except ImportError:
    HAS_PADDLESLIM = False


@unittest.skipIf(not HAS_PADDLESLIM, "Requires paddleslim package")
@unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
class TestQloraWeightQuantize(unittest.TestCase):
    """Tests for qlora weight quantization functions."""

    def test_qlora_weight_quantize_no_double_quant(self):
        """Test qlora_weight_quantize without double quantization."""
        from paddleformers.quantization.qlora import qlora_weight_quantize

        weight = paddle.randn([128, 256], dtype=paddle.float16).cuda()
        result = qlora_weight_quantize(weight=weight, quant_algo="nf4", double_quant=False, return_dict=True)
        self.assertIn("quant_weight", result)
        self.assertIn("weight_scale", result)

    def test_qlora_weight_quantize_with_double_quant(self):
        """Test qlora_weight_quantize with double quantization."""
        from paddleformers.quantization.qlora import qlora_weight_quantize

        weight = paddle.randn([128, 256], dtype=paddle.float16).cuda()
        result = qlora_weight_quantize(
            weight=weight, quant_algo="nf4", double_quant=True, return_dict=True, linear_name="test_linear"
        )
        self.assertIn("test_linear.quant_weight", result)
        self.assertIn("test_linear.qweight_scale", result)
        self.assertIn("test_linear.double_weight_scale", result)
        self.assertIn("test_linear.weight_scale_offset", result)

    def test_qlora_weight_quantize_no_dict(self):
        """Test qlora_weight_quantize with return_dict=False."""
        from paddleformers.quantization.qlora import qlora_weight_quantize

        weight = paddle.randn([128, 256], dtype=paddle.float16).cuda()
        quant_weight, state = qlora_weight_quantize(
            weight=weight, quant_algo="nf4", double_quant=False, return_dict=False
        )
        self.assertIsNotNone(quant_weight)
        self.assertIsNotNone(state)


@unittest.skipIf(not HAS_PADDLESLIM, "Requires paddleslim package")
@unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
class TestQloraWeightDequantize(unittest.TestCase):
    """Tests for qlora weight dequantization."""

    def test_qlora_quantize_dequantize_roundtrip(self):
        """Test that quantize then dequantize preserves shape."""
        from paddleformers.quantization.qlora import (
            qlora_weight_dequantize,
            qlora_weight_quantize,
        )

        weight = paddle.randn([128, 256], dtype=paddle.float16).cuda()
        quant_weight, state = qlora_weight_quantize(
            weight=weight, quant_algo="nf4", double_quant=False, return_dict=False
        )
        dequant_weight = qlora_weight_dequantize(quant_weight, "nf4", state)
        self.assertEqual(dequant_weight.shape, weight.shape)


@unittest.skipIf(not HAS_PADDLESLIM, "Requires paddleslim package")
@unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
class TestQloraWeightQuantizeDequantize(unittest.TestCase):
    """Tests for qlora_weight_quantize_dequantize."""

    def test_quantize_dequantize_shape(self):
        """Test that quantize_dequantize preserves shape and dtype."""
        from paddleformers.quantization.qlora import qlora_weight_quantize_dequantize

        weight = paddle.randn([64, 128], dtype=paddle.float16).cuda()
        result = qlora_weight_quantize_dequantize(weight, quant_algo="nf4")
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.dtype, weight.dtype)


@unittest.skipIf(not HAS_PADDLESLIM, "Requires paddleslim package")
@unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
class TestQloraWeightLinear(unittest.TestCase):
    """Tests for qlora_weight_linear."""

    def test_qlora_weight_linear_output_shape(self):
        """Test that qlora_weight_linear produces correct output shape."""
        from paddleformers.quantization.qlora import (
            qlora_weight_linear,
            qlora_weight_quantize,
        )

        in_features = 64
        out_features = 32
        weight = paddle.randn([in_features, out_features], dtype=paddle.float16).cuda()
        quant_weight, state = qlora_weight_quantize(
            weight=weight, quant_algo="nf4", double_quant=False, return_dict=False
        )

        x = paddle.randn([2, in_features], dtype=paddle.float16).cuda()
        output = qlora_weight_linear(
            x=x, quant_weight=quant_weight, dtype=paddle.float16, state=state, quant_algo="nf4"
        )
        self.assertEqual(output.shape, [2, out_features])

    def test_qlora_weight_linear_with_bias(self):
        """Test qlora_weight_linear with bias."""
        from paddleformers.quantization.qlora import (
            qlora_weight_linear,
            qlora_weight_quantize,
        )

        in_features = 64
        out_features = 32
        weight = paddle.randn([in_features, out_features], dtype=paddle.float16).cuda()
        quant_weight, state = qlora_weight_quantize(
            weight=weight, quant_algo="nf4", double_quant=False, return_dict=False
        )

        bias = paddle.randn([out_features], dtype=paddle.float16).cuda()
        x = paddle.randn([2, in_features], dtype=paddle.float16).cuda()
        output = qlora_weight_linear(
            x=x,
            quant_weight=quant_weight,
            dtype=paddle.float16,
            state=state,
            quant_algo="nf4",
            bias=bias,
        )
        self.assertEqual(output.shape, [2, out_features])


if __name__ == "__main__":
    unittest.main()
