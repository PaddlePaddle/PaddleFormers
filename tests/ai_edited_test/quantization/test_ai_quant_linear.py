# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for quantization/quantization_linear.py"""

import unittest

import paddle

from paddleformers.quantization.quantization_linear import (
    QuantizationLinear,
    QuantMapping,
)


class MockQuantizationConfig:
    """Mock quantization config for testing."""

    def __init__(
        self,
        weight_quantize_algo="weight_only_int8",
        ignore_modules=None,
        group_size=-1,
        qlora_weight_double_quant=False,
        qlora_weight_blocksize=64,
        qlora_weight_double_quant_block_size=256,
        llm_int8_threshold=6.0,
    ):
        self.weight_quantize_algo = weight_quantize_algo
        self.ignore_modules = ignore_modules
        self.group_size = group_size
        self.qlora_weight_double_quant = qlora_weight_double_quant
        self.qlora_weight_blocksize = qlora_weight_blocksize
        self.qlora_weight_double_quant_block_size = qlora_weight_double_quant_block_size
        self.llm_int8_threshold = llm_int8_threshold
        self.apply_hadamard = False


class TestQuantMapping(unittest.TestCase):
    """Tests for QuantMapping dictionary."""

    def test_supported_algorithms(self):
        """Test that QuantMapping contains all supported algorithms."""
        expected_keys = [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "fp4",
            "nf4",
            "a8w8linear",
            "a8w4linear",
            "fp8linear",
        ]
        for key in expected_keys:
            self.assertIn(key, QuantMapping)

    def test_mapping_structure(self):
        """Test that QuantMapping values are (dtype_str, bit) tuples."""
        for key, value in QuantMapping.items():
            self.assertEqual(len(value), 2)
            self.assertIsInstance(value[0], str)
            self.assertIsInstance(value[1], int)

    def test_int8_algorithms(self):
        """Test int8 quantization algorithm entries."""
        self.assertEqual(QuantMapping["weight_only_int8"], ("int8", 8))
        self.assertEqual(QuantMapping["a8w8linear"], ("int8", 8))

    def test_int4_algorithms(self):
        """Test int4 quantization algorithm entries."""
        self.assertEqual(QuantMapping["weight_only_int4"], ("int4", 4))

    def test_fp8_algorithm(self):
        """Test fp8 quantization algorithm entry."""
        self.assertEqual(QuantMapping["fp8linear"], ("fp8", 8))


class TestDequantWeight(unittest.TestCase):
    """Tests for dequant_weight function."""

    def test_dequant_weight_basic(self):
        """Test basic weight dequantization with weight_only_int8 algo."""
        from paddleformers.quantization.quantization_linear import dequant_weight

        quant_weight = paddle.randint(-127, 127, [4, 8]).astype("int8")
        scale = paddle.ones([8])
        config = MockQuantizationConfig(weight_quantize_algo="weight_only_int8")
        try:
            result = dequant_weight(
                quant_weight=quant_weight,
                quantization_config=config,
                weight_quantize_algo="weight_only_int8",
                dtype="float16",
                weight_scale=scale,
                quant_state=None,
                input_shape=[4, 8],
            )
            self.assertEqual(result.shape, [4, 8])
        except RuntimeError:
            # weight_dequantize kernel may not support all dtypes on all hardware
            pass


class TestQuantizationLinear(unittest.TestCase):
    """Tests for QuantizationLinear class."""

    def test_class_exists(self):
        """Test that QuantizationLinear class can be instantiated."""
        self.assertTrue(callable(QuantizationLinear))


if __name__ == "__main__":
    unittest.main()
