# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for quantization/quantization_utils.py"""

import unittest
from unittest.mock import patch

import paddle
import paddle.nn as nn

from paddleformers.quantization.quantization_utils import (
    convert_to_quantize_dequantize_state_dict,
    convert_to_quantize_state_dict,
    parse_weight_quantize_algo,
    replace_with_quantization_linear,
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
        apply_hadamard=False,
    ):
        self.weight_quantize_algo = weight_quantize_algo
        self.ignore_modules = ignore_modules
        self.group_size = group_size
        self.qlora_weight_double_quant = qlora_weight_double_quant
        self.qlora_weight_blocksize = qlora_weight_blocksize
        self.qlora_weight_double_quant_block_size = qlora_weight_double_quant_block_size
        self.llm_int8_threshold = llm_int8_threshold
        self.apply_hadamard = apply_hadamard


class SimpleModel(nn.Layer):
    """Simple model for quantization testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return self.linear(x)


class TestParseWeightQuantizeAlgo(unittest.TestCase):
    """Tests for parse_weight_quantize_algo."""

    def test_string_algo(self):
        """Test with string weight_quantize_algo."""
        config = MockQuantizationConfig(weight_quantize_algo="weight_only_int8")
        result = parse_weight_quantize_algo(config, "linear")
        self.assertEqual(result, "weight_only_int8")

    def test_ignored_module(self):
        """Test that ignored modules return None."""
        config = MockQuantizationConfig(ignore_modules=["lm_head"])
        result = parse_weight_quantize_algo(config, "lm_head")
        self.assertIsNone(result)

    def test_dict_algo_match(self):
        """Test dict algo with matching module."""
        config = MockQuantizationConfig(weight_quantize_algo={"weight_only_int8": ["linear"]})
        result = parse_weight_quantize_algo(config, "linear")
        self.assertEqual(result, "weight_only_int8")

    def test_dict_algo_no_match(self):
        """Test dict algo without matching module."""
        config = MockQuantizationConfig(weight_quantize_algo={"weight_only_int8": ["other"]})
        result = parse_weight_quantize_algo(config, "linear")
        self.assertIsNone(result)


class TestReplaceWithQuantizationLinear(unittest.TestCase):
    """Tests for replace_with_quantization_linear."""

    @unittest.skip("QuantizationLinear.__init__ requires specific GPU/dtype support not available in CI")
    def test_replace_linear(self):
        """Test that nn.Linear is replaced with QuantizationLinear."""
        from paddleformers.quantization.quantization_linear import QuantizationLinear

        model = SimpleModel()
        config = MockQuantizationConfig(weight_quantize_algo="weight_only_int8")
        # QuantizationLinear.__init__ tries to create int8 parameters which
        # fails with xavier init, so we patch it to avoid the actual __init__
        with patch.object(QuantizationLinear, "__init__", return_value=None):
            replace_with_quantization_linear(model, config)
            self.assertIsInstance(model.linear, QuantizationLinear)

    def test_skip_ignored_module(self):
        """Test that ignored modules are not replaced."""
        model = SimpleModel()
        config = MockQuantizationConfig(weight_quantize_algo="weight_only_int8", ignore_modules=["linear"])
        replace_with_quantization_linear(model, config)
        # Should still be nn.Linear since it's ignored
        self.assertIsInstance(model.linear, nn.Linear)


class TestConvertToQuantizeStateDict(unittest.TestCase):
    """Tests for convert_to_quantize_state_dict."""

    @unittest.skip("weight_quantize requires specific GPU/dtype support not available in CI")
    def test_none_algo_skipped(self):
        """Test that layers with None algo are skipped."""
        config = MockQuantizationConfig()
        with patch("paddleformers.quantization.quantization_utils.parse_weight_quantize_algo", return_value=None):
            state_dict = {"linear.weight": paddle.randn([4, 4])}
            result = convert_to_quantize_state_dict(state_dict, ["linear"], config, "float32")
            self.assertIn("linear.weight", result)


class TestConvertToQuantizeDequantizeStateDict(unittest.TestCase):
    """Tests for convert_to_quantize_dequantize_state_dict."""

    @unittest.skip("weight_quantize requires specific GPU/dtype support not available in CI")
    def test_none_algo_skipped(self):
        """Test that layers with None algo are skipped."""
        config = MockQuantizationConfig()
        with patch("paddleformers.quantization.quantization_utils.parse_weight_quantize_algo", return_value=None):
            state_dict = {"linear.weight": paddle.randn([4, 4])}
            result = convert_to_quantize_dequantize_state_dict(state_dict, ["linear"], config)
            self.assertIn("linear.weight", result)


if __name__ == "__main__":
    unittest.main()
