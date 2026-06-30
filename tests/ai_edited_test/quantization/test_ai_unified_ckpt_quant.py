# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for quantization/unified_checkpoint_quantization.py"""

import unittest

import numpy as np
import paddle

from paddleformers.quantization.unified_checkpoint_quantization import (
    dequant_unified_optimizer,
    quant_unified_optimizer,
)
from paddleformers.utils.env import (
    ASYMMETRY_QUANT_SCALE_MAX,
    ASYMMETRY_QUANT_SCALE_MIN,
    MOMENT1_KEYNAME,
    MOMENT2_KEYNAME,
    SYMMETRY_QUANT_SCALE,
)


class TestQuantUnifiedOptimizer(unittest.TestCase):
    """Tests for quant_unified_optimizer."""

    def test_quant_o0_returns_unchanged(self):
        """Test that O0 stage returns state_dict unchanged (no quantization)."""
        state_dict = {"param1_moment1_0": paddle.randn([4, 4])}
        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O0")
        self.assertIn("param1_moment1_0", result)

    def test_quant_o1_adds_scales(self):
        """Test that O1 stage adds scale keys to state_dict."""
        # quant_unified_optimizer uses numpy-based functions internally
        # so we need numpy arrays in the state dict
        m1 = np.random.randn(4, 4).astype(np.float32)
        m2 = np.abs(np.random.randn(4, 4).astype(np.float32)) + 0.01
        state_dict = {
            "param1/" + MOMENT1_KEYNAME: m1,
            "param1/" + MOMENT2_KEYNAME: m2,
        }
        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O1")
        # Check scales are added
        m1_key = "param1/" + MOMENT1_KEYNAME
        scale_key = m1_key + SYMMETRY_QUANT_SCALE
        self.assertIn(scale_key, result)

    def test_quant_non_optimizer_weight_no_change(self):
        """Test that non-optimizer_weight type doesn't get quantized."""
        state_dict = {"some_key": paddle.randn([4, 4])}
        result = quant_unified_optimizer(state_dict, "model_weight", "O1")
        # model_weight should not be quantized
        self.assertIn("some_key", result)

    def test_quant_o2_stage(self):
        """Test O2 stage quantization adds scale keys."""
        m1 = np.random.randn(32, 16).astype(np.float32)
        m2 = np.abs(np.random.randn(32, 16).astype(np.float32)) + 0.01
        state_dict = {
            "param1/" + MOMENT1_KEYNAME: m1,
            "param1/" + MOMENT2_KEYNAME: m2,
        }
        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O2")
        # O2 should add scales and remove m1 key
        self.assertNotIn("param1/" + MOMENT1_KEYNAME, result)


class TestDequantUnifiedOptimizer(unittest.TestCase):
    """Tests for dequant_unified_optimizer."""

    def test_dequant_o1_stage(self):
        """Test O1 stage dequantization."""
        # First quantize, then dequantize
        m1 = np.random.randn(4, 4).astype(np.float32)
        m2 = np.abs(np.random.randn(4, 4).astype(np.float32)) + 0.01

        state_dict = {
            "param1/" + MOMENT1_KEYNAME: m1,
            "param1/" + MOMENT2_KEYNAME: m2,
        }

        quant_result = quant_unified_optimizer(dict(state_dict), "optimizer_weight", "O1")

        # Extract scales
        scale_dict = {}
        for k in list(quant_result.keys()):
            if SYMMETRY_QUANT_SCALE in k or ASYMMETRY_QUANT_SCALE_MIN in k or ASYMMETRY_QUANT_SCALE_MAX in k:
                scale_dict[k] = quant_result.pop(k)

        dequant_result = dequant_unified_optimizer(quant_result, "O1", scale_dict)
        self.assertIn("param1/" + MOMENT1_KEYNAME, dequant_result)
        self.assertIn("param1/" + MOMENT2_KEYNAME, dequant_result)


if __name__ == "__main__":
    unittest.main()
