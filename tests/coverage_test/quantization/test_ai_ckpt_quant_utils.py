# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for quantization/checkpoint_quantization_utils.py"""

import unittest

import numpy as np
import paddle

from paddleformers.quantization.checkpoint_quantization_utils import (
    cal_abs_max_channel,
    cal_abs_min_max_channel,
    cal_ratio,
    group_wise_quant_dequant,
    merge_int4,
    qdq_weight,
    split_int8,
)


class TestCalRatio(unittest.TestCase):
    """Tests for cal_ratio function."""

    def test_cal_ratio_basic(self):
        """Test basic ratio calculation."""
        m = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 4.0, 9.0])
        result = cal_ratio(m, v)
        expected = 1.0 / (np.sqrt(v) + 1e-8)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cal_ratio_with_eps(self):
        """Test that cal_ratio handles small variance with epsilon."""
        m = np.array([1.0])
        v = np.array([0.0])
        result = cal_ratio(m, v, eps=1e-8)
        self.assertTrue(np.isfinite(result).all())


class TestGroupWiseQuantDequant(unittest.TestCase):
    """Tests for group_wise_quant_dequant function."""

    def test_quant_asymmetry(self):
        """Test asymmetric quantization."""
        inputs = np.random.randn(64, 8).astype(np.float32)
        quant_tensor, mins, maxs = group_wise_quant_dequant(
            inputs, quant_bits=4, group_size=32, quant=True, symmetry=False
        )
        self.assertEqual(quant_tensor.dtype, np.uint8)
        self.assertEqual(mins.shape, (2, 8))
        self.assertEqual(maxs.shape, (2, 8))

    def test_quant_symmetry(self):
        """Test symmetric quantization."""
        inputs = np.random.randn(64, 8).astype(np.float32)
        quant_tensor, scales = group_wise_quant_dequant(inputs, quant_bits=4, group_size=32, quant=True, symmetry=True)
        self.assertEqual(quant_tensor.dtype, np.int8)
        self.assertEqual(scales.shape, (2, 8))

    def test_dequant_asymmetry(self):
        """Test asymmetric dequantization roundtrip."""
        inputs = np.random.randn(64, 8).astype(np.float32)
        quant_tensor, mins, maxs = group_wise_quant_dequant(
            inputs, quant_bits=4, group_size=32, quant=True, symmetry=False
        )
        dequant_tensor = group_wise_quant_dequant(
            quant_tensor, mins=mins, maxs=maxs, quant_bits=4, group_size=32, quant=False, symmetry=False
        )
        self.assertEqual(dequant_tensor.shape, inputs.shape)

    def test_dequant_symmetry(self):
        """Test symmetric dequantization roundtrip."""
        inputs = np.random.randn(64, 8).astype(np.float32)
        quant_tensor, scales = group_wise_quant_dequant(inputs, quant_bits=4, group_size=32, quant=True, symmetry=True)
        dequant_tensor = group_wise_quant_dequant(
            quant_tensor, mins=scales, quant_bits=4, group_size=32, quant=False, symmetry=True
        )
        self.assertEqual(dequant_tensor.shape, inputs.shape)


class TestMergeInt4(unittest.TestCase):
    """Tests for merge_int4 function."""

    def test_merge_int4_basic(self):
        """Test merging two int4 values into int8."""
        x = np.array([1, -2, 3, -4], dtype=np.int8)
        y = np.array([5, 6, -7, 8], dtype=np.int8)
        result = merge_int4(x, y)
        self.assertEqual(result.dtype, np.int8)
        self.assertEqual(result.shape, x.shape)

    def test_merge_split_roundtrip(self):
        """Test that merge then split preserves values."""
        x = np.array([1, -2, 3, -4], dtype=np.int8)
        y = np.array([5, 6, -7, 8], dtype=np.int8)
        merged = merge_int4(x, y)
        high, low = split_int8(merged)
        # The high bits should match x
        np.testing.assert_array_equal(high.numpy(), x)


class TestSplitInt8(unittest.TestCase):
    """Tests for split_int8 function."""

    def test_split_int8_basic(self):
        """Test splitting int8 into two int4 values."""
        x = np.array([0x15, 0x2A], dtype=np.int8)  # 0x15 = 0001 0101, 0x2A = 0010 1010
        high, low = split_int8(x)
        self.assertIsInstance(high, paddle.Tensor)
        self.assertIsInstance(low, paddle.Tensor)


class TestCalAbsMinMaxChannel(unittest.TestCase):
    """Tests for cal_abs_min_max_channel function."""

    def test_basic(self):
        """Test channel-wise min max calculation."""
        inputs = np.random.randn(4, 8).astype(np.float32)
        maxs, mins = cal_abs_min_max_channel(inputs, quant_axis=1)
        self.assertEqual(maxs.shape, (8,))
        self.assertEqual(mins.shape, (8,))

    def test_quant_axis_0(self):
        """Test with quant_axis=0."""
        inputs = np.random.randn(4, 8).astype(np.float32)
        maxs, mins = cal_abs_min_max_channel(inputs, quant_axis=0)
        self.assertEqual(maxs.shape, (4,))
        self.assertEqual(mins.shape, (4,))

    def test_zero_handling(self):
        """Test that zero values are replaced with epsilon."""
        inputs = np.zeros((4, 8), dtype=np.float32)
        maxs, mins = cal_abs_min_max_channel(inputs, quant_axis=1)
        self.assertTrue(np.all(maxs > 0))
        self.assertTrue(np.all(mins != 0))


class TestCalAbsMaxChannel(unittest.TestCase):
    """Tests for cal_abs_max_channel function."""

    def test_basic(self):
        """Test basic abs max channel calculation."""
        inputs = np.random.randn(4, 8).astype(np.float32)
        result = cal_abs_max_channel(inputs, quant_axis=1)
        self.assertEqual(result.shape, (8,))

    def test_zero_handling(self):
        """Test that zero values are replaced with epsilon."""
        inputs = np.zeros((4, 8), dtype=np.float32)
        result = cal_abs_max_channel(inputs, quant_axis=1)
        self.assertTrue(np.all(result > 0))


class TestQdqWeight(unittest.TestCase):
    """Tests for qdq_weight function."""

    def test_quantize_basic(self):
        """Test basic quantization."""
        x = np.random.randn(4, 8).astype(np.float32)
        quant_x, scales = qdq_weight(x, quant_bit=8)
        self.assertEqual(quant_x.dtype, np.int8)
        self.assertEqual(scales.shape, (8,))

    def test_dequantize_basic(self):
        """Test basic dequantization roundtrip."""
        x = np.random.randn(4, 8).astype(np.float32)
        quant_x, scales = qdq_weight(x, quant_bit=8)
        dequant_x, _ = qdq_weight(quant_x, quant_bit=8, scales=scales, dequant=True)
        self.assertEqual(dequant_x.shape, x.shape)
        self.assertEqual(dequant_x.dtype, np.float32)


class TestAsymmetryQdqWeight(unittest.TestCase):
    """Tests for asymmetry_qdq_weight function (imported via qdq_weight)."""

    def test_asymmetry_quant_dequant_roundtrip(self):
        """Test asymmetric quantize-dequantize roundtrip."""
        from paddleformers.quantization.checkpoint_quantization_utils import (
            asymmetry_qdq_weight,
        )

        x = np.random.randn(4, 8).astype(np.float32)
        quant_x, mins, maxs = asymmetry_qdq_weight(x, quant_bit=8)
        self.assertEqual(quant_x.dtype, np.uint8)

        dequant_x, _ = asymmetry_qdq_weight(x=quant_x, quant_bit=8, mins=mins, maxs=maxs, dequant=True)
        self.assertEqual(dequant_x.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
