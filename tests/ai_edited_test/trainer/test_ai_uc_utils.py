# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/unified_checkpoint/utils.py"""

import unittest
from unittest.mock import MagicMock

from paddleformers.trainer.unified_checkpoint.utils import (
    FP32_MASTER,
    UnifiedCheckpointOption,
    generate_base_static_name,
    is_need_master_weight,
    unwrap_optimizer,
)


class TestFP32Master(unittest.TestCase):
    """Tests for FP32_MASTER constant."""

    def test_value(self):
        """Test FP32_MASTER constant value."""
        self.assertEqual(FP32_MASTER, "fp32_master_0")


class TestUnifiedCheckpointOption(unittest.TestCase):
    """Tests for UnifiedCheckpointOption enum."""

    def test_skip_save_model_weight(self):
        """Test skip_save_model_weight option."""
        self.assertEqual(UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value, "skip_save_model_weight")

    def test_master_weight_compatible(self):
        """Test master_weight_compatible option."""
        self.assertEqual(UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value, "master_weight_compatible")

    def test_remove_master_weight(self):
        """Test remove_master_weight option."""
        self.assertEqual(UnifiedCheckpointOption.REMOVE_MASTER_WEIGHT.value, "remove_master_weight")

    def test_async_save(self):
        """Test async_save option."""
        self.assertEqual(UnifiedCheckpointOption.ASYNC_SAVE.value, "async_save")

    def test_ignore_merge_optimizer(self):
        """Test ignore_merge_optimizer option."""
        self.assertEqual(UnifiedCheckpointOption.IGNORE_MERGE_OPTIMIZER.value, "ignore_merge_optimizer")


class TestUnwrapOptimizer(unittest.TestCase):
    """Tests for unwrap_optimizer function."""

    def test_no_wrapping(self):
        """Test with optimizer that has no _inner_opt or _optim."""
        optimizer = MagicMock(spec=[])
        result = unwrap_optimizer(optimizer)
        self.assertEqual(result, optimizer)

    def test_inner_opt_unwrap(self):
        """Test unwrapping through _inner_opt."""
        inner = MagicMock(spec=[])
        optimizer = MagicMock(spec=["_inner_opt"])
        optimizer._inner_opt = inner
        result = unwrap_optimizer(optimizer)
        self.assertEqual(result, inner)

    def test_optim_unwrap(self):
        """Test unwrapping through _optim."""
        inner = MagicMock(spec=[])
        optimizer = MagicMock(spec=["_optim"])
        optimizer._optim = inner
        result = unwrap_optimizer(optimizer)
        self.assertEqual(result, inner)

    def test_nested_unwrap(self):
        """Test unwrapping through multiple levels."""
        deepest = MagicMock(spec=[])
        mid = MagicMock(spec=["_optim"])
        mid._optim = deepest
        outer = MagicMock(spec=["_inner_opt"])
        outer._inner_opt = mid
        result = unwrap_optimizer(outer)
        self.assertEqual(result, deepest)


class TestIsNeedMasterWeight(unittest.TestCase):
    """Tests for is_need_master_weight function."""

    def test_no_multi_precision(self):
        """Test that optimizer without _multi_precision returns False."""
        optimizer = MagicMock(spec=[])
        result = is_need_master_weight(optimizer, is_fp16_or_bp16=True)
        self.assertFalse(result)

    def test_multi_precision_fp16(self):
        """Test that _multi_precision with fp16 returns True."""
        optimizer = MagicMock(spec=["_multi_precision"])
        optimizer._multi_precision = True
        result = is_need_master_weight(optimizer, is_fp16_or_bp16=True)
        self.assertTrue(result)

    def test_multi_precision_fp32(self):
        """Test that _multi_precision without fp16 returns False."""
        optimizer = MagicMock(spec=["_multi_precision"])
        optimizer._multi_precision = True
        result = is_need_master_weight(optimizer, is_fp16_or_bp16=False)
        self.assertFalse(result)

    def test_multi_precision_false(self):
        """Test that _multi_precision=False returns False."""
        optimizer = MagicMock(spec=["_multi_precision"])
        optimizer._multi_precision = False
        result = is_need_master_weight(optimizer, is_fp16_or_bp16=True)
        self.assertFalse(result)


class TestGenerateBaseStaticName(unittest.TestCase):
    """Tests for generate_base_static_name function."""

    def test_fp32_master_name(self):
        """Test generating base name from fp32 master weight."""
        base, typename = generate_base_static_name("linear.weight_fp32_master_0_moment1_0")
        self.assertEqual(base, "linear.weight")
        self.assertEqual(typename, "moment1_0")

    def test_moment1_name(self):
        """Test generating base name from moment1."""
        base, typename = generate_base_static_name("linear.weight_moment1_0")
        self.assertEqual(base, "linear.weight")
        self.assertEqual(typename, "moment1_0")

    def test_moment2_name(self):
        """Test generating base name from moment2."""
        base, typename = generate_base_static_name("linear.weight_moment2_0")
        self.assertEqual(base, "linear.weight")
        self.assertEqual(typename, "moment2_0")

    def test_beta1_name(self):
        """Test generating base name from beta1 pow accumulator."""
        base, typename = generate_base_static_name("linear.weight_beta1_pow_acc_0")
        self.assertEqual(base, "linear.weight")
        self.assertEqual(typename, "beta1_pow_acc_0")

    def test_beta2_name(self):
        """Test generating base name from beta2 pow accumulator."""
        base, typename = generate_base_static_name("linear.weight_beta2_pow_acc_0")
        self.assertEqual(base, "linear.weight")
        self.assertEqual(typename, "beta2_pow_acc_0")


if __name__ == "__main__":
    unittest.main()
