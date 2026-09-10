# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/utils/reshard/sharding_v2.py"""

import unittest

import paddle

from paddleformers.trainer.utils.reshard.sharding_v2 import (
    is_bata,
    merge_tensors,
    pad_tensor,
    slice_tensor,
)


class TestIsBata(unittest.TestCase):
    """Tests for is_bata function."""

    def test_beta1_pow_returns_true(self):
        """Test that beta1_pow_acc name returns True."""
        self.assertTrue(is_bata("param_beta1_pow_acc_0"))

    def test_beta2_pow_returns_true(self):
        """Test that beta2_pow_acc name returns True."""
        self.assertTrue(is_bata("param_beta2_pow_acc_0"))

    def test_moment_returns_false(self):
        """Test that moment name returns False."""
        self.assertFalse(is_bata("param_moment1_0"))

    def test_regular_name_returns_false(self):
        """Test that regular parameter name returns False."""
        self.assertFalse(is_bata("linear_weight"))

    def test_fp32_master_beta_returns_true(self):
        """Test that fp32 master beta name returns True."""
        self.assertTrue(is_bata("param_fp32_master_0_beta1_pow_acc_0"))
        self.assertTrue(is_bata("param_fp32_master_0_beta2_pow_acc_0"))


class TestMergeTensors(unittest.TestCase):
    """Tests for merge_tensors function."""

    def test_single_tensor(self):
        """Test merging a single tensor."""
        t = paddle.randn([8])
        result = merge_tensors("test_key", [t], [2, 4])
        self.assertEqual(list(result.shape), [2, 4])

    def test_multiple_tensors(self):
        """Test merging multiple tensors by concatenation."""
        t1 = paddle.randn([4])
        t2 = paddle.randn([4])
        result = merge_tensors("test_key", [t1, t2], [2, 4])
        self.assertEqual(list(result.shape), [2, 4])

    def test_padded_tensor_truncated(self):
        """Test that padded tensor is truncated to correct shape."""
        t = paddle.zeros([10])
        result = merge_tensors("test_key", [t], [2, 4])
        self.assertEqual(list(result.shape), [2, 4])


class TestPadTensor(unittest.TestCase):
    """Tests for pad_tensor function."""

    def test_basic_padding(self):
        """Test padding a tensor to a larger size."""
        tensor = paddle.randn([2, 3])
        padded_size = 8
        result = pad_tensor("test_key", tensor, padded_size)
        self.assertEqual(result.shape[0], padded_size)

    def test_no_padding_needed(self):
        """Test padding when tensor already matches size."""
        tensor = paddle.randn([4])
        result = pad_tensor("test_key", tensor, 4)
        self.assertEqual(result.shape[0], 4)


class TestSliceTensor(unittest.TestCase):
    """Tests for slice_tensor function."""

    def test_basic_slice(self):
        """Test basic tensor slicing."""
        tensor = paddle.randn([10])
        result = slice_tensor(tensor, 2, 7)
        self.assertEqual(result.shape[0], 5)

    def test_full_slice(self):
        """Test slicing entire tensor."""
        tensor = paddle.randn([5])
        result = slice_tensor(tensor, 0, 5)
        self.assertEqual(result.shape[0], 5)

    def test_single_element_slice(self):
        """Test slicing a single element."""
        tensor = paddle.randn([5])
        result = slice_tensor(tensor, 2, 3)
        self.assertEqual(result.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
