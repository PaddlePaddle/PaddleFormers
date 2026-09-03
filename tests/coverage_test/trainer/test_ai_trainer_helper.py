# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/utils/helper.py"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddleformers.trainer.utils.helper import (
    distributed_isfile,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    numpy_pad_and_concatenate,
    paddle_pad_and_concatenate,
)


class TestPaddlePadAndConcatenate(unittest.TestCase):
    """Tests for paddle_pad_and_concatenate function."""

    def test_same_shape_concat(self):
        """Test concatenation with same second dimension."""
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 3])
        result = paddle_pad_and_concatenate(t1, t2)
        self.assertEqual(result.shape, [6, 3])

    def test_different_shape_padding(self):
        """Test concatenation with different second dimension, requiring padding."""
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 5])
        result = paddle_pad_and_concatenate(t1, t2)
        self.assertEqual(result.shape, [6, 5])

    def test_1d_concat(self):
        """Test concatenation of 1D tensors."""
        t1 = paddle.randn([3])
        t2 = paddle.randn([5])
        result = paddle_pad_and_concatenate(t1, t2)
        self.assertEqual(result.shape, [8])

    def test_custom_padding_index(self):
        """Test with custom padding index."""
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 5])
        result = paddle_pad_and_concatenate(t1, t2, padding_index=0)
        self.assertEqual(result.shape, [6, 5])


class TestNumpyPadAndConcatenate(unittest.TestCase):
    """Tests for numpy_pad_and_concatenate function."""

    def test_same_shape_concat(self):
        """Test numpy concatenation with same second dimension."""
        a1 = np.random.randn(2, 3).astype(np.float32)
        a2 = np.random.randn(4, 3).astype(np.float32)
        result = numpy_pad_and_concatenate(a1, a2)
        self.assertEqual(result.shape, (6, 3))

    def test_different_shape_padding(self):
        """Test numpy concatenation with different second dimension."""
        a1 = np.random.randn(2, 3).astype(np.float32)
        a2 = np.random.randn(4, 5).astype(np.float32)
        result = numpy_pad_and_concatenate(a1, a2)
        self.assertEqual(result.shape, (6, 5))

    def test_1d_concat(self):
        """Test numpy concatenation of 1D arrays."""
        a1 = np.array([1.0, 2.0, 3.0])
        a2 = np.array([4.0, 5.0])
        result = numpy_pad_and_concatenate(a1, a2)
        self.assertEqual(result.shape, (5,))


class TestNestedConcat(unittest.TestCase):
    """Tests for nested_concat function."""

    def test_tensor_concat(self):
        """Test nested concatenation of tensors."""
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 3])
        result = nested_concat(t1, t2)
        self.assertEqual(result.shape, [6, 3])

    def test_numpy_concat(self):
        """Test nested concatenation of numpy arrays."""
        a1 = np.random.randn(2, 3).astype(np.float32)
        a2 = np.random.randn(4, 3).astype(np.float32)
        result = nested_concat(a1, a2)
        self.assertEqual(result.shape, (6, 3))

    def test_list_concat(self):
        """Test nested concatenation of lists of tensors."""
        t1 = [paddle.randn([2, 3]), paddle.randn([4, 3])]
        t2 = [paddle.randn([4, 3]), paddle.randn([6, 3])]
        result = nested_concat(t1, t2)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, [6, 3])
        self.assertEqual(result[1].shape, [10, 3])

    def test_type_mismatch_raises(self):
        """Test that mismatched types raise assertion."""
        t1 = paddle.randn([2, 3])
        a1 = np.random.randn(2, 3)
        with self.assertRaises(AssertionError):
            nested_concat(t1, a1)


class TestNestedDetach(unittest.TestCase):
    """Tests for nested_detach function."""

    def test_tensor_detach(self):
        """Test detaching a single tensor."""
        t = paddle.randn([2, 3])
        t.stop_gradient = False
        result = nested_detach(t)
        self.assertTrue(result.stop_gradient)

    def test_list_detach(self):
        """Test detaching a list of tensors."""
        t1 = paddle.randn([2, 3])
        t1.stop_gradient = False
        t2 = paddle.randn([4, 3])
        t2.stop_gradient = False
        result = nested_detach([t1, t2])
        self.assertIsInstance(result, list)
        self.assertTrue(result[0].stop_gradient)
        self.assertTrue(result[1].stop_gradient)

    def test_tuple_detach(self):
        """Test detaching a tuple of tensors."""
        t1 = paddle.randn([2, 3])
        t1.stop_gradient = False
        result = nested_detach((t1,))
        self.assertIsInstance(result, tuple)


class TestNestedNumpify(unittest.TestCase):
    """Tests for nested_numpify function."""

    def test_tensor_numpify(self):
        """Test numpifying a single tensor."""
        t = paddle.randn([2, 3])
        result = nested_numpify(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3))

    def test_float16_upcast(self):
        """Test that float16 tensors are upcast to float32."""
        t = paddle.randn([2, 3]).cast(paddle.float16)
        result = nested_numpify(t)
        self.assertEqual(result.dtype, np.float32)

    def test_list_numpify(self):
        """Test numpifying a list of tensors."""
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 3])
        result = nested_numpify([t1, t2])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)


class TestNestedTruncate(unittest.TestCase):
    """Tests for nested_truncate function."""

    def test_tensor_truncate(self):
        """Test truncating a single tensor."""
        t = paddle.randn([10])
        result = nested_truncate(t, 5)
        self.assertEqual(result.shape, [5])

    def test_list_truncate(self):
        """Test truncating a list of tensors."""
        t1 = paddle.randn([10])
        t2 = paddle.randn([8])
        result = nested_truncate([t1, t2], 5)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0].shape, [5])
        self.assertEqual(result[1].shape, [5])

    def test_tuple_truncate(self):
        """Test truncating a tuple of tensors."""
        t1 = paddle.randn([10])
        result = nested_truncate((t1,), 3)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0].shape, [3])


class TestDistributedIsfile(unittest.TestCase):
    """Tests for distributed_isfile function."""

    def test_existing_file_single_node(self):
        """Test that existing file returns True on single node."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"test")
            tmp_path = f.name
        try:
            with patch.dict(os.environ, {"PADDLE_TRAINERS_NUM": "1"}):
                result = distributed_isfile(tmp_path)
                self.assertTrue(result)
        finally:
            os.unlink(tmp_path)

    def test_nonexistent_file_single_node(self):
        """Test that non-existent file returns False on single node."""
        with patch.dict(os.environ, {"PADDLE_TRAINERS_NUM": "1"}):
            result = distributed_isfile("/nonexistent/path/file.json")
            self.assertFalse(result)

    def test_bin_json_equivalent(self):
        """Test that .json and .bin files are checked interchangeably."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"test")
            bin_path = f.name
        json_path = bin_path.replace(".bin", ".json")
        try:
            with patch.dict(os.environ, {"PADDLE_TRAINERS_NUM": "1"}):
                # Requesting .json but only .bin exists
                result = distributed_isfile(json_path)
                self.assertTrue(result)
        finally:
            os.unlink(bin_path)


if __name__ == "__main__":
    unittest.main()
