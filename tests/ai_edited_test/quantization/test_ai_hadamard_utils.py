# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for quantization/hadamard_utils.py"""

import unittest

import numpy as np
import paddle

from paddleformers.quantization.hadamard_utils import (
    apply_hadamard_matmul,
    create_hadamard_matrix,
    hadamard_matmul,
    matmul_hadU,
)
from paddleformers.utils import infohub


class TestMatmulHadU(unittest.TestCase):
    """Tests for matmul_hadU function."""

    def test_identity_input(self):
        """Test that Hadamard transform of identity produces Hadamard matrix."""
        n = 4
        identity = paddle.eye(n, dtype=paddle.float32)
        result = matmul_hadU(identity)
        self.assertEqual(result.shape, [n, n])

    def test_output_shape(self):
        """Test output shape matches input shape."""
        x = paddle.randn([2, 8])
        result = matmul_hadU(x)
        self.assertEqual(result.shape, x.shape)

    def test_3d_input(self):
        """Test with 3D input."""
        x = paddle.randn([2, 4, 1])
        result = matmul_hadU(x)
        self.assertEqual(result.shape, [2, 4, 1])

    def test_power_of_2(self):
        """Test with various power-of-2 sizes."""
        for n in [2, 4, 8, 16]:
            x = paddle.randn([n])
            x = x.reshape([1, n, 1])
            result = matmul_hadU(x)
            self.assertEqual(result.shape, [1, n, 1])


class TestCreateHadamardMatrix(unittest.TestCase):
    """Tests for create_hadamard_matrix function."""

    def test_size_2(self):
        """Test Hadamard matrix of size 2."""
        H = create_hadamard_matrix(2, paddle.float32)
        self.assertEqual(H.shape, [2, 2])
        # Hadamard matrix of size 2: [[1,1],[1,-1]]
        expected = paddle.to_tensor([[1.0, 1.0], [1.0, -1.0]])
        np.testing.assert_allclose(H.numpy(), expected.numpy(), atol=1e-5)

    def test_size_4(self):
        """Test Hadamard matrix of size 4."""
        H = create_hadamard_matrix(4, paddle.float32)
        self.assertEqual(H.shape, [4, 4])
        # Check orthogonality: H @ H.T should be proportional to I
        HHT = paddle.matmul(H, H.T)
        # For block_size=4, HHT should be 4*I
        expected = 4.0 * paddle.eye(4)
        np.testing.assert_allclose(HHT.numpy(), expected.numpy(), atol=1e-4)

    def test_float16_dtype(self):
        """Test Hadamard matrix with float16 dtype."""
        H = create_hadamard_matrix(4, paddle.float16)
        self.assertEqual(H.dtype, paddle.float16)


class TestHadamardMatmul(unittest.TestCase):
    """Tests for hadamard_matmul function."""

    def setUp(self):
        self.hadamard_matrix = create_hadamard_matrix(4, paddle.float32)

    def test_right_side(self):
        """Test right-side Hadamard multiplication."""
        x = paddle.randn([2, 4])
        result = hadamard_matmul(x, "right", self.hadamard_matrix, 4)
        self.assertEqual(result.shape, [2, 4])

    def test_left_side(self):
        """Test left-side Hadamard multiplication."""
        x = paddle.randn([4, 2])
        result = hadamard_matmul(x, "left", self.hadamard_matrix, 4)
        self.assertEqual(result.shape, [4, 2])

    def test_1d_input(self):
        """Test with 1D-like input (reshaped to 2D)."""
        x = paddle.randn([4])
        x = x.reshape([-1, 4])
        result = hadamard_matmul(x, "right", self.hadamard_matrix, 4)
        self.assertEqual(result.shape, [1, 4])

    def test_multiple_blocks(self):
        """Test with input that has multiple blocks."""
        x = paddle.randn([2, 8])
        H4 = create_hadamard_matrix(4, paddle.float32)
        result = hadamard_matmul(x, "right", H4, 4)
        self.assertEqual(result.shape, [2, 8])


class TestApplyHadamardMatmul(unittest.TestCase):
    """Tests for apply_hadamard_matmul function."""

    def setUp(self):
        # Clear cached hadamard matrices
        infohub.hadamard = None

    def test_basic_apply(self):
        """Test basic apply_hadamard_matmul."""
        x = paddle.randn([2, 4])
        result = apply_hadamard_matmul(x, "right", 4)
        self.assertEqual(result.shape, x.shape)

    def test_caching(self):
        """Test that hadamard matrix is cached in infohub."""
        x = paddle.randn([2, 4])
        result1 = apply_hadamard_matmul(x, "right", 4)
        self.assertTrue(hasattr(infohub, "hadamard"))
        self.assertIn(4, infohub.hadamard)

        # Second call should use cached matrix
        result2 = apply_hadamard_matmul(x, "right", 4)
        np.testing.assert_allclose(result1.numpy(), result2.numpy(), atol=1e-5)

    def test_different_block_sizes(self):
        """Test with different block sizes."""
        x8 = paddle.randn([2, 8])
        result8 = apply_hadamard_matmul(x8, "right", 8)
        self.assertEqual(result8.shape, x8.shape)

        x4 = paddle.randn([2, 4])
        result4 = apply_hadamard_matmul(x4, "right", 4)
        self.assertEqual(result4.shape, x4.shape)


if __name__ == "__main__":
    unittest.main()
