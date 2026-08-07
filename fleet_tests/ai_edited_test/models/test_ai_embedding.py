# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.models.kimi_k25.embedding import (
    Learnable2DInterpPosEmbDivided_fixed,
    VisionEmbeddingSpec,
    get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
)


class TestVisionEmbeddingSpec(unittest.TestCase):
    """Test VisionEmbeddingSpec dataclass."""

    def test_default_rope_embedding_is_none(self):
        """Test default rope_embedding is None."""
        spec = VisionEmbeddingSpec()
        self.assertIsNone(spec.rope_embedding)

    def test_with_rope_embedding(self):
        """Test setting rope_embedding."""
        mock_rope = MagicMock()
        spec = VisionEmbeddingSpec(rope_embedding=mock_rope)
        self.assertEqual(spec.rope_embedding, mock_rope)


class TestGet1dSincosPosEmbedFromGrid(unittest.TestCase):
    """Test get_1d_sincos_pos_embed_from_grid function."""

    def test_output_shape(self):
        """Test output shape is (M, D)."""
        pos = paddle.arange(10, dtype=paddle.float32)
        result = get_1d_sincos_pos_embed_from_grid(16, pos)
        self.assertEqual(result.shape, [10, 16])

    def test_embed_dim_must_be_even(self):
        """Test that odd embed_dim raises AssertionError."""
        pos = paddle.arange(10, dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            get_1d_sincos_pos_embed_from_grid(7, pos)

    def test_single_position(self):
        """Test with single position."""
        pos = paddle.to_tensor([0.0])
        result = get_1d_sincos_pos_embed_from_grid(8, pos)
        self.assertEqual(result.shape, [1, 8])

    def test_sin_cos_structure(self):
        """Test that first half is sin and second half is cos."""
        pos = paddle.arange(5, dtype=paddle.float32)
        result = get_1d_sincos_pos_embed_from_grid(8, pos)
        # First 4 columns should be sin, last 4 should be cos
        self.assertEqual(result.shape, [5, 8])


class TestGet1dSincosPosEmbed(unittest.TestCase):
    """Test get_1d_sincos_pos_embed function."""

    def test_without_cls_token(self):
        """Test output shape without cls_token."""
        result = get_1d_sincos_pos_embed(16, t_size=5)
        self.assertEqual(result.shape, [5, 16])

    def test_with_cls_token(self):
        """Test output shape with cls_token."""
        result = get_1d_sincos_pos_embed(16, t_size=5, cls_token=True)
        self.assertEqual(result.shape, [6, 16])

    def test_cls_token_first_row_is_zero(self):
        """Test that first row is all zeros when cls_token=True."""
        result = get_1d_sincos_pos_embed(8, t_size=3, cls_token=True)
        self.assertTrue(paddle.allclose(result[0], paddle.zeros([8])))


class TestLearnable2DInterpPosEmb(unittest.TestCase):
    """Test Learnable2DInterpPosEmbDivided_fixed class."""

    def test_init_creates_weight(self):
        """Test that init creates weight parameter."""
        emb = Learnable2DInterpPosEmbDivided_fixed(
            height=4,
            width=4,
            num_frames=2,
            dim=32,
        )
        self.assertIsNotNone(emb.weight)
        self.assertEqual(emb.weight.shape, [4, 4, 32])

    def test_init_creates_time_weight(self):
        """Test that init creates time_weight buffer."""
        emb = Learnable2DInterpPosEmbDivided_fixed(
            height=4,
            width=4,
            num_frames=2,
            dim=32,
        )
        self.assertIsNotNone(emb.time_weight)
        # time_weight should be [num_frames, 1, dim]
        self.assertEqual(emb.time_weight.shape[0], 2)

    def test_forward_same_size(self):
        """Test forward when grid_thws matches weight size."""
        emb = Learnable2DInterpPosEmbDivided_fixed(
            height=4,
            width=4,
            num_frames=1,
            dim=32,
        )
        x = paddle.randn([16, 32])
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        result = emb(x, grid_thws)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 16)

    def test_forward_t_greater_than_num_frames_raises(self):
        """Test forward raises when t > num_frames."""
        emb = Learnable2DInterpPosEmbDivided_fixed(
            height=4,
            width=4,
            num_frames=1,
            dim=32,
        )
        x = paddle.randn([32, 32])
        grid_thws = paddle.to_tensor([[2, 4, 4]])  # t=2 > num_frames=1
        with self.assertRaises(AssertionError):
            emb(x, grid_thws)


if __name__ == "__main__":
    unittest.main()
