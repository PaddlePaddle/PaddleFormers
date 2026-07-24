# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import MagicMock, patch

import paddle


class TestRotaryEmbeddingInit(unittest.TestCase):
    """Tests for RotaryEmbedding initialization."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_basic_init(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
        )
        self.assertFalse(rope.rotary_interleaved)
        self.assertIsNone(rope.seq_len_interpolation_factor)
        self.assertIsNotNone(rope.inv_freq)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_init_with_partial_rotary(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=0.5,
        )
        self.assertIsNotNone(rope.inv_freq)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_init_with_interleaved(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            rotary_interleaved=True,
        )
        self.assertTrue(rope.rotary_interleaved)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_init_with_interpolation_factor(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            seq_len_interpolation_factor=2.0,
        )
        self.assertEqual(rope.seq_len_interpolation_factor, 2.0)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_init_with_rope_scaling(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            rope_scaling=True,
            rope_scaling_factor=8.0,
        )
        self.assertIsNotNone(rope.inv_freq)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_init_with_custom_rotary_base(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            rotary_base=500000.0,
        )
        self.assertIsNotNone(rope.inv_freq)


class TestRotaryEmbeddingForward(unittest.TestCase):
    """Tests for RotaryEmbedding forward pass."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_basic(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        result = rope(max_seq_len=128)
        # Expected shape: [1, 128, 1, 64]
        self.assertEqual(result.shape, [1, 128, 1, 64])

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_with_offset(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        result = rope(max_seq_len=128, offset=10)
        self.assertEqual(result.shape, [1, 128, 1, 64])

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_with_interpolation(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            seq_len_interpolation_factor=2.0,
        )
        result = rope(max_seq_len=128)
        self.assertEqual(result.shape, [1, 128, 1, 64])

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_interleaved(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
            rotary_interleaved=True,
        )
        result = rope(max_seq_len=128)
        self.assertEqual(result.shape, [1, 128, 1, 64])

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_small_seq(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=32, rotary_percent=0.5)
        result = rope(max_seq_len=16)
        # With rotary_percent=0.5, dim = int(32 * 0.5) = 16
        # After cat((freqs, freqs), axis=-1), shape is [1, 16, 1, 16]
        self.assertEqual(result.shape, [1, 16, 1, 16])


class TestRotaryEmbeddingGetCosSin(unittest.TestCase):
    """Tests for RotaryEmbedding get_cos_sin method."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_get_cos_sin_basic(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        cos, sin = rope.get_cos_sin(max_seq_len=32)
        self.assertEqual(cos.shape, [32, 32])
        self.assertEqual(sin.shape, [32, 32])

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_get_cos_sin_with_offset(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        cos, sin = rope.get_cos_sin(max_seq_len=32, offset=10)
        self.assertEqual(cos.shape, [32, 32])
        self.assertEqual(sin.shape, [32, 32])


class TestRotaryEmbeddingGetFreqsNonRepeated(unittest.TestCase):
    """Tests for RotaryEmbedding get_freqs_non_repeated method."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_get_freqs_non_repeated(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        freqs = rope.get_freqs_non_repeated(max_seq_len=32)
        self.assertEqual(freqs.shape, [32, 32])

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_get_freqs_with_offset(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        freqs = rope.get_freqs_non_repeated(max_seq_len=32, offset=5)
        self.assertEqual(freqs.shape, [32, 32])


class TestRotaryEmbeddingApplyScaling(unittest.TestCase):
    """Tests for RotaryEmbedding._apply_scaling method."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_apply_scaling_default_params(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        freqs = rope.inv_freq.clone()
        scaled = rope._apply_scaling(freqs)
        # Scaled frequencies should differ from original
        self.assertEqual(scaled.shape, freqs.shape)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_apply_scaling_custom_factor(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        freqs = rope.inv_freq.clone()
        scaled = rope._apply_scaling(freqs, factor=16.0)
        self.assertEqual(scaled.shape, freqs.shape)


class TestRotaryEmbeddingGetRotarySeqLen(unittest.TestCase):
    """Tests for RotaryEmbedding.get_rotary_seq_len method."""

    def _make_config(self, **overrides):
        from paddleformers.fleet.transformer.transformer_config import (
            TransformerConfig,
        )

        defaults = {
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
        }
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_get_rotary_seq_len_basic(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        config = self._make_config()
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        transformer_input = paddle.randn([2, 16, 64], dtype="float32")
        seq_len = rope.get_rotary_seq_len(transformer_input, config)
        self.assertEqual(seq_len, 16)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_get_rotary_seq_len_with_packed_seq(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        config = self._make_config()
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        transformer_input = paddle.randn([2, 16, 64], dtype="float32")

        packed_params = MagicMock()
        packed_params.max_seqlen_q = 32
        packed_params.max_seqlen_kv = 24

        seq_len = rope.get_rotary_seq_len(
            transformer_input, config, packed_seq_params=packed_params
        )
        self.assertEqual(seq_len, 32)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_get_rotary_seq_len_with_cp_group(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        mock_group = MagicMock()
        mock_group.world_size = 2
        mock_ps.get_context_parallel_group.return_value = mock_group
        config = self._make_config()
        rope = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        transformer_input = paddle.randn([2, 16, 64], dtype="float32")
        seq_len = rope.get_rotary_seq_len(transformer_input, config)
        # With cp_group.world_size=2, should be 16 * 2 = 32
        self.assertEqual(seq_len, 32)


class TestMultimodalRotaryEmbedding(unittest.TestCase):
    """Tests for MultimodalRotaryEmbedding."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_init(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        mrope = MultimodalRotaryEmbedding(
            head_dim=64,
            rotary_percent=1.0,
        )
        self.assertIsNotNone(mrope.inv_freq)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        head_dim = 64
        half_dim = head_dim // 2  # 32
        mrope = MultimodalRotaryEmbedding(
            head_dim=head_dim,
            rotary_percent=1.0,
        )

        batch_size = 2
        seq_len = 16
        position_ids = paddle.randint(0, 100, [3, batch_size, seq_len])
        mrope_section = [
            half_dim // 3,
            half_dim // 3,
            half_dim - 2 * (half_dim // 3),
        ]

        result = mrope(position_ids, mrope_section)
        # Expected shape: [batch_size, seq_len, head_dim]
        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[1], seq_len)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_forward_with_interpolation(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        head_dim = 64
        half_dim = head_dim // 2
        mrope = MultimodalRotaryEmbedding(
            head_dim=head_dim,
            rotary_percent=1.0,
            seq_len_interpolation_factor=2.0,
        )

        batch_size = 2
        seq_len = 16
        position_ids = paddle.randint(0, 100, [3, batch_size, seq_len])
        mrope_section = [
            half_dim // 3,
            half_dim // 3,
            half_dim - 2 * (half_dim // 3),
        ]

        result = mrope(position_ids, mrope_section)
        self.assertEqual(result.shape[0], batch_size)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rotary_pos_embedding.parallel_state"
    )
    def test_apply_interleaved_mrope(self, mock_ps):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        mock_ps.get_context_parallel_group.return_value = None
        head_dim = 64
        half_dim = head_dim // 2
        mrope = MultimodalRotaryEmbedding(head_dim=head_dim, rotary_percent=1.0)

        batch_size = 2
        seq_len = 8
        freqs = paddle.randn(
            [3, batch_size, seq_len, half_dim], dtype="float32"
        )
        mrope_section = [
            half_dim // 3,
            half_dim // 3,
            half_dim - 2 * (half_dim // 3),
        ]

        result = mrope.apply_interleaved_mrope(freqs, mrope_section)
        self.assertEqual(result.shape, [batch_size, seq_len, half_dim])


class TestRope2DPosEmbRepeated(unittest.TestCase):
    """Tests for Rope2DPosEmbRepeated."""

    def test_init(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        rope2d = Rope2DPosEmbRepeated(
            head_dim=64,
            max_height=32,
            max_width=32,
        )
        self.assertEqual(rope2d.dim, 64)
        self.assertEqual(rope2d.max_height, 32)
        self.assertEqual(rope2d.max_width, 32)

    def test_init_requires_divisible_by_4(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        with self.assertRaises(AssertionError):
            Rope2DPosEmbRepeated(head_dim=30, max_height=32, max_width=32)

    def test_precompute_freqs_cis(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        rope2d = Rope2DPosEmbRepeated(
            head_dim=64,
            max_height=8,
            max_width=8,
        )
        freqs_cis = rope2d._precompute_freqs_cis()
        self.assertEqual(freqs_cis.shape, [8, 8, 32])

    def test_forward(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        rope2d = Rope2DPosEmbRepeated(
            head_dim=64,
            max_height=16,
            max_width=16,
        )
        # grid_thws: [t, h, w] pairs
        grid_thws = paddle.to_tensor([[1, 8, 8], [2, 4, 4]], dtype="int64")
        result = rope2d(grid_thws)
        # 1*8*8 + 2*4*4 = 64 + 32 = 96 tokens
        self.assertEqual(result.shape[0], 96)
        self.assertEqual(result.shape[1], 32)

    def test_get_cos_sin(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        rope2d = Rope2DPosEmbRepeated(
            head_dim=64,
            max_height=8,
            max_width=8,
        )
        grid_thws = paddle.to_tensor([[1, 4, 4]], dtype="int64")
        cos, sin = rope2d.get_cos_sin(grid_thws)
        self.assertEqual(cos.shape[0], 16)
        self.assertEqual(sin.shape[0], 16)

    def test_get_cos_sin_caches(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        rope2d = Rope2DPosEmbRepeated(
            head_dim=64,
            max_height=8,
            max_width=8,
        )
        grid_thws = paddle.to_tensor([[1, 4, 4]], dtype="int64")
        # First call computes and caches
        cos1, sin1 = rope2d.get_cos_sin(grid_thws)
        # Second call should use cache
        cos2, sin2 = rope2d.get_cos_sin(grid_thws)
        self.assertTrue(paddle.allclose(cos1, cos2))
        self.assertTrue(paddle.allclose(sin1, sin2))

    def test_forward_multiple_grids(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        rope2d = Rope2DPosEmbRepeated(
            head_dim=64,
            max_height=16,
            max_width=16,
        )
        grid_thws = paddle.to_tensor(
            [
                [3, 4, 4],
                [1, 8, 8],
                [2, 2, 2],
            ],
            dtype="int64",
        )
        result = rope2d(grid_thws)
        expected_len = 3 * 16 + 1 * 64 + 2 * 4  # 48 + 64 + 8 = 120
        self.assertEqual(result.shape[0], expected_len)

    def test_rotary_pos_cos_sin_set_after_forward(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        rope2d = Rope2DPosEmbRepeated(
            head_dim=64,
            max_height=8,
            max_width=8,
        )
        self.assertIsNone(rope2d.rotary_pos_cos)
        self.assertIsNone(rope2d.rotary_pos_sin)

        grid_thws = paddle.to_tensor([[1, 4, 4]], dtype="int64")
        rope2d(grid_thws)

        self.assertIsNotNone(rope2d.rotary_pos_cos)
        self.assertIsNotNone(rope2d.rotary_pos_sin)


if __name__ == "__main__":
    unittest.main()
