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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock


class TestRotaryEmbeddingApplyScaling(unittest.TestCase):
    """Test RotaryEmbedding._apply_scaling method."""

    def test_apply_scaling_returns_correct_shape(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        freqs = emb.inv_freq
        result = emb._apply_scaling(freqs, factor=8)
        self.assertEqual(result.shape, freqs.shape)

    def test_apply_scaling_with_custom_factor(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        freqs = emb.inv_freq
        result = emb._apply_scaling(freqs, factor=4)
        self.assertEqual(result.shape, freqs.shape)

    def test_apply_scaling_with_custom_freq_params(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        freqs = emb.inv_freq
        result = emb._apply_scaling(
            freqs,
            factor=8,
            low_freq_factor=2,
            high_freq_factor=8,
            original_max_position_embeddings=4096,
        )
        self.assertEqual(result.shape, freqs.shape)


class TestRotaryEmbeddingGetFreqsNonRepeated(unittest.TestCase):
    """Test RotaryEmbedding.get_freqs_non_repeated method."""

    def test_basic_freqs(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        result = emb.get_freqs_non_repeated(max_seq_len=128)
        self.assertEqual(result.shape, [128, 32])

    def test_with_offset(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        result = emb.get_freqs_non_repeated(max_seq_len=128, offset=10)
        self.assertEqual(result.shape, [128, 32])

    def test_with_interpolation_factor(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0, seq_len_interpolation_factor=2.0)
        result = emb.get_freqs_non_repeated(max_seq_len=128)
        self.assertEqual(result.shape, [128, 32])


class TestRotaryEmbeddingGetCosSin(unittest.TestCase):
    """Test RotaryEmbedding.get_cos_sin method."""

    def test_cos_sin_shapes(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        cos, sin = emb.get_cos_sin(max_seq_len=128)
        self.assertEqual(cos.shape, [128, 32])
        self.assertEqual(sin.shape, [128, 32])


class TestRotaryEmbeddingGetRotarySeqLen(unittest.TestCase):
    """Test RotaryEmbedding.get_rotary_seq_len method."""

    def test_basic_seq_len(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        mock_config = MagicMock()
        mock_config.sequence_parallel = False
        tensor = paddle.randn([2, 32, 64])
        result = emb.get_rotary_seq_len(tensor, mock_config, packed_seq_params=None)
        self.assertEqual(result, 32)

    def test_sequence_parallel_seq_len(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        mock_config = MagicMock()
        mock_config.sequence_parallel = True
        mock_config.tensor_model_parallel_size = 2
        tensor = paddle.randn([32, 2, 64])
        result = emb.get_rotary_seq_len(tensor, mock_config, packed_seq_params=None)
        self.assertEqual(result, 64)

    def test_with_packed_seq_params(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0)
        mock_config = MagicMock()
        mock_packed = MagicMock()
        mock_packed.max_seqlen_q = 128
        mock_packed.max_seqlen_kv = 128
        tensor = paddle.randn([2, 32, 64])
        result = emb.get_rotary_seq_len(tensor, mock_config, packed_seq_params=mock_packed)
        self.assertEqual(result, 128)


class TestMultimodalRotaryEmbedding(unittest.TestCase):
    """Test MultimodalRotaryEmbedding class."""

    def test_init(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        emb = MultimodalRotaryEmbedding(head_dim=64, rotary_percent=1.0)
        self.assertFalse(emb.rotary_interleaved)

    def test_init_with_interleaved(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        emb = MultimodalRotaryEmbedding(head_dim=64, rotary_percent=1.0, rotary_interleaved=True)
        self.assertTrue(emb.rotary_interleaved)

    def test_forward_basic(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        emb = MultimodalRotaryEmbedding(head_dim=64, rotary_percent=1.0)
        position_ids = paddle.randn([3, 2, 16])  # [3, batch, seqlens]
        result = emb(position_ids, mrope_section=[16, 32, 16])
        self.assertIsNotNone(result)

    def test_forward_asserts_mrope_section(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        emb = MultimodalRotaryEmbedding(head_dim=64, rotary_percent=1.0)
        position_ids = paddle.randn([3, 2, 16])
        with self.assertRaises(AssertionError):
            emb(position_ids, mrope_section=None)

    def test_apply_interleaved_mrope(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        emb = MultimodalRotaryEmbedding(head_dim=64, rotary_percent=1.0)
        freqs = paddle.randn([3, 2, 16, 32])  # [3, bs, seq_len, head_dim//2]
        result = emb.apply_interleaved_mrope(freqs, [16, 32, 16])
        self.assertEqual(result.shape, [2, 16, 32])


class TestRope2DPosEmbRepeated(unittest.TestCase):
    """Test Rope2DPosEmbRepeated class."""

    def test_init(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        emb = Rope2DPosEmbRepeated(head_dim=64, max_height=64, max_width=64)
        self.assertEqual(emb.dim, 64)
        self.assertEqual(emb.max_height, 64)
        self.assertEqual(emb.max_width, 64)

    def test_dim_not_divisible_by_4_raises(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        with self.assertRaises(AssertionError):
            Rope2DPosEmbRepeated(head_dim=30, max_height=64, max_width=64)

    def test_precompute_freqs_cis(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        emb = Rope2DPosEmbRepeated(head_dim=64, max_height=4, max_width=4)
        result = emb._precompute_freqs_cis()
        self.assertEqual(result.shape, [4, 4, 32])

    def test_get_freqs_cis(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        emb = Rope2DPosEmbRepeated(head_dim=64, max_height=4, max_width=4)
        grid_thws = paddle.to_tensor([[1, 2, 2], [1, 4, 4]])
        result = emb.get_freqs_cis(grid_thws)
        # 1*2*2 + 1*4*4 = 4+16 = 20 total tokens
        self.assertEqual(result.shape[0], 20)
        self.assertEqual(result.shape[1], 32)

    def test_get_cos_sin(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        emb = Rope2DPosEmbRepeated(head_dim=64, max_height=4, max_width=4)
        grid_thws = paddle.to_tensor([[1, 2, 2]])
        cos, sin = emb.get_cos_sin(grid_thws)
        self.assertIsNotNone(cos)
        self.assertIsNotNone(sin)

    def test_forward(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            Rope2DPosEmbRepeated,
        )

        emb = Rope2DPosEmbRepeated(head_dim=64, max_height=4, max_width=4)
        grid_thws = paddle.to_tensor([[1, 2, 2]])
        result = emb(grid_thws)
        self.assertIsNotNone(result)
        self.assertIsNotNone(emb.rotary_pos_cos)
        self.assertIsNotNone(emb.rotary_pos_sin)


class TestRotaryEmbeddingWithRopeScaling(unittest.TestCase):
    """Test RotaryEmbedding with rope_scaling enabled."""

    def test_init_with_rope_scaling(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0, rope_scaling=True)
        self.assertIsNotNone(emb.inv_freq)

    def test_forward_with_rope_scaling(self):
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        emb = RotaryEmbedding(head_dim=64, rotary_percent=1.0, rope_scaling=True)
        result = emb(max_seq_len=128)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
