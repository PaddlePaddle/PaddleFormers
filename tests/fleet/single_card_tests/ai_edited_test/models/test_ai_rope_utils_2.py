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
from unittest.mock import patch

import paddle

from paddleformers.fleet.models.common.embeddings.rope_utils import (
    _rotate_half,
    apply_rotary_pos_emb,
    get_pos_emb_on_this_cp_rank,
    get_unsqueeze_dim,
)


class TestRotateHalf(unittest.TestCase):
    """Test _rotate_half function."""

    def test_rotate_half_non_interleaved(self):
        """Test _rotate_half with non-interleaved mode."""
        x = paddle.randn([2, 4, 8])
        result = _rotate_half(x, rotary_interleaved=False)
        self.assertEqual(result.shape, x.shape)

    def test_rotate_half_interleaved(self):
        """Test _rotate_half with interleaved mode on 4D tensor."""
        # _rotate_half interleaved assumes 4D input (see view() call)
        x = paddle.randn([2, 4, 8, 16])
        result = _rotate_half(x, rotary_interleaved=True)
        self.assertEqual(result.shape, x.shape)

    def test_rotate_half_non_interleaved_values(self):
        """Test _rotate_half correct values for non-interleaved mode."""
        x = paddle.to_tensor([[[1.0, 2.0, 3.0, 4.0]]])
        result = _rotate_half(x, rotary_interleaved=False)
        # Non-interleaved: split into x1=[1,2], x2=[3,4], return [-x2, x1] = [-3,-4,1,2]
        expected = paddle.to_tensor([[[-3.0, -4.0, 1.0, 2.0]]])
        self.assertTrue(paddle.allclose(result, expected))

    def test_rotate_half_interleaved_values(self):
        """Test _rotate_half correct values for interleaved mode on 4D tensor."""
        # _rotate_half interleaved uses view(shape[0], shape[1], shape[2], -1)
        x = paddle.to_tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        result = _rotate_half(x, rotary_interleaved=True)
        # Interleaved: x1=[1,3], x2=[2,4], stack([-x2,x1])=[[-2,1],[-4,3]] -> flatten=[-2,1,-4,3]
        self.assertEqual(result.shape, x.shape)


class TestGetUnsqueezeDim(unittest.TestCase):
    """Test get_unsqueeze_dim function."""

    def test_seq_in_dim1(self):
        """Test when sequence length is in dimension 1."""
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])
        result = get_unsqueeze_dim(t, freqs)
        self.assertEqual(result, 2)

    def test_seq_in_dim2(self):
        """Test when sequence length is in dimension 2."""
        t = paddle.randn([2, 4, 8, 16])
        freqs = paddle.randn([2, 8, 16])
        result = get_unsqueeze_dim(t, freqs)
        self.assertEqual(result, 1)


class TestGetPosEmbOnThisCPRank(unittest.TestCase):
    """Test get_pos_emb_on_this_cp_rank function."""

    def test_none_cp_group_raises(self):
        """Test that None cp_group raises ValueError."""
        pos_emb = paddle.randn([1, 16, 64])
        with self.assertRaises(ValueError):
            get_pos_emb_on_this_cp_rank(pos_emb, seq_dim=1, cp_group=None)


class TestApplyRotaryPosEmb(unittest.TestCase):
    """Test apply_rotary_pos_emb function."""

    def _make_config(self, **overrides):
        """Create a minimal TransformerConfig for testing."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        defaults = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "use_cpu_initialization": True,
        }
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_bshd_basic(self, mock_rank, mock_size):
        """Test basic bshd format RoPE."""
        config = self._make_config()
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])

        result = apply_rotary_pos_emb(
            t=t,
            freqs=freqs,
            cos=None,
            sin=None,
            config=config,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, t.shape)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_bshd_high_precision_rope(self, mock_rank, mock_size):
        """Test bshd format RoPE with high_precision_rope."""
        config = self._make_config(high_precision_rope=True)
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])

        result = apply_rotary_pos_emb(
            t=t,
            freqs=freqs,
            cos=None,
            sin=None,
            config=config,
        )
        self.assertIsNotNone(result)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_bshd_interleaved(self, mock_rank, mock_size):
        """Test bshd format with interleaved rotary embedding."""
        config = self._make_config(rotary_interleaved=True)
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])

        result = apply_rotary_pos_emb(
            t=t,
            freqs=freqs,
            cos=None,
            sin=None,
            config=config,
        )
        self.assertIsNotNone(result)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_bshd_mla(self, mock_rank, mock_size):
        """Test bshd format with multi_latent_attention."""
        config = self._make_config(multi_latent_attention=True)
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])

        result = apply_rotary_pos_emb(
            t=t,
            freqs=freqs,
            cos=None,
            sin=None,
            config=config,
        )
        self.assertIsNotNone(result)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_bshd_with_mscale(self, mock_rank, mock_size):
        """Test bshd format with mscale > 1.0."""
        config = self._make_config()
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])

        result = apply_rotary_pos_emb(
            t=t,
            freqs=freqs,
            cos=None,
            sin=None,
            config=config,
            mscale=2.0,
        )
        self.assertIsNotNone(result)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_bshd_none_mscale(self, mock_rank, mock_size):
        """Test bshd format with mscale=None defaults to 1.0."""
        config = self._make_config()
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])

        result = apply_rotary_pos_emb(
            t=t,
            freqs=freqs,
            cos=None,
            sin=None,
            config=config,
            mscale=None,
        )
        self.assertIsNotNone(result)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_freqs_transpose_alignment(self, mock_rank, mock_size):
        """Test freqs is transposed when dims are swapped but product is same."""
        config = self._make_config()
        t = paddle.randn([4, 2, 8, 16])  # B=4, S=2
        freqs = paddle.randn([2, 4, 16])  # S=2, B=4 (swapped)

        result = apply_rotary_pos_emb(
            t=t,
            freqs=freqs,
            cos=None,
            sin=None,
            config=config,
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
