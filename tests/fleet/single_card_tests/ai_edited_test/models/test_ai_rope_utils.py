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
from unittest.mock import patch

import paddle

from paddleformers.fleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    _apply_rotary_pos_emb_thd,
    _get_thd_freqs_on_this_cp_rank,
    apply_rotary_pos_emb,
)


class TestApplyRotaryPosEmbBSHD(unittest.TestCase):
    """Test _apply_rotary_pos_emb_bshd function directly."""

    def test_basic_bshd(self):
        """Test basic bshd RoPE application."""
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])
        result = _apply_rotary_pos_emb_bshd(
            t=t,
            freqs=freqs,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, t.shape)

    def test_bshd_with_high_precision(self):
        """Test bshd with high_precision_rope=True."""
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])
        result = _apply_rotary_pos_emb_bshd(
            t=t,
            freqs=freqs,
            high_precision_rope=True,
        )
        self.assertIsNotNone(result)

    def test_bshd_interleaved(self):
        """Test bshd with interleaved rotary embedding."""
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])
        result = _apply_rotary_pos_emb_bshd(
            t=t,
            freqs=freqs,
            rotary_interleaved=True,
        )
        self.assertIsNotNone(result)

    def test_bshd_mla(self):
        """Test bshd with multi_latent_attention."""
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])
        result = _apply_rotary_pos_emb_bshd(
            t=t,
            freqs=freqs,
            multi_latent_attention=True,
        )
        self.assertIsNotNone(result)

    def test_bshd_with_mscale_none(self):
        """Test bshd with mscale=None defaults to 1.0."""
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])
        result = _apply_rotary_pos_emb_bshd(
            t=t,
            freqs=freqs,
            mscale=None,
        )
        self.assertIsNotNone(result)

    def test_bshd_with_mscale(self):
        """Test bshd with mscale > 1.0."""
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])
        result = _apply_rotary_pos_emb_bshd(
            t=t,
            freqs=freqs,
            mscale=2.0,
        )
        self.assertIsNotNone(result)

    def test_bshd_freqs_transpose(self):
        """Test bshd with freqs needing transpose for alignment."""
        t = paddle.randn([4, 2, 8, 16])  # B=4, S=2
        freqs = paddle.randn([2, 4, 16])  # S=2, B=4 (swapped)
        result = _apply_rotary_pos_emb_bshd(
            t=t,
            freqs=freqs,
        )
        self.assertIsNotNone(result)

    def test_bshd_rot_dim_less_than_hidden(self):
        """Test bshd when rot_dim < hidden_dim (partial rotation)."""
        t = paddle.randn([2, 8, 4, 32])
        freqs = paddle.randn([2, 8, 16])  # rot_dim=16 < 32
        result = _apply_rotary_pos_emb_bshd(
            t=t,
            freqs=freqs,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, t.shape)

    def test_bshd_high_precision_with_interleaved(self):
        """Test bshd with both high_precision and interleaved."""
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([2, 8, 16])
        result = _apply_rotary_pos_emb_bshd(
            t=t,
            freqs=freqs,
            high_precision_rope=True,
            rotary_interleaved=True,
        )
        self.assertIsNotNone(result)


class TestGetThdFreqsOnThisCPRank(unittest.TestCase):
    """Test _get_thd_freqs_on_this_cp_rank function."""

    def test_single_rank_no_offset(self):
        """Test single rank with offset=0."""
        x = paddle.randn([2, 8, 64])
        freqs = paddle.randn([1, 16, 64])
        result = _get_thd_freqs_on_this_cp_rank(
            cp_rank=0,
            cp_size=1,
            x=x,
            freqs=freqs,
            offset=0,
        )
        self.assertIsNotNone(result)

    def test_single_rank_with_offset(self):
        """Test single rank with offset > 0."""
        x = paddle.randn([2, 8, 64])
        freqs = paddle.randn([1, 32, 64])
        result = _get_thd_freqs_on_this_cp_rank(
            cp_rank=0,
            cp_size=1,
            x=x,
            freqs=freqs,
            offset=4,
        )
        self.assertIsNotNone(result)

    def test_multi_rank(self):
        """Test multi-rank CP slicing."""
        x = paddle.randn([2, 8, 64])
        freqs = paddle.randn([1, 16, 64])
        result = _get_thd_freqs_on_this_cp_rank(
            cp_rank=0,
            cp_size=2,
            x=x,
            freqs=freqs,
            offset=0,
        )
        self.assertIsNotNone(result)

    def test_multi_rank_with_offset(self):
        """Test multi-rank with offset."""
        x = paddle.randn([2, 8, 64])
        freqs = paddle.randn([1, 32, 64])
        result = _get_thd_freqs_on_this_cp_rank(
            cp_rank=1,
            cp_size=2,
            x=x,
            freqs=freqs,
            offset=4,
        )
        self.assertIsNotNone(result)


class TestApplyRotaryPosEmbTHD(unittest.TestCase):
    """Test _apply_rotary_pos_emb_thd function with 4D tensors."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_thd_basic_4d(self, mock_rank, mock_size):
        """Test basic THD format RoPE with 4D tensor."""
        # t shape: [batch, total_seq, num_heads, head_dim]
        t = paddle.randn([2, 8, 4, 16])
        cu_seqlens = paddle.to_tensor([0, 4, 8], dtype="int32")
        freqs = paddle.randn([1, 32, 16])  # freqs.size(1) != total_seq_len

        result = _apply_rotary_pos_emb_thd(
            t=t,
            cu_seqlens=cu_seqlens,
            total_seq_len=8,
            freqs=freqs,
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
    def test_thd_exact_mapping_4d(self, mock_rank, mock_size):
        """Test THD with exact mapping on 4D tensor."""
        t = paddle.randn([2, 8, 4, 16])
        cu_seqlens = paddle.to_tensor([0, 4, 8], dtype="int32")
        freqs = paddle.randn([1, 8, 16])  # freqs.size(1) == total_seq_len

        result = _apply_rotary_pos_emb_thd(
            t=t,
            cu_seqlens=cu_seqlens,
            total_seq_len=8,
            freqs=freqs,
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
    def test_thd_traditional_mapping_4d(self, mock_rank, mock_size):
        """Test THD with traditional mapping on 4D tensor."""
        t = paddle.randn([2, 8, 4, 16])
        cu_seqlens = paddle.to_tensor([0, 4, 8], dtype="int32")
        freqs = paddle.randn([1, 32, 16])  # freqs.size(1) != total_seq_len

        result = _apply_rotary_pos_emb_thd(
            t=t,
            cu_seqlens=cu_seqlens,
            total_seq_len=8,
            freqs=freqs,
        )
        self.assertIsNotNone(result)


class TestApplyRotaryPosEmbWithTHD(unittest.TestCase):
    """Test apply_rotary_pos_emb router with THD format."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_router_with_cu_seqlens(self, mock_rank, mock_size):
        """Test that apply_rotary_pos_emb routes to THD when cu_seqlens is provided."""
        from paddleformers.fleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        t = paddle.randn([2, 8, 4, 16])
        freqs = paddle.randn([1, 32, 16])
        cu_seqlens = paddle.to_tensor([0, 4, 8], dtype="int32")

        result = apply_rotary_pos_emb(
            t=t,
            freqs=freqs,
            cos=None,
            sin=None,
            config=config,
            cu_seqlens=cu_seqlens,
            total_seq_len=8,
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
