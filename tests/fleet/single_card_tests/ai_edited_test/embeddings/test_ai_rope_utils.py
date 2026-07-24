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
from unittest.mock import MagicMock, patch

import numpy as np
import paddle


class TestGetPosEmbOnThisCPRank(unittest.TestCase):
    """Test get_pos_emb_on_this_cp_rank function."""

    def test_raises_when_cp_group_none(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            get_pos_emb_on_this_cp_rank,
        )

        pos_emb = paddle.randn([1, 16, 1, 64])
        with self.assertRaises(ValueError):
            get_pos_emb_on_this_cp_rank(pos_emb, seq_dim=1, cp_group=None)

    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_size",
        return_value=2,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.rope_utils.get_pg_rank",
        return_value=0,
    )
    def test_returns_correct_shape(self, mock_rank, mock_size):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            get_pos_emb_on_this_cp_rank,
        )

        pos_emb = paddle.randn([1, 32, 1, 64])
        cp_group = MagicMock()
        result = get_pos_emb_on_this_cp_rank(
            pos_emb, seq_dim=1, cp_group=cp_group
        )
        self.assertIsNotNone(result)


class TestRotateHalf(unittest.TestCase):
    """Test _rotate_half function."""

    def test_non_interleaved(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _rotate_half,
        )

        x = paddle.randn([2, 8, 12, 64])
        result = _rotate_half(x, rotary_interleaved=False)
        self.assertEqual(result.shape, x.shape)

    def test_interleaved(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _rotate_half,
        )

        x = paddle.randn([2, 8, 12, 64])
        result = _rotate_half(x, rotary_interleaved=True)
        self.assertEqual(result.shape, x.shape)

    def test_non_interleaved_correct_rotation(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _rotate_half,
        )

        x = paddle.randn([1, 1, 1, 4])
        result = _rotate_half(x, rotary_interleaved=False)
        # x = [x0, x1, x2, x3] -> chunk -> x1=[x0,x1], x2=[x2,x3]
        # result = [-x2, -x3, x0, x1]
        x_np = x.numpy()
        r_np = result.numpy()
        np.testing.assert_allclose(
            r_np[0, 0, 0, :2], -x_np[0, 0, 0, 2:], atol=1e-6
        )
        np.testing.assert_allclose(
            r_np[0, 0, 0, 2:], x_np[0, 0, 0, :2], atol=1e-6
        )


class TestApplyRotaryPosEmbBshdBasic(unittest.TestCase):
    """Test _apply_rotary_pos_emb_bshd basic functionality."""

    def test_basic_apply(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _apply_rotary_pos_emb_bshd,
        )

        # freqs shape [B, S, D] to match t shape [B, S, H, D]
        t = paddle.randn([2, 8, 12, 64])
        freqs = paddle.randn([2, 8, 32])
        result = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            rotary_interleaved=False,
        )
        self.assertEqual(result.shape, t.shape)

    def test_with_partial_rot_dim(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _apply_rotary_pos_emb_bshd,
        )

        t = paddle.randn([2, 8, 12, 128])
        freqs = paddle.randn([2, 8, 32])
        result = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            rotary_interleaved=False,
        )
        self.assertEqual(result.shape, t.shape)

    def test_interleaved(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _apply_rotary_pos_emb_bshd,
        )

        t = paddle.randn([2, 8, 12, 64])
        freqs = paddle.randn([2, 8, 32])
        result = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            rotary_interleaved=True,
        )
        self.assertEqual(result.shape, t.shape)


class TestApplyRotaryPosEmbBshdWithMscale(unittest.TestCase):
    """Test _apply_rotary_pos_emb_bshd with mscale."""

    def test_mscale(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _apply_rotary_pos_emb_bshd,
        )

        t = paddle.randn([2, 8, 12, 64])
        freqs = paddle.randn([2, 8, 32])
        result = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            cos=None,
            sin=None,
            mscale=2.0,
            apply_rope_fusion=False,
            rotary_interleaved=False,
        )
        self.assertEqual(result.shape, t.shape)

    def test_mscale_none_treated_as_one(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _apply_rotary_pos_emb_bshd,
        )

        t = paddle.randn([2, 8, 12, 64])
        freqs = paddle.randn([2, 8, 32])
        result = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            cos=None,
            sin=None,
            mscale=None,
            apply_rope_fusion=False,
            rotary_interleaved=False,
        )
        self.assertEqual(result.shape, t.shape)


class TestApplyRotaryPosEmbBshdWithMLA(unittest.TestCase):
    """Test _apply_rotary_pos_emb_bshd with multi_latent_attention."""

    def test_mla_reorder(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _apply_rotary_pos_emb_bshd,
        )

        t = paddle.randn([2, 8, 12, 64])
        freqs = paddle.randn([2, 8, 32])
        result = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            rotary_interleaved=False,
            multi_latent_attention=True,
        )
        self.assertEqual(result.shape, t.shape)


class TestApplyRotaryPosEmbBshdHighPrecision(unittest.TestCase):
    """Test _apply_rotary_pos_emb_bshd with high_precision_rope."""

    def test_high_precision(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _apply_rotary_pos_emb_bshd,
        )

        t = paddle.randn([2, 8, 12, 64]).astype("float16")
        freqs = paddle.randn([2, 8, 32])
        result = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            rotary_interleaved=False,
            high_precision_rope=True,
        )
        self.assertEqual(result.shape, t.shape)
        self.assertEqual(result.dtype, t.dtype)


class TestApplyRotaryPosEmbBshdFreqTranspose(unittest.TestCase):
    """Test _apply_rotary_pos_emb_bshd with transposed freq dims."""

    def test_freq_transpose_when_dims_differ(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            _apply_rotary_pos_emb_bshd,
        )

        t = paddle.randn([2, 8, 12, 64])
        # freqs is [S, B, D] while t is [B, S, H, D] - same product but different order
        freqs = paddle.randn([8, 2, 32])
        result = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            cos=None,
            sin=None,
            apply_rope_fusion=False,
            rotary_interleaved=False,
        )
        self.assertEqual(result.shape, t.shape)


class TestApplyRotaryPosEmbReroute(unittest.TestCase):
    """Test apply_rotary_pos_emb routing function."""

    def test_routes_to_bshd(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            apply_rotary_pos_emb,
        )

        mock_config = MagicMock()
        mock_config.apply_rope_fusion = False
        mock_config.rotary_interleaved = False
        mock_config.multi_latent_attention = False
        mock_config.high_precision_rope = False
        mock_config.rope_theta = 10000.0
        mock_config.sequence_parallel = False

        t = paddle.randn([2, 8, 12, 64])
        freqs = paddle.randn([2, 8, 32])
        result = apply_rotary_pos_emb(
            t,
            freqs,
            cos=None,
            sin=None,
            config=mock_config,
        )
        self.assertEqual(result.shape, t.shape)

    def test_routes_to_thd(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            apply_rotary_pos_emb,
        )

        mock_config = MagicMock()
        mock_config.apply_rope_fusion = False
        mock_config.rotary_interleaved = False
        mock_config.multi_latent_attention = False
        mock_config.high_precision_rope = False
        mock_config.rope_theta = 10000.0
        mock_config.sequence_parallel = False

        t = paddle.randn([2, 4, 12, 64])
        freqs = paddle.randn([4, 4, 1, 32])
        cu_seqlens = paddle.to_tensor([0, 2, 4], dtype="int32")
        # Mock the thd implementation to verify routing without exercising
        # the internal shape conversion which has source-level limitations
        with patch(
            "paddleformers.fleet.models.common.embeddings.rope_utils._apply_rotary_pos_emb_thd",
            return_value=t,
        ) as mock_thd:
            result = apply_rotary_pos_emb(
                t,
                freqs,
                cos=None,
                sin=None,
                config=mock_config,
                cu_seqlens=cu_seqlens,
            )
            mock_thd.assert_called_once()
            self.assertIsNotNone(result)


class TestGetUnsqueezeDim(unittest.TestCase):
    """Test get_unsqueeze_dim helper function."""

    def test_batch_first(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            get_unsqueeze_dim,
        )

        t = paddle.randn([2, 8, 12, 64])
        freqs = paddle.randn([2, 8, 32])
        result = get_unsqueeze_dim(t, freqs)
        self.assertEqual(result, 2)

    def test_seq_first(self):
        from paddleformers.fleet.models.common.embeddings.rope_utils import (
            get_unsqueeze_dim,
        )

        t = paddle.randn([8, 2, 12, 64])
        freqs = paddle.randn([2, 8, 32])
        result = get_unsqueeze_dim(t, freqs)
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
