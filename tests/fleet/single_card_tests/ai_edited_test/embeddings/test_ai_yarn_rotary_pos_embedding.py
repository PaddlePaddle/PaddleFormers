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


import math
import unittest
from unittest.mock import MagicMock


class TestYarnFindCorrectionDim(unittest.TestCase):
    """Test _yarn_find_correction_dim function."""

    def test_basic_computation(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_dim,
        )

        result = _yarn_find_correction_dim(
            num_rotations=1.0,
            dim=64,
            rotary_base=10000,
            max_position_embeddings=2048,
        )
        expected = (64 * math.log(2048 / (2 * math.pi))) / (2 * math.log(10000))
        self.assertAlmostEqual(result, expected, places=5)

    def test_different_base(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_dim,
        )

        result = _yarn_find_correction_dim(
            num_rotations=1.0,
            dim=64,
            rotary_base=500000,
            max_position_embeddings=2048,
        )
        self.assertIsInstance(result, float)

    def test_different_max_pos(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_dim,
        )

        result = _yarn_find_correction_dim(
            num_rotations=2.0,
            dim=128,
            rotary_base=10000,
            max_position_embeddings=4096,
        )
        self.assertIsInstance(result, float)


class TestYarnFindCorrectionRange(unittest.TestCase):
    """Test _yarn_find_correction_range function."""

    def test_basic_range(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_range,
        )

        # Use low_rot > high_rot so that low < high
        low, high = _yarn_find_correction_range(
            low_rot=4.0,
            high_rot=1.0,
            dim=64,
            rotary_base=10000,
            max_position_embeddings=2048,
        )
        self.assertLessEqual(low, high)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 63)

    def test_with_round_to_int_false(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_range,
        )

        # Use low_rot > high_rot so that low < high
        low, high = _yarn_find_correction_range(
            low_rot=4.0,
            high_rot=1.0,
            dim=64,
            rotary_base=10000,
            max_position_embeddings=2048,
            round_to_int=False,
        )
        self.assertLessEqual(low, high)

    def test_clamped_values(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_find_correction_range,
        )

        # Very low rotations should give low correction dim
        low, high = _yarn_find_correction_range(
            low_rot=0.1,
            high_rot=0.2,
            dim=64,
            rotary_base=10000,
            max_position_embeddings=2048,
        )
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 63)


class TestYarnLinearRampMask(unittest.TestCase):
    """Test _yarn_linear_ramp_mask function."""

    def test_basic_mask(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_linear_ramp_mask,
        )

        result = _yarn_linear_ramp_mask(min=10.0, max=20.0, dim=64)
        self.assertEqual(result.shape, [64])

    def test_mask_clamped(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_linear_ramp_mask,
        )

        result = _yarn_linear_ramp_mask(min=10.0, max=20.0, dim=64)
        # Values below min should be 0, above max should be 1
        self.assertTrue(paddle.all(result >= 0))
        self.assertTrue(paddle.all(result <= 1))

    def test_equal_min_max(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_linear_ramp_mask,
        )

        result = _yarn_linear_ramp_mask(min=10.0, max=10.0, dim=32)
        self.assertEqual(result.shape, [32])


class TestYarnGetMscale(unittest.TestCase):
    """Test _yarn_get_mscale function."""

    def test_scale_leq_one(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_mscale,
        )

        self.assertEqual(_yarn_get_mscale(scale=0.5), 1.0)
        self.assertEqual(_yarn_get_mscale(scale=1.0), 1.0)

    def test_scale_gt_one(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_mscale,
        )

        result = _yarn_get_mscale(scale=2.0, mscale=1.0)
        expected = 0.1 * 1.0 * math.log(2.0) + 1.0
        self.assertAlmostEqual(result, expected, places=5)

    def test_with_custom_mscale(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_mscale,
        )

        result = _yarn_get_mscale(scale=4.0, mscale=0.5)
        expected = 0.1 * 0.5 * math.log(4.0) + 1.0
        self.assertAlmostEqual(result, expected, places=5)


class TestYarnGetConcentrationFactor(unittest.TestCase):
    """Test _yarn_get_concentration_factor function."""

    def test_basic_factor(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor,
        )

        result = _yarn_get_concentration_factor(scaling_factor=1.0, mscale=1.0, mscale_all_dim=0.0)
        # scale <= 1: both mscale calls return 1.0, so factor = 1.0/1.0 = 1.0
        self.assertAlmostEqual(result, 1.0)

    def test_factor_gt_one(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor,
        )

        result = _yarn_get_concentration_factor(scaling_factor=2.0, mscale=1.0, mscale_all_dim=0.0)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_cached_results(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor,
        )

        r1 = _yarn_get_concentration_factor(2.0, 1.0, 0.0)
        r2 = _yarn_get_concentration_factor(2.0, 1.0, 0.0)
        self.assertEqual(r1, r2)


class TestYarnGetConcentrationFactorFromConfig(unittest.TestCase):
    """Test _yarn_get_concentration_factor_from_config function."""

    def test_config_with_yarn_fields(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor_from_config,
        )

        mock_config = MagicMock()
        mock_config.yarn_rotary_scaling_factor = 2.0
        mock_config.yarn_mscale = 1.0
        mock_config.yarn_mscale_all_dim = 0.0

        result = _yarn_get_concentration_factor_from_config(mock_config)
        self.assertIsInstance(result, float)

    def test_config_without_yarn_fields(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            _yarn_get_concentration_factor_from_config,
        )

        mock_config = MagicMock(spec=[])
        # Don't set yarn_rotary_scaling_factor at all so hasattr returns False
        del mock_config.yarn_rotary_scaling_factor
        del mock_config.yarn_mscale
        del mock_config.yarn_mscale_all_dim
        # Make hasattr return False for these fields
        type(mock_config).yarn_rotary_scaling_factor = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no attr"))
        )
        type(mock_config).yarn_mscale = property(lambda self: (_ for _ in ()).throw(AttributeError("no attr")))
        type(mock_config).yarn_mscale_all_dim = property(lambda self: (_ for _ in ()).throw(AttributeError("no attr")))

        result = _yarn_get_concentration_factor_from_config(mock_config)
        self.assertEqual(result, 1.0)


class TestYarnRotaryEmbeddingInit(unittest.TestCase):
    """Test YarnRotaryEmbedding initialization."""

    def test_basic_init(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        emb = YarnRotaryEmbedding(head_dim=64)
        self.assertEqual(emb.dim, 64)
        self.assertEqual(emb.scaling_factor, 1.0)
        self.assertEqual(emb.original_max_position_embeddings, 4096)
        self.assertEqual(emb.beta_fast, 32.0)
        self.assertEqual(emb.beta_slow, 1.0)

    def test_custom_params(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        emb = YarnRotaryEmbedding(
            head_dim=128,
            scaling_factor=4.0,
            original_max_position_embeddings=8192,
            beta_fast=16.0,
            beta_slow=2.0,
        )
        self.assertEqual(emb.dim, 128)
        self.assertEqual(emb.scaling_factor, 4.0)
        self.assertEqual(emb.original_max_position_embeddings, 8192)

    def test_inv_freq_attributes(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        emb = YarnRotaryEmbedding(head_dim=64)
        self.assertIsNotNone(emb.inv_freq_extra)
        self.assertIsNotNone(emb.inv_freq_inter)


class TestYarnRotaryEmbeddingForward(unittest.TestCase):
    """Test YarnRotaryEmbedding forward method."""

    def test_forward_returns_tuple(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        emb = YarnRotaryEmbedding(head_dim=64)
        result = emb(max_seq_len=128)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_forward_shape(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        emb = YarnRotaryEmbedding(head_dim=64)
        emb_val, mscale = emb(max_seq_len=128)
        # emb should be [1, seq_len, 1, dim]
        self.assertEqual(emb_val.shape[0], 1)
        self.assertEqual(emb_val.shape[1], 128)

    def test_forward_with_interleaved(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        emb = YarnRotaryEmbedding(head_dim=64, rotary_interleaved=True)
        emb_val, mscale = emb(max_seq_len=128)
        self.assertIsNotNone(emb_val)


class TestYarnRotaryEmbeddingCachedCosSin(unittest.TestCase):
    """Test YarnRotaryEmbedding.get_cached_cos_sin method."""

    def test_cache_creation(self):
        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        emb = YarnRotaryEmbedding(head_dim=64)
        cos, sin = emb.get_cached_cos_sin(seq_len=128)
        self.assertIsNotNone(cos)
        self.assertIsNotNone(sin)

    def test_cache_reuse(self):
        import paddle

        from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )

        emb = YarnRotaryEmbedding(head_dim=64)
        cos1, sin1 = emb.get_cached_cos_sin(seq_len=128)
        cos2, sin2 = emb.get_cached_cos_sin(seq_len=128)
        self.assertTrue(paddle.equal(cos1, cos2).all())


if __name__ == "__main__":
    unittest.main()
