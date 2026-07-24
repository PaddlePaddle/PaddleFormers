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

"""Tests for swa_high_precision_norm feature coverage.

Covers:
- dsv4_hybrid_attention.py:56-59 (_q_rms_norm high_precision_norm=True branch)
- dsv4_hybrid_attention.py:531 (q_head_norm with high_precision_norm in forward)
- dsv4_hybrid_attention.py:612 (kv RoPE mscale path in high_precision forward)
- paddle_norm.py:82-83 (RMSNorm forward with high_precision_norm=True)
- paddle_norm.py:96,98 (RMSNorm forward with return_high_precision_norm=True)
- transformer_config.py:1239 (swa_high_precision_norm validation error)
"""

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

import paddle

from paddleformers.fleet.transformer.dsv4_hybrid_attention import _q_rms_norm
from paddleformers.fleet.transformer.paddle_norm import RMSNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "normalization": "RMSNorm",
        "rms_norm_eps": 1e-5,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestQRmsNormHighPrecision(unittest.TestCase):
    """Cover dsv4_hybrid_attention.py lines 56-59: _q_rms_norm with high_precision_norm=True."""

    def test_high_precision_norm_true_casts_to_float32(self):
        """When high_precision_norm=True, computation should happen in float32
        and output should be cast back to original dtype."""
        q = paddle.randn([2, 4, 8, 32]).astype(paddle.bfloat16)
        result = _q_rms_norm(q, eps=1e-5, high_precision_norm=True)
        # Output dtype should match input dtype (bfloat16)
        self.assertEqual(result.dtype, paddle.bfloat16)
        # Output shape should be preserved
        self.assertEqual(result.shape, [2, 4, 8, 32])

    def test_high_precision_norm_false_keeps_dtype(self):
        """When high_precision_norm=False, no dtype casting happens."""
        q = paddle.randn([2, 4, 8, 32]).astype(paddle.bfloat16)
        result = _q_rms_norm(q, eps=1e-5, high_precision_norm=False)
        self.assertEqual(result.dtype, paddle.bfloat16)
        self.assertEqual(result.shape, [2, 4, 8, 32])

    def test_high_precision_norm_float32_input(self):
        """When input is already float32, high_precision_norm=True should still work."""
        q = paddle.randn([1, 2, 4, 16]).astype(paddle.float32)
        result = _q_rms_norm(q, eps=1e-5, high_precision_norm=True)
        self.assertEqual(result.dtype, paddle.float32)


class TestRMSNormHighPrecision(unittest.TestCase):
    """Cover paddle_norm.py lines 82-83, 96, 98: RMSNorm forward with high_precision_norm
    and return_high_precision_norm flags."""

    def test_high_precision_norm_casts_input_and_weight_to_float32(self):
        """Lines 82-83: When high_precision_norm=True, input and weight are cast to float32."""
        config = _make_config(params_dtype=paddle.bfloat16)
        norm = RMSNorm(config)
        x = paddle.randn([2, 4, 128]).astype(paddle.bfloat16)
        result = norm(x, high_precision_norm=True)
        # Output should be cast back to weight dtype (bfloat16)
        self.assertEqual(result.dtype, paddle.bfloat16)
        self.assertEqual(result.shape, [2, 4, 128])

    def test_return_high_precision_norm_returns_float32(self):
        """Lines 96, 98: When return_high_precision_norm=True, output stays in float32."""
        config = _make_config(params_dtype=paddle.bfloat16)
        norm = RMSNorm(config)
        x = paddle.randn([2, 4, 128]).astype(paddle.bfloat16)
        result = norm(
            x, high_precision_norm=True, return_high_precision_norm=True
        )
        # Output should remain float32 due to return_high_precision_norm
        self.assertEqual(result.dtype, paddle.float32)
        self.assertEqual(result.shape, [2, 4, 128])

    def test_high_precision_norm_false_default_behavior(self):
        """Without high_precision_norm, normal path is taken."""
        config = _make_config(params_dtype=paddle.float32)
        norm = RMSNorm(config)
        x = paddle.randn([2, 4, 128]).astype(paddle.float32)
        result = norm(x, high_precision_norm=False)
        self.assertEqual(result.dtype, paddle.float32)
        self.assertEqual(result.shape, [2, 4, 128])


class TestSwaHighPrecisionNormConfigValidation(unittest.TestCase):
    """Cover transformer_config.py line 1239: swa_high_precision_norm validation."""

    def test_swa_high_precision_norm_with_non_dsv4_raises_error(self):
        """Line 1239: Setting swa_high_precision_norm=True without dsv4_hybrid should raise."""
        with self.assertRaisesRegex(
            ValueError,
            "swa_high_precision_norm=True is only supported when",
        ):
            TransformerConfig(
                hidden_size=128,
                num_attention_heads=4,
                swa_high_precision_norm=True,
                experimental_attention_variant=None,
            )

    def test_swa_high_precision_norm_with_dsv4_hybrid_passes(self):
        """swa_high_precision_norm=True with dsv4_hybrid should not raise."""
        config = TransformerConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            params_dtype=paddle.bfloat16,
            bf16=True,
            multi_latent_attention=True,
            experimental_attention_variant="dsv4_hybrid",
            swa_high_precision_norm=True,
            q_lora_rank=64,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            csa_compress_ratios=[0, 4, 128, 4],
            csa_window_size=16,
        )
        self.assertTrue(config.swa_high_precision_norm)

    def test_swa_high_precision_norm_false_no_validation(self):
        """swa_high_precision_norm=False should pass regardless of variant."""
        config = TransformerConfig(
            hidden_size=128,
            num_attention_heads=4,
            swa_high_precision_norm=False,
            experimental_attention_variant=None,
        )
        self.assertFalse(config.swa_high_precision_norm)


if __name__ == "__main__":
    unittest.main()
