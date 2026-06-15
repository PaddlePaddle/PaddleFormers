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

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddleformers.fleet.transformer.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSublayersSpec,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

_GDN_MODULE = "paddleformers.fleet.transformer.gated_delta_net"


class NoBiasLinear(nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias_attr=False)

    def forward(self, x):
        return self.linear(x), None

    def backward_dw(self):
        pass


class SimpleRMSNorm(nn.Layer):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, x):
        x_float = x.astype(paddle.float32)
        rms = paddle.rsqrt(x_float.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return (x_float * rms * self.weight.astype(paddle.float32)).astype(x.dtype)


class _FakeGroup:
    ranks = [0]
    nranks = 1
    rank = 0


class _FakePGCollection:
    def __init__(self):
        self.tp = _FakeGroup()


H, B, S = 64, 2, 32


def _make_gdn():
    config = TransformerConfig(
        hidden_size=H,
        num_attention_heads=4,
        num_hidden_layers=2,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        deterministic_mode=True,
    )
    spec = GatedDeltaNetSublayersSpec(
        in_proj=NoBiasLinear,
        out_norm=SimpleRMSNorm,
        out_proj=NoBiasLinear,
    )
    return GatedDeltaNet(
        config=config,
        sublayers_spec=spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=_FakePGCollection(),
        conv_kernel_dim=4,
        key_head_dim=16,
        value_head_dim=16,
        num_key_heads=4,
        num_value_heads=4,
    )


def _make_gdn_sp():
    """GDN with SP simulated (config.sequence_parallel=True, sp_size=2)."""
    config = TransformerConfig(
        hidden_size=H,
        num_attention_heads=4,
        num_hidden_layers=2,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        deterministic_mode=True,
    )
    config.sequence_parallel = True
    spec = GatedDeltaNetSublayersSpec(
        in_proj=NoBiasLinear,
        out_norm=SimpleRMSNorm,
        out_proj=NoBiasLinear,
    )
    gdn = GatedDeltaNet(
        config=config,
        sublayers_spec=spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=_FakePGCollection(),
        conv_kernel_dim=4,
        key_head_dim=16,
        value_head_dim=16,
        num_key_heads=4,
        num_value_heads=4,
    )
    gdn.sp_size = 2  # simulate TP=2 + SP
    return gdn


def _make_startend_indices(batch, seq_len, padding_start=None):
    """Create attn_mask_startend_row_indices [b, 1, s, 1]. padding_start=None means all valid."""
    indices = paddle.full([batch, 1, seq_len, 1], fill_value=seq_len, dtype="int64")
    if padding_start is not None:
        for pos in range(padding_start, seq_len):
            indices[:, :, pos, :] = pos
    return indices


class TestBuildPaddingMaskNoSP(unittest.TestCase):
    """_build_padding_mask without sequence parallel."""

    def setUp(self):
        self.gdn = _make_gdn()

    def test_both_none_returns_none(self):
        self.assertIsNone(self.gdn._build_padding_mask(None, None, B, S))

    def test_attention_mask_ndim3_raises(self):
        with self.assertRaises(ValueError):
            self.gdn._build_padding_mask(paddle.ones([B, 1, S]), None, B, S)

    def test_attention_mask_all_valid_returns_none(self):
        mask = paddle.ones([B, S], dtype="int64")
        self.assertIsNone(self.gdn._build_padding_mask(mask, None, B, S))

    def test_attention_mask_with_padding(self):
        mask = paddle.ones([B, S], dtype="int64")
        mask[0, -8:] = 0
        result = self.gdn._build_padding_mask(mask, None, B, S)
        self.assertEqual(list(result.shape), [B, S, 1])
        np.testing.assert_array_equal(result[0, :24, 0].numpy(), np.ones(24))
        np.testing.assert_array_equal(result[0, 24:, 0].numpy(), np.zeros(8))
        np.testing.assert_array_equal(result[1, :, 0].numpy(), np.ones(S))

    def test_startend_all_valid_returns_none(self):
        indices = _make_startend_indices(B, S, padding_start=None)
        self.assertIsNone(self.gdn._build_padding_mask(None, indices, B, S))

    def test_startend_with_padding(self):
        indices = _make_startend_indices(B, S, padding_start=24)
        result = self.gdn._build_padding_mask(None, indices, B, S)
        self.assertEqual(list(result.shape), [B, S, 1])
        np.testing.assert_array_equal(result[0, :24, 0].numpy(), np.ones(24))
        np.testing.assert_array_equal(result[0, 24:, 0].numpy(), np.zeros(8))

    def test_startend_per_sample_padding(self):
        indices = _make_startend_indices(B, S, padding_start=None)
        for pos in range(20, S):
            indices[0, :, pos, :] = pos  # only sample 0 has padding
        result = self.gdn._build_padding_mask(None, indices, B, S)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result[0, 20:, 0].numpy(), np.zeros(12))
        np.testing.assert_array_equal(result[1, :, 0].numpy(), np.ones(S))

    def test_attention_mask_takes_priority(self):
        attn_mask = paddle.ones([B, S], dtype="int64")
        attn_mask[0, -4:] = 0
        indices = _make_startend_indices(B, S, padding_start=None)  # all valid
        result = self.gdn._build_padding_mask(attn_mask, indices, B, S)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result[0, -4:, 0].numpy(), np.zeros(4))

    def test_boundary_indices_eq_pos_is_padding(self):
        indices = paddle.full([1, 1, S, 1], fill_value=S, dtype="int64")
        indices[0, 0, 0, 0] = 1  # 1 > 0 → valid
        indices[0, 0, 1, 0] = 1  # 1 > 1 → False → padding
        result = self.gdn._build_padding_mask(None, indices, 1, S)
        self.assertEqual(result[0, 0, 0].item(), 1.0)
        self.assertEqual(result[0, 1, 0].item(), 0.0)


class TestBuildPaddingMaskWithSP(unittest.TestCase):
    """_build_padding_mask with simulated SP=True, TP=2."""

    def setUp(self):
        self.gdn = _make_gdn_sp()

    def test_attention_mask_rank0_all_valid(self):
        mask = paddle.ones([B, S], dtype="int64")
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0):
            self.assertIsNone(self.gdn._build_padding_mask(mask, None, B, S))

    def test_attention_mask_rank1_with_padding(self):
        mask = paddle.ones([B, S], dtype="int64")
        mask[0, -4:] = 0  # positions 28-31 padding
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=1):
            result = self.gdn._build_padding_mask(mask, None, B, S)
        # SP layout: [s_local=16, b=2, 1]
        self.assertEqual(list(result.shape), [16, B, 1])
        # Rank1 sees [16,32), positions 28-31 → local 12-15
        np.testing.assert_array_equal(result[:12, 0, 0].numpy(), np.ones(12))
        np.testing.assert_array_equal(result[12:, 0, 0].numpy(), np.zeros(4))
        np.testing.assert_array_equal(result[:, 1, 0].numpy(), np.ones(16))

    def test_startend_rank0_all_valid(self):
        indices = _make_startend_indices(B, S, padding_start=None)
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0):
            self.assertIsNone(self.gdn._build_padding_mask(None, indices, B, S))

    def test_startend_rank1_with_padding(self):
        indices = _make_startend_indices(B, S, padding_start=24)
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=1):
            result = self.gdn._build_padding_mask(None, indices, B, S)
        # Rank1 [16,32): 16-23 valid, 24-31 padding → local 0-7 valid, 8-15 padding
        self.assertEqual(list(result.shape), [16, B, 1])
        np.testing.assert_array_equal(result[:8, 0, 0].numpy(), np.ones(8))
        np.testing.assert_array_equal(result[8:, 0, 0].numpy(), np.zeros(8))

    def test_startend_rank0_no_padding_rank1_has_padding(self):
        indices = _make_startend_indices(B, S, padding_start=20)
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0):
            self.assertIsNone(self.gdn._build_padding_mask(None, indices, B, S))
        with patch(f"{_GDN_MODULE}.get_pg_rank", return_value=1):
            result = self.gdn._build_padding_mask(None, indices, B, S)
        self.assertIsNotNone(result)
        # Rank1 [16,32): 16-19 valid, 20-31 padding → local 0-3 valid, 4-15 padding
        np.testing.assert_array_equal(result[:4, 0, 0].numpy(), np.ones(4))
        np.testing.assert_array_equal(result[4:, 0, 0].numpy(), np.zeros(12))

    def test_attention_mask_shape_mismatch_raises(self):
        """SP path: attention_mask full_seq != seq_len should raise ValueError."""
        wrong_len_mask = paddle.ones([B, S + 4], dtype="int64")
        with (
            patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0),
            self.assertRaises(ValueError),
        ):
            self.gdn._build_padding_mask(wrong_len_mask, None, B, S)

    def test_startend_shape_mismatch_raises(self):
        """SP path: startend full_seq != seq_len should raise ValueError."""
        wrong_indices = _make_startend_indices(B, S + 4, padding_start=None)
        with (
            patch(f"{_GDN_MODULE}.get_pg_rank", return_value=0),
            self.assertRaises(ValueError),
        ):
            self.gdn._build_padding_mask(None, wrong_indices, B, S)


class TestForwardPaddingMask(unittest.TestCase):
    """Forward-level tests for mask application."""

    def setUp(self):
        self.gdn = _make_gdn()
        self.gdn.eval()

    def test_all_valid_mask_equals_no_mask(self):
        x = paddle.randn([B, S, H])
        out_none, _ = self.gdn(x, attention_mask=None)
        out_valid, _ = self.gdn(x, attention_mask=paddle.ones([B, S], dtype="int64"))
        np.testing.assert_allclose(out_none.numpy(), out_valid.numpy(), rtol=1e-5, atol=1e-5)

    def test_padding_changes_output(self):
        x = paddle.randn([B, S, H])
        mask = paddle.ones([B, S], dtype="int64")
        mask[0, :] = 0
        out_masked, _ = self.gdn(x, attention_mask=mask)
        out_none, _ = self.gdn(x, attention_mask=None)
        # Sample 0 differs, sample 1 same
        self.assertFalse(np.allclose(out_masked[0].numpy(), out_none[0].numpy(), atol=1e-6))
        np.testing.assert_allclose(out_masked[1].numpy(), out_none[1].numpy(), rtol=1e-5, atol=1e-5)

    def test_startend_indices_with_padding(self):
        x = paddle.randn([B, S, H])
        indices = _make_startend_indices(B, S, padding_start=24)
        out_pad, _ = self.gdn(x, attention_mask=None, attn_mask_startend_row_indices=indices)
        out_none, _ = self.gdn(x, attention_mask=None)
        self.assertFalse(np.allclose(out_pad.numpy(), out_none.numpy(), atol=1e-6))

    def test_tokens_before_padding_unaffected(self):
        x = paddle.randn([1, S, H])
        mask = paddle.ones([1, S], dtype="int64")
        mask[0, 28:] = 0
        out_pad, _ = self.gdn(x, attention_mask=mask)
        out_none, _ = self.gdn(x, attention_mask=None)
        # Causal: tokens far before padding are unaffected
        np.testing.assert_allclose(
            out_none[0, :24].numpy(),
            out_pad[0, :24].numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_backward_gradients(self):
        x = paddle.randn([B, S, H])
        x.stop_gradient = False
        mask = paddle.ones([B, S], dtype="int64")
        mask[0, -8:] = 0
        out, _ = self.gdn(x, attention_mask=mask)
        out.sum().backward()
        self.assertTrue(paddle.isfinite(x.grad).all().item())
        # Padding positions get zero gradient
        np.testing.assert_array_equal(x.grad[0, -8:].numpy(), np.zeros([8, H]))
        # Valid positions get non-zero gradient
        self.assertFalse(np.allclose(x.grad[0, :24].numpy(), 0))


if __name__ == "__main__":
    unittest.main()
