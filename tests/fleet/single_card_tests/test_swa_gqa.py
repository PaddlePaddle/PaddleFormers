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

"""Unit tests for GQA Sliding Window Attention (SWA) for MiMo.

Covers the following commits:
- support GQA SWA for MiMo
- fix mtp arg pass
- fix num_empty_layers_add_in_head for swa
- fix softmax_offset init
"""

import unittest
from unittest.mock import patch

import paddle

from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.transformer.utils import (
    is_layer_window_attention,
    startend_row_indices_add_sliding_window,
)
from paddleformers.fleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)

strategy = paddle.distributed.fleet.DistributedStrategy()
initialize_fleet(strategy=strategy)


# ============================================================================
# Helper classes (same pattern as existing tests)
# ============================================================================


class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias


class RMSNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs.get("normalized_shape", kwargs.get("hidden_size"))
        eps = kwargs.get("norm_eps", kwargs.get("eps"))
        self.weight = paddle.nn.Parameter(paddle.ones([hidden_size]))
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


# ============================================================================
# Test: is_layer_window_attention (0-indexed after fix)
# ============================================================================


class TestIsLayerWindowAttention(unittest.TestCase):
    """Tests for is_layer_window_attention with 0-indexed layer_number."""

    def test_no_sliding_window_returns_false(self):
        """When sliding_window is None, always returns False."""
        self.assertFalse(is_layer_window_attention(None, 3, 0))
        self.assertFalse(is_layer_window_attention(None, 3, 5))

    def test_no_skip_freq_returns_true(self):
        """When window_attn_skip_freq is None, all layers are SWA."""
        self.assertTrue(is_layer_window_attention((4096, 0), None, 0))
        self.assertTrue(is_layer_window_attention((4096, 0), None, 10))

    def test_int_skip_freq_layer_0_is_full_attention(self):
        """With int skip_freq, layer 0 should be full attention (0 % N == 0)."""
        # skip_freq=4: layers 0,4,8... are full attention (not SWA)
        self.assertFalse(is_layer_window_attention((4096, 0), 4, 0))

    def test_int_skip_freq_non_zero_mod(self):
        """With int skip_freq, layers where layer_number % freq != 0 are SWA."""
        # skip_freq=4: layer 1,2,3 are SWA
        self.assertTrue(is_layer_window_attention((4096, 0), 4, 1))
        self.assertTrue(is_layer_window_attention((4096, 0), 4, 2))
        self.assertTrue(is_layer_window_attention((4096, 0), 4, 3))
        # layer 4 is full attention
        self.assertFalse(is_layer_window_attention((4096, 0), 4, 4))

    def test_list_skip_freq_0_indexed(self):
        """With list skip_freq, uses 0-indexed access."""
        # Pattern: [1, 1, 0, 1, 0] means layers 0,1,3 are SWA; layers 2,4 are not
        pattern = [1, 1, 0, 1, 0]
        self.assertTrue(is_layer_window_attention((4096, 0), pattern, 0))
        self.assertTrue(is_layer_window_attention((4096, 0), pattern, 1))
        self.assertFalse(is_layer_window_attention((4096, 0), pattern, 2))
        self.assertTrue(is_layer_window_attention((4096, 0), pattern, 3))
        self.assertFalse(is_layer_window_attention((4096, 0), pattern, 4))

    def test_list_skip_freq_all_swa(self):
        """All layers set to 1 means all layers are SWA."""
        pattern = [1, 1, 1, 1]
        for i in range(4):
            self.assertTrue(is_layer_window_attention((4096, 0), pattern, i))

    def test_list_skip_freq_no_swa(self):
        """All layers set to 0 means no layers are SWA."""
        pattern = [0, 0, 0, 0]
        for i in range(4):
            self.assertFalse(is_layer_window_attention((4096, 0), pattern, i))

    def test_invalid_skip_freq_type_raises(self):
        """Non-int/list/None skip_freq should raise ValueError."""
        with self.assertRaises(ValueError):
            is_layer_window_attention((4096, 0), "invalid", 0)

    # ---- int sliding_window branch (HF causal one-sided) ----
    # int W is truthy exactly like tuple (W, 0), so all skip_freq logic below
    # must match the corresponding tuple cases above.

    def test_int_no_skip_freq_returns_true(self):
        """int sliding_window with no skip_freq: all layers are SWA."""
        self.assertTrue(is_layer_window_attention(4096, None, 0))
        self.assertTrue(is_layer_window_attention(4096, None, 10))

    def test_int_zero_returns_false(self):
        """int sliding_window == 0 is falsy: never SWA (window closed)."""
        self.assertFalse(is_layer_window_attention(0, None, 0))
        self.assertFalse(is_layer_window_attention(0, 4, 3))

    def test_int_equivalent_to_tuple_left_zero(self):
        """int W and tuple (W, 0) must give identical is_swa decisions."""
        for skip_freq in (None, 4, [1, 1, 0, 1, 0]):
            n = 5 if isinstance(skip_freq, list) else 6
            for layer in range(n):
                self.assertEqual(
                    is_layer_window_attention(4096, skip_freq, layer),
                    is_layer_window_attention((4096, 0), skip_freq, layer),
                    f"int vs tuple mismatch: skip_freq={skip_freq}, layer={layer}",
                )


# ============================================================================
# Test: startend_row_indices_add_sliding_window
# ============================================================================


class TestStartendRowIndicesAddSlidingWindow(unittest.TestCase):
    """Tests for the new startend_row_indices_add_sliding_window function."""

    def test_no_sliding_window_returns_unchanged(self):
        """When sliding_window is None, returns input unchanged."""
        indices = paddle.ones([2, 1, 8, 1], dtype=paddle.int32) * 100
        result = startend_row_indices_add_sliding_window(indices, None, 0.0, 4)
        self.assertTrue(paddle.equal_all(result, indices).item())

    def test_empty_sliding_window_returns_unchanged(self):
        """When sliding_window is empty tuple (falsy), returns input unchanged."""
        indices = paddle.ones([2, 1, 8, 1], dtype=paddle.int32) * 100
        result = startend_row_indices_add_sliding_window(indices, (), 0.0, 4)
        self.assertTrue(paddle.equal_all(result, indices).item())

    def test_invalid_num_vec_raises(self):
        """When last dim not in [1, 2], should raise ValueError."""
        indices = paddle.ones([2, 1, 8, 3], dtype=paddle.int32)
        with self.assertRaises(ValueError):
            startend_row_indices_add_sliding_window(
                indices,
                (4096, 0),
                0.0,
                4,
            )

    def test_single_head_expansion(self):
        """When heads==1, should expand to kv_num_heads."""
        bsz, seq, kv_num_heads = 2, 8, 4
        # Large values so SWA should clip them
        indices = paddle.ones([bsz, 1, seq, 1], dtype=paddle.int32) * 10000
        window_size = 4
        result = startend_row_indices_add_sliding_window(
            indices, (window_size, 0), 0.0, kv_num_heads
        )
        # Output shape should be [bsz, kv_num_heads, seq, 1]
        self.assertEqual(list(result.shape), [bsz, kv_num_heads, seq, 1])

    def test_sliding_window_clips_values(self):
        """SWA should clip start indices to window boundary."""
        bsz, seq, kv_num_heads = 1, 8, 1
        window_size = 3
        # Set all start indices to a large value (simulating full attention)
        indices = paddle.ones([bsz, 1, seq, 1], dtype=paddle.int32) * 10000

        result = startend_row_indices_add_sliding_window(
            indices, (window_size, 0), 0.0, kv_num_heads
        )

        # Expected: LTS_SWA = arange(window_size, seq + window_size)
        # = [3, 4, 5, 6, 7, 8, 9, 10]
        # where(10000 < LTS_SWA, 10000, LTS_SWA) -> LTS_SWA (since 10000 > all)
        expected = paddle.arange(
            window_size, seq + window_size, dtype=paddle.int32
        ).reshape([1, 1, seq, 1])
        self.assertTrue(
            paddle.equal_all(result, expected).item(),
            f"Expected {expected.numpy()}, got {result.numpy()}",
        )

    def test_small_indices_not_clipped(self):
        """When original start indices are smaller than window boundary, they remain."""
        bsz, seq, kv_num_heads = 1, 8, 1
        window_size = 100  # Very large window
        # Small indices that are within the window
        indices = paddle.arange(1, seq + 1, dtype=paddle.int32).reshape(
            [bsz, 1, seq, 1]
        )

        result = startend_row_indices_add_sliding_window(
            indices, (window_size, 0), 0.0, kv_num_heads
        )

        # LTS_SWA = arange(100, 108) which is always larger than indices [1..8]
        # So where(indices < LTS_SWA) is True -> keep indices
        self.assertTrue(
            paddle.equal_all(result, indices).item(),
            f"Small indices should not be clipped. Got {result.numpy()}",
        )

    def test_head_wise_swa_ratio_partial(self):
        """When head_wise_swa_ratio > 0, some heads should not be clipped (non-SWA heads)."""
        bsz, seq, kv_num_heads = 1, 8, 4
        window_size = 3
        head_wise_swa_ratio = 0.5  # 2 out of 4 heads are SWA

        # Large values for all heads
        indices = (
            paddle.ones([bsz, kv_num_heads, seq, 1], dtype=paddle.int32) * 10000
        )

        result = startend_row_indices_add_sliding_window(
            indices, (window_size, 0), head_wise_swa_ratio, kv_num_heads
        )

        # swa_head_num = int(0.5 * 4) = 2
        # non_swa_head_num = 4 - 2 = 2
        # First 2 heads (non-SWA) should retain original values
        non_swa_result = result[:, :2, :, :]
        expected_non_swa = (
            paddle.ones([bsz, 2, seq, 1], dtype=paddle.int32) * 10000
        )
        self.assertTrue(
            paddle.equal_all(non_swa_result, expected_non_swa).item(),
            "Non-SWA heads should retain original indices",
        )

        # Last 2 heads (SWA) should be clipped
        swa_result = result[:, 2:, :, :]
        expected_swa = (
            paddle.arange(window_size, seq + window_size, dtype=paddle.int32)
            .reshape([1, 1, seq, 1])
            .expand([bsz, 2, seq, 1])
        )
        self.assertTrue(
            paddle.equal_all(swa_result, expected_swa).item(),
            f"SWA heads should be clipped. Got {swa_result.numpy()}",
        )

    def test_head_wise_swa_ratio_zero_all_swa(self):
        """When head_wise_swa_ratio=0.0, swa_head_num=0, no partial restore."""
        bsz, seq, kv_num_heads = 1, 4, 2
        window_size = 2
        # All heads should be fully clipped
        indices = paddle.ones([bsz, 1, seq, 1], dtype=paddle.int32) * 10000

        result = startend_row_indices_add_sliding_window(
            indices, (window_size, 0), 0.0, kv_num_heads
        )

        # All heads should be SWA-clipped
        expected = (
            paddle.arange(window_size, seq + window_size, dtype=paddle.int32)
            .reshape([1, 1, seq, 1])
            .expand([bsz, kv_num_heads, seq, 1])
        )
        self.assertTrue(
            paddle.equal_all(result, expected).item(),
            "All heads should be SWA-clipped when ratio=0",
        )

    # ---- int sliding_window branch (window_size = int, right ignored) ----
    # int W must be interpreted with window_size = W, identical to tuple (W, 0).

    def test_int_clips_values(self):
        """int sliding_window should clip start indices exactly like tuple (W, 0)."""
        bsz, seq, kv_num_heads = 1, 8, 1
        window_size = 3
        indices = paddle.ones([bsz, 1, seq, 1], dtype=paddle.int32) * 10000

        result = startend_row_indices_add_sliding_window(
            indices, window_size, 0.0, kv_num_heads
        )

        expected = paddle.arange(
            window_size, seq + window_size, dtype=paddle.int32
        ).reshape([1, 1, seq, 1])
        self.assertTrue(
            paddle.equal_all(result, expected).item(),
            f"Expected {expected.numpy()}, got {result.numpy()}",
        )

    def test_int_equivalent_to_tuple_left_zero(self):
        """int W output must equal tuple (W, 0) output across representative cases."""
        bsz, seq, kv_num_heads = 1, 8, 4
        window_size = 3
        for ratio in (0.0, 0.5):
            indices_int = (
                paddle.ones([bsz, 1, seq, 1], dtype=paddle.int32) * 10000
            )
            indices_tuple = (
                paddle.ones([bsz, 1, seq, 1], dtype=paddle.int32) * 10000
            )
            out_int = startend_row_indices_add_sliding_window(
                indices_int, window_size, ratio, kv_num_heads
            )
            out_tuple = startend_row_indices_add_sliding_window(
                indices_tuple, (window_size, 0), ratio, kv_num_heads
            )
            self.assertTrue(
                paddle.equal_all(out_int, out_tuple).item(),
                f"int vs tuple mismatch at ratio={ratio}",
            )

    def test_int_zero_returns_unchanged(self):
        """int sliding_window == 0 is falsy: returns input unchanged."""
        indices = paddle.ones([2, 1, 8, 1], dtype=paddle.int32) * 100
        result = startend_row_indices_add_sliding_window(indices, 0, 0.0, 4)
        self.assertTrue(paddle.equal_all(result, indices).item())


# ============================================================================
# Test: TransformerConfig SWA fields and validation
# ============================================================================


class TestTransformerConfigSWAFields(unittest.TestCase):
    """Tests for new SWA-related fields in TransformerConfig."""

    def test_default_swa_fields(self):
        """SWA fields should have correct defaults."""
        config = TransformerConfig(num_hidden_layers=4)
        self.assertEqual(config.swa_head_dim, config.head_dim)
        self.assertEqual(config.swa_v_head_dim, config.v_head_dim)
        self.assertEqual(
            config.swa_num_attention_heads, config.num_attention_heads
        )
        self.assertEqual(
            config.swa_num_key_value_heads, config.num_key_value_heads
        )
        self.assertEqual(config.swa_rope_theta, 10000)
        self.assertEqual(config.head_wise_swa_ratio, 0.0)
        self.assertIsNone(config.attention_value_scale)
        self.assertFalse(config.add_full_attention_sink_bias)
        self.assertTrue(config.add_swa_attention_sink_bias)

    def test_v_head_dim_defaults_to_head_dim(self):
        """When v_head_dim is not set, it should default to head_dim."""
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=128,
            num_attention_heads=4,
        )
        # head_dim = 128 // 4 = 32
        self.assertEqual(config.v_head_dim, 32)

    def test_v_head_dim_explicit(self):
        """When v_head_dim is explicitly set, it should be preserved."""
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=128,
            num_attention_heads=4,
            v_head_dim=64,
        )
        self.assertEqual(config.v_head_dim, 64)

    def test_swa_rope_theta_custom(self):
        """Custom swa_rope_theta should be preserved."""
        config = TransformerConfig(
            num_hidden_layers=4,
            swa_rope_theta=500000,
        )
        self.assertEqual(config.swa_rope_theta, 500000)

    def test_attention_value_scale_custom(self):
        """Custom attention_value_scale should be preserved."""
        config = TransformerConfig(
            num_hidden_layers=4,
            attention_value_scale=0.5,
        )
        self.assertAlmostEqual(config.attention_value_scale, 0.5)


class TestTransformerConfigSWAValidation(unittest.TestCase):
    """Tests for SWA validation in __post_init__ with MTP layers."""

    def test_mtp_requires_list_window_attn_skip_freq(self):
        """When num_nextn_predict_layers > 0, window_attn_skip_freq must be a list."""
        with self.assertRaises(TypeError):
            TransformerConfig(
                num_hidden_layers=4,
                num_nextn_predict_layers=1,
                sliding_window=(4096, 0),
                window_attn_skip_freq=3,  # int should fail
            )

    def test_mtp_window_attn_skip_freq_wrong_length(self):
        """window_attn_skip_freq list length must equal num_hidden + num_nextn."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=4,
                num_nextn_predict_layers=1,
                sliding_window=(4096, 0),
                window_attn_skip_freq=[1, 1, 0, 1],  # length 4, expected 5
            )

    def test_mtp_window_attn_skip_freq_correct_length(self):
        """window_attn_skip_freq list with correct length should pass."""
        config = TransformerConfig(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            sliding_window=(4096, 0),
            window_attn_skip_freq=[1, 1, 0, 1, 1],  # length 5 = 4 + 1
        )
        self.assertEqual(config.window_attn_skip_freq, [1, 1, 0, 1, 1])

    def test_no_mtp_no_validation(self):
        """When num_nextn_predict_layers=0, no validation on window_attn_skip_freq type."""
        # int is valid when no MTP
        config = TransformerConfig(
            num_hidden_layers=4,
            sliding_window=(4096, 0),
            window_attn_skip_freq=4,
        )
        self.assertEqual(config.window_attn_skip_freq, 4)

    def test_window_attn_skip_freq_non_positive_int_raises(self):
        """window_attn_skip_freq as int must be positive."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=4,
                sliding_window=(4096, 0),
                window_attn_skip_freq=0,
            )
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=4,
                sliding_window=(4096, 0),
                window_attn_skip_freq=-1,
            )

    def test_no_mtp_list_wrong_length_raises(self):
        """Without MTP, list length must equal num_hidden_layers."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=4,
                num_nextn_predict_layers=0,
                sliding_window=(4096, 0),
                window_attn_skip_freq=[1, 0, 1],  # length 3, expected 4
            )

    def test_head_wise_swa_ratio_out_of_range_raises(self):
        """head_wise_swa_ratio must be between 0.0 and 1.0."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=4,
                sliding_window=(4096, 0),
                head_wise_swa_ratio=1.5,
            )
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=4,
                sliding_window=(4096, 0),
                head_wise_swa_ratio=-0.1,
            )

    def test_int_sliding_window_accepted(self):
        """int sliding_window should be accepted and stored as-is (no tuple coercion)."""
        config = TransformerConfig(
            num_hidden_layers=4,
            sliding_window=4096,
            window_attn_skip_freq=4,
        )
        self.assertEqual(config.sliding_window, 4096)

    def test_int_sliding_window_mtp_validation(self):
        """int sliding_window with MTP still enforces list window_attn_skip_freq."""
        with self.assertRaises(TypeError):
            TransformerConfig(
                num_hidden_layers=4,
                num_nextn_predict_layers=1,
                sliding_window=4096,
                window_attn_skip_freq=3,  # int should fail under MTP
            )
        config = TransformerConfig(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            sliding_window=4096,
            window_attn_skip_freq=[1, 1, 0, 1, 1],
        )
        self.assertEqual(config.sliding_window, 4096)


# ============================================================================
# Test: SelfAttention with SWA mode
# ============================================================================


class TestSelfAttentionSWA(unittest.TestCase):
    """Tests for SelfAttention layer operating in SWA mode."""

    def _make_config(self, is_swa_layer=True, is_mtp=False, gated=False):
        """Create a config where layer 0 will be SWA based on skip_freq pattern."""
        num_hidden_layers = 4
        num_nextn_predict_layers = 1 if is_mtp else 0
        total_layers = num_hidden_layers + num_nextn_predict_layers

        if is_swa_layer:
            # Layer 0 will be SWA (pattern[0] = 1)
            pattern = [1] * total_layers
        else:
            # Layer 0 will NOT be SWA (pattern[0] = 0)
            pattern = [0] * total_layers

        # Use swa_head_dim == config.head_dim for forward tests because
        # DotProductAttention's non-flash code path uses
        # config.head_dim * config.num_attention_heads for reshape.
        # In production, flash attention uses reshape([bsz, q_len, -1]).
        # Also swa_num_key_value_heads == swa_num_attention_heads for the
        # non-flash bmm path (GQA differences tested in attribute-only tests).
        config = TransformerConfig(
            num_hidden_layers=num_hidden_layers,
            num_nextn_predict_layers=num_nextn_predict_layers,
            hidden_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=128,
            v_head_dim=128,
            sliding_window=(4096, 0),
            window_attn_skip_freq=pattern,
            swa_head_dim=128,
            swa_v_head_dim=128,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
            swa_rope_theta=500000,
            head_wise_swa_ratio=0.0,
        )
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = gated
        config.attention_value_scale = None
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False
        return config

    def _build_attn(self, config, layer_number=0, is_mtp_layer=False):
        return SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=layer_number,
            is_mtp_layer=is_mtp_layer,
        )

    def test_swa_layer_uses_swa_dims(self):
        """SWA layer should use swa_head_dim and swa_num_attention_heads."""
        config = self._make_config(is_swa_layer=True)
        attn = self._build_attn(config, layer_number=0)

        self.assertTrue(attn.is_swa)
        self.assertEqual(attn.head_dim, 128)  # swa_head_dim
        self.assertEqual(attn.v_head_dim, 128)  # swa_v_head_dim
        self.assertEqual(attn.num_attention_heads, 4)  # swa_num_attention_heads
        self.assertEqual(attn.num_key_value_heads, 4)  # swa_num_key_value_heads

    def test_non_swa_layer_uses_normal_dims(self):
        """Non-SWA layer should use normal head_dim and num_attention_heads."""
        config = self._make_config(is_swa_layer=False)
        attn = self._build_attn(config, layer_number=0)

        self.assertFalse(attn.is_swa)
        self.assertEqual(attn.head_dim, 128)  # config.head_dim
        self.assertEqual(attn.v_head_dim, 128)  # config.v_head_dim
        self.assertEqual(attn.num_attention_heads, 4)
        self.assertEqual(attn.num_key_value_heads, 4)

    def test_swa_projection_sizes(self):
        """SWA layer should compute correct projection sizes."""
        config = self._make_config(is_swa_layer=True)
        attn = self._build_attn(config, layer_number=0)

        # query_projection_size = swa_head_dim * swa_num_attention_heads = 128 * 4 = 512
        self.assertEqual(attn.query_projection_size, 128 * 4)
        # key_projection_size = swa_head_dim * swa_num_key_value_heads = 128 * 4 = 512
        self.assertEqual(attn.key_projection_size, 128 * 4)
        # value_projection_size = swa_v_head_dim * swa_num_key_value_heads = 128 * 4 = 512
        self.assertEqual(attn.value_projection_size, 128 * 4)
        # out_projection_size = swa_v_head_dim * swa_num_attention_heads = 128 * 4 = 512
        self.assertEqual(attn.out_projection_size, 128 * 4)

    def test_non_swa_projection_sizes(self):
        """Non-SWA layer should compute correct projection sizes."""
        config = self._make_config(is_swa_layer=False)
        attn = self._build_attn(config, layer_number=0)

        # query_projection_size = head_dim * num_attention_heads = 128 * 4 = 512
        self.assertEqual(attn.query_projection_size, 128 * 4)
        # key_projection_size = head_dim * num_key_value_heads = 128 * 4 = 512
        self.assertEqual(attn.key_projection_size, 128 * 4)
        # value_projection_size = v_head_dim * num_key_value_heads = 128 * 4 = 512
        self.assertEqual(attn.value_projection_size, 128 * 4)
        # out_projection_size = v_head_dim * num_attention_heads = 128 * 4 = 512
        self.assertEqual(attn.out_projection_size, 128 * 4)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_swa_forward_shape(self):
        """SWA layer forward should produce correct output shape."""
        config = self._make_config(is_swa_layer=True)
        attn = self._build_attn(config, layer_number=0).cuda()
        attn.bfloat16()

        seq_len, batch_size = 32, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        # swa_rotary_pos_emb with swa_head_dim
        swa_rotary_pos_emb = (
            paddle.randn((1, seq_len, 1, config.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        output, bias = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary_pos_emb,
        )

        self.assertEqual(
            output.shape, [batch_size, seq_len, config.hidden_size]
        )

    def test_non_swa_forward_shape(self):
        """Non-SWA layer forward should produce correct output shape."""
        config = self._make_config(is_swa_layer=False)
        attn = self._build_attn(config, layer_number=0)

        seq_len, batch_size = 32, 2
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config.head_dim))

        output, bias = attn(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
        )

        self.assertEqual(
            output.shape, [batch_size, seq_len, config.hidden_size]
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_swa_backward(self):
        """SWA layer should produce valid gradients."""
        config = self._make_config(is_swa_layer=True)
        attn = self._build_attn(config, layer_number=0).cuda()
        attn.bfloat16()

        seq_len, batch_size = 16, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        hidden_states.stop_gradient = False
        swa_rotary_pos_emb = (
            paddle.randn((1, seq_len, 1, config.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        output, bias = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary_pos_emb,
        )
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(
            paddle.all(paddle.isfinite(hidden_states.grad)).item(),
            "SWA backward: input gradient contains NaN or Inf",
        )

    def test_mtp_layer_swa_detection(self):
        """MTP layer should use correct layer number for SWA detection."""
        config = self._make_config(is_swa_layer=True, is_mtp=True)
        # For MTP layer, for_swa_layer_number = layer_number + num_hidden_layers
        # layer_number=0 -> for_swa_layer_number=4, pattern[4]=1 -> is_swa
        attn = self._build_attn(config, layer_number=0, is_mtp_layer=True)
        self.assertTrue(attn.is_swa)

    def test_swa_layer_with_rope_freqs_cis_raises(self):
        """SWA layer should raise when rope_freqs_cis is provided."""
        config = self._make_config(is_swa_layer=True)
        attn = self._build_attn(config, layer_number=0)

        seq_len, batch_size = 16, 2
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))
        rope_freqs_cis = paddle.randn((1, seq_len, 1, config.swa_head_dim))

        with self.assertRaises(ValueError) as ctx:
            attn(
                hidden_states,
                attention_mask=None,
                rope_freqs_cis=rope_freqs_cis,
            )
        self.assertIn(
            "Sliding Window Not Support rope_freqs_cis",
            str(ctx.exception),
        )


# ============================================================================
# Test: SelfAttention with SWA + different head_dim/v_head_dim (attribute only)
# ============================================================================


class TestSelfAttentionSWADifferentHeadDims(unittest.TestCase):
    """Tests for SWA with swa_head_dim=192, swa_v_head_dim=128 (attribute verification).

    NOTE: Forward tests are not run here because DotProductAttention's non-flash
    code path uses config.head_dim for reshape. In production, flash attention
    uses reshape([bsz, q_len, -1]) which handles different v_head_dim correctly.
    """

    def _make_config(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            v_head_dim=128,
            sliding_window=(4096, 0),
            window_attn_skip_freq=[1, 1, 0, 1],
            swa_head_dim=192,
            swa_v_head_dim=128,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=2,
        )
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = False
        config.attention_value_scale = None
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False
        return config

    def test_swa_different_head_v_head_dims(self):
        """SWA with head_dim=192 and v_head_dim=128 should set correct attributes."""
        config = self._make_config()
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,  # pattern[0]=1 -> SWA
        )

        self.assertTrue(attn.is_swa)
        self.assertEqual(attn.head_dim, 192)
        self.assertEqual(attn.v_head_dim, 128)
        self.assertEqual(attn.num_attention_heads, 4)
        self.assertEqual(attn.num_key_value_heads, 2)

    def test_swa_different_dims_projection_sizes(self):
        """Projection sizes should differ for q/k vs v when head_dim != v_head_dim."""
        config = self._make_config()
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        )

        # query_projection_size = swa_head_dim * swa_num_attention_heads = 192 * 4 = 768
        self.assertEqual(attn.query_projection_size, 192 * 4)
        # key_projection_size = swa_head_dim * swa_num_key_value_heads = 192 * 2 = 384
        self.assertEqual(attn.key_projection_size, 192 * 2)
        # value_projection_size = swa_v_head_dim * swa_num_key_value_heads = 128 * 2 = 256
        self.assertEqual(attn.value_projection_size, 128 * 2)
        # out_projection_size = swa_v_head_dim * swa_num_attention_heads = 128 * 4 = 512
        self.assertEqual(attn.out_projection_size, 128 * 4)

    def test_non_swa_layer_unchanged(self):
        """Non-SWA layer should use config's base head_dim/v_head_dim."""
        config = self._make_config()
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=2,  # pattern[2]=0 -> not SWA
        )

        self.assertFalse(attn.is_swa)
        self.assertEqual(attn.head_dim, 128)
        self.assertEqual(attn.v_head_dim, 128)
        self.assertEqual(attn.query_projection_size, 128 * 4)
        self.assertEqual(attn.key_projection_size, 128 * 2)
        self.assertEqual(attn.value_projection_size, 128 * 2)
        self.assertEqual(attn.out_projection_size, 128 * 4)


# ============================================================================
# Test: SelfAttention with SWA + GQA + gated attention
# ============================================================================


class TestSelfAttentionSWAGated(unittest.TestCase):
    """Tests for SWA with gated attention."""

    def _make_config(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=128,
            v_head_dim=128,
            sliding_window=(4096, 0),
            window_attn_skip_freq=[1, 1, 0, 1],
            swa_head_dim=128,
            swa_v_head_dim=128,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
        )
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = True
        config.attention_value_scale = None
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False
        return config

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_swa_gated_forward_shape(self):
        """SWA + gated attention should produce correct output shape."""
        config = self._make_config()
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,  # pattern[0]=1 -> SWA
        ).cuda()
        attn.bfloat16()

        self.assertTrue(attn.is_swa)

        seq_len, batch_size = 32, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        swa_rotary_pos_emb = (
            paddle.randn((1, seq_len, 1, config.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        output, bias = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary_pos_emb,
        )

        self.assertEqual(
            output.shape, [batch_size, seq_len, config.hidden_size]
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_swa_gated_backward(self):
        """SWA + gated attention backward should produce valid gradients."""
        config = self._make_config()
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        ).cuda()
        attn.bfloat16()

        seq_len, batch_size = 16, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        hidden_states.stop_gradient = False
        swa_rotary_pos_emb = (
            paddle.randn((1, seq_len, 1, config.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        output, bias = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary_pos_emb,
        )
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(
            paddle.all(paddle.isfinite(hidden_states.grad)).item(),
            "SWA gated backward: input gradient contains NaN or Inf",
        )


# ============================================================================
# Test: SelfAttention with attention_value_scale
# ============================================================================


class TestSelfAttentionValueScale(unittest.TestCase):
    """Tests for attention_value_scale feature."""

    def _make_config(self, value_scale=None):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=128,
            v_head_dim=128,
            attention_value_scale=value_scale,
        )
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = False
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False
        return config

    def test_value_scale_none_no_effect(self):
        """When value_scale is None, v_scale should be None."""
        config = self._make_config(value_scale=None)
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )
        self.assertIsNone(attn.v_scale)

    def test_value_scale_set(self):
        """When value_scale is set, v_scale should hold that value."""
        config = self._make_config(value_scale=0.5)
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )
        self.assertAlmostEqual(attn.v_scale, 0.5)

    def test_value_scale_forward(self):
        """Forward pass with value_scale should produce correct output shape."""
        config = self._make_config(value_scale=0.5)
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )

        seq_len, batch_size = 16, 2
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config.head_dim))

        output, bias = attn(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
        )

        self.assertEqual(
            output.shape, [batch_size, seq_len, config.hidden_size]
        )
        self.assertTrue(
            paddle.all(paddle.isfinite(output)).item(),
            "Output with value_scale contains NaN or Inf",
        )


# ============================================================================
# Test: DotProductAttention softmax_offset init (fix softmax_offset init)
# ============================================================================


class TestDotProductAttentionSoftmaxOffset(unittest.TestCase):
    """Tests for softmax_offset initialization with different softmax types."""

    def _make_config(
        self, softmax_type="vanilla", add_full_sink=False, add_swa_sink=False
    ):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=128,
            v_head_dim=128,
            softmax_type=softmax_type,
            add_full_attention_sink_bias=add_full_sink,
            add_swa_attention_sink_bias=add_swa_sink,
        )
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.gated_attention = False
        config.attention_value_scale = None
        config.head_wise_swa_ratio = 0.0
        return config

    def test_vanilla_softmax_no_offset(self):
        """vanilla softmax should have softmax_offset=None."""
        config = self._make_config(softmax_type="vanilla")
        dpa = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_mtp_layer=False,
            is_swa=False,
        )
        self.assertIsNone(dpa.softmax_offset)

    def test_off_by_one_softmax_zeros(self):
        """off-by-one softmax should have softmax_offset as zeros tensor."""
        config = self._make_config(softmax_type="off-by-one")
        dpa = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_mtp_layer=False,
            is_swa=False,
        )
        self.assertIsNotNone(dpa.softmax_offset)
        self.assertTrue(
            paddle.equal_all(
                dpa.softmax_offset,
                paddle.zeros([dpa.num_attention_heads_per_partition]),
            ).item()
        )

    def test_learnable_softmax_is_parameter(self):
        """learnable softmax should create a trainable parameter."""
        config = self._make_config(softmax_type="learnable")
        config.perform_initialization = False
        dpa = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_mtp_layer=False,
            is_swa=False,
        )
        self.assertIsNotNone(dpa.softmax_offset)
        self.assertFalse(dpa.softmax_offset.stop_gradient)

    def test_add_full_sink_bias_promotes_to_learnable(self):
        """add_full_attention_sink_bias=True should promote non-SWA to learnable."""
        config = self._make_config(
            softmax_type="vanilla",
            add_full_sink=True,
        )
        config.perform_initialization = False
        dpa = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_mtp_layer=False,
            is_swa=False,
        )
        # Should be learnable (not None)
        self.assertIsNotNone(dpa.softmax_offset)
        self.assertFalse(dpa.softmax_offset.stop_gradient)

    def test_add_swa_sink_bias_promotes_swa_to_learnable(self):
        """add_swa_attention_sink_bias=True should promote SWA layer to learnable."""
        config = self._make_config(
            softmax_type="vanilla",
            add_swa_sink=True,
        )
        config.perform_initialization = False
        config.sliding_window = (4096, 0)
        config.window_attn_skip_freq = [1, 1, 1, 1]
        dpa = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_mtp_layer=False,
            is_swa=True,
        )
        # Should be learnable (not None)
        self.assertIsNotNone(dpa.softmax_offset)
        self.assertFalse(dpa.softmax_offset.stop_gradient)

    def test_swa_no_sink_bias_stays_vanilla(self):
        """SWA layer without sink bias should stay vanilla (None offset)."""
        config = self._make_config(
            softmax_type="vanilla",
            add_swa_sink=False,
        )
        config.sliding_window = (4096, 0)
        config.window_attn_skip_freq = [1, 1, 1, 1]
        dpa = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_mtp_layer=False,
            is_swa=True,
        )
        self.assertIsNone(dpa.softmax_offset)

    def test_invalid_softmax_type_raises(self):
        """Invalid softmax_type should raise ValueError."""
        config = self._make_config(softmax_type="unknown_type")
        with self.assertRaises(ValueError):
            DotProductAttention(
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                is_mtp_layer=False,
                is_swa=False,
            )


# ============================================================================
# Test: SelfAttention with partial RoPE (rotary_percent < 1.0)
# ============================================================================


class TestSelfAttentionPartialRoPE(unittest.TestCase):
    """Tests for SWA with rotary_percent < 1.0 (split q/k for rope)."""

    def _make_config(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=128,
            v_head_dim=128,
            rotary_percent=0.5,  # Only half of head_dim gets RoPE
            sliding_window=(4096, 0),
            window_attn_skip_freq=[1, 0, 1, 0],
            swa_head_dim=128,
            swa_v_head_dim=128,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
        )
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = False
        config.attention_value_scale = None
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False
        return config

    def test_partial_rope_qk_rope_head_dim(self):
        """With rotary_percent=0.5, qk_rope_head_dim should be half of head_dim."""
        config = self._make_config()
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,  # SWA layer
        )
        # qk_rope_head_dim = int(128 * 0.5) = 64
        self.assertEqual(attn.qk_rope_head_dim, 64)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_partial_rope_forward(self):
        """Forward with partial RoPE should work correctly."""
        config = self._make_config()
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        ).cuda()
        attn.bfloat16()

        seq_len, batch_size = 16, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        # RoPE emb with qk_rope_head_dim (64, not full 128)
        swa_rotary_pos_emb = (
            paddle.randn((1, seq_len, 1, 64)).cuda().cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        output, bias = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary_pos_emb,
        )

        self.assertEqual(
            output.shape, [batch_size, seq_len, config.hidden_size]
        )
        self.assertTrue(
            paddle.all(paddle.isfinite(output)).item(),
            "Partial RoPE output contains NaN or Inf",
        )


# ============================================================================
# Test: Computational correctness of SWA forward/backward
# ============================================================================


class TestSWAComputationalCorrectness(unittest.TestCase):
    """Verify that the SWA code paths produce correct computational results.

    These tests go beyond attribute checks to validate actual forward/backward
    numerical behavior introduced in the GQA SWA commits.
    """

    def _make_config(
        self,
        is_swa_layer=True,
        gated=False,
        value_scale=None,
        rotary_percent=1.0,
    ):
        num_hidden_layers = 4
        pattern = [1, 0, 1, 0] if is_swa_layer else [0, 0, 0, 0]

        config = TransformerConfig(
            num_hidden_layers=num_hidden_layers,
            hidden_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=128,
            v_head_dim=128,
            sliding_window=(4096, 0),
            window_attn_skip_freq=pattern,
            swa_head_dim=128,
            swa_v_head_dim=128,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
            swa_rope_theta=500000,
            head_wise_swa_ratio=0.0,
            attention_value_scale=value_scale,
            rotary_percent=rotary_percent,
        )
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = gated
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False
        return config

    def _build_attn(self, config, layer_number=0, is_mtp_layer=False):
        return SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=layer_number,
            is_mtp_layer=is_mtp_layer,
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_swa_layer_uses_swa_rotary_not_normal_rotary(self):
        """SWA layer should use swa_rotary_pos_emb and ignore rotary_pos_emb.

        Changing rotary_pos_emb should NOT affect SWA layer output.
        Changing swa_rotary_pos_emb SHOULD affect SWA layer output.
        """
        config = self._make_config(is_swa_layer=True)
        attn = self._build_attn(config, layer_number=0).cuda()
        attn.bfloat16()

        seq_len, batch_size = 16, 2
        paddle.manual_seed(42)
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )

        swa_rotary = (
            paddle.randn((1, seq_len, 1, config.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        rotary_a = (
            paddle.randn((1, seq_len, 1, config.head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        rotary_b = (
            paddle.randn((1, seq_len, 1, config.head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        out_a, _ = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            rotary_pos_emb=rotary_a,
            swa_rotary_pos_emb=swa_rotary,
        )
        out_b, _ = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            rotary_pos_emb=rotary_b,
            swa_rotary_pos_emb=swa_rotary,
        )
        self.assertTrue(
            paddle.allclose(
                out_a.cast(paddle.float32),
                out_b.cast(paddle.float32),
                atol=1e-6,
            ).item(),
            "SWA layer output should NOT change when only rotary_pos_emb changes",
        )

    def test_non_swa_layer_uses_normal_rotary_ignores_swa(self):
        """Non-SWA layer should use rotary_pos_emb and ignore swa_rotary_pos_emb.

        Changing swa_rotary_pos_emb should NOT affect non-SWA layer output.
        """
        config = self._make_config(is_swa_layer=True)
        # layer_number=1 -> pattern[1]=0 -> NOT SWA
        attn = self._build_attn(config, layer_number=1)
        self.assertFalse(attn.is_swa)

        seq_len, batch_size = 16, 2
        paddle.manual_seed(42)
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))

        rotary = paddle.randn((1, seq_len, 1, config.head_dim))
        swa_rotary_a = paddle.randn((1, seq_len, 1, config.swa_head_dim))
        swa_rotary_b = paddle.randn((1, seq_len, 1, config.swa_head_dim))

        out_a, _ = attn(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=rotary,
            swa_rotary_pos_emb=swa_rotary_a,
        )
        out_b, _ = attn(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=rotary,
            swa_rotary_pos_emb=swa_rotary_b,
        )
        self.assertTrue(
            paddle.allclose(
                out_a.cast(paddle.float32),
                out_b.cast(paddle.float32),
                atol=1e-6,
            ).item(),
            "Non-SWA layer output should NOT change when swa_rotary_pos_emb changes",
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_value_scale_multiplies_value(self):
        """attention_value_scale should scale value, producing different output.

        With v_scale=k, output should differ from v_scale=None.
        """
        config_no_scale = self._make_config(value_scale=None)
        config_with_scale = self._make_config(value_scale=0.5)

        paddle.manual_seed(100)
        attn_no_scale = self._build_attn(config_no_scale, layer_number=0).cuda()
        attn_no_scale.bfloat16()

        paddle.manual_seed(100)
        attn_with_scale = self._build_attn(
            config_with_scale, layer_number=0
        ).cuda()
        attn_with_scale.bfloat16()

        seq_len, batch_size = 16, 2
        paddle.manual_seed(200)
        hidden_states = (
            paddle.randn((batch_size, seq_len, config_no_scale.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        swa_rotary = (
            paddle.randn((1, seq_len, 1, config_no_scale.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        out_no_scale, _ = attn_no_scale(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary,
        )
        out_with_scale, _ = attn_with_scale(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary,
        )

        # Outputs must differ due to value scaling
        self.assertFalse(
            paddle.allclose(
                out_no_scale.cast(paddle.float32),
                out_with_scale.cast(paddle.float32),
                atol=1e-5,
            ).item(),
            "Output with value_scale=0.5 should differ from value_scale=None",
        )
        # Both should be finite
        self.assertTrue(paddle.all(paddle.isfinite(out_no_scale)).item())
        self.assertTrue(paddle.all(paddle.isfinite(out_with_scale)).item())

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_swa_forward_deterministic(self):
        """Same input produces same output (deterministic forward)."""
        config = self._make_config(is_swa_layer=True)
        attn = self._build_attn(config, layer_number=0).cuda()
        attn.bfloat16()

        seq_len, batch_size = 16, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        swa_rotary = (
            paddle.randn((1, seq_len, 1, config.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        out1, bias1 = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary,
        )
        out2, bias2 = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary,
        )

        self.assertTrue(
            paddle.allclose(
                out1.cast(paddle.float32), out2.cast(paddle.float32), atol=1e-6
            ).item(),
            "Same input should produce same output (deterministic)",
        )
        self.assertTrue(
            paddle.allclose(
                bias1.cast(paddle.float32),
                bias2.cast(paddle.float32),
                atol=1e-6,
            ).item(),
            "Same input should produce same bias (deterministic)",
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_swa_backward_all_params_have_nonzero_grad(self):
        """All parameters in SWA attention should receive non-zero gradients."""
        config = self._make_config(is_swa_layer=True)
        attn = self._build_attn(config, layer_number=0).cuda()
        attn.bfloat16()

        seq_len, batch_size = 16, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        hidden_states.stop_gradient = False
        swa_rotary = (
            paddle.randn((1, seq_len, 1, config.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        output, _ = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary,
        )
        loss = output.sum()
        loss.backward()

        # Input gradient should be non-zero and finite
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(
            paddle.all(paddle.isfinite(hidden_states.grad)).item(),
            "Input gradient contains NaN or Inf",
        )
        self.assertFalse(
            paddle.all(hidden_states.grad == 0).item(),
            "Input gradient should not be all zeros",
        )

        # All trainable parameters should have non-zero gradients
        for name, param in attn.named_parameters():
            if param.stop_gradient:
                continue
            self.assertIsNotNone(
                param.grad, f"Parameter '{name}' has no gradient"
            )
            self.assertTrue(
                paddle.all(paddle.isfinite(param.grad)).item(),
                f"Parameter '{name}' gradient contains NaN or Inf",
            )
            self.assertFalse(
                paddle.all(param.grad == 0).item(),
                f"Parameter '{name}' gradient should not be all zeros",
            )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_swa_gated_attention_gate_modulates_output(self):
        """Gated SWA attention output should differ from non-gated SWA."""
        config_gated = self._make_config(is_swa_layer=True, gated=True)
        config_ungated = self._make_config(is_swa_layer=True, gated=False)

        paddle.manual_seed(42)
        attn_gated = self._build_attn(config_gated, layer_number=0).cuda()
        attn_gated.bfloat16()

        paddle.manual_seed(42)
        attn_ungated = self._build_attn(config_ungated, layer_number=0).cuda()
        attn_ungated.bfloat16()

        seq_len, batch_size = 16, 2
        paddle.manual_seed(123)
        hidden_states = (
            paddle.randn((batch_size, seq_len, config_gated.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        swa_rotary = (
            paddle.randn((1, seq_len, 1, config_gated.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        out_gated, _ = attn_gated(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary,
        )
        out_ungated, _ = attn_ungated(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary,
        )

        self.assertFalse(
            paddle.allclose(
                out_gated.cast(paddle.float32),
                out_ungated.cast(paddle.float32),
                atol=1e-6,
            ).item(),
            "Gated and ungated SWA outputs should differ",
        )
        self.assertTrue(paddle.all(paddle.isfinite(out_gated)).item())

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_swa_gated_backward_correctness(self):
        """Gated SWA backward should propagate non-zero finite gradients."""
        config = self._make_config(is_swa_layer=True, gated=True)
        attn = self._build_attn(config, layer_number=0).cuda()
        attn.bfloat16()

        seq_len, batch_size = 16, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        hidden_states.stop_gradient = False
        swa_rotary = (
            paddle.randn((1, seq_len, 1, config.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        output, _ = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary,
        )
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(paddle.all(paddle.isfinite(hidden_states.grad)).item())
        self.assertFalse(paddle.all(hidden_states.grad == 0).item())

        for name, param in attn.named_parameters():
            if param.stop_gradient:
                continue
            self.assertIsNotNone(param.grad, f"'{name}' has no grad")
            self.assertTrue(
                paddle.all(paddle.isfinite(param.grad)).item(),
                f"'{name}' grad has NaN/Inf",
            )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_partial_rope_only_rotates_partial_dims(self):
        """With rotary_percent=0.5, only first half of q/k dims get rotated.

        Verify: partial and full RoPE produce different outputs.
        """
        config_partial = self._make_config(
            is_swa_layer=True, rotary_percent=0.5
        )
        config_full = self._make_config(is_swa_layer=True, rotary_percent=1.0)

        paddle.manual_seed(42)
        attn_partial = self._build_attn(config_partial, layer_number=0).cuda()
        attn_partial.bfloat16()

        paddle.manual_seed(42)
        attn_full = self._build_attn(config_full, layer_number=0).cuda()
        attn_full.bfloat16()

        seq_len, batch_size = 16, 2
        paddle.manual_seed(123)
        hidden_states = (
            paddle.randn((batch_size, seq_len, config_partial.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )

        swa_rotary_partial = (
            paddle.randn((1, seq_len, 1, 64)).cuda().cast(paddle.bfloat16)
        )
        swa_rotary_full = (
            paddle.randn((1, seq_len, 1, 128)).cuda().cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        out_partial, _ = attn_partial(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary_partial,
        )
        out_full, _ = attn_full(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=swa_rotary_full,
        )

        self.assertTrue(paddle.all(paddle.isfinite(out_partial)).item())
        self.assertTrue(paddle.all(paddle.isfinite(out_full)).item())

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_partial_rope_zero_emb_vs_nonzero_emb(self):
        """With partial RoPE, forward with zero and non-zero embedding both succeed."""
        config = self._make_config(is_swa_layer=True, rotary_percent=0.5)
        attn = self._build_attn(config, layer_number=0).cuda()
        attn.bfloat16()

        seq_len, batch_size = 16, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )

        rope_dim = int(config.swa_head_dim * config.rotary_percent)  # 64
        zero_rotary = (
            paddle.zeros((1, seq_len, 1, rope_dim)).cuda().cast(paddle.bfloat16)
        )
        nonzero_rotary = (
            paddle.randn((1, seq_len, 1, rope_dim)).cuda().cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        out_zero, _ = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=zero_rotary,
        )
        out_nonzero, _ = attn(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            swa_rotary_pos_emb=nonzero_rotary,
        )

        self.assertTrue(paddle.all(paddle.isfinite(out_zero)).item())
        self.assertTrue(paddle.all(paddle.isfinite(out_nonzero)).item())

    def test_startend_row_indices_numerical_correctness(self):
        """Verify startend_row_indices_add_sliding_window computes correct values.

        For a sequence of length 8 with window_size=3:
        - LTS_SWA[i] = i + window_size = [3, 4, 5, 6, 7, 8, 9, 10]
        - If original indices are [0, 1, 2, 3, 4, 5, 6, 7] (representing causal mask start),
          result = min(original, LTS_SWA) for each position
        """
        bsz, seq, kv_num_heads = 1, 8, 1
        window_size = 3

        # Simulate causal mask: start at position 0 for all (full attention)
        # After SWA, start should be clipped to max(0, pos - window_size + 1)
        # represented as LTS: min(original_LTS, pos + window_size)
        original_indices = paddle.arange(0, seq, dtype=paddle.int32).reshape(
            [bsz, 1, seq, 1]
        )

        result = startend_row_indices_add_sliding_window(
            original_indices, (window_size, 0), 0.0, kv_num_heads
        )

        # LTS_SWA = [3, 4, 5, 6, 7, 8, 9, 10]
        # original = [0, 1, 2, 3, 4, 5, 6, 7]
        # where(original < LTS_SWA) → original is always < LTS_SWA → keep original
        expected = original_indices
        self.assertTrue(
            paddle.equal_all(result, expected).item(),
            f"Expected {expected.numpy().flatten()}, got {result.numpy().flatten()}",
        )

        # Now test with larger original indices (e.g., full attention LTS = seq)
        # original_indices = [8, 8, 8, 8, 8, 8, 8, 8] (all attending from start)
        full_attn_indices = (
            paddle.ones([bsz, 1, seq, 1], dtype=paddle.int32) * seq
        )
        result2 = startend_row_indices_add_sliding_window(
            full_attn_indices, (window_size, 0), 0.0, kv_num_heads
        )

        # LTS_SWA = [3, 4, 5, 6, 7, 8, 9, 10]
        # original = [8, 8, 8, 8, 8, 8, 8, 8]
        # where(8 < LTS_SWA[i]): True for i>=6 (LTS=9,10), False for i<6
        # result: [3, 4, 5, 6, 7, 8, 8, 8] — wait, let me recalculate
        # where(original < LTS_SWA, original, LTS_SWA)
        # pos 0: where(8 < 3) → False → LTS_SWA=3
        # pos 1: where(8 < 4) → False → LTS_SWA=4
        # pos 2: where(8 < 5) → False → LTS_SWA=5
        # pos 3: where(8 < 6) → False → LTS_SWA=6
        # pos 4: where(8 < 7) → False → LTS_SWA=7
        # pos 5: where(8 < 8) → False → LTS_SWA=8
        # pos 6: where(8 < 9) → True → original=8
        # pos 7: where(8 < 10) → True → original=8
        expected2 = paddle.to_tensor(
            [3, 4, 5, 6, 7, 8, 8, 8], dtype=paddle.int32
        ).reshape([bsz, 1, seq, 1])
        self.assertTrue(
            paddle.equal_all(result2, expected2).item(),
            f"Expected {expected2.numpy().flatten()}, got {result2.numpy().flatten()}",
        )

    def test_head_wise_swa_correctness(self):
        """Verify head_wise_swa_ratio correctly splits SWA/non-SWA heads.

        With ratio=0.5 and 4 heads:
        - swa_head_num = int(0.5 * 4) = 2
        - non_swa_head_num = 4 - 2 = 2
        - First 2 heads: non-SWA (original mask preserved)
        - Last 2 heads: SWA (clipped by window)
        """
        bsz, seq, kv_num_heads = 2, 16, 4
        window_size = 4
        head_wise_swa_ratio = 0.5

        # All heads initially have large LTS values (full attention)
        original = (
            paddle.ones([bsz, kv_num_heads, seq, 1], dtype=paddle.int32) * 1000
        )

        result = startend_row_indices_add_sliding_window(
            original, (window_size, 0), head_wise_swa_ratio, kv_num_heads
        )

        # non-SWA heads (first 2) should keep original values
        non_swa_heads = result[:, :2, :, :]
        self.assertTrue(
            paddle.all(non_swa_heads == 1000).item(),
            "Non-SWA heads should retain original full-attention indices",
        )

        # SWA heads (last 2) should be clipped by window
        swa_heads = result[:, 2:, :, :]
        expected_swa = (
            paddle.arange(window_size, seq + window_size, dtype=paddle.int32)
            .reshape([1, 1, seq, 1])
            .expand([bsz, 2, seq, 1])
        )
        self.assertTrue(
            paddle.equal_all(swa_heads, expected_swa).item(),
            f"SWA heads should be window-clipped.\n"
            f"Expected: {expected_swa[0, 0, :, 0].numpy()}\n"
            f"Got: {swa_heads[0, 0, :, 0].numpy()}",
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "Requires CUDA for SWA flash attention",
    )
    def test_mtp_layer_swa_detection_uses_offset(self):
        """MTP layer uses layer_number + num_hidden_layers for SWA pattern lookup.

        Non-MTP layer uses layer_number - num_empty_layers_add_in_head.
        """
        num_hidden_layers = 4
        # Pattern: layers 0-3 = [0,0,0,0] (non-SWA), layer 4 (MTP) = [1] (SWA)
        pattern = [0, 0, 0, 0, 1]
        config = TransformerConfig(
            num_hidden_layers=num_hidden_layers,
            num_nextn_predict_layers=1,
            hidden_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=128,
            v_head_dim=128,
            sliding_window=(4096, 0),
            window_attn_skip_freq=pattern,
            swa_head_dim=128,
            swa_v_head_dim=128,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
        )
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = False
        config.attention_value_scale = None
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False

        # MTP layer with layer_number=0 → for_swa_layer_number = 0 + 4 = 4 → pattern[4]=1 → SWA
        attn_mtp = self._build_attn(
            config, layer_number=0, is_mtp_layer=True
        ).cuda()
        attn_mtp.bfloat16()

        self.assertTrue(attn_mtp.is_swa)

        # Non-MTP layer with layer_number=0 → for_swa = 0 - 0 = 0 → pattern[0]=0 → NOT SWA
        attn_normal = self._build_attn(
            config, layer_number=0, is_mtp_layer=False
        ).cuda()
        attn_normal.bfloat16()

        self.assertFalse(attn_normal.is_swa)

        # Verify they produce different outputs with swa_rotary
        seq_len, batch_size = 16, 2
        hidden_states = (
            paddle.randn((batch_size, seq_len, config.hidden_size))
            .cuda()
            .cast(paddle.bfloat16)
        )
        swa_rotary = (
            paddle.randn((1, seq_len, 1, config.swa_head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        rotary = (
            paddle.randn((1, seq_len, 1, config.head_dim))
            .cuda()
            .cast(paddle.bfloat16)
        )
        startend = (
            paddle.arange(1, seq_len + 1, dtype=paddle.int32)
            .reshape([1, 1, seq_len, 1])
            .expand([batch_size, 1, seq_len, 1])
            .cuda()
        )

        out_mtp, _ = attn_mtp(
            hidden_states,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
            rotary_pos_emb=rotary,
            swa_rotary_pos_emb=swa_rotary,
        )
        out_normal, _ = attn_normal(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=rotary,
            swa_rotary_pos_emb=swa_rotary,
        )

        # MTP (SWA) uses swa_rotary, normal uses rotary → different outputs
        self.assertFalse(
            paddle.allclose(
                out_mtp.cast(paddle.float32),
                out_normal.cast(paddle.float32),
                atol=1e-5,
            ).item(),
            "MTP-SWA layer and normal layer should produce different outputs",
        )


class TestDotProductAttentionFlashMaskSWA(unittest.TestCase):
    """Test that DotProductAttention applies sliding_window in flashmask path."""

    def _make_swa_dot_product_attention(self):
        """Create a DotProductAttention with is_swa=True."""
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=64,
            v_head_dim=64,
            sliding_window=(4096, 0),
            window_attn_skip_freq=[1, 0, 1, 0],
            swa_head_dim=64,
            swa_v_head_dim=64,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
        )
        config.softmax_scale = None
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.perform_initialization = True
        config.params_dtype = "float32"
        config.init_method = init_method_normal(0.02)
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False
        config.head_wise_swa_ratio = 0.0
        config._attn_implementation = "flash"
        config.flashmask_use_varlen = False

        attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=True,
        )
        return attn

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.flashmask_attention"
    )
    def test_flashmask_path_applies_sliding_window(self, mock_fm):
        """Verify startend_row_indices_add_sliding_window is called when is_swa."""
        attn = self._make_swa_dot_product_attention()

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 64
        query = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )
        key = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )
        value = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )
        # startend_row_indices: [bsz, 1, seq_len, 1] - LTS format
        attn_mask_startend_row_indices = paddle.zeros(
            [batch_size, 1, seq_len, 1], dtype="int32"
        )

        # Mock flashmask_attention to return proper shape [b, s, h, d]
        mock_fm.return_value = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )

        output = attn(
            query,
            key,
            value,
            attention_mask=None,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        # flashmask_attention should have been called
        mock_fm.assert_called_once()
        # Check that the startend_row_indices passed to flashmask has been modified
        # (sliding window applied) - heads should be expanded
        call_kwargs = mock_fm.call_args
        passed_indices = call_kwargs[1]["startend_row_indices"]
        self.assertEqual(passed_indices.shape[1], num_heads)
        # Output shape: [batch, seq, hidden]
        self.assertEqual(
            output.shape, [batch_size, seq_len, num_heads * head_dim]
        )

    def _make_cp_swap2p_attention(self, sliding_window):
        """SWA + MLA + contiguous_a2a attention forced onto the CP path."""
        attn = self._make_swa_dot_product_attention()
        # Force the CP contiguous_swap2p SWA branch without a real CP group.
        attn.context_parallel_size = 2
        attn.config.multi_latent_attention = True
        attn.config.cp_balance_mode = "contiguous_a2a"
        # shape[-1] == 2 + experimental_dataflow avoids the .cuda() concat path
        # inside expand_attn_mask_startend_row_indices_for_cp.
        attn.config.experimental_dataflow = True
        attn.sliding_window = sliding_window
        return attn

    def test_cp_swap2p_rejects_infinite_window(self):
        """CP contiguous_swap2p SWA must reject an infinite (`-1`) window with a
        clear ValueError instead of failing deep inside the flashmask kernel."""
        attn = self._make_cp_swap2p_attention(-1)

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 64
        query = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )
        key = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )
        value = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )
        row_indices = paddle.zeros([batch_size, 1, seq_len, 2], dtype="int32")

        with self.assertRaises(ValueError) as ctx:
            attn(
                query,
                key,
                value,
                attention_mask=None,
                attn_mask_startend_row_indices=row_indices,
            )
        self.assertIn("positive sliding_window", str(ctx.exception))

    def test_cp_swap2p_rejects_tuple_infinite_window(self):
        """Tuple form `(-1, 0)` (infinite left window) is rejected the same way."""
        attn = self._make_cp_swap2p_attention((-1, 0))

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 64
        query = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )
        key = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )
        value = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim], dtype="bfloat16"
        )
        row_indices = paddle.zeros([batch_size, 1, seq_len, 2], dtype="int32")

        with self.assertRaises(ValueError):
            attn(
                query,
                key,
                value,
                attention_mask=None,
                attn_mask_startend_row_indices=row_indices,
            )


class TestGPTEmbeddingSWA(unittest.TestCase):
    """Tests for GPTEmbedding SWA rotary pos embedding paths."""

    def _make_swa_embedding(self, window_attn_skip_freq=None):
        """Create a GPTEmbedding with sliding_window and SWA RoPE."""
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )
        from paddleformers.fleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=64,
            v_head_dim=64,
            sliding_window=(4096, 0),
            window_attn_skip_freq=window_attn_skip_freq,
            swa_head_dim=64,
            swa_v_head_dim=64,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
        )
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.apply_rope_fusion = False
        config.sequence_parallel = False
        config.multimodal_embedding = False
        config.mtp_load_weight_only = False
        config.experimental_dataflow = False

        spec = GPTEmbeddingSpec(
            language_embedding=LayerSpec(layer=LanguageModelEmbedding),
            rope_embedding=LayerSpec(layer=RotaryEmbedding),
        )
        emb = GPTEmbedding(
            sublayers_spec=spec,
            config=config,
            vocab_size=1000,
            max_sequence_length=512,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            swa_rotary_base=10000,
        )
        return emb, config

    def test_swa_rotary_pos_emb_constructed(self):
        """When sliding_window is set, swa_rotary_pos_emb should be constructed."""
        emb, _ = self._make_swa_embedding(window_attn_skip_freq=[1, 0, 1, 0])
        self.assertIsNotNone(emb.swa_rotary_pos_emb)

    def test_swa_warning_when_skip_freq_none(self):
        """Should warn when sliding_window set but window_attn_skip_freq is None."""
        import warnings as w

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            emb, _ = self._make_swa_embedding(window_attn_skip_freq=None)
            swa_warnings = [
                x
                for x in caught
                if "window_attn_skip_freq is None" in str(x.message)
            ]
            self.assertGreater(len(swa_warnings), 0)
        self.assertIsNotNone(emb.swa_rotary_pos_emb)

    def test_swa_rotary_forward_produces_swa_emb(self):
        """Forward pass with SWA should produce swa_rotary_pos_emb in output."""
        emb, config = self._make_swa_embedding(
            window_attn_skip_freq=[1, 0, 1, 0]
        )

        batch_size, seq_len = 2, 16
        input_ids = paddle.randint(0, 1000, [batch_size, seq_len])

        dict_args = {
            "input_ids": input_ids,
            "position_ids": None,
            "attention_mask": None,
        }

        output = emb(dict_args)
        # Output should contain swa_rotary_pos_emb
        self.assertIn("swa_rotary_pos_emb", output)
        self.assertIsNotNone(output["swa_rotary_pos_emb"])

    def test_swa_rotary_forward_with_rope_fusion(self):
        """Forward with apply_rope_fusion should produce swa cos/sin."""
        emb, config = self._make_swa_embedding(
            window_attn_skip_freq=[1, 0, 1, 0]
        )
        config.apply_rope_fusion = True

        batch_size, seq_len = 2, 16
        input_ids = paddle.randint(0, 1000, [batch_size, seq_len])

        dict_args = {
            "input_ids": input_ids,
            "position_ids": None,
            "attention_mask": None,
        }

        output = emb(dict_args)
        self.assertIn("swa_rotary_pos_cos", output)
        self.assertIn("swa_rotary_pos_sin", output)
        self.assertIsNotNone(output["swa_rotary_pos_cos"])
        self.assertIsNotNone(output["swa_rotary_pos_sin"])


if __name__ == "__main__":
    unittest.main()
