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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.multi_latent_attention import (
    FP8OverlapProj,
    MLASelfAttentionSublayersSpec,
    MultiLatentAttention,
    _ec_compatible_rope_apply,
)


class TestMLASelfAttentionSublayersSpec(unittest.TestCase):
    """Tests for MLASelfAttentionSublayersSpec dataclass."""

    def test_default_values(self):
        spec = MLASelfAttentionSublayersSpec()
        self.assertIsNone(spec.q_a_layernorm)
        self.assertIsNone(spec.kv_a_layernorm)
        self.assertIsNone(spec.q_proj)
        self.assertIsNone(spec.q_a_proj)
        self.assertIsNone(spec.q_b_proj)
        self.assertIsNone(spec.kv_a_proj_with_mqa)
        self.assertIsNone(spec.kv_b_proj)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)
        self.assertIsNone(spec.gate_proj)

    def test_custom_values(self):
        spec = MLASelfAttentionSublayersSpec(
            q_a_layernorm=MagicMock(),
            kv_a_layernorm=MagicMock(),
            q_proj=MagicMock(),
        )
        self.assertIsNotNone(spec.q_a_layernorm)
        self.assertIsNotNone(spec.kv_a_layernorm)
        self.assertIsNotNone(spec.q_proj)


class TestECCompatibleRopeApply(unittest.TestCase):
    """Tests for _ec_compatible_rope_apply function."""

    def test_basic_apply(self):
        """Test basic RoPE application."""
        batch, seq_len, num_heads, head_dim = 1, 4, 2, 8
        q_pe = paddle.randn([batch, seq_len, num_heads, head_dim])
        k_pe = paddle.randn([batch, seq_len, 1, head_dim])

        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, seq_len)
        self.assertEqual(q_out.shape, q_pe.shape)
        self.assertEqual(k_out.shape, k_pe.shape)

    def test_apply_with_custom_base(self):
        """Test RoPE with custom rope_base."""
        batch, seq_len, num_heads, head_dim = 1, 4, 2, 8
        q_pe = paddle.randn([batch, seq_len, num_heads, head_dim])
        k_pe = paddle.randn([batch, seq_len, 1, head_dim])

        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, seq_len, rope_base=500000.0)
        self.assertEqual(q_out.shape, q_pe.shape)
        self.assertEqual(k_out.shape, k_pe.shape)

    def test_output_dtype_matches_input(self):
        """Test output dtype matches input dtype."""
        batch, seq_len, num_heads, head_dim = 1, 4, 2, 8
        q_pe = paddle.randn([batch, seq_len, num_heads, head_dim]).cast("float32")
        k_pe = paddle.randn([batch, seq_len, 1, head_dim]).cast("float32")

        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, seq_len)
        self.assertEqual(q_out.dtype, q_pe.dtype)
        self.assertEqual(k_out.dtype, k_pe.dtype)


class TestFP8OverlapProj(unittest.TestCase):
    """Tests for FP8OverlapProj PyLayer."""

    def test_forward_basic(self):
        """Test FP8OverlapProj forward pass."""
        x = paddle.randn([2, 4, 8])
        weight = paddle.randn([8, 4])
        out = FP8OverlapProj.apply(x, weight)
        self.assertEqual(out.shape, [2, 4, 4])

    def test_forward_matches_linear(self):
        """Test FP8OverlapProj forward matches paddle.nn.functional.linear."""
        x = paddle.randn([2, 4, 8])
        weight = paddle.randn([8, 4])
        out_custom = FP8OverlapProj.apply(x, weight)
        out_linear = paddle.nn.functional.linear(x, weight)
        self.assertTrue(paddle.allclose(out_custom, out_linear, atol=1e-5).item())


class TestMultiLatentAttentionRopeTypeValidation(unittest.TestCase):
    """Tests for MultiLatentAttention rope_type validation."""

    def test_unsupported_rope_type_raises(self):
        """Test that unsupported rope_type raises ValueError."""
        # Test the rope_type validation logic directly by patching
        # only the parts that would fail
        from paddleformers.fleet.transformer.multi_latent_attention import (
            MLASelfAttention,
        )

        config = MagicMock()
        config.rope_type = "invalid_rope"
        config.rotary_scaling_factor = 1.0
        config.mscale_all_dim = False
        config.v_head_dim = 16
        config.num_attention_heads = 4
        config.qk_nope_head_dim = 8
        config.qk_rope_head_dim = 8
        config.q_lora_rank = 8

        spec = MLASelfAttentionSublayersSpec(
            q_a_layernorm=MagicMock(),
            kv_a_layernorm=MagicMock(),
            core_attention=MagicMock(),
            o_proj=MagicMock(),
        )

        # Patch MLASelfAttention.__init__ to just call super().__init__
        # with a patched Attention.__init__ that sets self.config
        def patched_attention_init(self_inner, *a, **kw):
            paddle.nn.Layer.__init__(self_inner)
            self_inner.config = config
            self_inner.v_head_dim = config.v_head_dim
            self_inner.num_attention_heads = config.num_attention_heads
            self_inner.is_swa = False
            self_inner.is_mtp_layer = False

        with (
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.Attention.__init__",
                patched_attention_init,
            ),
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.build_spec_layer",
                return_value=MagicMock(),
            ),
        ):
            with self.assertRaises(ValueError) as ctx:
                MLASelfAttention(
                    config=config,
                    sublayers_spec=spec,
                    layer_number=1,
                )
            self.assertIn("Unsupported RoPE type", str(ctx.exception))


class TestMultiLatentAttentionForwardAssertions(unittest.TestCase):
    """Tests for MultiLatentAttention forward assertions."""

    def _make_mla(self):
        """Create a MultiLatentAttention with mocked internals."""
        config = MagicMock()
        config.sequence_parallel = False
        config.recompute_granularity = None
        config.gated_attention = False
        config.dw_p2p_overlap = False
        config.use_bias = True

        # Create a concrete subclass to avoid abstract class instantiation error
        class ConcreteMLA(MultiLatentAttention):
            def get_query_key_value_tensors(self, *args, **kwargs):
                return None

        with patch(
            "paddleformers.fleet.transformer.multi_latent_attention.Attention.__init__",
            return_value=None,
        ):
            mla = ConcreteMLA.__new__(ConcreteMLA)
            mla.config = config
            mla.layer_number = 1
            mla.recompute_core_attention = False
            mla.use_rr_flash_attention = False
            mla.training = True
            mla.attn_mask_type = AttnMaskType = MagicMock()
            mla.core_attention = MagicMock(return_value=paddle.randn([1, 4, 2, 16]))
            mla.o_proj = MagicMock(return_value=(paddle.randn([1, 4, 64]), None))
            mla.gate_proj = None
            mla.recompute_gated_attn = False
            return mla

    def test_forward_raises_with_rotary_pos_emb(self):
        """Test that forward raises when rotary_pos_emb is passed."""
        mla = self._make_mla()
        with self.assertRaises(AssertionError):
            mla.forward(
                hidden_states=paddle.randn([1, 4, 64]),
                attention_mask=None,
                rotary_pos_emb=paddle.randn([1, 4, 1, 8]),
            )

    def test_forward_raises_with_attention_bias(self):
        """Test that forward raises when attention_bias is passed."""
        mla = self._make_mla()
        with self.assertRaises(AssertionError):
            mla.forward(
                hidden_states=paddle.randn([1, 4, 64]),
                attention_mask=None,
                attention_bias=paddle.randn([1, 1, 4, 4]),
            )

    def test_forward_raises_with_rotary_cos_sin(self):
        """Test that forward raises when rotary_pos_cos/sin is passed."""
        mla = self._make_mla()
        with self.assertRaises(AssertionError):
            mla.forward(
                hidden_states=paddle.randn([1, 4, 64]),
                attention_mask=None,
                rotary_pos_cos=paddle.randn([1, 4, 1, 8]),
            )


if __name__ == "__main__":
    unittest.main()
