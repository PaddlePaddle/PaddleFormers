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

import paddle

from paddleformers.fleet.transformer.multi_latent_attention import (
    FP8OverlapProj,
    MultiLatentAttention,
    _ec_compatible_rope_apply,
)


class TestMLASelfAttentionBackwardDW(unittest.TestCase):
    """Tests for MLASelfAttention backward_dw methods."""

    def _make_mla_self_attn(self, q_lora_rank=None):
        """Create a MLASelfAttention with mocked internals."""
        config = MagicMock()
        config.head_dim = 16
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.hidden_size = 64
        config.v_head_dim = 16
        config.qk_nope_head_dim = 8
        config.qk_rope_head_dim = 8
        config.q_lora_rank = q_lora_rank
        config.kv_lora_rank = 16
        config.rope_type = "rope"
        config.rotary_interleaved = False
        config.rope_theta = 10000.0
        config.rotary_scaling_factor = 1.0
        config.mscale_all_dim = False
        config.gated_attention = False
        config.dw_p2p_overlap = False
        config.use_bias = True
        config.recompute_granularity = None
        config.recompute_modules = None
        config.sequence_parallel = False

        with (
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.Attention.__init__",
                return_value=None,
            ),
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.RotaryEmbedding"
            ),
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.build_spec_layer"
            ),
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.ProcessGroupCollection.use_mpu_process_groups"
            ),
        ):
            from paddleformers.fleet.transformer.multi_latent_attention import (
                MLASelfAttention,
            )

            mla = MLASelfAttention.__new__(MLASelfAttention)
            mla.config = config
            mla.kv_b_proj = MagicMock()
            mla.kv_a_proj_with_mqa = MagicMock()
            mla.o_proj = MagicMock()
            if q_lora_rank is not None:
                mla.q_a_proj = MagicMock()
                mla.q_b_proj = MagicMock()
            else:
                mla.q_proj = MagicMock()
            return mla

    def test_backward_dw_with_q_lora_rank(self):
        """Test backward_dw with q_lora_rank (uses q_a_proj and q_b_proj)."""
        mla = self._make_mla_self_attn(q_lora_rank=32)
        mla.backward_dw()
        mla.kv_b_proj.backward_dw.assert_called_once()
        mla.kv_a_proj_with_mqa.backward_dw.assert_called_once()
        mla.q_a_proj.backward_dw.assert_called_once()
        mla.q_b_proj.backward_dw.assert_called_once()
        mla.o_proj.backward_dw.assert_called_once()

    def test_backward_dw_without_q_lora_rank(self):
        """Test backward_dw without q_lora_rank (uses q_proj)."""
        mla = self._make_mla_self_attn(q_lora_rank=None)
        mla.backward_dw()
        mla.kv_b_proj.backward_dw.assert_called_once()
        mla.kv_a_proj_with_mqa.backward_dw.assert_called_once()
        mla.q_proj.backward_dw.assert_called_once()
        mla.o_proj.backward_dw.assert_called_once()


class TestFP8OverlapProjBackward(unittest.TestCase):
    """Tests for FP8OverlapProj backward pass."""

    def test_backward_with_stop_gradient_weight(self):
        """Test FP8OverlapProj backward with weight that has stop_gradient."""
        x = paddle.randn([2, 4, 8])
        weight = paddle.randn([8, 4])
        weight.stop_gradient = True

        out = FP8OverlapProj.apply(x, weight)
        self.assertEqual(out.shape, [2, 4, 4])


class TestMultiLatentAttentionGate(unittest.TestCase):
    """Tests for MultiLatentAttention _gate method."""

    def _make_mla_with_gate(self):
        """Create a MultiLatentAttention with gate."""
        config = MagicMock()
        config.sequence_parallel = False
        config.recompute_granularity = None
        config.gated_attention = True
        config.dw_p2p_overlap = False
        config.use_bias = True
        config.sigmoid_gate_fusion = False

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
            mla.attn_mask_type = MagicMock()
            mla.core_attention = MagicMock(
                return_value=paddle.randn([1, 4, 2, 16])
            )
            mla.o_proj = MagicMock(
                return_value=(paddle.randn([1, 4, 64]), None)
            )
            mla.gate_proj = MagicMock(
                return_value=(paddle.randn([1, 4, 64]), None)
            )
            mla.recompute_gated_attn = False
            return mla

    def test_gate_without_fusion(self):
        """Test _gate without sigmoid_gate_fusion."""
        mla = self._make_mla_with_gate()
        mla.config.sigmoid_gate_fusion = False

        hidden_states = paddle.randn([1, 4, 64])
        core_attn_out = paddle.randn([1, 4, 64])

        result = mla._gate(hidden_states, core_attn_out)
        self.assertEqual(result.shape, [1, 4, 64])


class TestECCompatibleRopeApplyEdgeCases(unittest.TestCase):
    """Edge case tests for _ec_compatible_rope_apply."""

    def test_single_token_sequence(self):
        """Test RoPE with single-token sequence."""
        batch, seq_len, num_heads, head_dim = 1, 1, 2, 8
        q_pe = paddle.randn([batch, seq_len, num_heads, head_dim])
        k_pe = paddle.randn([batch, seq_len, 1, head_dim])

        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, seq_len)
        self.assertEqual(q_out.shape, q_pe.shape)
        self.assertEqual(k_out.shape, k_pe.shape)

    def test_large_batch(self):
        """Test RoPE with larger batch size."""
        batch, seq_len, num_heads, head_dim = 4, 8, 2, 8
        q_pe = paddle.randn([batch, seq_len, num_heads, head_dim])
        k_pe = paddle.randn([batch, seq_len, 1, head_dim])

        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, seq_len)
        self.assertEqual(q_out.shape, q_pe.shape)
        self.assertEqual(k_out.shape, k_pe.shape)


if __name__ == "__main__":
    unittest.main()
