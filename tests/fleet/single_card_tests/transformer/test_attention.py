# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle

from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import init_method_normal, scaled_init_method_normal


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
        self.weight = paddle.nn.Parameter(paddle.zeros([hidden_size]))
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )

        # TODO(liangshuhao): make these args formal
        self.config.num_key_value_heads = self.config.num_attention_heads
        self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.config.softmax_scale = None
        self.config.use_bias = True
        self.config.no_rope_freq = None
        self.config.recompute_granularity = None
        self.config.fused_single_qkv_rope = False
        self.config.rotary_interleaved = False
        self.config.multi_latent_attention = False
        self.config.init_method = init_method_normal(0.02)
        self.config.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
        self.config.rms_norm_eps = 1e-5
        self.config.context_parallel_size = 1
        self.config.apply_query_key_layer_scaling = False
        self.config.sliding_window = None
        self.config.window_attn_skip_freq = None
        self.config.fp16 = False
        self.config.bf16 = False
        self.config.masked_softmax_fusion = False
        self.config.attention_softmax_in_fp32 = True
        self.config.attention_dropout = 0.1
        self.config.softmax_type = "vanilla"

        self.self_attn = SelfAttention(
            self.config,
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

    def test_self_attention(self):
        config = self.self_attn.config
        sequence_length = 127
        micro_batch_size = 2
        hidden_size = self.self_attn.config.hidden_size

        hidden_states = paddle.randn(
            (micro_batch_size, sequence_length, hidden_size),
        )
        rotary_pos_emb = paddle.randn((1, sequence_length, 1, self.config.head_dim))

        output, bias = self.self_attn(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)

        # Check if output and bias have the correct shape
        assert output.shape[0] == micro_batch_size
        assert output.shape[1] == sequence_length
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


class TestSelfAttentionQKNormPerLayer(unittest.TestCase):
    """Test SelfAttention with qk_norm_type='per_layer'."""

    def setUp(self):
        self.config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )
        self.config.num_key_value_heads = self.config.num_attention_heads
        self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.config.softmax_scale = None
        self.config.use_bias = True
        self.config.no_rope_freq = None
        self.config.recompute_granularity = None
        self.config.fused_single_qkv_rope = False
        self.config.rotary_interleaved = False
        self.config.multi_latent_attention = False
        self.config.init_method = init_method_normal(0.02)
        self.config.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
        self.config.rms_norm_eps = 1e-5
        self.config.context_parallel_size = 1
        self.config.apply_query_key_layer_scaling = False
        self.config.sliding_window = None
        self.config.window_attn_skip_freq = None
        self.config.fp16 = False
        self.config.bf16 = False
        self.config.masked_softmax_fusion = False
        self.config.attention_softmax_in_fp32 = True
        self.config.attention_dropout = 0.1
        self.config.softmax_type = "vanilla"
        self.config.qk_norm_type = "per_layer"

        self.self_attn = SelfAttention(
            self.config,
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

    def test_self_attention_qk_norm_per_layer(self):
        config = self.self_attn.config
        sequence_length = 127
        micro_batch_size = 2
        hidden_size = self.self_attn.config.hidden_size

        hidden_states = paddle.randn(
            (micro_batch_size, sequence_length, hidden_size),
        )
        rotary_pos_emb = paddle.randn((1, sequence_length, 1, self.config.head_dim))

        output, bias = self.self_attn(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)

        # Check if output and bias have the correct shape
        assert output.shape[0] == micro_batch_size
        assert output.shape[1] == sequence_length
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


class TestMLASelfAttention(unittest.TestCase):
    def setUp(self):
        self.config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=1,
        )

        self.config.num_key_value_heads = self.config.num_attention_heads
        self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.config.softmax_scale = None
        self.config.use_bias = True
        self.config.no_rope_freq = None
        self.config.recompute_granularity = None
        self.config.fused_single_qkv_rope = False
        self.config.rotary_interleaved = False
        self.config.multi_latent_attention = True
        self.config.init_method = init_method_normal(0.02)
        self.config.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
        self.config.rms_norm_eps = 1e-5
        self.config.context_parallel_size = 1
        self.config.apply_query_key_layer_scaling = False
        self.config.sliding_window = None
        self.config.window_attn_skip_freq = None
        self.config.fp16 = False
        self.config.bf16 = False
        self.config.masked_softmax_fusion = False
        self.config.attention_softmax_in_fp32 = True
        self.config.attention_dropout = 0.1
        self.config.softmax_type = "vanilla"

    def test_self_attention(self):
        self.self_attn = MLASelfAttention(
            self.config,
            MLASelfAttentionSublayersSpec(
                q_proj=BiasedLinear,
                q_a_proj=BiasedLinear,
                q_b_proj=BiasedLinear,
                kv_a_proj_with_mqa=BiasedLinear,
                kv_b_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_a_layernorm=RMSNorm,
                kv_a_layernorm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )
        config = self.self_attn.config
        sequence_length = 127
        micro_batch_size = 2
        hidden_size = self.self_attn.config.hidden_size

        hidden_states = paddle.randn(
            (micro_batch_size, sequence_length, hidden_size),
        )

        output, bias = self.self_attn(
            hidden_states,
            attention_mask=None,
        )

        # Check if output and bias have the correct shape
        assert output.shape[0] == micro_batch_size
        assert output.shape[1] == sequence_length
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_self_attention_sp(self):
        self.config.sequence_parallel = True
        self.self_attn = MLASelfAttention(
            self.config,
            MLASelfAttentionSublayersSpec(
                q_proj=BiasedLinear,
                q_a_proj=BiasedLinear,
                q_b_proj=BiasedLinear,
                kv_a_proj_with_mqa=BiasedLinear,
                kv_b_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_a_layernorm=RMSNorm,
                kv_a_layernorm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )
        config = self.self_attn.config
        sequence_length = 127
        micro_batch_size = 2
        hidden_size = self.self_attn.config.hidden_size

        hidden_states = paddle.randn(
            (micro_batch_size, sequence_length, hidden_size),
        )

        output, bias = self.self_attn(
            hidden_states,
            attention_mask=None,
        )

        # Check if output and bias have the correct shape
        assert output.shape[0] == micro_batch_size
        assert output.shape[1] == sequence_length
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


class TestGatedSelfAttention(unittest.TestCase):
    """Test SelfAttention with gated_attention=True (forward and backward)."""

    def _make_config(self, gqa=False):
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )
        config.num_key_value_heads = 2 if gqa else config.num_attention_heads
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = True
        return config

    def _build_attn(self, config):
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
            layer_number=1,
        )

    def test_gated_attention_forward_shape(self):
        """Gated attention output should have the same shape as standard attention."""
        config = self._make_config()
        attn = self._build_attn(config)

        seq_len, batch_size = 64, 2
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config.head_dim))

        output, bias = attn(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)

        self.assertEqual(output.shape, [batch_size, seq_len, config.hidden_size])
        self.assertEqual(bias.shape[0], config.hidden_size)
        self.assertTrue(
            paddle.all(paddle.isfinite(output)).item(),
            "Output contains NaN or Inf",
        )

    def test_gated_attention_backward(self):
        """Gated attention should produce valid gradients for all parameters."""
        config = self._make_config()
        attn = self._build_attn(config)

        seq_len, batch_size = 32, 2
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))
        hidden_states.stop_gradient = False
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config.head_dim))

        output, bias = attn(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
        loss = output.sum()
        loss.backward()

        # Check input gradient exists and is finite
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(
            paddle.all(paddle.isfinite(hidden_states.grad)).item(),
            "Input gradient contains NaN or Inf",
        )

        # Check all parameter gradients exist and are finite
        for name, param in attn.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertTrue(
                paddle.all(paddle.isfinite(param.grad)).item(),
                f"Parameter {name} gradient contains NaN or Inf",
            )

    def test_gated_attention_gqa(self):
        """Gated attention should work with grouped query attention (GQA)."""
        config = self._make_config(gqa=True)
        attn = self._build_attn(config)

        seq_len, batch_size = 32, 2
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))
        hidden_states.stop_gradient = False
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config.head_dim))

        output, bias = attn(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
        loss = output.sum()
        loss.backward()

        self.assertEqual(output.shape, [batch_size, seq_len, config.hidden_size])
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(
            paddle.all(paddle.isfinite(output)).item(),
            "GQA gated attention output contains NaN or Inf",
        )

    def test_gate_has_effect(self):
        """Verify that the gate actually modulates the output (not a no-op)."""
        config_gated = self._make_config()
        config_ungated = self._make_config()
        config_ungated.gated_attention = False

        paddle.manual_seed(42)
        attn_gated = self._build_attn(config_gated)
        paddle.manual_seed(42)
        attn_ungated = self._build_attn(config_ungated)

        seq_len, batch_size = 32, 2
        paddle.manual_seed(123)
        hidden_states = paddle.randn((batch_size, seq_len, config_gated.hidden_size))
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config_gated.head_dim))

        out_gated, _ = attn_gated(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
        out_ungated, _ = attn_ungated(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)

        # Outputs should differ because gated has extra gate projection
        self.assertFalse(
            paddle.allclose(out_gated, out_ungated, atol=1e-6).item(),
            "Gated and ungated outputs should differ",
        )


class TestMLAUseVarlenSelfAttention(TestMLASelfAttention):
    def setUp(self):
        super().setUp()
        self.config.flashmask_use_varlen = True


if __name__ == "__main__":
    unittest.main()
