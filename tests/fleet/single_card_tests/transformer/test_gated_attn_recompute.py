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

"""Tests for gated_attn selective recompute in SelfAttention."""

import unittest

import paddle

from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.enums import AttnMaskType
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
        eps = kwargs.get("norm_eps", kwargs.get("eps", 1e-5))
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


def _make_config(recompute_gated=False):
    """Build config with gated_attention enabled, optionally with selective recompute."""
    config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=128,
        num_attention_heads=4,
    )
    config.num_key_value_heads = config.num_attention_heads
    config.head_dim = config.hidden_size // config.num_attention_heads
    config.softmax_scale = None
    config.use_bias = True
    config.no_rope_freq = None
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

    if recompute_gated:
        config.recompute_granularity = "selective"
        config.recompute_modules = ["gated_attn"]
    else:
        config.recompute_granularity = None
        config.recompute_modules = None

    return config


def _build_attn(config):
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


class TestGatedAttnRecompute(unittest.TestCase):
    """Test gated_attn selective recompute: numerical correctness and memory saving."""

    def _run_forward_backward(self, attn, hidden_states, rotary_pos_emb):
        """Run forward + backward and return output and input grad."""
        hidden_states = hidden_states.clone()
        hidden_states.stop_gradient = False
        output, _ = attn(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
        loss = output.sum()
        loss.backward()
        return output.detach(), hidden_states.grad.detach()

    def test_recompute_numerical_correctness(self):
        """Recompute path should produce identical output and gradients as non-recompute."""
        paddle.manual_seed(42)
        config_ref = _make_config(recompute_gated=False)
        attn_ref = _build_attn(config_ref)
        attn_ref.train()

        paddle.manual_seed(42)
        config_rc = _make_config(recompute_gated=True)
        attn_rc = _build_attn(config_rc)
        attn_rc.train()

        seq_len, batch_size = 32, 2
        paddle.manual_seed(123)
        hidden_states = paddle.randn([batch_size, seq_len, 128])
        rotary_pos_emb = paddle.randn([1, seq_len, 1, 32])

        out_ref, grad_ref = self._run_forward_backward(attn_ref, hidden_states, rotary_pos_emb)
        out_rc, grad_rc = self._run_forward_backward(attn_rc, hidden_states, rotary_pos_emb)

        self.assertTrue(
            paddle.equal_all(out_ref, out_rc).item(),
            f"Output mismatch: max diff = {(out_ref - out_rc).abs().max().item()}",
        )
        self.assertTrue(
            paddle.equal_all(grad_ref, grad_rc).item(),
            f"Grad mismatch: max diff = {(grad_ref - grad_rc).abs().max().item()}",
        )


if __name__ == "__main__":
    unittest.main()
