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

from paddleformers.fleet.transformer.attention import (
    Attention,
    CrossAttention,
    CrossAttentionSublayersSpec,
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import init_method_normal, scaled_init_method_normal


class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias


class SimpleRMSNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs.get("normalized_shape", kwargs.get("hidden_size"))
        eps = kwargs.get("norm_eps", kwargs.get("eps", 1e-5))
        self.weight = paddle.nn.Parameter(paddle.ones([hidden_size]))
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "softmax_scale": None,
        "use_bias": True,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": None,
        "apply_rope_fusion": False,
        "rotary_interleaved": False,
        "multi_latent_attention": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "rms_norm_eps": 1e-5,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "softmax_type": "vanilla",
        "fa_version": None,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestSelfAttentionSublayersSpec(unittest.TestCase):
    """Tests for SelfAttentionSublayersSpec."""

    def test_defaults(self):
        spec = SelfAttentionSublayersSpec()
        self.assertIsNone(spec.qkv_proj)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)
        self.assertIsNone(spec.q_norm)
        self.assertIsNone(spec.k_norm)

    def test_custom_values(self):
        spec = SelfAttentionSublayersSpec(
            qkv_proj=BiasedLinear,
            core_attention=DotProductAttention,
            o_proj=BiasedLinear,
            q_norm=SimpleRMSNorm,
            k_norm=SimpleRMSNorm,
        )
        self.assertEqual(spec.qkv_proj, BiasedLinear)


class TestCrossAttentionSublayersSpec(unittest.TestCase):
    """Tests for CrossAttentionSublayersSpec."""

    def test_defaults(self):
        spec = CrossAttentionSublayersSpec()
        self.assertIsNone(spec.linear_q)
        self.assertIsNone(spec.linear_kv)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)


class TestSelfAttention(unittest.TestCase):
    """Tests for SelfAttention."""

    def _build_attn(self, **overrides):
        config = _make_config(**overrides)
        return SelfAttention(
            config=config,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=None,
                k_norm=None,
            ),
            layer_number=1,
        )

    def test_constructor(self):
        attn = self._build_attn()
        self.assertIsInstance(attn, SelfAttention)
        self.assertIsInstance(attn, Attention)

    def test_forward_with_rope(self):
        attn = self._build_attn()
        attn.eval()
        x = paddle.randn([2, 4, 128])
        rotary_pos_emb = (paddle.randn([2, 4, 1, 16]),) * 2
        out, bias = attn(x, None, rotary_pos_emb=rotary_pos_emb)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_with_recompute(self):
        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
        )
        attn = SelfAttention(
            config=config,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            layer_number=1,
        )
        self.assertTrue(attn.recompute_core_attention)

    def test_forward_with_recompute_block(self):
        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
            recompute_method="block",
            recompute_num_layers=2,
        )
        attn = SelfAttention(
            config=config,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            layer_number=1,
        )
        self.assertTrue(attn.recompute_core_attention)

    def test_q_norm_k_norm(self):
        attn = SelfAttention(
            config=_make_config(),
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=SimpleRMSNorm,
                k_norm=SimpleRMSNorm,
            ),
            layer_number=1,
        )
        attn.eval()
        x = paddle.randn([2, 4, 128])
        out, bias = attn(x, None)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_set_for_recompute_input_layernorm_raises(self):
        attn = self._build_attn()
        with self.assertRaises(NotImplementedError):
            attn.set_for_recompute_input_layernorm()

    def test_get_query_key_value_tensors_split_true(self):
        attn = self._build_attn()
        x = paddle.randn([2, 4, 128])
        q, k, v = attn.get_query_key_value_tensors(x, split_qkv=True)
        self.assertEqual(q.shape[-1], 32)  # head_dim
        self.assertEqual(k.shape[-1], 32)
        self.assertEqual(v.shape[-1], 32)

    def test_get_query_key_value_tensors_split_false(self):
        attn = self._build_attn()
        x = paddle.randn([2, 4, 128])
        result = attn.get_query_key_value_tensors(x, split_qkv=False)
        mixed_qkv, split_arg_list = result
        self.assertEqual(len(split_arg_list), 3)


class TestSelfAttentionGated(unittest.TestCase):
    """Tests for SelfAttention with gated attention."""

    def test_gated_attention_forward(self):
        config = _make_config(gated_attention=True)
        attn = SelfAttention(
            config=config,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            layer_number=1,
        )
        attn.eval()
        x = paddle.randn([2, 4, 128])
        out, bias = attn(x, None)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_gated_attention_qkv_output_has_gate(self):
        config = _make_config(gated_attention=True)
        attn = SelfAttention(
            config=config,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            layer_number=1,
        )
        x = paddle.randn([2, 4, 128])
        q, k, v, gate = attn.get_query_key_value_tensors(x, split_qkv=True)
        self.assertIsNotNone(gate)


class TestCrossAttention(unittest.TestCase):
    """Tests for CrossAttention."""

    def test_constructor(self):
        config = _make_config()
        attn = CrossAttention(
            config=config,
            sublayers_spec=CrossAttentionSublayersSpec(
                linear_q=BiasedLinear,
                linear_kv=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            layer_number=1,
        )
        self.assertIsInstance(attn, CrossAttention)

    def test_group_query_raises(self):
        config = _make_config(num_key_value_heads=2)
        with self.assertRaises(ValueError):
            CrossAttention(
                config=config,
                sublayers_spec=CrossAttentionSublayersSpec(
                    linear_q=BiasedLinear,
                    linear_kv=BiasedLinear,
                    core_attention=DotProductAttention,
                    o_proj=BiasedLinear,
                ),
                layer_number=1,
            )

    def test_backward_dw_raises(self):
        config = _make_config()
        attn = CrossAttention(
            config=config,
            sublayers_spec=CrossAttentionSublayersSpec(
                linear_q=BiasedLinear,
                linear_kv=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            layer_number=1,
        )
        # CrossAttention does not have backward_dw (it's only on SelfAttention)
        with self.assertRaises(AttributeError):
            attn.backward_dw()

    def test_split_qkv_false_raises(self):
        config = _make_config()
        attn = CrossAttention(
            config=config,
            sublayers_spec=CrossAttentionSublayersSpec(
                linear_q=BiasedLinear,
                linear_kv=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            layer_number=1,
        )
        with self.assertRaises(AssertionError):
            attn.get_query_key_value_tensors(
                paddle.randn([2, 4, 128]),
                paddle.randn([2, 4, 128]),
                split_qkv=False,
            )


class TestSelfAttentionWithFlashAttnRecompute(unittest.TestCase):
    """Tests for flash_attn recompute settings."""

    def test_flash_attn_recompute_list(self):
        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["core_attn", "flash_attn"],
        )
        attn = SelfAttention(
            config=config,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            layer_number=1,
        )
        self.assertTrue(attn.use_rr_flash_attention)


class TestSelfAttentionSequenceParallel(unittest.TestCase):
    """Tests for SelfAttention with sequence parallel."""

    def test_sp_forward(self):
        config = _make_config(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            num_key_value_heads=4,
        )
        attn = SelfAttention(
            config=config,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            layer_number=1,
        )
        attn.eval()
        x = paddle.randn([4, 2, 128])  # [seq, batch, hidden]
        out, bias = attn(x, None)
        self.assertEqual(out.shape, [4, 2, 128])


if __name__ == "__main__":
    unittest.main()
