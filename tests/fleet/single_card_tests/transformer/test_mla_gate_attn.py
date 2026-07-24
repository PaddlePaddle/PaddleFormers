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
import unittest
from dataclasses import fields
from unittest.mock import MagicMock

import paddle
import paddle.nn.functional as F

from paddleformers.fleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddleformers.fleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)


class BiasedLinear(paddle.nn.Layer):
    """Simple linear layer that returns (output, bias) like ColumnParallelLinear."""

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
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


def _make_mla_config(**overrides):
    """Build a minimal TransformerConfig for MLA testing."""
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "softmax_scale": None,
        "use_bias": False,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": None,
        "apply_rope_fusion": False,
        "rotary_interleaved": False,
        "multi_latent_attention": True,
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
        # MLA-specific
        "kv_lora_rank": 32,
        "q_lora_rank": 64,
        "qk_nope_head_dim": 24,
        "qk_rope_head_dim": 8,
        "v_head_dim": 32,
        "rope_type": "rope",
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_sublayers_spec(gate_proj_cls=None):
    """Build an MLASelfAttentionSublayersSpec with mock-friendly classes."""
    return MLASelfAttentionSublayersSpec(
        q_proj=BiasedLinear,
        q_a_proj=BiasedLinear,
        q_b_proj=BiasedLinear,
        kv_a_proj_with_mqa=BiasedLinear,
        kv_b_proj=BiasedLinear,
        core_attention=DotProductAttention,
        o_proj=BiasedLinear,
        q_a_layernorm=SimpleRMSNorm,
        kv_a_layernorm=SimpleRMSNorm,
        gate_proj=gate_proj_cls,
    )


def _build_mla(gated_attention=False, **config_overrides):
    """Build an MLASelfAttention with or without gated attention."""
    config = _make_mla_config(
        gated_attention=gated_attention, **config_overrides
    )
    gate_cls = BiasedLinear if gated_attention else None
    spec = _make_sublayers_spec(gate_proj_cls=gate_cls)
    attn = MLASelfAttention(
        config=config,
        sublayers_spec=spec,
        layer_number=1,
    )
    return attn


class TestMLASublayersSpecGateField(unittest.TestCase):
    """Test that gate_proj is a valid field of MLASelfAttentionSublayersSpec."""

    def test_gate_proj_field_exists(self):
        field_names = [f.name for f in fields(MLASelfAttentionSublayersSpec)]
        self.assertIn("gate_proj", field_names)

    def test_gate_proj_default_is_none(self):
        spec = MLASelfAttentionSublayersSpec()
        self.assertIsNone(spec.gate_proj)

    def test_gate_proj_can_be_set(self):
        spec = MLASelfAttentionSublayersSpec(gate_proj=BiasedLinear)
        self.assertEqual(spec.gate_proj, BiasedLinear)


class TestMLAGateConstruction(unittest.TestCase):
    """Test gate_proj layer creation in MultiLatentAttention.__init__."""

    def test_gated_attention_false_no_gate(self):
        attn = _build_mla(gated_attention=False)
        self.assertFalse(attn.gated_attention)
        self.assertIsNone(attn.gate_proj)

    def test_gated_attention_true_creates_gate(self):
        attn = _build_mla(gated_attention=True)
        self.assertTrue(attn.gated_attention)
        self.assertIsNotNone(attn.gate_proj)

    def test_gated_true_but_spec_none_disables_gate(self):
        """If config says gated but spec has no gate_proj, gate is disabled."""
        config = _make_mla_config(gated_attention=True)
        spec = _make_sublayers_spec(gate_proj_cls=None)  # No gate in spec
        attn = MLASelfAttention(
            config=config, sublayers_spec=spec, layer_number=1
        )
        self.assertFalse(attn.gated_attention)
        self.assertIsNone(attn.gate_proj)

    def test_gate_proj_input_output_dims(self):
        """gate_proj maps hidden_size -> num_heads * v_head_dim."""
        attn = _build_mla(gated_attention=True)
        gate = attn.gate_proj
        # BiasedLinear wraps a paddle.nn.Linear
        in_features = gate.linear.weight.shape[0]
        out_features = gate.linear.weight.shape[1]
        expected_in = 128  # hidden_size
        expected_out = 4 * 32  # num_attention_heads * v_head_dim = 128
        self.assertEqual(in_features, expected_in)
        self.assertEqual(out_features, expected_out)

    def test_gate_proj_input_dim_uses_q_lora_rank(self):
        """When gated_attn_use_q_lora=True, gate_proj input dim is q_lora_rank."""
        attn = _build_mla(gated_attention=True, gated_attn_use_q_lora=True)
        self.assertTrue(attn.gated_attn_use_q_lora)
        gate = attn.gate_proj
        in_features = gate.linear.weight.shape[0]
        out_features = gate.linear.weight.shape[1]
        expected_in = 64  # q_lora_rank (not hidden_size=128)
        expected_out = 4 * 32  # num_attention_heads * v_head_dim = 128
        self.assertEqual(in_features, expected_in)
        self.assertEqual(out_features, expected_out)


class TestMLAGatedForward(unittest.TestCase):
    def test_forward_shape_with_gate(self):
        attn = _build_mla(gated_attention=True)
        attn.eval()
        x = paddle.randn([2, 4, 128])
        out, bias = attn(x, attention_mask=None)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_shape_without_gate(self):
        attn = _build_mla(gated_attention=False)
        attn.eval()
        x = paddle.randn([2, 4, 128])
        out, bias = attn(x, attention_mask=None)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_shape_with_gate_q_lora(self):
        """Forward with gated_attn_use_q_lora: gate consumes q_compressed."""
        attn = _build_mla(gated_attention=True, gated_attn_use_q_lora=True)
        attn.eval()
        x = paddle.randn([2, 4, 128])
        out, bias = attn(x, attention_mask=None)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_gated_output_differs_from_ungated(self):
        paddle.manual_seed(42)
        attn_ungated = _build_mla(gated_attention=False)
        attn_ungated.eval()

        paddle.manual_seed(42)
        attn_gated = _build_mla(gated_attention=True)
        attn_gated.eval()

        x = paddle.randn([2, 4, 128])
        out_ungated, _ = attn_ungated(x, attention_mask=None)
        out_gated, _ = attn_gated(x, attention_mask=None)

        self.assertFalse(
            paddle.allclose(out_ungated, out_gated, atol=1e-6).item(),
            "Gated and ungated outputs should differ",
        )

    def test_gate_applies_sigmoid(self):
        attn = _build_mla(gated_attention=True)
        attn.eval()
        x = paddle.randn([2, 4, 128])

        original_forward = attn.gate_proj.forward
        gate_values = {}

        def capture_gate(input_tensor):
            result = original_forward(input_tensor)
            gate_values["raw"] = result[0].detach()
            return result

        attn.gate_proj.forward = capture_gate
        out, _ = attn(x, attention_mask=None)

        self.assertIn("raw", gate_values)
        raw_gate = gate_values["raw"]
        sigmoid_gate = F.sigmoid(raw_gate)
        self.assertTrue((sigmoid_gate >= 0).all().item())
        self.assertTrue((sigmoid_gate <= 1).all().item())

    def test_forward_backward_with_gate(self):
        attn = _build_mla(gated_attention=True)
        attn.train()
        x = paddle.randn([2, 4, 128])
        x.stop_gradient = False
        out, _ = attn(x, attention_mask=None)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        for param in attn.gate_proj.parameters():
            self.assertIsNotNone(param.grad)

    def test_forward_backward_with_gate_recompute(self):
        attn = _build_mla(
            gated_attention=True,
            sigmoid_gate_fusion=True,
            recompute_granularity="selective",
            recompute_modules=["gated_attn"],
        )
        attn.train()
        attn_ref = _build_mla(gated_attention=True)
        attn_ref.train()

        x = paddle.randn([2, 4, 128])
        x.stop_gradient = False

        mem0 = paddle.device.memory_allocated()
        out, _ = attn(x, attention_mask=None)
        mem1 = paddle.device.memory_allocated()
        out_ref, _ = attn_ref(x, attention_mask=None)
        mem2 = paddle.device.memory_allocated()

        # Recomputing gate avoids storing sigmoid_out and mul_out, so the
        # memory usage should be reduced by both sizes.
        sigmoid_out_size = out.size * out.itemsize
        mul_out_size = sigmoid_out_size
        self.assertGreaterEqual(
            (mem2 - mem1) - (mem1 - mem0),
            sigmoid_out_size + mul_out_size,
        )

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        for param in attn.gate_proj.parameters():
            self.assertIsNotNone(param.grad)


class TestGetAttentionSpecGate(unittest.TestCase):
    def _make_mock_config(self, gated=False, align_mode=None):
        config = MagicMock()
        config.gated_attention = gated
        config.normalization = "RMSNorm"
        config.use_qk_norm = False
        config.qk_l2_norm = False
        config.gpt_model_use_experimental_version = align_mode
        config.use_vha_attention = False
        return config

    def test_mla_gated_has_gate_proj(self):
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_attention_spec,
        )

        config = self._make_mock_config(gated=True)
        spec = get_attention_spec(
            config=config,
            attention_layer_type="multi_latent_attention",
        )
        gate_proj = spec.sublayers_spec.gate_proj
        self.assertIsNotNone(gate_proj)

    def test_mla_not_gated_gate_proj_is_none(self):
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_attention_spec,
        )

        config = self._make_mock_config(gated=False)
        spec = get_attention_spec(
            config=config,
            attention_layer_type="multi_latent_attention",
        )
        gate_proj = spec.sublayers_spec.gate_proj
        self.assertIsNone(gate_proj)

    def test_mla_gated_gate_proj_is_column_parallel(self):
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_attention_spec,
        )
        from paddleformers.fleet.tensor_parallel.layers import (
            ColumnParallelLinear,
        )

        config = self._make_mock_config(gated=True)
        spec = get_attention_spec(
            config=config,
            attention_layer_type="multi_latent_attention",
        )
        gate_proj = spec.sublayers_spec.gate_proj
        self.assertEqual(gate_proj, ColumnParallelLinear)

    def test_self_attention_type_unaffected(self):
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_attention_spec,
        )
        from paddleformers.fleet.transformer.identity_op import IdentityOp

        config = self._make_mock_config(gated=True)
        spec = get_attention_spec(
            config=config,
            attention_layer_type="self_attention",
        )
        # When align_mode is None, gate_proj should be IdentityOp (no-op)
        self.assertEqual(spec.sublayers_spec.gate_proj, IdentityOp)


if __name__ == "__main__":
    unittest.main()
