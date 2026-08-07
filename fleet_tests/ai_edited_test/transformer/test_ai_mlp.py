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
import paddle.nn.functional as F

from paddleformers.fleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddleformers.fleet.transformer.mlp import MLP, MLPSublayersSpec
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import init_method_normal, scaled_init_method_normal


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "use_bias": True,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "gated_linear_unit": False,
        "bias_activation_fusion": False,
        "activation_func_clamp_value": None,
        "glu_linear_offset": 0.0,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestMLPConstructor(unittest.TestCase):
    """Tests for MLP constructor."""

    def test_basic_construction(self):
        config = _make_config()
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        self.assertIsInstance(mlp, MLP)

    def test_construction_with_expert_raises(self):
        config = _make_config()
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        with self.assertRaises(ValueError):
            MLP(config, spec, is_expert=True, intermediate_size=None)

    def test_construction_with_expert_and_intermediate_size(self):
        config = _make_config()
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        # Should not raise when intermediate_size is provided
        mlp = MLP(config, spec, is_expert=True, intermediate_size=128)
        self.assertIsInstance(mlp, MLP)

    def test_custom_input_size(self):
        config = _make_config()
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec, input_size=32)
        self.assertEqual(mlp.input_size, 32)

    def test_custom_hidden_size(self):
        config = _make_config()
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec, hidden_size=128)
        self.assertEqual(mlp.hidden_size, 128)

    def test_gated_linear_unit_construction(self):
        config = _make_config(gated_linear_unit=True)
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        self.assertIsInstance(mlp, MLP)

    def test_no_bias_construction(self):
        config = _make_config(use_bias=False)
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        self.assertIsInstance(mlp, MLP)


class TestMLPForward(unittest.TestCase):
    """Tests for MLP forward pass."""

    def setUp(self):
        self.config = _make_config()
        spec = get_gpt_layer_local_spec(
            self.config
        ).sublayers_spec.mlp.sublayers_spec
        self.mlp = MLP(self.config, spec)
        self.mlp.eval()

    def test_forward_shape(self):
        x = paddle.randn([4, 8, 64])
        self.mlp.eval()
        out, bias = self.mlp(x)
        self.assertEqual(out.shape, [4, 8, 64])

    def test_forward_bias_shape(self):
        x = paddle.randn([4, 8, 64])
        self.mlp.eval()
        out, bias = self.mlp(x)
        self.assertEqual(bias.shape, [64])

    def test_forward_dtype(self):
        x = paddle.randn([4, 8, 64]).cast(paddle.float32)
        self.mlp.eval()
        out, bias = self.mlp(x)
        self.assertEqual(out.dtype, paddle.float32)

    def test_backward(self):
        x = paddle.randn([4, 8, 64])
        x.stop_gradient = False
        self.mlp.train()
        out, bias = self.mlp(x + 0.0)
        paddle.autograd.backward((out, bias))
        self.assertIsNotNone(x.grad)


class TestMLPGatedLinearUnit(unittest.TestCase):
    """Tests for MLP with gated linear unit."""

    def setUp(self):
        self.config = _make_config(gated_linear_unit=True)
        spec = get_gpt_layer_local_spec(
            self.config
        ).sublayers_spec.mlp.sublayers_spec
        self.mlp = MLP(self.config, spec)
        self.mlp.eval()

    def test_forward_shape(self):
        x = paddle.randn([4, 8, 64])
        out, bias = self.mlp(x)
        self.assertEqual(out.shape, [4, 8, 64])

    def test_backward(self):
        x = paddle.randn([4, 8, 64])
        x.stop_gradient = False
        self.mlp.train()
        out, bias = self.mlp(x + 0.0)
        paddle.autograd.backward((out, bias))
        self.assertIsNotNone(x.grad)


class TestMLPBiasActivationFusion(unittest.TestCase):
    """Tests for MLP with bias activation fusion."""

    def test_silu_fusion(self):
        config = _make_config(
            gated_linear_unit=True,
            bias_activation_fusion=True,
            hidden_act=F.silu,
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        mlp.eval()
        x = paddle.randn([4, 8, 64])
        out, bias = mlp(x)
        self.assertEqual(out.shape, [4, 8, 64])

    def test_gelu_fusion(self):
        config = _make_config(
            gated_linear_unit=True,
            bias_activation_fusion=True,
            hidden_act=F.gelu,
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        mlp.eval()
        x = paddle.randn([4, 8, 64])
        out, bias = mlp(x)
        self.assertEqual(out.shape, [4, 8, 64])

    def test_non_gated_gelu_fusion(self):
        config = _make_config(
            gated_linear_unit=False,
            bias_activation_fusion=True,
            use_bias=True,
            hidden_act=F.gelu,
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        mlp.eval()
        x = paddle.randn([4, 8, 64])
        out, bias = mlp(x)
        self.assertEqual(out.shape, [4, 8, 64])

    def test_unsupported_fusion_raises(self):
        config = _make_config(
            gated_linear_unit=False,
            bias_activation_fusion=True,
            hidden_act=F.relu,
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        x = paddle.randn([4, 8, 64])
        with self.assertRaises(ValueError):
            mlp(x)


class TestMLPPerTokenScale(unittest.TestCase):
    """Tests for MLP forward with per_token_scale."""

    def test_per_token_scale_no_fusion(self):
        config = _make_config(
            gated_linear_unit=False, bias_activation_fusion=False
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        mlp.eval()
        x = paddle.randn([4, 8, 64])
        scale = paddle.ones([4, 8])
        out, bias = mlp(x, per_token_scale=scale)
        self.assertEqual(out.shape, [4, 8, 64])

    def test_per_token_scale_gated(self):
        config = _make_config(
            gated_linear_unit=True, bias_activation_fusion=False
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        mlp.eval()
        x = paddle.randn([4, 8, 64])
        scale = paddle.ones([4, 8])
        out, bias = mlp(x, per_token_scale=scale)
        self.assertEqual(out.shape, [4, 8, 64])

    def test_per_token_scale_with_clamp(self):
        config = _make_config(
            gated_linear_unit=True,
            bias_activation_fusion=False,
            activation_func_clamp_value=1.0,
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        mlp.eval()
        x = paddle.randn([4, 8, 64])
        scale = paddle.ones([4, 8])
        out, bias = mlp(x, per_token_scale=scale)
        self.assertEqual(out.shape, [4, 8, 64])


class TestMLPGLUOffset(unittest.TestCase):
    """Tests for MLP with GLU linear offset."""

    def test_gated_with_offset(self):
        config = _make_config(
            gated_linear_unit=True,
            bias_activation_fusion=False,
            glu_linear_offset=0.1,
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        mlp = MLP(config, spec)
        mlp.eval()
        x = paddle.randn([4, 8, 64])
        out, bias = mlp(x)
        self.assertEqual(out.shape, [4, 8, 64])


class TestMLPSublayersSpec(unittest.TestCase):
    """Tests for MLPSublayersSpec dataclass."""

    def test_defaults(self):
        spec = MLPSublayersSpec()
        self.assertIsNone(spec.up_gate_proj)
        self.assertIsNone(spec.hidden_act)
        self.assertIsNone(spec.down_proj)

    def test_custom_values(self):
        spec = MLPSublayersSpec(
            up_gate_proj=paddle.nn.Linear,
            hidden_act=F.gelu,
            down_proj=paddle.nn.Linear,
        )
        self.assertEqual(spec.up_gate_proj, paddle.nn.Linear)
        self.assertEqual(spec.hidden_act, F.gelu)
        self.assertEqual(spec.down_proj, paddle.nn.Linear)


if __name__ == "__main__":
    unittest.main()
