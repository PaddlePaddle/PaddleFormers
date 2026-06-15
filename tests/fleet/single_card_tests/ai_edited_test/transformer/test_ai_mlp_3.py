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

import paddle
import paddle.nn.functional as F

from paddleformers.fleet.transformer.mlp import MLP
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
        "hidden_act": F.gelu,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_mlp_spec(config):
    from paddleformers.fleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    spec = get_gpt_layer_local_spec(config)
    return spec.sublayers_spec.mlp.sublayers_spec


class TestMLPBiasActivationFusion(unittest.TestCase):
    """Tests for MLP with bias_activation_fusion."""

    def test_forward_with_bias_gelu_fusion(self):
        """Test MLP forward with bias+gelu fusion."""
        config = _make_config(
            gated_linear_unit=False,
            hidden_act=F.gelu,
            use_bias=True,
            bias_activation_fusion=True,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(config=config, sublayers_spec=spec)

        hidden_states = paddle.randn([2, 4, 64])
        output, bias = mlp(hidden_states)
        self.assertEqual(output.shape, [2, 4, 64])

    def test_forward_with_bias_swiglu_fusion(self):
        """Test MLP forward with bias+swiglu fusion."""
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=True,
            bias_activation_fusion=True,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(config=config, sublayers_spec=spec)

        hidden_states = paddle.randn([2, 4, 64])
        output, bias = mlp(hidden_states)
        self.assertEqual(output.shape, [2, 4, 64])


class TestMLPWithClampValue(unittest.TestCase):
    """Tests for MLP with activation_func_clamp_value."""

    def test_forward_with_clamp_value(self):
        """Test MLP forward with activation_func_clamp_value set."""
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=False,
            bias_activation_fusion=False,
            activation_func_clamp_value=5.0,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(config=config, sublayers_spec=spec)

        hidden_states = paddle.randn([2, 4, 64])
        output, bias = mlp(hidden_states)
        self.assertEqual(output.shape, [2, 4, 64])


class TestMLPWithGLULinearOffset(unittest.TestCase):
    """Tests for MLP with glu_linear_offset."""

    def test_forward_with_glu_linear_offset(self):
        """Test MLP forward with non-zero glu_linear_offset."""
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=False,
            bias_activation_fusion=False,
            glu_linear_offset=0.5,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(config=config, sublayers_spec=spec)

        hidden_states = paddle.randn([2, 4, 64])
        output, bias = mlp(hidden_states)
        self.assertEqual(output.shape, [2, 4, 64])


class TestMLPExpertWithPerTokenScaleAndBias(unittest.TestCase):
    """Tests for MLP expert with bias."""

    def test_forward_expert_with_bias(self):
        """Test MLP expert construction with bias."""
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=True,
            bias_activation_fusion=True,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(
            config=config,
            sublayers_spec=spec,
            is_expert=True,
            intermediate_size=128,
        )
        # Verify the MLP has expected sublayers for expert mode
        self.assertTrue(mlp.config.use_bias)
        self.assertTrue(hasattr(mlp, "up_gate_proj"))
        self.assertTrue(hasattr(mlp, "down_proj"))

    def test_forward_expert_without_bias(self):
        """Test MLP expert construction without bias."""
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=False,
            bias_activation_fusion=False,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(
            config=config,
            sublayers_spec=spec,
            is_expert=True,
            intermediate_size=128,
        )
        self.assertFalse(mlp.config.use_bias)

    def test_expert_backward_dw(self):
        """Test MLP expert has backward_dw method."""
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=True,
            bias_activation_fusion=True,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(
            config=config,
            sublayers_spec=spec,
            is_expert=True,
            intermediate_size=128,
        )
        self.assertTrue(hasattr(mlp, "backward_dw"))


if __name__ == "__main__":
    unittest.main()
