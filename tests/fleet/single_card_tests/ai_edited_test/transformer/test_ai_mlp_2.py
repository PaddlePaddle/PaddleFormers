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

import paddle
import paddle.nn.functional as F

from paddleformers.fleet.transformer.mlp import MLP
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "use_bias": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "gated_linear_unit": True,
        "bias_activation_fusion": False,
        "activation_func_clamp_value": None,
        "glu_linear_offset": 0.0,
        "hidden_act": F.silu,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_mlp_spec(config):
    from paddleformers.fleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
    )

    spec = get_gpt_layer_local_spec(config)
    return spec.sublayers_spec.mlp.sublayers_spec


class TestMLPWithSwigluPath(unittest.TestCase):
    """Tests for MLP with swiglu activation path."""

    def test_forward_with_gated_linear_unit(self):
        """Test MLP forward with gated_linear_unit=True."""
        config = _make_config(gated_linear_unit=True)
        spec = _make_mlp_spec(config)
        mlp = MLP(config=config, sublayers_spec=spec)

        hidden_states = paddle.randn([2, 4, 64])
        output, bias = mlp(hidden_states)
        self.assertEqual(output.shape, [2, 4, 64])

    def test_forward_without_gated_linear_unit(self):
        """Test MLP forward with gated_linear_unit=False."""
        config = _make_config(
            gated_linear_unit=False,
            hidden_act=F.gelu,
            use_bias=True,
            bias_activation_fusion=False,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(config=config, sublayers_spec=spec)

        hidden_states = paddle.randn([2, 4, 64])
        output, bias = mlp(hidden_states)
        self.assertEqual(output.shape, [2, 4, 64])


class TestMLPWithExperimentalVersion(unittest.TestCase):
    """Tests for MLP with experimental version swiglu path."""

    def test_forward_paddle_swiglu(self):
        """Test MLP forward with gpt_model_use_experimental_version."""
        config = _make_config(
            gated_linear_unit=True,
            gpt_model_use_experimental_version=True,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(config=config, sublayers_spec=spec)

        hidden_states = paddle.randn([2, 4, 64])
        output, bias = mlp(hidden_states)
        self.assertEqual(output.shape, [2, 4, 64])


class TestMLPWithPerTokenScale(unittest.TestCase):
    """Tests for MLP with per_token_scale."""

    def test_forward_with_per_token_scale_no_fusion(self):
        """Test MLP forward with per_token_scale and no bias fusion."""
        config = _make_config(
            gated_linear_unit=True,
            use_bias=False,
            bias_activation_fusion=False,
        )
        spec = _make_mlp_spec(config)
        mlp = MLP(config=config, sublayers_spec=spec)

        hidden_states = paddle.randn([2, 4, 64])
        per_token_scale = paddle.randn([2, 4])
        output, bias = mlp(hidden_states, per_token_scale=per_token_scale)
        self.assertEqual(output.shape, [2, 4, 64])


class TestMLPWithCustomInputSize(unittest.TestCase):
    """Tests for MLP with custom input_size."""

    def test_forward_with_custom_input_size(self):
        """Test MLP forward with custom input_size."""
        config = _make_config(gated_linear_unit=True)
        spec = _make_mlp_spec(config)
        mlp = MLP(
            config=config,
            sublayers_spec=spec,
            input_size=32,
        )

        hidden_states = paddle.randn([2, 4, 32])
        output, bias = mlp(hidden_states)
        self.assertEqual(output.shape, [2, 4, 64])


class TestMLPWithCustomHiddenSize(unittest.TestCase):
    """Tests for MLP with custom hidden_size output."""

    def test_forward_with_custom_hidden_size(self):
        """Test MLP forward with custom hidden_size."""
        config = _make_config(gated_linear_unit=True)
        spec = _make_mlp_spec(config)
        mlp = MLP(
            config=config,
            sublayers_spec=spec,
            hidden_size=32,
        )

        hidden_states = paddle.randn([2, 4, 64])
        output, bias = mlp(hidden_states)
        self.assertEqual(output.shape, [2, 4, 32])


class TestMLPBackwardDW(unittest.TestCase):
    """Tests for MLP backward_dw."""

    def test_backward_dw_method_exists(self):
        """Test MLP has backward_dw method."""
        config = _make_config(gated_linear_unit=True)
        spec = _make_mlp_spec(config)
        mlp = MLP(config=config, sublayers_spec=spec)
        self.assertTrue(hasattr(mlp, "backward_dw"))
        self.assertTrue(callable(mlp.backward_dw))


class TestMLPConstructionValidation(unittest.TestCase):
    """Tests for MLP construction validation."""

    def test_expert_without_intermediate_size_raises(self):
        """Test that constructing expert MLP without intermediate_size raises."""
        config = _make_config(gated_linear_unit=True)
        config.intermediate_size = None
        spec = _make_mlp_spec(config)

        with self.assertRaises(ValueError):
            MLP(
                config=config,
                sublayers_spec=spec,
                is_expert=True,
            )


if __name__ == "__main__":
    unittest.main()
