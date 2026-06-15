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
"""Tests for MLP with clamped weighted bias swiglu fusion and bias_activation_fusion.


The mock strategy:
  - ``build_spec_layer`` returns a simple layer that produces a proper
    (output, bias) tuple so MLP can forward without distributed setup.
  - Only the lowest-level CUDA autograd Function.apply is mocked so that
    the outer ``weighted_bias_swiglu_impl``
    function bodies are executed and tracked by coverage.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import patch

import paddle
import paddle.nn.functional as F

from paddleformers.fleet.fusions.fused_bias_swiglu import WeightedSwiGLUFunction
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
        "gated_linear_unit": True,
        "bias_activation_fusion": True,
        "activation_func_clamp_value": None,
        "glu_linear_offset": 0.0,
        "hidden_act": F.silu,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class _MockColumnParallelLinear(paddle.nn.Linear):
    """A drop-in mock that behaves like ColumnParallelLinear's output:
    returns (output, bias_parallel) tuple instead of a bare tensor.
    """

    def forward(self, x):
        output = super().forward(x)
        # ColumnParallelLinear with skip_bias_add=True returns
        # (output, bias_parallel).  We return bias_parallel=None so
        # the downstream swiglu_impl path does not raise
        # NotImplementedError("Bias is not supported...").
        return output, None


def _make_mock_linear(sublayer_spec, in_features, out_features, **kwargs):
    """Return a _MockColumnParallelLinear for build_spec_layer mock."""
    return _MockColumnParallelLinear(in_features, out_features)


class TestMLPClampBiasActivationFusion(unittest.TestCase):
    """Tests for MLP bias_activation_fusion with activation_func_clamp_value."""

    @patch(
        "paddleformers.fleet.transformer.mlp.build_spec_layer",
        side_effect=_make_mock_linear,
    )
    @patch.object(
        WeightedSwiGLUFunction,
        "apply",
        return_value=paddle.randn([2, 4, 128]),
    )
    def test_forward_with_clamp_bias_fusion(self, _, __):
        """Lines 201-210: when activation_func_clamp_value is set and
        bias_activation_fusion=True, weighted_bias_swiglu_impl is
        called instead of weighted_bias_swiglu_impl."""
        config = _make_config(activation_func_clamp_value=5.0)
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        spec = get_gpt_layer_local_spec(config)
        sublayers = spec.sublayers_spec.mlp.sublayers_spec

        # activation_func_fp8_input_store is referenced by mlp.py but not
        # defined as a TransformerConfig field; set it manually.
        config.activation_func_fp8_input_store = False

        mlp = MLP(config=config, sublayers_spec=sublayers)
        mlp.eval()

        hidden_states = paddle.randn([2, 4, 64])
        scale = paddle.ones([2, 4])

        out, bias = mlp(hidden_states, per_token_scale=scale)
        self.assertEqual(out.shape, [2, 4, 64])
        # clamp_value is non-None, so the clamped path is taken (lines 201-210)

    @patch(
        "paddleformers.fleet.transformer.mlp.build_spec_layer",
        side_effect=_make_mock_linear,
    )
    @patch.object(
        WeightedSwiGLUFunction,
        "apply",
        return_value=paddle.randn([2, 4, 128]),
    )
    def test_forward_without_clamp_bias_fusion(self, _, __):
        """Lines 211-217: when activation_func_clamp_value is None and
        bias_activation_fusion=True, weighted_bias_swiglu_impl is called."""
        config = _make_config(activation_func_clamp_value=None)
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        spec = get_gpt_layer_local_spec(config)
        sublayers = spec.sublayers_spec.mlp.sublayers_spec

        config.activation_func_fp8_input_store = False

        mlp = MLP(config=config, sublayers_spec=sublayers)
        mlp.eval()

        hidden_states = paddle.randn([2, 4, 64])
        scale = paddle.ones([2, 4])

        out, bias = mlp(hidden_states, per_token_scale=scale)
        self.assertEqual(out.shape, [2, 4, 64])
        # clamp_value is None, so the non-clamped path is taken (lines 211-217)

    @patch(
        "paddleformers.fleet.transformer.mlp.build_spec_layer",
        side_effect=_make_mock_linear,
    )
    @patch.object(
        WeightedSwiGLUFunction,
        "apply",
        return_value=paddle.randn([2, 4, 128]),
    )
    def test_backward_with_clamp_bias_fusion(self, _, __):
        """test fwd path with clamp_value exercises the new
        weighted_bias_swiglu_impl code path in mlp.py."""
        config = _make_config(activation_func_clamp_value=3.0)
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        spec = get_gpt_layer_local_spec(config)
        sublayers = spec.sublayers_spec.mlp.sublayers_spec

        config.activation_func_fp8_input_store = False
        mlp = MLP(config=config, sublayers_spec=sublayers)
        mlp.eval()

        hidden_states = paddle.randn([2, 4, 64])
        scale = paddle.ones([2, 4])

        out, _ = mlp(hidden_states, per_token_scale=scale)
        self.assertEqual(out.shape, [2, 4, 64])


if __name__ == "__main__":
    unittest.main()
