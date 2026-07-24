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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import unittest

import paddle

from paddleformers.fleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddleformers.fleet.transformer.mlp import MLP
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class TestParallelMLP(unittest.TestCase):
    transformer_config = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=12,
        intermediate_size=48,
        num_attention_heads=4,
        use_bias=True,
    )
    expected_num_weights = 1212

    def setUp(self):
        self.mlp = MLP(
            self.transformer_config,
            get_gpt_layer_local_spec(
                self.transformer_config
            ).sublayers_spec.mlp.sublayers_spec,
        )

    def test_constructor(self):
        assert isinstance(self.mlp, MLP)

        num_weights = sum([p.numel() for p in self.mlp.parameters()])
        assert num_weights == self.expected_num_weights

    def test_forward_backward(self):
        mlp = self.mlp
        # [sequence length, batch size, hidden size]
        hidden_states = paddle.ones((32, 12, mlp.config.hidden_size))
        hidden_states.stop_gradient = False

        # add 0.0 to make hidden_states non-leaf
        output, output_bias = mlp(hidden_states + 0.0)
        assert output.shape[0] == 32
        assert output.shape[1] == 12
        assert output.shape[2] == mlp.config.hidden_size
        assert output.dtype == paddle.float32
        assert output_bias.shape[0] == mlp.config.hidden_size

        paddle.autograd.backward((output, output_bias))
        assert hidden_states.grad is not None


class TestBiasFusedGatedMLP(TestParallelMLP):
    transformer_config = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=12,
        intermediate_size=48,
        num_attention_heads=4,
        bias_activation_fusion=True,
        gated_linear_unit=True,
        use_bias=True,
    )
    expected_num_weights = 1836


if __name__ == "__main__":
    unittest.main()
