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

from paddleformers.nn.mlp import MLP
from paddleformers.transformers import LlamaConfig


class TestMLP(unittest.TestCase):
    def setUp(self):
        # Create a mock config with default values
        self.config = LlamaConfig()
        self.config.hidden_size = 768
        self.config.intermediate_size = 3072
        self.config.tensor_parallel_degree = 1
        self.config.mlp_bias = False
        self.config.fuse_swiglu = False
        self.config.hidden_act = "silu"
        self.config.fuse_linear = False

        # Default test input
        self.batch_size = 2
        self.seq_len = 10
        self.test_input = paddle.randn([self.batch_size, self.seq_len, self.config.hidden_size])

    def test_initialization_default(self):
        # Test default initialization
        mlp = MLP(self.config)

        # Check basic attributes
        self.assertFalse(mlp.fuse_swiglu)
        self.assertEqual(mlp.act_type, "silu")

        # Check layer existence
        self.assertTrue(hasattr(mlp, "up_proj"))
        self.assertTrue(hasattr(mlp, "gate_proj"))
        self.assertTrue(hasattr(mlp, "down_proj"))

    def test_initialization_fuse_ffn(self):
        # Test initialization with custom sizes
        custom_hidden = self.config.hidden_size
        custom_intermediate = self.config.intermediate_size
        mlp = MLP(self.config, hidden_size=custom_hidden, intermediate_size=custom_intermediate, fuse_up_gate=True)

        self.assertEqual(mlp.hidden_size, custom_hidden)
        self.assertEqual(mlp.intermediate_size, custom_intermediate)
        self.assertEqual(mlp.up_gate_proj.weight.shape, [custom_hidden, custom_intermediate * 2])
        self.assertEqual(mlp.down_proj.weight.shape, [custom_intermediate, custom_hidden])

    def test_initialization_non_fuse_ffn(self):
        # Test initialization with custom sizes
        custom_hidden = self.config.hidden_size
        custom_intermediate = self.config.intermediate_size
        mlp = MLP(self.config, hidden_size=custom_hidden, intermediate_size=custom_intermediate, fuse_up_gate=False)

        self.assertEqual(mlp.hidden_size, custom_hidden)
        self.assertEqual(mlp.intermediate_size, custom_intermediate)
        self.assertEqual(mlp.up_proj.weight.shape, [custom_hidden, custom_intermediate])
        self.assertEqual(mlp.gate_proj.weight.shape, [custom_hidden, custom_intermediate])
        self.assertEqual(mlp.down_proj.weight.shape, [custom_intermediate, custom_hidden])

        mlp(self.test_input)

    def test_custom_proj_names(self):
        # Test initialization with custom projection names
        custom_names = {"gate_up_proj_name": "custom_gate_up", "down_proj_name": "custom_down"}
        mlp = MLP(self.config, **custom_names, fuse_up_gate=True)
        mlp(self.test_input)

        self.assertTrue(hasattr(mlp, "custom_gate_up"))
        self.assertTrue(hasattr(mlp, "custom_down"))


if __name__ == "__main__":
    unittest.main()
