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

from paddleformers.fleet.transformer.layer import FleetLayer
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class DummyLayer(FleetLayer):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.linear = paddle.nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)


class TestFleetLayer(unittest.TestCase):
    def setUp(self):
        transformer_config = TransformerConfig(
            num_hidden_layers=2, hidden_size=12, num_attention_heads=4
        )
        self.fleet_layer = DummyLayer(config=transformer_config)

    def test_fleet_layer(self):
        fleet_layer = self.fleet_layer
        assert fleet_layer
        assert fleet_layer.config.hidden_size == 12
        assert fleet_layer.linear.weight.dtype == paddle.float32

        x = paddle.ones((2, 2)).cuda()
        assert fleet_layer(x).dtype == paddle.float32


if __name__ == "__main__":
    unittest.main()
