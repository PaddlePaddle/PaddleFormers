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

from paddleformers.fleet.transformer.paddle_norm import RMSNorm, WrappedPaddleNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class TestFleetLayer(unittest.TestCase):
    def setUp(self):
        self.transformer_config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            normalization="RMSNorm",
        )
        self.norm = WrappedPaddleNorm(
            config=self.transformer_config,
            hidden_size=self.transformer_config.hidden_size,
        )

    def test_fleet_layer(self):
        assert isinstance(self.norm, RMSNorm)
        x = paddle.uniform(shape=[1, self.transformer_config.hidden_size])
        self.norm(x)


if __name__ == "__main__":
    unittest.main()
