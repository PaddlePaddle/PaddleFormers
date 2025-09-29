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

from paddleformers.nn.norm import LayerNorm, Norm, RMSNorm
from paddleformers.transformers import LlamaConfig


class TestNormLayers(unittest.TestCase):
    def test_layer_norm_initialization(self):
        config = LlamaConfig()
        # Test LayerNorm initialization
        config.hidden_size = 10
        layer_norm = Norm.create(config, norm_type="layer_norm")
        assert isinstance(layer_norm, LayerNorm)

        input = paddle.randn([1, 10, config.hidden_size])
        layer_norm(input)

    def test_layer_norm_sequence_parallel(self):
        # Test LayerNorm with sequence parallel
        config = LlamaConfig()
        config.sequence_parallel = True
        layer_norm = Norm.create(config, norm_type="layer_norm")
        assert isinstance(layer_norm, LayerNorm)

        input = paddle.randn([1, 10, config.hidden_size])
        layer_norm(input)

    def test_rms_norm_initialization(self):
        # Test RMSNorm initialization
        config = LlamaConfig()
        rms_norm = Norm.create(config)
        assert isinstance(rms_norm, RMSNorm)

        input = paddle.randn([1, 10, config.hidden_size])
        rms_norm(input)

    def test_rms_norm_sequence_parallel(self):
        # Test RMSNorm with sequence parallel
        config = LlamaConfig()
        config.sequence_parallel = True
        rms_norm = Norm.create(config)
        input = paddle.randn([1, 10, config.hidden_size])
        rms_norm(input)
        assert isinstance(rms_norm, RMSNorm)


if __name__ == "__main__":
    unittest.main()
