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
import paddle.nn as nn

from paddleformers.nn.activation import ACT2CLS, ACT2FN


class TestActivationFunctions(unittest.TestCase):
    def test_act2fn_instantiation(self):
        # Test all activation functions can be instantiated
        batch_size = 1
        feature_size = 32
        for act_name in ACT2CLS.keys():
            test_input = paddle.randn([batch_size, feature_size])
            activation = ACT2FN[act_name]
            self.assertTrue(isinstance(activation, nn.Layer))

            # Test forward pass for each activation
            output = activation(test_input)
            self.assertEqual(output.shape, [batch_size, feature_size])


if __name__ == "__main__":
    unittest.main()
