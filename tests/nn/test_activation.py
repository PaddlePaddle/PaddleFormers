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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleformers.nn.activation import ACT2CLS, ACT2FN, FusedSwiglu, Swiglu


class TestActivationFunctions(unittest.TestCase):
    def test_swiglu(self):
        swiglu = Swiglu()
        # Test with separate gate and x
        output = swiglu(self.test_gate, self.test_input)

        # Verify output shape
        self.assertEqual(output.shape, [self.batch_size, self.feature_size])

        # Verify computation (silu(gate) * x)
        expected = F.silu(self.test_gate) * self.test_input
        np.testing.assert_allclose(output.numpy(), expected.numpy(), rtol=1e-5)

    def test_fused_swiglu(self):
        fused_swiglu_layer = FusedSwiglu()
        # Test with concatenated input
        # Assuming first half is gate, second half is x
        concat_input = paddle.concat([self.test_gate, self.test_input], axis=-1)
        output = fused_swiglu_layer(concat_input)

        # Verify output shape
        self.assertEqual(output.shape, [self.batch_size, self.feature_size])

        # Verify computation matches regular swiglu
        expected = F.silu(self.test_gate) * self.test_input
        np.testing.assert_allclose(output.numpy(), expected.numpy(), rtol=1e-5)

    def test_act2fn_instantiation(self):
        # Test all activation functions can be instantiated
        for act_name in ACT2CLS.keys():
            activation = ACT2FN[act_name]
            self.assertTrue(isinstance(activation, nn.Layer))

            # Test forward pass for each activation
            if act_name == "fused_swiglu":
                # Special case for fused_swiglu which needs concatenated input
                concat_input = paddle.concat([self.test_gate, self.test_input], axis=-1)
                output = activation(concat_input)
                self.assertEqual(output.shape, [self.batch_size, self.feature_size])
            elif act_name == "swish":
                # Swish (Swiglu) needs separate gate and x
                output = activation(self.test_gate, self.test_input)
                self.assertEqual(output.shape, [self.batch_size, self.feature_size])
            else:
                # Standard activations
                output = activation(self.test_input)
                self.assertEqual(output.shape, [self.batch_size, self.feature_size])


if __name__ == "__main__":
    unittest.main()
