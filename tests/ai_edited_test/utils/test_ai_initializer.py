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

import math
import unittest

import numpy as np
import paddle
import paddle.nn as nn

from paddleformers.utils.initializer import (
    _calculate_correct_fan,
    _calculate_fan_in_and_fan_out,
    _calculate_gain,
    _no_grad_fill_,
    _no_grad_normal_,
    _no_grad_uniform_,
    bias_init_with_prob,
    constant_,
    conv_init_,
    kaiming_normal_,
    kaiming_uniform_,
    linear_init_,
    normal_,
    ones_,
    reset_initialized_parameter,
    uniform_,
    vector_,
    xavier_normal_,
    xavier_uniform_,
    zeros_,
)


class TestNoGradUniform(unittest.TestCase):
    def test_values_in_range(self):
        t = paddle.zeros([100])
        result = _no_grad_uniform_(t, -1.0, 1.0)
        self.assertTrue(paddle.all(result >= -1.0).item())
        self.assertTrue(paddle.all(result <= 1.0).item())

    def test_returns_tensor(self):
        t = paddle.zeros([10])
        result = _no_grad_uniform_(t, 0.0, 1.0)
        self.assertIsInstance(result, paddle.Tensor)


class TestNoGradNormal(unittest.TestCase):
    def test_returns_tensor(self):
        t = paddle.zeros([1000])
        result = _no_grad_normal_(t, mean=0.0, std=1.0)
        self.assertIsInstance(result, paddle.Tensor)

    def test_mean_close_to_target(self):
        t = paddle.zeros([10000])
        _no_grad_normal_(t, mean=5.0, std=1.0)
        self.assertAlmostEqual(t.mean().item(), 5.0, delta=0.2)


class TestNoGradFill(unittest.TestCase):
    def test_fill_value(self):
        t = paddle.zeros([10])
        result = _no_grad_fill_(t, 3.14)
        self.assertTrue(paddle.all(paddle.abs(result - 3.14) < 1e-6).item())


class TestUniform(unittest.TestCase):
    def test_delegates(self):
        t = paddle.zeros([100])
        result = uniform_(t, -2.0, 2.0)
        self.assertTrue(paddle.all(result >= -2.0).item())
        self.assertTrue(paddle.all(result <= 2.0).item())


class TestNormal(unittest.TestCase):
    def test_delegates(self):
        t = paddle.zeros([1000])
        result = normal_(t, mean=0.0, std=1.0)
        self.assertIsInstance(result, paddle.Tensor)


class TestConstant(unittest.TestCase):
    def test_fill_constant(self):
        t = paddle.zeros([5])
        result = constant_(t, 42.0)
        self.assertTrue(paddle.all(result == 42.0).item())


class TestOnes(unittest.TestCase):
    def test_fill_ones(self):
        t = paddle.zeros([5])
        result = ones_(t)
        self.assertTrue(paddle.all(result == 1.0).item())


class TestZeros(unittest.TestCase):
    def test_fill_zeros(self):
        t = paddle.ones([5])
        result = zeros_(t)
        self.assertTrue(paddle.all(result == 0.0).item())


class TestVector(unittest.TestCase):
    def test_set_vector(self):
        t = paddle.zeros([4])
        result = vector_(t, [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(result[0].item(), 1.0)
        self.assertAlmostEqual(result[3].item(), 4.0)


class TestCalculateFanInAndFanOut(unittest.TestCase):
    def test_2d_tensor(self):
        t = paddle.randn([64, 128])
        fan_in, fan_out = _calculate_fan_in_and_fan_out(t)
        self.assertEqual(fan_in, 128)
        self.assertEqual(fan_out, 64)

    def test_2d_tensor_reverse(self):
        t = paddle.randn([64, 128])
        fan_in, fan_out = _calculate_fan_in_and_fan_out(t, reverse=True)
        self.assertEqual(fan_in, 64)
        self.assertEqual(fan_out, 128)

    def test_4d_tensor(self):
        t = paddle.randn([64, 128, 3, 3])
        fan_in, fan_out = _calculate_fan_in_and_fan_out(t)
        self.assertEqual(fan_in, 128 * 3 * 3)
        self.assertEqual(fan_out, 64 * 3 * 3)

    def test_1d_tensor_raises(self):
        t = paddle.randn([10])
        with self.assertRaises(ValueError):
            _calculate_fan_in_and_fan_out(t)


class TestCalculateCorrectFan(unittest.TestCase):
    def test_fan_in(self):
        t = paddle.randn([64, 128])
        fan = _calculate_correct_fan(t, "fan_in")
        self.assertEqual(fan, 128)

    def test_fan_out(self):
        t = paddle.randn([64, 128])
        fan = _calculate_correct_fan(t, "fan_out")
        self.assertEqual(fan, 64)

    def test_invalid_mode_raises(self):
        t = paddle.randn([64, 128])
        with self.assertRaises(ValueError):
            _calculate_correct_fan(t, "invalid_mode")


class TestCalculateGain(unittest.TestCase):
    def test_linear(self):
        self.assertEqual(_calculate_gain("linear"), 1)

    def test_conv1d(self):
        self.assertEqual(_calculate_gain("conv1d"), 1)

    def test_sigmoid(self):
        self.assertEqual(_calculate_gain("sigmoid"), 1)

    def test_tanh(self):
        self.assertAlmostEqual(_calculate_gain("tanh"), 5.0 / 3)

    def test_relu(self):
        self.assertAlmostEqual(_calculate_gain("relu"), math.sqrt(2.0))

    def test_leaky_relu_default(self):
        expected = math.sqrt(2.0 / (1 + 0.01**2))
        self.assertAlmostEqual(_calculate_gain("leaky_relu"), expected)

    def test_leaky_relu_custom_param(self):
        expected = math.sqrt(2.0 / (1 + 0.2**2))
        self.assertAlmostEqual(_calculate_gain("leaky_relu", 0.2), expected)

    def test_selu(self):
        self.assertAlmostEqual(_calculate_gain("selu"), 3.0 / 4)

    def test_unsupported_raises(self):
        with self.assertRaises(ValueError):
            _calculate_gain("unsupported")


class TestXavierUniform(unittest.TestCase):
    def test_output_in_range(self):
        t = paddle.zeros([64, 128])
        result = xavier_uniform_(t)
        self.assertIsInstance(result, paddle.Tensor)
        # Values should be finite
        self.assertTrue(paddle.all(paddle.isfinite(result)).item())


class TestXavierNormal(unittest.TestCase):
    def test_output_is_finite(self):
        t = paddle.zeros([64, 128])
        result = xavier_normal_(t)
        self.assertIsInstance(result, paddle.Tensor)
        self.assertTrue(paddle.all(paddle.isfinite(result)).item())


class TestKaimingUniform(unittest.TestCase):
    def test_output_in_range(self):
        t = paddle.zeros([64, 128])
        result = kaiming_uniform_(t)
        self.assertIsInstance(result, paddle.Tensor)
        self.assertTrue(paddle.all(paddle.isfinite(result)).item())


class TestKaimingNormal(unittest.TestCase):
    def test_output_is_finite(self):
        t = paddle.zeros([64, 128])
        result = kaiming_normal_(t)
        self.assertIsInstance(result, paddle.Tensor)
        self.assertTrue(paddle.all(paddle.isfinite(result)).item())


class TestLinearInit(unittest.TestCase):
    def test_initializes_weight_and_bias(self):
        linear = nn.Linear(64, 32)
        linear_init_(linear)
        # Weight should not be all zeros
        self.assertFalse(paddle.all(linear.weight == 0).item())
        # Bias should not be all zeros
        if linear.bias is not None:
            self.assertFalse(paddle.all(linear.bias == 0).item())


class TestConvInit(unittest.TestCase):
    def test_initializes_weight(self):
        conv = nn.Conv2D(3, 16, kernel_size=3, padding=1)
        conv_init_(conv)
        self.assertFalse(paddle.all(conv.weight == 0).item())

    def test_initializes_bias_when_present(self):
        conv = nn.Conv2D(3, 16, kernel_size=3, padding=1, bias_attr=True)
        conv_init_(conv)
        if conv.bias is not None:
            self.assertFalse(paddle.all(conv.bias == 0).item())


class TestBiasInitWithProb(unittest.TestCase):
    def test_default_prob(self):
        result = bias_init_with_prob()
        expected = float(-np.log((1 - 0.01) / 0.01))
        self.assertAlmostEqual(result, expected)

    def test_custom_prob(self):
        result = bias_init_with_prob(0.5)
        expected = float(-np.log((1 - 0.5) / 0.5))
        self.assertAlmostEqual(result, expected)


class TestResetInitializedParameter(unittest.TestCase):
    def test_resets_conv2d(self):
        model = nn.Conv2D(3, 16, kernel_size=3, padding=1)
        # Set all weights to 1
        constant_(model.weight, 1.0)
        self.assertTrue(paddle.all(model.weight == 1.0).item())

        reset_initialized_parameter(model, include_self=True)
        # Weights should have changed
        self.assertFalse(paddle.all(model.weight == 1.0).item())

    def test_resets_linear(self):
        model = nn.Linear(32, 16)
        constant_(model.weight, 1.0)
        reset_initialized_parameter(model, include_self=True)
        self.assertFalse(paddle.all(model.weight == 1.0).item())

    def test_resets_embedding(self):
        model = nn.Embedding(100, 32)
        constant_(model.weight, 1.0)
        reset_initialized_parameter(model, include_self=True)
        self.assertFalse(paddle.all(model.weight == 1.0).item())

    def test_resets_layernorm(self):
        model = nn.LayerNorm(32)
        zeros_(model.weight)
        reset_initialized_parameter(model, include_self=True)
        # Weight should be 1.0 after reset
        self.assertTrue(paddle.all(model.weight == 1.0).item())

    def test_resets_batchnorm2d(self):
        model = nn.BatchNorm2D(16)
        zeros_(model.weight)
        reset_initialized_parameter(model, include_self=True)
        self.assertTrue(paddle.all(model.weight == 1.0).item())

    def test_sequential_model(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        constant_(model[0].weight, 1.0)
        reset_initialized_parameter(model, include_self=True)
        self.assertFalse(paddle.all(model[0].weight == 1.0).item())


if __name__ == "__main__":
    unittest.main()
