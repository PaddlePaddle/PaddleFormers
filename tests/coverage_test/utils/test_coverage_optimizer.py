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
from unittest.mock import MagicMock, patch

import paddle
import paddle.nn as nn

from paddleformers.utils.optimizer import AdamWCustom, AdamWMini


class TestAdamWMiniAdamwPython(unittest.TestCase):
    """Test the adamw_python method of AdamWMini."""

    def setUp(self):
        # Create a simple model and optimizer so we can call the method
        # as an instance method rather than an unbound method
        self.linear = nn.Linear(4, 2)
        self.optimizer = AdamWMini(
            parameters=self.linear.parameters(),
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
        )

    def test_skip_update_returns_early(self):
        param = paddle.zeros([4], dtype="float32")
        grad = paddle.ones([4], dtype="float32")
        lr = paddle.to_tensor(0.001)
        moment1 = paddle.zeros([4], dtype="float32")
        moment2 = paddle.zeros([1], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9])
        beta2_pow = paddle.to_tensor([0.999])

        # skip_update=True should return immediately
        self.optimizer.adamw_python(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            True,  # master_weight, skip_update
            0.9,
            0.999,
            1e-8,  # beta1, beta2, epsilon
            1.0,
            0.01,  # lr_ratio, coeff
            True,
            False,  # with_decay, multi_precision
        )
        # param should remain unchanged
        self.assertTrue(paddle.all(param == 0).item())

    def test_no_decay_sets_coeff_zero(self):
        param = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        grad = paddle.to_tensor([0.1, 0.1, 0.1, 0.1])
        lr = paddle.to_tensor(0.001)
        moment1 = paddle.zeros([4], dtype="float32")
        moment2 = paddle.zeros([1], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9])
        beta2_pow = paddle.to_tensor([0.999])

        param_before = param.clone()
        self.optimizer.adamw_python(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            False,  # master_weight, skip_update
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,  # lr_ratio, coeff
            False,
            False,  # with_decay=False, multi_precision
        )
        # param should change
        self.assertFalse(paddle.all(param == param_before).item())

    def test_basic_update(self):
        param = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        grad = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
        lr = paddle.to_tensor(0.001)
        moment1 = paddle.zeros([4], dtype="float32")
        moment2 = paddle.zeros([1], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9])
        beta2_pow = paddle.to_tensor([0.999])

        self.optimizer.adamw_python(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            False,
        )
        # moments should be updated
        self.assertFalse(paddle.all(moment1 == 0).item())
        # beta powers should be updated
        self.assertAlmostEqual(beta1_pow.item(), 0.9 * 0.9, places=5)
        self.assertAlmostEqual(beta2_pow.item(), 0.999 * 0.999, places=5)

    def test_with_master_weight(self):
        param = paddle.to_tensor([1.0, 2.0], dtype="float16")
        grad = paddle.to_tensor([0.1, 0.2], dtype="float16")
        lr = paddle.to_tensor(0.001, dtype="float32")
        moment1 = paddle.zeros([2], dtype="float32")
        moment2 = paddle.zeros([1], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9])
        beta2_pow = paddle.to_tensor([0.999])
        master_weight = paddle.to_tensor([1.0, 2.0], dtype="float32")

        self.optimizer.adamw_python(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master_weight,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            True,  # with_decay, multi_precision
        )
        # master_weight should be updated
        self.assertFalse(paddle.all(master_weight == paddle.to_tensor([1.0, 2.0])).item())


class TestAdamWCustomAdamwCustom(unittest.TestCase):
    """Test the adamw_custom method of AdamWCustom."""

    def setUp(self):
        self.linear = nn.Linear(4, 2)
        self.mock_quant_config = MagicMock()
        self.mock_quant_config.weight_quantize_algo = "a8w8linear"
        self.mock_quant_config.apply_hadamard = False
        with patch("paddle.distributed.get_world_size", return_value=1):
            self.optimizer = AdamWCustom(
                quantization_config=self.mock_quant_config,
                tensorwise_offload_optimizer=False,
                parameters=self.linear.parameters(),
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                weight_decay=0.01,
                multi_precision=True,
            )

    def test_skip_update_returns_early(self):
        param = paddle.zeros([4], dtype="float32")
        grad = paddle.ones([4], dtype="float32")
        lr = paddle.to_tensor(0.001)
        moment1 = paddle.zeros([4], dtype="float32")
        moment2 = paddle.zeros([4], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9])
        beta2_pow = paddle.to_tensor([0.999])

        self.optimizer.adamw_custom(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            True,  # master_weight, skip_update
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            False,
            False,  # with_decay, multi_precision, skip_update_param
        )
        self.assertTrue(paddle.all(param == 0).item())

    def test_no_decay_sets_coeff_zero(self):
        param = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        grad = paddle.to_tensor([0.1, 0.1, 0.1, 0.1])
        lr = paddle.to_tensor(0.001)
        moment1 = paddle.zeros([4], dtype="float32")
        moment2 = paddle.zeros([4], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9])
        beta2_pow = paddle.to_tensor([0.999])

        param_before = param.clone()
        self.optimizer.adamw_custom(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            False,
            False,
            False,  # with_decay=False
        )
        self.assertFalse(paddle.all(param == param_before).item())

    def test_basic_update(self):
        param = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        grad = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
        lr = paddle.to_tensor(0.001)
        moment1 = paddle.zeros([4], dtype="float32")
        moment2 = paddle.zeros([4], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9])
        beta2_pow = paddle.to_tensor([0.999])

        self.optimizer.adamw_custom(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            False,
            False,
        )
        self.assertFalse(paddle.all(moment1 == 0).item())
        self.assertFalse(paddle.all(moment2 == 0).item())

    def test_with_master_weight_and_skip_update_param(self):
        param = paddle.to_tensor([1.0, 2.0], dtype="float16")
        grad = paddle.to_tensor([0.1, 0.2], dtype="float16")
        lr = paddle.to_tensor(0.001, dtype="float32")
        moment1 = paddle.zeros([2], dtype="float32")
        moment2 = paddle.zeros([2], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9])
        beta2_pow = paddle.to_tensor([0.999])
        master_weight = paddle.to_tensor([1.0, 2.0], dtype="float32")
        param_clone = param.clone()

        self.optimizer.adamw_custom(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master_weight,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            True,
            True,  # skip_update_param=True -> param not updated
        )
        # master_weight should change
        self.assertFalse(paddle.all(master_weight == paddle.to_tensor([1.0, 2.0])).item())
        # param should NOT be updated because skip_update_param=True
        self.assertTrue(paddle.all(param == param_clone).item())

    def test_with_master_weight_no_skip(self):
        param = paddle.to_tensor([1.0, 2.0], dtype="float16")
        grad = paddle.to_tensor([0.1, 0.2], dtype="float16")
        lr = paddle.to_tensor(0.001, dtype="float32")
        moment1 = paddle.zeros([2], dtype="float32")
        moment2 = paddle.zeros([2], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9])
        beta2_pow = paddle.to_tensor([0.999])
        master_weight = paddle.to_tensor([1.0, 2.0], dtype="float32")

        self.optimizer.adamw_custom(
            param,
            grad,
            lr,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master_weight,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            True,
            False,  # skip_update_param=False -> param updated
        )
        # param should be updated
        self.assertFalse(paddle.all(param == paddle.to_tensor([1.0, 2.0], dtype="float16")).item())


class TestAdamWCustomIsDtypeFp16OrBf16(unittest.TestCase):
    """Test _is_dtype_fp16_or_bf16 method."""

    def setUp(self):
        self.linear = nn.Linear(4, 2)
        self.mock_quant_config = MagicMock()
        self.mock_quant_config.weight_quantize_algo = "a8w8linear"
        self.mock_quant_config.apply_hadamard = False
        with patch("paddle.distributed.get_world_size", return_value=1):
            self.optimizer = AdamWCustom(
                quantization_config=self.mock_quant_config,
                tensorwise_offload_optimizer=False,
                parameters=self.linear.parameters(),
                learning_rate=0.001,
                multi_precision=True,
            )

    def test_fp16_returns_true(self):
        result = self.optimizer._is_dtype_fp16_or_bf16(paddle.float16)
        self.assertTrue(result)

    def test_bf16_returns_true(self):
        result = self.optimizer._is_dtype_fp16_or_bf16(paddle.bfloat16)
        self.assertTrue(result)

    def test_fp32_returns_false(self):
        result = self.optimizer._is_dtype_fp16_or_bf16(paddle.float32)
        self.assertFalse(result)

    def test_int8_returns_true(self):
        result = self.optimizer._is_dtype_fp16_or_bf16(paddle.int8)
        self.assertTrue(result)

    def test_invalid_dtype_raises(self):
        with self.assertRaises(AssertionError):
            self.optimizer._is_dtype_fp16_or_bf16("invalid")


if __name__ == "__main__":
    unittest.main()
