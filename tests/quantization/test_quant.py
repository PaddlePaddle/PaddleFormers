# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.peft.lora import LoRAConfig, LoRAModel
from paddleformers.transformers import AutoConfig, AutoModelForCausalLM
from paddleformers.transformers.configuration_utils import QuantizationConfig
from tests.testing_utils import gpu_device_initializer


class TestQuantedModel(unittest.TestCase):
    @classmethod
    @gpu_device_initializer(log_prefix="TestQuantedModel")
    def setUpClass(cls):
        quantization_config = dict(
            weight_quantize_algo={"weight_only_int8": [".*mlp.*", ".*self_attn.*"]},
            ignore_modules=[".*out_linear.*"],
        )
        quantization_config = QuantizationConfig.from_dict(quantization_config)
        model_config = AutoConfig.from_pretrained(
            "PaddleFormers/tiny-random-qwen3",
            dtype="bfloat16",
            quantization_config=quantization_config,
        )
        cls.model = AutoModelForCausalLM.from_pretrained(
            "PaddleFormers/tiny-random-qwen3",
            config=model_config,
            load_checkpoint_format="flex_checkpoint",
        )

    def test_quant_model(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        input_ids = paddle.to_tensor([input_ids])
        with paddle.no_grad():
            out = self.model(input_ids, return_dict=True).logits.float()

        # Expected mean on dim = -1
        EXPECTED_MEAN = paddle.to_tensor(
            [[-0.00021177, -0.00002655, -0.00033170, -0.00071293, -0.00059850, -0.00084414, -0.00059258, -0.00059982]]
        )
        self.assertTrue(paddle.allclose(out.mean(-1), EXPECTED_MEAN, atol=1e-5, rtol=1e-5))

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = paddle.to_tensor([0.06298828, -0.06176758, 0.06396484, 0.02416992, -0.13281250,
                                          -0.02258301, -0.22656250, 0.09570312, 0.12500000, -0.04736328,
                                          -0.07958984, 0.30468750, 0.01513672, -0.05932617, 0.05761719,
                                          -0.08349609, -0.14160156, -0.25000000, -0.07861328, -0.31250000,
                                           0.23144531, 0.29882812, 0.20214844, 0.27929688, 0.18847656,
                                          -0.17773438, -0.00988770, -0.04248047, 0.04589844, 0.05737305])  # fmt: skip
        self.assertTrue(paddle.allclose(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5))

        self.assertTrue(type(self.model.model.layers[0].self_attn.qkv_proj).__name__ == "QuantizationLinear")

        lora_config = LoRAConfig(
            target_modules=[".*qkv_proj.*"],
            r=4,
            lora_alpha=8,
        )
        lora_model = LoRAModel(self.model, lora_config)

        with paddle.no_grad():
            out = self.model(input_ids, return_dict=True).logits.float()

        # Expected mean on dim = -1
        EXPECTED_MEAN = paddle.to_tensor(
            [[-0.00021177, -0.00002655, -0.00033170, -0.00071293, -0.00059850, -0.00084414, -0.00059258, -0.00059982]]
        )
        self.assertTrue(paddle.allclose(out.mean(-1), EXPECTED_MEAN, atol=1e-5, rtol=1e-5))

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = paddle.to_tensor([0.06298828, -0.06176758, 0.06396484, 0.02416992, -0.13281250,
                                          -0.02258301, -0.22656250, 0.09570312, 0.12500000, -0.04736328,
                                          -0.07958984, 0.30468750, 0.01513672, -0.05932617, 0.05761719,
                                          -0.08349609, -0.14160156, -0.25000000, -0.07861328, -0.31250000,
                                           0.23144531, 0.29882812, 0.20214844, 0.27929688, 0.18847656,
                                          -0.17773438, -0.00988770, -0.04248047, 0.04589844, 0.05737305])  # fmt: skip
        self.assertTrue(paddle.allclose(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5))

        self.assertTrue(type(lora_model.model.model.layers[0].self_attn.qkv_proj).__name__ == "QuantizationLoRALinear")
