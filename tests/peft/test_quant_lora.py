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

import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import paddle
from paddle.quantization import QAT, QuantConfig
from paddle.quantization.config import SingleLayerConfig
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
from paddle.quantization.quanters.abs_max import FakeQuanterWithAbsMaxObserverLayer

from paddleformers.peft.lora import LoRAConfig, LoRALinear, LoRAModel
from paddleformers.peft.lora.lora_quant_layers import QuantedLoRALinear
from paddleformers.transformers import AutoModel


class TestQuantedLoraLayer(unittest.TestCase):
    def test_forward(self):
        quant_lora_layer = QuantedLoRALinear(
            layer=LoRALinear(in_features=16, out_features=8, r=4, lora_alpha=8),
            q_config=SingleLayerConfig(weight=FakeQuanterWithAbsMaxObserver(moving_rate=0.9), activation=None),
        )
        x = paddle.randn([2, 4, 16], "float32")
        quant_output = quant_lora_layer(x)
        self.assertFalse(quant_lora_layer.lora_A.stop_gradient)
        self.assertFalse(quant_lora_layer.lora_B.stop_gradient)
        self.assertTrue(quant_lora_layer.weight.stop_gradient)
        self.assertFalse(quant_lora_layer.bias.stop_gradient)
        self.assertEqual(quant_output.shape, [2, 4, 8])

    def test_forward_no_quant(self):
        lora_layer = LoRALinear(
            in_features=16,
            out_features=8,
            r=4,
            lora_alpha=8,
        )
        quant_lora_layer = QuantedLoRALinear(
            layer=lora_layer, q_config=SingleLayerConfig(weight=None, activation=None)
        )
        x = paddle.randn([2, 4, 16], "float32")
        output = lora_layer(x)
        quant_output = quant_lora_layer(x)
        self.assertTrue(paddle.allclose(output, quant_output))

    def test_dropout_raise_exception(self):
        with self.assertRaises(ValueError):
            QuantedLoRALinear(
                layer=LoRALinear(in_features=16, out_features=8, r=4, lora_alpha=8, lora_dropout=0.1),
                q_config=SingleLayerConfig(weight=None, activation=None),
            )

    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            q_config = SingleLayerConfig(weight=FakeQuanterWithAbsMaxObserver(moving_rate=0.9), activation=None)
            quant_lora_layer = QuantedLoRALinear(
                layer=LoRALinear(in_features=16, out_features=8, r=4, lora_alpha=8), q_config=q_config
            )
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(quant_lora_layer.state_dict(), weights_path)
            new_quant_lora_layer = QuantedLoRALinear(
                layer=LoRALinear(in_features=16, out_features=8, r=4, lora_alpha=8), q_config=q_config
            )
            state_dict = paddle.load(weights_path)
            new_quant_lora_layer.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(new_quant_lora_layer(x), quant_lora_layer(x)))

    def test_merge_weights(self):
        lora_layer = LoRALinear(in_features=16, out_features=8, r=4, lora_alpha=8)
        quant_lora_layer = QuantedLoRALinear(
            layer=lora_layer, q_config=SingleLayerConfig(weight=None, activation=None)
        )
        x = paddle.randn([2, 4, 16], "float32")

        quant_lora_layer.merge()
        merge_output = lora_layer(x)
        quant_lora_layer.unmerge()
        unmerge_output = lora_layer(x)
        self.assertTrue(paddle.allclose(merge_output, unmerge_output))


class TestQuantedLoRAModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lora_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
        )
        cls.model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        cls.lora_model = LoRAModel(cls.model, lora_config)
        cls.lora_model.mark_only_lora_as_trainable()
        # lora_B parameter is initialized to 0, therefore AB = 0 and W + AB = W
        # Since we want to test W + AB logic, we set lora_B to random values.
        lora_b_state_dict = {}
        for name, state in cls.lora_model.state_dict().items():
            if "lora_B" in name:
                lora_b_state_dict[name] = paddle.randn(state.shape)
        cls.lora_model.set_dict(lora_b_state_dict)

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_count_model_layers(self):
        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)
        q_config.add_type_config(LoRALinear, weight=FakeQuanterWithAbsMaxObserver(moving_rate=0.9))
        qat = QAT(q_config)
        self.lora_model.train()
        quant_lora_model = qat.quantize(self.lora_model, inplace=False)
        quantizer_cnt = self._count_layers(quant_lora_model, FakeQuanterWithAbsMaxObserverLayer)
        # 2 LoRA layers (q_proj, v_proj) per transformer layer
        self.assertEqual(quantizer_cnt, 2 * self.model.config.num_hidden_layers)

    def test_forward_no_quant(self):
        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)
        q_config.add_type_config(LoRALinear, weight=None, activation=None)
        qat = QAT(q_config)
        self.lora_model.train()
        quant_lora_model = qat.quantize(self.lora_model, inplace=False)
        quant_lora_model.merge()
        self.lora_model.merge()
        quant_lora_model.eval()
        self.lora_model.eval()

        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 5]))
        original_model_outputs = self.lora_model(input_ids)[0]
        quant_model_outputs = quant_lora_model(input_ids)[0]
        self.assertTrue(paddle.allclose(original_model_outputs, quant_model_outputs, atol=1e-5))

    def test_forward_weight_quant(self):
        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)
        q_config.add_type_config(LoRALinear, weight=FakeQuanterWithAbsMaxObserver(moving_rate=0.9))
        qat = QAT(q_config)
        self.lora_model.train()
        quant_lora_model = qat.quantize(self.lora_model, inplace=False)
        quant_lora_model.eval()
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 5]))
        original_model_outputs = self.lora_model(input_ids)[0]
        quant_model_outputs = quant_lora_model(input_ids)[0]
        self.assertEqual(original_model_outputs.shape, quant_model_outputs.shape)

    def test_quant_lora_model_stop_gradient(self):
        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)
        q_config.add_type_config(LoRALinear, weight=FakeQuanterWithAbsMaxObserver(moving_rate=0.9))
        qat = QAT(q_config)
        self.lora_model.train()
        quant_lora_model = qat.quantize(self.lora_model, inplace=False)
        for name, weight in quant_lora_model.state_dict().items():
            if "lora" in name:
                self.assertFalse(weight.stop_gradient)
            else:
                self.assertTrue(weight.stop_gradient)
