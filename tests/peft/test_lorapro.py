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

import copy
import math
import os
import re
import shutil
import tempfile
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import paddle
from parameterized import parameterized

from paddleformers.peft.lora import LoRAConfig, LoRALinear, LoRAModel
from paddleformers.transformers import AutoModel, BertModel
from paddleformers.utils.optimizer import AdamWLoRAPro


class TestLoRAProLayer(unittest.TestCase):
    def test_r_raise_exception(self):
        with self.assertRaises(ValueError):
            LoRALinear(in_features=16, out_features=8, r=0, lora_dropout=0.1, lora_alpha=8, lorapro=True)

    def test_forward(self):
        lorapro_layer = LoRALinear(in_features=16, out_features=8, r=4, lora_dropout=0.1, lora_alpha=8, lorapro=True)
        x = paddle.randn([2, 4, 16], "float32")
        output = lorapro_layer(x)
        self.assertFalse(lorapro_layer.lora_A.stop_gradient)
        self.assertFalse(lorapro_layer.lora_B.stop_gradient)
        self.assertTrue(lorapro_layer.weight.stop_gradient)
        self.assertFalse(lorapro_layer.bias.stop_gradient)
        self.assertEqual(output.shape, [2, 4, 8])

    def test_train_eval(self):
        x = paddle.randn([2, 4, 16], "float32")
        lorapro_layer = LoRALinear(in_features=16, out_features=8, r=4, lorapro=True)
        lorapro_layer.train()
        train_result = lorapro_layer(x)
        train_weight = copy.deepcopy(lorapro_layer.weight)  # deep copy since this is a pointer
        lorapro_layer.eval()
        eval_result = lorapro_layer(x)
        eval_weight = lorapro_layer.weight
        self.assertTrue(paddle.allclose(train_result, eval_result))
        self.assertTrue(paddle.allclose(train_weight, eval_weight))

    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            lorapro_layer = LoRALinear(in_features=16, out_features=8, r=4, lorapro=True)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(lorapro_layer.state_dict(), weights_path)
            new_lorapro_layer = LoRALinear(in_features=16, out_features=8, r=4, lorapro=True)
            state_dict = paddle.load(weights_path)
            new_lorapro_layer.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(new_lorapro_layer(x), lorapro_layer(x)))

    def test_load_regular_linear(self):
        with TemporaryDirectory() as tempdir:
            regular_linear = paddle.nn.Linear(in_features=16, out_features=8)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(regular_linear.state_dict(), weights_path)
            state_dict = paddle.load(weights_path)
            # should be identical to regular linear
            lorapro_layer_r8 = LoRALinear(in_features=16, out_features=8, r=8, lorapro=True)
            lorapro_layer_r4 = LoRALinear(in_features=16, out_features=8, r=4, lorapro=True)
            lorapro_layer_r8.set_dict(state_dict)
            lorapro_layer_r4.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(lorapro_layer_r8(x), regular_linear(x)))
            self.assertTrue(paddle.allclose(lorapro_layer_r4(x), regular_linear(x)))


class TestLoRAProModel(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_lorapro_model_restore(self):
        lorapro_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
            enable_lora_list=[None, [True, False]],
            head_dim=2,
            lorapro=True,
        )
        model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        model.eval()
        original_results_1 = model(input_ids)
        lorapro_model = LoRAModel(model, lorapro_config)
        restored_model = lorapro_model.restore_original_model()
        restored_model.eval()
        original_results_2 = restored_model(input_ids)
        self.assertIsNotNone(original_results_1)
        self.assertIsNotNone(original_results_2)
        self.assertIsInstance(restored_model, BertModel)
        self.assertTrue(paddle.allclose(original_results_1[0], original_results_2[0]))

    @parameterized.expand([(None,), ("all",), ("lora",)])
    def test_lorapro_model_constructor(self, bias):
        lorapro_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
            enable_lora_list=[None, [True, False]],
            trainable_bias=bias,
            head_dim=2,
            lorapro=True,
        )
        # turn off plm dropout for to test train vs test
        model = AutoModel.from_pretrained(
            "Paddleformers/tiny-random-bert",
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        lorapro_model = LoRAModel(model, lorapro_config)
        lorapro_model.mark_only_lora_as_trainable()
        for name, weight in lorapro_model.state_dict().items():
            if any([re.fullmatch(target_module, name) for target_module in lorapro_config.target_modules]):
                if "lora" in name:
                    self.assertFalse(weight.stop_gradient)
                elif "bias" in name and bias in ["lora", "all"]:
                    self.assertFalse(weight.stop_gradient)
                else:
                    self.assertTrue(weight.stop_gradient)
            else:
                if "bias" in name and bias == "all":
                    self.assertFalse(weight.stop_gradient)
                else:
                    self.assertTrue(weight.stop_gradient)
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        lorapro_model.train()
        train_forward_results = lorapro_model(input_ids)
        self.assertIsNotNone(train_forward_results)
        lorapro_model.eval()
        eval_forward_results = lorapro_model(input_ids)
        self.assertIsNotNone(eval_forward_results)
        self.assertTrue(paddle.allclose(train_forward_results[0], eval_forward_results[0]))

    def test_lorapro_model_save_load(self):
        with TemporaryDirectory() as tempdir:
            input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
            lorapro_config = LoRAConfig(target_modules=[".*q_proj.*", ".*v_proj.*"], r=4, lora_alpha=8, lorapro=True)
            model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
            lorapro_model = LoRAModel(model, lorapro_config)
            lorapro_model.eval()
            original_results = lorapro_model(input_ids)
            lorapro_model.save_pretrained(tempdir)

            loaded_lorapro_model = LoRAModel.from_pretrained(model, tempdir)
            loaded_lorapro_model.eval()
            loaded_results = loaded_lorapro_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], loaded_results[0]))

            config_loaded_lorapro_model = LoRAModel.from_pretrained(model, tempdir, lora_config=lorapro_config)
            config_loaded_lorapro_model.eval()
            config_loaded_results = config_loaded_lorapro_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], config_loaded_results[0]))

    @parameterized.expand([("zero",), ("sylvester",), ("symmetry",)])
    def test_lorapro_modes(self, x_mode):
        """Test if AdamWLoRAPro optimizer with different x_modes can perform optimization steps"""
        lorapro_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
            enable_lora_list=[None, [True, False]],
            head_dim=2,
            lorapro=True,
        )

        model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        lorapro_model = LoRAModel(model, lorapro_config)
        lorapro_model.mark_only_lora_as_trainable()

        input_ids = paddle.to_tensor(np.random.randint(100, 200, [2, 20]))

        lorapro_model.train()

        scaling_factor = lorapro_config.lora_alpha / lorapro_config.r
        if lorapro_config.rslora:
            scaling_factor = lorapro_config.lora_alpha / math.sqrt(lorapro_config.r)

        optimizer = AdamWLoRAPro(
            learning_rate=1e-4, parameters=lorapro_model.parameters(), scaling_factor=scaling_factor, x_mode=x_mode
        )

        outputs = lorapro_model(input_ids)
        loss = outputs[0].mean()

        loss.backward()

        optimizer.step()

        train_forward_results = lorapro_model(input_ids)
        self.assertIsNotNone(train_forward_results)
        self.assertIsInstance(optimizer, AdamWLoRAPro)
        self.assertEqual(optimizer.x_mode, x_mode)

    def test_lorapro_module_raise_exception(self):
        lorapro_config = LoRAConfig(
            target_modules=[".*norm1.*"], r=4, lora_alpha=8, enable_lora_list=None, lorapro=True
        )
        model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        with self.assertRaises(ValueError):
            LoRAModel(model, lorapro_config)


class TestLoRAProConfig(unittest.TestCase):
    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            lorapro_config = LoRAConfig()
            lorapro_config.save_pretrained(tempdir)
            loaded_lorapro_config = LoRAConfig.from_pretrained(tempdir)
            self.assertEqual(lorapro_config, loaded_lorapro_config)


if __name__ == "__main__":
    unittest.main()
