# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os
import re
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import paddle
from parameterized import parameterized

from paddleformers.peft.lora import LoRAConfig, LoRALinear, LoRAModel
from paddleformers.transformers import AutoModel, BertModel


class TestMoraLayer(unittest.TestCase):
    def test_r_raise_exception(self):
        with self.assertRaises(ValueError):
            LoRALinear(in_features=16, out_features=8, r=0, lora_dropout=0.1, lora_alpha=8, use_mora=True)

    def test_forward(self):
        mora_layer = LoRALinear(in_features=16, out_features=8, r=4, lora_dropout=0.1, lora_alpha=8, use_mora=True)
        x = paddle.randn([2, 4, 16], "float32")
        output = mora_layer(x)
        self.assertFalse(mora_layer.lora_A.stop_gradient)
        self.assertTrue(mora_layer.weight.stop_gradient)
        self.assertFalse(mora_layer.bias.stop_gradient)
        self.assertEqual(output.shape, [2, 4, 8])

    def test_train_eval(self):
        x = paddle.randn([2, 4, 16], "float32")
        mora_layer = LoRALinear(in_features=16, out_features=8, r=4, use_mora=True)
        mora_layer.train()
        train_result = mora_layer(x)
        train_weight = copy.deepcopy(mora_layer.weight)  # deep copy since this is a pointer
        mora_layer.eval()
        eval_result = mora_layer(x)
        eval_weight = mora_layer.weight
        self.assertTrue(paddle.allclose(train_result, eval_result))
        self.assertTrue(paddle.allclose(train_weight, eval_weight))

    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            mora_layer = LoRALinear(in_features=16, out_features=8, r=4, use_mora=True)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(mora_layer.state_dict(), weights_path)
            new_mora_layer = LoRALinear(in_features=16, out_features=8, r=4, use_mora=True)
            state_dict = paddle.load(weights_path)
            new_mora_layer.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(new_mora_layer(x), mora_layer(x)))

    def test_load_regular_linear(self):
        with TemporaryDirectory() as tempdir:
            regular_linear = paddle.nn.Linear(in_features=16, out_features=8)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(regular_linear.state_dict(), weights_path)
            state_dict = paddle.load(weights_path)
            # should be identical to regular linear
            mora_layer_r8 = LoRALinear(in_features=16, out_features=8, r=8, use_mora=True)
            mora_layer_r4 = LoRALinear(in_features=16, out_features=8, r=4, use_mora=True)
            mora_layer_r8.set_dict(state_dict)
            mora_layer_r4.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(mora_layer_r8(x), regular_linear(x)))
            self.assertTrue(paddle.allclose(mora_layer_r4(x), regular_linear(x)))

    def test_merge(self):
        mora_layer_r8 = LoRALinear(in_features=16, out_features=8, r=8, use_mora=True)
        mora_layer_r8.merge()

    def test_unmerge(self):
        mora_layer_r8 = LoRALinear(in_features=16, out_features=8, r=8, use_mora=True)
        mora_layer_r8.merged = True
        mora_layer_r8.unmerge()
        mora_layer_r8 = LoRALinear(in_features=16, out_features=8, r=8)
        mora_layer_r8.merged = True
        mora_layer_r8.unmerge()


class TestMoraModel(unittest.TestCase):
    def test_mora_model_restore(self):
        mora_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
            enable_lora_list=[None, [True, False]],
            head_dim=2,
            use_mora=True,
        )
        model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        model.eval()
        original_results_1 = model(input_ids)
        mora_model = LoRAModel(model, mora_config)
        restored_model = mora_model.restore_original_model()
        restored_model.eval()
        original_results_2 = restored_model(input_ids)
        self.assertIsNotNone(original_results_1)
        self.assertIsNotNone(original_results_2)
        self.assertIsInstance(restored_model, BertModel)
        self.assertTrue(paddle.allclose(original_results_1[0], original_results_2[0]))

    @parameterized.expand([(None,), ("all",), ("lora",)])
    def test_mora_model_constructor(self, bias):
        mora_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
            enable_lora_list=[None, [True, False]],
            trainable_bias=bias,
            head_dim=2,
            use_mora=True,
        )
        # turn off plm dropout for to test train vs test
        model = AutoModel.from_pretrained(
            "Paddleformers/tiny-random-bert",
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        mora_model = LoRAModel(model, mora_config)
        mora_model.mark_only_lora_as_trainable()
        for name, weight in mora_model.state_dict().items():
            if any([re.fullmatch(target_module, name) for target_module in mora_config.target_modules]):
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
        mora_model.train()
        train_forward_results = mora_model(input_ids)
        self.assertIsNotNone(train_forward_results)
        mora_model.eval()
        eval_forward_results = mora_model(input_ids)
        self.assertIsNotNone(eval_forward_results)
        self.assertTrue(paddle.allclose(train_forward_results[0], eval_forward_results[0]))

    def test_mora_model_save_load(self):
        with TemporaryDirectory() as tempdir:
            input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
            mora_config = LoRAConfig(target_modules=[".*q_proj.*", ".*v_proj.*"], r=4, lora_alpha=8, use_mora=True)
            model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
            mora_model = LoRAModel(model, mora_config)
            mora_model.eval()
            original_results = mora_model(input_ids)
            mora_model.save_pretrained(tempdir)

            loaded_mora_model = LoRAModel.from_pretrained(model, tempdir)
            loaded_mora_model.eval()
            loaded_results = loaded_mora_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], loaded_results[0]))

            config_loaded_mora_model = LoRAModel.from_pretrained(model, tempdir, lora_config=mora_config)
            config_loaded_mora_model.eval()
            config_loaded_results = config_loaded_mora_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], config_loaded_results[0]))

    def test_lora_module_raise_exception(self):
        mora_config = LoRAConfig(target_modules=[".*norm1.*"], r=4, lora_alpha=8, enable_lora_list=None, use_mora=True)
        model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        with self.assertRaises(ValueError):
            LoRAModel(model, mora_config)


class TestMoraConfig(unittest.TestCase):
    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            mora_config = LoRAConfig()
            mora_config.save_pretrained(tempdir)
            loaded_mora_config = LoRAConfig.from_pretrained(tempdir)
            self.assertEqual(mora_config, loaded_mora_config)


if __name__ == "__main__":
    unittest.main()
