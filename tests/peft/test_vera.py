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
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import paddle
from paddle import nn
from parameterized import parameterized

from paddleformers.peft.vera import VeRAConfig, VeRALinear, VeRAModel
from paddleformers.transformers import AutoModel


class TestVeraLayer(unittest.TestCase):
    def test_r_raise_exception(self):
        with self.assertRaises(ValueError):
            VeRALinear(
                in_features=16,
                out_features=16,
                r=0,
                vera_dropout=0.1,
                vera_alpha=4,
                base_linear_module=nn.Linear(in_features=16, out_features=16),
            )

    def test_forward(self):
        vera_layer = VeRALinear(
            in_features=16,
            out_features=16,
            r=4,
            vera_dropout=0.1,
            vera_alpha=4,
            base_linear_module=nn.Linear(16, 16),
            pissa_init=True,
        )
        x = paddle.randn([2, 4, 16], "float32")
        output = vera_layer(x)
        self.assertFalse(vera_layer.vera_b.stop_gradient)
        self.assertFalse(vera_layer.vera_d.stop_gradient)
        self.assertTrue(vera_layer.weight.stop_gradient)
        self.assertFalse(vera_layer.bias.stop_gradient)
        self.assertEqual(output.shape, [2, 4, 16])

    def test_train_eval(self):
        x = paddle.randn([2, 4, 16], "float32")
        vera_layer = VeRALinear(
            in_features=16, out_features=16, r=4, base_linear_module=nn.Linear(in_features=16, out_features=16)
        )
        vera_layer.train()
        train_result = vera_layer(x)
        train_weight = copy.deepcopy(vera_layer.weight)  # deep copy since this is a pointer
        vera_layer.eval()
        eval_result = vera_layer(x)
        eval_weight = vera_layer.weight
        self.assertTrue(paddle.allclose(train_result, eval_result))
        self.assertTrue(paddle.allclose(train_weight, eval_weight))

    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            vera_layer = VeRALinear(
                in_features=16, out_features=16, r=4, base_linear_module=nn.Linear(in_features=16, out_features=16)
            )
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(vera_layer.state_dict(), weights_path)
            new_vera_layer = VeRALinear(
                in_features=16, out_features=16, r=4, base_linear_module=nn.Linear(in_features=16, out_features=16)
            )
            state_dict = paddle.load(weights_path)
            new_vera_layer.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(new_vera_layer(x), vera_layer(x)))

    def test_load_regular_linear(self):
        with TemporaryDirectory() as tempdir:
            regular_linear = paddle.nn.Linear(in_features=16, out_features=16)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(regular_linear.state_dict(), weights_path)
            state_dict = paddle.load(weights_path)
            # should be identical to regular linear
            vera_layer_r8 = VeRALinear(
                in_features=16, out_features=16, r=8, base_linear_module=nn.Linear(in_features=16, out_features=16)
            )
            vera_layer_r4 = VeRALinear(
                in_features=16, out_features=16, r=4, base_linear_module=nn.Linear(in_features=16, out_features=16)
            )
            vera_layer_r8.set_dict(state_dict)
            vera_layer_r4.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(vera_layer_r8(x), regular_linear(x)))
            self.assertTrue(paddle.allclose(vera_layer_r4(x), regular_linear(x)))


class TestVeraModel(unittest.TestCase):
    @parameterized.expand([(None,), ("all",), ("vera",)])
    def test_vera_model_constructor(self, bias):
        vera_config = VeRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"], r=4, vera_alpha=4, head_dim=2, pissa_init=True
        )
        # turn off plm dropout for to test train vs test
        model = AutoModel.from_pretrained(
            "Paddleformers/tiny-random-bert",
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        vera_model = VeRAModel(model, vera_config)
        vera_model.mark_only_vera_as_trainable()

        for name, weight in vera_model.state_dict().items():
            if any([re.fullmatch(target_module, name) for target_module in vera_config.target_modules]):
                if "vera_b" in name or "vera_d" in name:
                    self.assertFalse(weight.stop_gradient)
                else:
                    self.assertTrue(weight.stop_gradient)

        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        vera_model.train()
        train_forward_results = vera_model(input_ids)
        self.assertIsNotNone(train_forward_results)
        vera_model.eval()
        eval_forward_results = vera_model(input_ids)
        self.assertIsNotNone(eval_forward_results)
        self.assertTrue(paddle.allclose(train_forward_results[0], eval_forward_results[0]))

    def test_vera_model_save_load(self):
        with TemporaryDirectory() as tempdir:
            input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
            vera_config = VeRAConfig(
                target_modules=[".*q_proj.*", ".*v_proj.*"],
                r=4,
                vera_alpha=4,
            )
            model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
            vera_model = VeRAModel(model, vera_config)
            vera_model.eval()
            original_results = vera_model(input_ids)
            vera_model.save_pretrained(tempdir)

            loaded_vera_model = VeRAModel.from_pretrained(model, tempdir)
            loaded_vera_model.eval()
            loaded_results = loaded_vera_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], loaded_results[0]))

            config_loaded_vera_model = VeRAModel.from_pretrained(model, tempdir, vera_config=vera_config)
            config_loaded_vera_model.eval()
            config_loaded_results = config_loaded_vera_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], config_loaded_results[0]))

    def test_restore_original_model(self):
        vera_config = VeRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            vera_alpha=4,
        )
        model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        vera_model = VeRAModel(model, vera_config)
        with self.assertRaises(NotImplementedError):
            vera_model.restore_original_model()

    def test_vera_module_raise_exception(self):
        vera_config = VeRAConfig(target_modules=[".*norm1.*"], r=4, vera_alpha=4)
        model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        with self.assertRaises(ValueError):
            VeRAModel(model, vera_config)

    def test_pissa_raise_exception(self):
        vera_config = VeRAConfig(target_modules=[".*q_proj.*"], r=4, vera_alpha=8, pissa_init=True)
        model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        with self.assertRaises(AssertionError):
            VeRAModel(model, vera_config)


class TestVeRAConfig(unittest.TestCase):
    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            vera_config = VeRAConfig()
            vera_config.save_pretrained(tempdir)
            loaded_vera_config = VeRAConfig.from_pretrained(tempdir)
            self.assertEqual(vera_config, loaded_vera_config)

    def test_save_load_err(self):
        with NamedTemporaryFile("w+t") as f:
            with self.assertRaises(ValueError):
                VeRAConfig.from_pretrained(f.name)

    def test_save_pretrained_file_error(self):
        with NamedTemporaryFile("w+t") as f:
            vera_config = VeRAConfig()
            with self.assertRaises(AssertionError):
                vera_config.save_pretrained(f.name)
