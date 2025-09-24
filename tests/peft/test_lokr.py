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
import random
import re
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import paddle

from paddleformers.peft.lokr import LoKrConfig, LoKrLinear, LoKrModel
from paddleformers.transformers import AutoModel, BertModel

DEFAULT_LINEAR_TEST_CONFIG = {
    "in_features": 4864,
    "out_features": 4864,
    "lokr_dim": 8,
    "lokr_alpha": 8,
    "factor": -1,
    "decompose_both": False,
}
DEFAULT_MODEL_TEST_CONFIG = {
    "base_model_name_or_path": "Paddleformers/tiny-random-bert",
    "target_modules": [".*q_proj*.", ".*v_proj*."],
    "lokr_alpha": 8,
    "lokr_dim": 8,
    "decompose_both": False,
    "factor": -1,
}

defaultTestLayer = LoKrLinear(**DEFAULT_LINEAR_TEST_CONFIG)


class TestLoKrLayer(unittest.TestCase):
    def test_r_raise_exception(self):
        with self.assertRaises(ValueError):
            LoKrLinear(in_features=16, out_features=8, lokr_dim=0, lokr_alpha=8)

    def test_forward(self):
        def myForward():
            input = paddle.randn([2, 4, DEFAULT_LINEAR_TEST_CONFIG["in_features"]], "float32")
            self.assertEqual(defaultTestLayer.scale, 1.0)
            output = defaultTestLayer(input)
            self.assertEqual(output.shape, [2, 4, DEFAULT_LINEAR_TEST_CONFIG["out_features"]])

        def randomForward():
            for _ in range(50):
                inFeatureRand = random.randint(100, 200)
                outFeatureRand = random.randint(100, 200)
                decompose_both_rand = random.choice([True, False])
                factorRand = random.choice([-1, random.randint(2, min(inFeatureRand, outFeatureRand))])
                lokr_layer = LoKrLinear(
                    in_features=inFeatureRand,
                    out_features=outFeatureRand,
                    lokr_dim=8,
                    lokr_alpha=8,
                    factor=factorRand,
                    decompose_both=decompose_both_rand,
                )
                input = paddle.randn([2, 4, inFeatureRand], "float32")
                self.assertEqual(lokr_layer.scale, 1.0)
                output = lokr_layer(input)
                self.assertEqual(output.shape, [2, 4, outFeatureRand])

        myForward()
        randomForward()

    def test_train_eval(self):
        def myTrainEval():
            x = paddle.randn([2, 4, DEFAULT_LINEAR_TEST_CONFIG["in_features"]], "float32")
            defaultTestLayer.train()
            train_result = defaultTestLayer(x)
            train_weight = copy.deepcopy(defaultTestLayer.weight)  # deep copy since this is a pointer
            defaultTestLayer.eval()
            eval_result = defaultTestLayer(x)
            eval_weight = defaultTestLayer.weight
            self.assertTrue(paddle.allclose(train_result, eval_result))
            self.assertTrue(paddle.allclose(train_weight, eval_weight))

        def randomTrainEval():
            for _ in range(100):
                inFeatureRand = random.randint(10, 50)
                outFeatureRand = random.randint(10, 50)
                decompose_both_rand = random.choice([True, False])
                factorRand = random.choice([-1, random.randint(2, min(inFeatureRand, outFeatureRand))])
                lokr_layer = LoKrLinear(
                    in_features=inFeatureRand,
                    out_features=outFeatureRand,
                    lokr_dim=8,
                    lokr_alpha=8,
                    factor=factorRand,
                    decompose_both=decompose_both_rand,
                )
                x = paddle.randn([2, 4, inFeatureRand], "float32")
                lokr_layer.train()
                train_result = lokr_layer(x)
                train_weight = copy.deepcopy(lokr_layer.weight)  # deep copy since this is a pointer
                lokr_layer.eval()
                eval_result = lokr_layer(x)
                eval_weight = lokr_layer.weight
                self.assertTrue(paddle.allclose(train_result, eval_result))
                self.assertTrue(paddle.allclose(train_weight, eval_weight))

        myTrainEval()
        randomTrainEval()

    def test_save_load(self):
        for _ in range(10):
            with TemporaryDirectory() as tempdir:
                weights_path = os.path.join(tempdir, "model.pdparams")
                paddle.save(defaultTestLayer.state_dict(), weights_path)
                new_lokr_layer = defaultTestLayer
                state_dict = paddle.load(weights_path)
                new_lokr_layer.set_dict(state_dict)
                x = paddle.randn([2, 4, DEFAULT_LINEAR_TEST_CONFIG["in_features"]], "float32")
                self.assertTrue(paddle.allclose(new_lokr_layer(x), defaultTestLayer(x)))  # something goes wrong here

    def test_load_regular_linear(self):
        for i in range(10):
            with TemporaryDirectory() as tempdir:
                inFeatureRand = random.randint(10, 30)
                outFeatureRand = random.randint(10, 50)
                regular_linear = paddle.nn.Linear(in_features=inFeatureRand, out_features=outFeatureRand)
                weights_path = os.path.join(tempdir, "model.pdparams")
                paddle.save(regular_linear.state_dict(), weights_path)
                state_dict = paddle.load(weights_path)
                lokr_layer = LoKrLinear(
                    in_features=inFeatureRand,
                    out_features=outFeatureRand,
                    lokr_dim=8,
                    lokr_alpha=8,
                    factor=-1,
                    decompose_both=False,
                )
                lokr_layer.set_dict(state_dict)
                x = paddle.randn([2, 4, inFeatureRand], "float32")
                self.assertTrue(paddle.allclose(lokr_layer(x), regular_linear(x)))


class TestLoKrModel(unittest.TestCase):
    def test_tp_raise_exception(self):
        with self.assertRaises(NotImplementedError):
            lokr_config = LoKrConfig(**DEFAULT_MODEL_TEST_CONFIG, tensor_parallel_degree=2)
            model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
            lokr_model = LoKrModel(model, lokr_config)
            lokr_model.eval()

    def test_lokr_model_restore(self):
        lokr_config = LoKrConfig(**DEFAULT_MODEL_TEST_CONFIG)
        model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        model.eval()
        original_results_1 = model(input_ids)
        lokr_model = LoKrModel(model, lokr_config)
        restored_model = lokr_model.restore_original_model()
        restored_model.eval()
        original_results_2 = restored_model(input_ids)
        self.assertIsNotNone(original_results_1)
        self.assertIsNotNone(original_results_2)
        self.assertIsInstance(restored_model, BertModel)
        self.assertTrue(paddle.allclose(original_results_1[0], original_results_2[0]))

    def test_lokr_model_constructor(self):
        lokr_config = LoKrConfig(**DEFAULT_MODEL_TEST_CONFIG)
        model = AutoModel.from_pretrained(
            "Paddleformers/tiny-random-bert",
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        lokr_model = LoKrModel(model, lokr_config)
        for name, weight in lokr_model.state_dict().items():
            if any([re.fullmatch(target_module, name) for target_module in lokr_config.target_modules]):
                # general rule of thumb: any weight in state_dict with name having "lokr" should enable training, vice versa.
                if "lokr" in name:
                    self.assertFalse(weight.stop_gradient)
                else:
                    self.assertTrue(weight.stop_gradient)

    def test_lokr_model_save_load(self):
        with TemporaryDirectory() as tempdir:
            input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
            lokr_config = LoKrConfig(**DEFAULT_MODEL_TEST_CONFIG)
            model = AutoModel.from_pretrained("Paddleformers/tiny-random-bert")
            lokr_model = LoKrModel(model, lokr_config)
            lokr_model.eval()
            original_results = lokr_model(input_ids)
            lokr_model.save_pretrained(tempdir)

            loaded_lokr_model = LoKrModel.from_pretrained(model, tempdir)
            loaded_lokr_model.eval()
            loaded_results = loaded_lokr_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], loaded_results[0]))

            config_loaded_lokr_model = LoKrModel.from_pretrained(model, tempdir, lokr_config=lokr_config)
            config_loaded_lokr_model.eval()
            config_loaded_results = config_loaded_lokr_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], config_loaded_results[0]))


class TestLoKrConfig(unittest.TestCase):
    def test_to_dict(self):
        config = LoKrConfig()
        expected_dict = {
            "base_model_name_or_path": None,
            "target_modules": None,
            "trainable_modules": None,
            "trainable_bias": None,
            "lokr_dim": 8,
            "factor": -1,
            "decompose_both": False,
            "lokr_alpha": 0.0,
            "merge_weight": False,
            "tensor_parallel_degree": -1,
            "dtype": None,
        }
        self.assertEqual(config.to_dict(), expected_dict)

    def test_invalid_directory_save_pretrained(self):
        config = LoKrConfig()
        with TemporaryDirectory() as tempdir:
            # Create a file instead of directory
            invalid_dir = os.path.join(tempdir, "invalid_dir")
            with open(invalid_dir, "w") as f:
                f.write("This is a file, not a directory.")
            with self.assertRaises(AssertionError):
                config.save_pretrained(invalid_dir)

    def test_from_pretrained_not_found(self):
        with TemporaryDirectory() as tempdir:
            with self.assertRaises(ValueError):
                LoKrConfig.from_pretrained(tempdir)  # No config file in directory

    def test_scaling_property(self):
        lokr_config = LoKrConfig(lokr_alpha=10, lokr_dim=2)
        self.assertEqual(lokr_config.scaling, 5.0)
        lokr_config = LoKrConfig(lokr_alpha=0, lokr_dim=8)
        self.assertEqual(lokr_config.scaling, 0.0)
        lokr_config = LoKrConfig(lokr_alpha=0, lokr_dim=0)
        self.assertEqual(lokr_config.scaling, 1.0)

    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            lokr_config = LoKrConfig(**DEFAULT_MODEL_TEST_CONFIG)
            lokr_config.save_pretrained(tempdir)
            loaded_lokr_config = LoKrConfig.from_pretrained(tempdir)
            self.assertEqual(lokr_config, loaded_lokr_config)


if __name__ == "__main__":
    unittest.main()
