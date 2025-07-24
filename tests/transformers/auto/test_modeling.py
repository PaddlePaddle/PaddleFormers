# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Hugging Face inc.
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
import json
import os
import tempfile
import unittest

from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForPretraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    BertConfig,
    BertModel,
    LlamaModel,
)
from paddleformers.transformers.auto.configuration import CONFIG_MAPPING
from paddleformers.transformers.auto.modeling import MODEL_MAPPING
from paddleformers.utils.env import CONFIG_NAME, PADDLE_WEIGHTS_NAME

from ...utils.test_module.custom_configuration import CustomConfig
from ...utils.test_module.custom_model import CustomModel
from ..bert.test_modeling import BertModelTester


class AutoModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert")

    def test_from_pretrained_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.model.save_pretrained(tmp_dir)
            model = AutoModel.from_pretrained(tmp_dir)
            self.assertIsInstance(model, BertModel)

    def test_from_pretrained_no_init_class_with_model_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = copy.deepcopy(self.model)
            # when init_class is not found, we rely on the filename to get the import class
            model_save_path = os.path.join(tmp_dir, "tiny-random-bert")
            model.save_pretrained(model_save_path)
            config = model.config.to_dict()
            config.pop("architectures")
            with open(os.path.join(model_save_path, "config.json"), "w", encoding="utf-8") as writer:
                writer.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
            reloaded_model = AutoModel.from_pretrained(model_save_path)
            self.assertIsInstance(reloaded_model, BertModel)

    def test_model_from_pretrained_cache_dir(self):
        model_name = "__internal_testing__/tiny-random-bert"
        with tempfile.TemporaryDirectory() as tempdir:
            AutoModel.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, PADDLE_WEIGHTS_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))

    @unittest.skip("skipping due to connection error!")
    def test_from_hf_hub(self):
        model = AutoModel.from_pretrained("dfargveazd/tiny-random-llama", from_hf_hub=True, convert_from_torch=False)
        self.assertIsInstance(model, LlamaModel)

    @unittest.skip("skipping due to connection error!")
    def test_from_aistudio(self):
        model = AutoModel.from_pretrained("test_paddleformers/tiny-random-llama", from_aistudio=True)
        self.assertIsInstance(model, LlamaModel)

    @unittest.skip("skipping due to connection error!")
    def test_from_modelscope(self):
        model = AutoModel.from_pretrained("sqlhuman/tiny-random-llama", from_modelscope=True)
        self.assertIsInstance(model, LlamaModel)

    def test_new_model_registration(self):
        AutoConfig.register("custom", CustomConfig)

        auto_classes = [
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForPretraining,
            AutoModelForQuestionAnswering,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        ]

        try:
            for auto_class in auto_classes:
                with self.subTest(auto_class.__name__):
                    # Wrong config class will raise an error
                    with self.assertRaises(ValueError):
                        auto_class.register(BertConfig, CustomModel)
                    auto_class.register(CustomConfig, CustomModel)
                    # Trying to register something existing in the Transformers library will raise an error
                    with self.assertRaises(ValueError):
                        auto_class.register(BertConfig, BertModel)

                    # Now that the config is registered, it can be used as any other config with the auto-API
                    tiny_config = BertModelTester(self).get_config()
                    config = CustomConfig(**tiny_config.to_dict())
                    model = auto_class.from_config(config)
                    self.assertIsInstance(model, CustomModel)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        model.save_pretrained(tmp_dir)
                        new_model = auto_class.from_pretrained(tmp_dir)
                        # The model is a CustomModel but from the new dynamically imported class.
                        self.assertIsInstance(new_model, CustomModel)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            for mapping in (MODEL_MAPPING,):
                if CustomConfig in mapping._extra_content:
                    del mapping._extra_content[CustomConfig]
