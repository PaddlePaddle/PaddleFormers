# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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
import json
import os
import tempfile
import unittest.mock as mock

from requests.exceptions import HTTPError

from paddleformers.transformers import BertConfig
from paddleformers.transformers.configuration_utils import PretrainedConfig


class CustomConfig(PretrainedConfig):
    model_type = "custom"

    def __init__(self, attribute=1, **kwargs):
        self.attribute = attribute
        super().__init__(**kwargs)


class NoSuperInitConfig(PretrainedConfig):
    model_type = "custom"

    def __init__(self, attribute=1, **kwargs):
        self.attribute = attribute


config_common_kwargs = {
    "return_dict": False,
    "output_hidden_states": True,
    "output_attentions": True,
    "use_bfloat16": True,
    "tf_legacy_loss": True,
    "pruned_heads": {"a": 1},
    "tie_word_embeddings": False,
    "is_decoder": True,
    "cross_attention_hidden_size": 128,
    "add_cross_attention": True,
    "tie_encoder_decoder": True,
    "max_length": 50,
    "min_length": 3,
    "do_sample": True,
    "early_stopping": True,
    "num_beams": 3,
    "num_beam_groups": 3,
    "diversity_penalty": 0.5,
    "temperature": 2.0,
    "top_k": 10,
    "top_p": 0.7,
    "typical_p": 0.2,
    "repetition_penalty": 0.8,
    "length_penalty": 0.8,
    "no_repeat_ngram_size": 5,
    "encoder_no_repeat_ngram_size": 5,
    "bad_words_ids": [1, 2, 3],
    "num_return_sequences": 3,
    "chunk_size_feed_forward": 5,
    "output_scores": True,
    "return_dict_in_generate": True,
    "forced_bos_token_id": 2,
    "forced_eos_token_id": 3,
    "remove_invalid_values": True,
    "architectures": ["BertModel"],
    "finetuning_task": "translation",
    "id2label": {0: "label"},
    "label2id": {"label": "0"},
    "tokenizer_class": "BertTokenizerFast",
    "prefix": "prefix",
    "bos_token_id": 6,
    "pad_token_id": 7,
    "eos_token_id": 8,
    "sep_token_id": 9,
    "decoder_start_token_id": 10,
    "exponential_decay_length_penalty": (5, 1.01),
    "task_specific_params": {"translation": "some_params"},
    "problem_type": "regression",
}


class ConfigTester(object):
    def __init__(self, parent, config_class=None, has_text_modality=True, common_properties=None, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.has_text_modality = has_text_modality
        self.inputs_dict = kwargs
        self.common_properties = common_properties

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        common_properties = (
            ["hidden_size", "num_attention_heads", "num_hidden_layers"]
            if self.common_properties is None
            else self.common_properties
        )

        # Add common fields for text models
        if self.has_text_modality:
            common_properties.extend(["vocab_size"])

        # Test that config has the common properties as getters
        for prop in common_properties:
            self.parent.assertTrue(hasattr(config, prop), msg=f"`{prop}` does not exist")

        # Test that config has the common properties as setter
        for idx, name in enumerate(common_properties):
            try:
                setattr(config, name, idx)
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

        # Test if config class can be called with Config(prop_name=..)
        for idx, name in enumerate(common_properties):
            try:
                config = self.config_class(**{name: idx})
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

    def create_and_test_config_to_json_string(self):
        config = self.config_class(**self.inputs_dict)
        obj = json.loads(config.to_json_string())
        for key, value in self.inputs_dict.items():
            self.parent.assertEqual(obj[key], value)

    def create_and_test_config_to_json_file(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "config.json")
            config_first.to_json_file(json_file_path)
            config_second = self.config_class.from_json_file(json_file_path)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_from_and_save_pretrained(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config_first.save_pretrained(tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_with_num_classes(self):
        config = self.config_class(**self.inputs_dict, num_classes=5)
        self.parent.assertEqual(len(config.id2label), 5)
        self.parent.assertEqual(len(config.label2id), 5)

        config.num_classes = 3
        self.parent.assertEqual(len(config.id2label), 3)
        self.parent.assertEqual(len(config.label2id), 3)

    def check_config_can_be_init_without_params(self):
        if self.config_class.is_composition:
            return
        config = self.config_class()
        self.parent.assertIsNotNone(config)

    def check_config_arguments_init(self):
        kwargs = copy.deepcopy(config_common_kwargs)
        config = self.config_class(**kwargs)
        wrong_values = []
        for key, value in config_common_kwargs.items():
            if getattr(config, key) != value:
                wrong_values.append((key, getattr(config, key), value))
        if len(wrong_values) > 0:
            errors = "\n".join([f"- {v[0]}: got {v[1]} instead of {v[2]}" for v in wrong_values])
            raise ValueError(f"The following keys were not properly set in the config:\n{errors}")

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()
        self.create_and_test_config_from_and_save_pretrained()
        self.create_and_test_config_with_num_classes()
        self.check_config_can_be_init_without_params()
        self.check_config_arguments_init()

    def create_and_test_config_with_num_labels(self):
        config = self.config_class(**self.inputs_dict, num_labels=5)
        self.parent.assertEqual(len(config.id2label), 5)
        self.parent.assertEqual(len(config.label2id), 5)

        config.num_labels = 3
        self.parent.assertEqual(len(config.id2label), 3)
        self.parent.assertEqual(len(config.label2id), 3)


class ConfigTestUtils:
    def test_config_common_kwargs_is_complete(self):
        base_config = PretrainedConfig()
        missing_keys = [key for key in base_config.__dict__ if key not in config_common_kwargs]
        # If this part of the test fails, you have arguments to addin config_common_kwargs above.
        self.assertListEqual(
            missing_keys,
            [
                "use_cache",
                "is_encoder_decoder",
                "classifier_dropout",
                "dtype",
                "_name_or_path",
                "paddleformers_version",
            ],
        )
        keys_with_defaults = [key for key, value in config_common_kwargs.items() if value == getattr(base_config, key)]
        if len(keys_with_defaults) > 0:
            raise ValueError(
                "The following keys are set with the default values in"
                " `test_configuration_common.config_common_kwargs` pick another value for them:"
                f" {', '.join(keys_with_defaults)}."
            )

    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = []
        response_mock.raise_for_status.side_effect = HTTPError

        # Download this model to make sure it's in the cache.
        _ = BertConfig.from_pretrained("Paddleformers/tiny-random-bert")

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("transformers.utils.hub.requests.head", return_value=response_mock) as mock_head:
            _ = BertConfig.from_pretrained("Paddleformers/tiny-random-bert")
            # This check we did call the fake head request
            mock_head.assert_called()
