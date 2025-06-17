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

import os
import tempfile
import unittest

import paddlenlp
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers.auto.configuration import CONFIG_MAPPING, AutoConfig
from paddlenlp.transformers.auto.tokenizer import TOKENIZER_MAPPING
from paddlenlp.transformers.bert.configuration import BertConfig
from paddlenlp.transformers.bert.tokenizer import BertTokenizer
from paddlenlp.transformers.bert.tokenizer_fast import BertTokenizerFast
from paddlenlp.utils.env import TOKENIZER_CONFIG_NAME

from ...utils.test_module.custom_configuration import CustomConfig
from ...utils.test_module.custom_tokenizer import CustomTokenizer
from ...utils.test_module.custom_tokenizer_fast import (
    CustomTokenizerFast,
    CustomTokenizerFastWithoutSlow,
)


class AutoTokenizerTest(unittest.TestCase):
    @unittest.skip("skipping due to connection error!")
    def test_from_aistudio(self):
        tokenizer = AutoTokenizer.from_pretrained("PaddleNLP/tiny-random-bert", from_aistudio=True)
        self.assertIsInstance(tokenizer, paddlenlp.transformers.BertTokenizer)

    def test_from_pretrained_cache_dir(self):
        model_name = "__internal_testing__/tiny-random-bert"
        with tempfile.TemporaryDirectory() as tempdir:
            AutoTokenizer.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, TOKENIZER_CONFIG_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))

    def test_from_pretrained_tokenizer_fast(self):
        tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2", use_fast=True)
        self.assertIsInstance(tokenizer, BertTokenizerFast)

    def test_new_tokenizer_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)

            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            # Trying to register something existing in the PaddleNLP library will raise an error
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, slow_tokenizer_class=BertTokenizer)

            tokenizer = CustomTokenizer.from_pretrained("julien-c/bert-xsmall-dummy")
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir)

                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
                self.assertIsInstance(new_tokenizer, CustomTokenizer)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]

    def test_new_tokenizer_fast_registration(self):
        try:
            # Trying to register nothing
            with self.assertRaises(ValueError):
                AutoTokenizer.register(CustomConfig)
            # Trying to register tokenizer with wrong type
            with self.assertRaises(ValueError):
                AutoTokenizer.register(CustomConfig, fast_tokenizer_class=CustomTokenizer)
            with self.assertRaises(ValueError):
                AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizerFast)
            with self.assertRaises(ValueError):
                AutoTokenizer.register(
                    CustomConfig,
                    slow_tokenizer_class=CustomTokenizer,
                    fast_tokenizer_class=CustomTokenizerFastWithoutSlow,
                )
            AutoConfig.register("custom", CustomConfig)

            # Can register in two steps
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, None))
            AutoTokenizer.register(CustomConfig, fast_tokenizer_class=CustomTokenizerFast)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, CustomTokenizerFast))

            del TOKENIZER_MAPPING._extra_content[CustomConfig]
            # Can register in one step
            AutoTokenizer.register(
                CustomConfig, slow_tokenizer_class=CustomTokenizer, fast_tokenizer_class=CustomTokenizerFast
            )
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, CustomTokenizerFast))

            # Trying to register something existing in the PaddleNLP library will raise an error
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, fast_tokenizer_class=BertTokenizerFast)
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, slow_tokenizer_class=BertTokenizer)

            # We pass through a llama tokenizer fast cause there is no converter slow to fast for our new toknizer
            # and that model does not have a tokenizer.json
            with tempfile.TemporaryDirectory() as tmp_dir:
                llama_tokenizer = BertTokenizerFast.from_pretrained("julien-c/bert-xsmall-dummy", from_hf_hub=True)
                llama_tokenizer.save_pretrained(tmp_dir)
                tokenizer = CustomTokenizerFast.from_pretrained(tmp_dir)

            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir, legacy_format=True)

                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, use_fast=True)
                self.assertIsInstance(new_tokenizer, CustomTokenizerFast)

                # TODO: fix this test. Now keep loaded tokenizer type
                # new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, use_fast=False)
                # self.assertIsInstance(new_tokenizer, CustomTokenizer)
        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
