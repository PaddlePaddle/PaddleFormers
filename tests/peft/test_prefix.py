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
from tempfile import TemporaryDirectory

import paddle

from paddleformers.peft.prefix import (
    PrefixConfig,
    PrefixModelForCausalLM,
    llama_postprocess_past_key_value,
)
from paddleformers.transformers import LlamaConfig, LlamaForCausalLM


class TestPrefixModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = LlamaConfig(
            vocab_size=200,
            hidden_size=32,
            intermediate_size=86,
            num_hidden_layers=1,
            num_attention_heads=1,
            dtype="float32",
        )

        cls.model = LlamaForCausalLM(cls.config)
        cls.prefix_config = PrefixConfig(
            num_prefix_tokens=2,
            num_attention_heads=cls.model.config.num_attention_heads,
            num_hidden_layers=cls.model.config.num_hidden_layers,
            hidden_size=cls.model.config.hidden_size,
            prefix_projection_hidden_size=cls.model.config.hidden_size,
            dtype="float32",
        )
        cls.prefix_model = PrefixModelForCausalLM(
            model=cls.model,
            prefix_config=cls.prefix_config,
            postprocess_past_key_value=llama_postprocess_past_key_value,
        )

    def test_prefix_config(self):
        with TemporaryDirectory() as tempdir:
            self.prefix_config.save_pretrained(tempdir)
            loaded_prefix_config = PrefixConfig.from_pretrained(tempdir)
            self.assertEqual(self.prefix_config, loaded_prefix_config)

    def test_prefix_model_save_load(self):
        with TemporaryDirectory() as tempdir:
            input_ids = paddle.randint(100, 200, [1, 20])
            self.prefix_model.eval()
            self.prefix_model.save_pretrained(tempdir)
            loaded_prefix_model = PrefixModelForCausalLM.from_pretrained(
                self.model, tempdir, llama_postprocess_past_key_value
            )
            loaded_prefix_model.eval()

            original_results = self.prefix_model(input_ids)
            loaded_results = loaded_prefix_model(input_ids)

            self.assertIsNotNone(original_results)
            self.assertEqual(original_results[0].shape, [1, 20, self.config.vocab_size])
            self.assertIsNotNone(loaded_results)
            self.assertEqual(loaded_results[0].shape, [1, 20, self.config.vocab_size])
            self.assertTrue(paddle.allclose(original_results[0], loaded_results[0]))

    def test_prefix_model_attention_mask(self):
        inputs = {
            "input_ids": paddle.randint(100, 200, [1, 20]),
            "attention_mask": paddle.ones([1, 20]),
            "position_ids": paddle.arange(20).unsqueeze(0),
        }
        logits_2d = self.prefix_model(**inputs)[0]
        inputs["attention_mask"] = paddle.tril(paddle.ones([1, 20, 20]))
        logits_3d = self.prefix_model(**inputs)[0]
        inputs["attention_mask"] = paddle.tril(paddle.ones([1, 1, 20, 20]))
        logits_4d = self.prefix_model(**inputs)[0]
        self.assertTrue(paddle.allclose(logits_2d, logits_3d))
        self.assertTrue(paddle.allclose(logits_3d, logits_4d))

    def test_prefix_model_generate(self):
        inputs = {
            "input_ids": paddle.randint(100, 200, [1, 20]),
            "attention_mask": paddle.ones([1, 20]),
            "position_ids": paddle.arange(20).unsqueeze(0),
        }
        self.prefix_model.generate(
            **inputs,
            max_length=5,
            decode_strategy="sampling",
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
        )
