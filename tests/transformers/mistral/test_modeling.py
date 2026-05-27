# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import tempfile
import unittest

import paddle

from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    MistralConfig,
    MistralForCausalLM,
    MistralForQuestionAnswering,
    MistralModel,
    MistralForSequenceClassification,
    MistralForTokenClassification,
)


def tiny_mistral_config(**kwargs):
    config_kwargs = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        sliding_window=16,
        use_sliding_window=True,
        fuse_rms_norm=False,
    )
    config_kwargs.update(kwargs)
    return MistralConfig(**config_kwargs)


class MistralModelingTest(unittest.TestCase):
    def test_causal_lm_forward(self):
        config = tiny_mistral_config()
        model = MistralForCausalLM(config)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        outputs = model(input_ids=input_ids, return_dict=True)

        self.assertEqual(list(outputs.logits.shape), [2, 8, config.vocab_size])

    def test_sequence_and_token_classification_forward(self):
        config = tiny_mistral_config(num_labels=3, pad_token_id=0)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        seq_model = MistralForSequenceClassification(config)
        seq_outputs = seq_model(input_ids=input_ids, return_dict=True)
        self.assertEqual(list(seq_outputs.logits.shape), [2, config.num_labels])

        token_model = MistralForTokenClassification(config)
        token_outputs = token_model(input_ids=input_ids, return_dict=True)
        self.assertEqual(list(token_outputs.logits.shape), [2, 8, config.num_labels])

    def test_from_pretrained_and_auto_model(self):
        config = tiny_mistral_config()
        model = MistralForCausalLM(config)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            class_loaded = MistralForCausalLM.from_pretrained(tmpdir)
            class_outputs = class_loaded(input_ids=input_ids, return_dict=True)
            self.assertEqual(list(class_outputs.logits.shape), [2, 8, config.vocab_size])

            auto_loaded = AutoModelForCausalLM.from_pretrained(tmpdir)
            self.assertEqual(type(auto_loaded).__name__, "MistralForCausalLM")
            auto_outputs = auto_loaded(input_ids=input_ids, return_dict=True)
            self.assertEqual(list(auto_outputs.logits.shape), [2, 8, config.vocab_size])

    def test_auto_config_with_model_type(self):
        config = tiny_mistral_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            config_path = os.path.join(tmpdir, "config.json")

            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict.pop("architectures", None)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f)

            auto_config = AutoConfig.from_pretrained(tmpdir)

            self.assertEqual(type(auto_config).__name__, "MistralConfig")
            self.assertEqual(auto_config.model_type, "mistral")

    def test_auto_model_base_dispatch(self):
        config = tiny_mistral_config()
        model = MistralForCausalLM(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = AutoModel.from_pretrained(tmpdir)
            self.assertIsInstance(loaded, MistralModel)

    def test_auto_model_task_dispatch(self):
        config = tiny_mistral_config(num_labels=3, pad_token_id=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            MistralForSequenceClassification(config).save_pretrained(tmpdir)
            seq_loaded = AutoModelForSequenceClassification.from_pretrained(tmpdir)
            self.assertIsInstance(seq_loaded, MistralForSequenceClassification)

        with tempfile.TemporaryDirectory() as tmpdir:
            MistralForTokenClassification(config).save_pretrained(tmpdir)
            token_loaded = AutoModelForTokenClassification.from_pretrained(tmpdir)
            self.assertIsInstance(token_loaded, MistralForTokenClassification)

        with tempfile.TemporaryDirectory() as tmpdir:
            MistralForQuestionAnswering(config).save_pretrained(tmpdir)
            qa_loaded = AutoModelForQuestionAnswering.from_pretrained(tmpdir)
            self.assertIsInstance(qa_loaded, MistralForQuestionAnswering)


if __name__ == "__main__":
    unittest.main()
