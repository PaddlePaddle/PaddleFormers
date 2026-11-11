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

import numpy as np
import paddle

from paddleformers.nn.pp_model import EmbeddingPipe, parse_args
from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    MistralConfig,
    MistralForCausalLM,
    MistralForCausalLMPipe,
    MistralForSequenceClassification,
    MistralModel,
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
    def test_pipeline_rotary_embeddings_match_base_model(self):
        for rope_scaling in (None, {"rope_type": "linear", "factor": 2.0}):
            with self.subTest(rope_scaling=rope_scaling):
                config = tiny_mistral_config(rope_scaling=rope_scaling)
                model = MistralModel(config)
                embedding_pipe = EmbeddingPipe(config, rotary_emb_cls=MistralForCausalLMPipe._rotary_emb_cls)
                input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
                position_ids = paddle.to_tensor([[0, 4, 63, 80]], dtype="int64")

                embeddings, _, _, position_embeddings, _ = parse_args(embedding_pipe((input_ids, None, position_ids)))
                expected_cos, expected_sin = model.rotary_emb(embeddings, position_ids)

                np.testing.assert_allclose(
                    position_embeddings[0].numpy(),
                    expected_cos.numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    position_embeddings[1].numpy(),
                    expected_sin.numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_causal_lm_forward(self):
        config = tiny_mistral_config()
        model = MistralForCausalLM(config)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        outputs = model(input_ids=input_ids, return_dict=True)

        self.assertEqual(list(outputs.logits.shape), [2, 8, config.vocab_size])

    def test_causal_lm_forwards_loss_mask_to_criterion(self):
        class CaptureCriterion(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.loss_mask = None

            def forward(self, logits, labels, loss_mask=None):
                self.loss_mask = loss_mask
                return paddle.to_tensor(0.0), None

        config = tiny_mistral_config()
        model = MistralForCausalLM(config)
        criterion = CaptureCriterion()
        model.criterion = criterion
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")
        labels = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")
        loss_mask = paddle.ones([2, 8], dtype="float32")

        outputs = model(input_ids=input_ids, labels=labels, loss_mask=loss_mask, return_dict=True)

        self.assertIsNotNone(outputs.loss)
        self.assertIs(criterion.loss_mask, loss_mask)

    def test_base_model_tuple_output_includes_cache(self):
        config = tiny_mistral_config()
        model = MistralModel(config)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        outputs = model(input_ids=input_ids, use_cache=True, return_dict=False)

        self.assertEqual(list(outputs[0].shape), [2, 8, config.hidden_size])
        self.assertIsNotNone(outputs[1])

    def test_sequence_classification_forward(self):
        config = tiny_mistral_config(num_labels=3, pad_token_id=0)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        seq_model = MistralForSequenceClassification(config)
        seq_outputs = seq_model(input_ids=input_ids, return_dict=True)
        self.assertEqual(list(seq_outputs.logits.shape), [2, config.num_labels])

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

    def test_auto_model_sequence_classification_dispatch(self):
        config = tiny_mistral_config(num_labels=3, pad_token_id=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            MistralForSequenceClassification(config).save_pretrained(tmpdir)
            seq_loaded = AutoModelForSequenceClassification.from_pretrained(tmpdir)
            self.assertIsInstance(seq_loaded, MistralForSequenceClassification)


if __name__ == "__main__":
    unittest.main()
