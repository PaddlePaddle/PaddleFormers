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

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import paddle
import paddle.nn.functional as F

from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForConditionalGeneration,
    Ovis2Config,
    Ovis2ForConditionalGeneration,
    Ovis2Model,
    Ovis2VisionConfig,
    Ovis2VisionModel,
)
from tests.transformers.test_configuration_common import ConfigTester


class Ovis2ModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.seq_length = 12
        self.vocab_size = 64
        self.hidden_size = 32
        self.image_size = 16
        self.patch_size = 4
        self.hidden_stride = 2
        self.image_token_id = 1
        self.visual_indicator_token_ids = [2, 3, 4, 5, 6]
        self.image_seq_length = (
            self.image_size // (self.patch_size * self.hidden_stride)
        ) ** 2

    def get_config(self):
        return Ovis2Config(
            text_config={
                "model_type": "qwen2",
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
                "pad_token_id": 0,
                "tie_word_embeddings": True,
            },
            vision_config={
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_channels": 3,
                "hidden_size": self.hidden_size,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "vocab_size": 32,
                "hidden_stride": self.hidden_stride,
                "num_visual_indicator_tokens": len(self.visual_indicator_token_ids),
                "tokenize_function": "softmax",
            },
            image_token_id=self.image_token_id,
            visual_indicator_token_ids=self.visual_indicator_token_ids,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            tie_word_embeddings=True,
        )

    def prepare_inputs(self):
        input_ids = paddle.randint(
            7,
            self.vocab_size,
            shape=[self.batch_size, self.seq_length],
            dtype="int64",
        )
        input_ids[:, 1 : 1 + self.image_seq_length] = self.image_token_id
        input_ids[:, 1 + self.image_seq_length] = self.visual_indicator_token_ids[0]
        attention_mask = paddle.ones_like(input_ids)
        pixel_values = paddle.randn(
            [self.batch_size, 3, self.image_size, self.image_size], dtype="float32"
        )
        return input_ids, attention_mask, pixel_values


class Ovis2ModelTest(unittest.TestCase):
    def setUp(self):
        paddle.seed(13)
        self.model_tester = Ovis2ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Ovis2Config,
            has_text_modality=False,
            common_properties=["hidden_size", "vocab_size"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_round_trip_preserves_nested_config_types(self):
        config = self.model_tester.get_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = AutoConfig.from_pretrained(tmpdir)

        self.assertIsInstance(loaded, Ovis2Config)
        self.assertIsInstance(loaded.vision_config, Ovis2VisionConfig)
        self.assertEqual(loaded.text_config.model_type, "qwen2")
        self.assertEqual(loaded.image_token_id, self.model_tester.image_token_id)

    def test_vision_model_forward(self):
        config = self.model_tester.get_config().vision_config
        model = Ovis2VisionModel(config)
        model.eval()
        pixel_values = paddle.randn([2, 3, config.image_size, config.image_size])

        outputs = model(pixel_values, return_dict=True)

        self.assertEqual(tuple(outputs.last_hidden_state.shape), (2, 4, 128))
        self.assertEqual(tuple(outputs.pooler_output.shape), (2, 4, 27))
        probabilities = outputs.pooler_output.sum(axis=-1)
        self.assertTrue(paddle.allclose(probabilities, paddle.ones_like(probabilities), atol=1e-6))

    def test_multimodal_model_forward(self):
        config = self.model_tester.get_config()
        input_ids, attention_mask, pixel_values = self.model_tester.prepare_inputs()
        model = Ovis2ForConditionalGeneration(config)
        model.eval()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )

        self.assertEqual(
            tuple(outputs.logits.shape),
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size),
        )
        self.assertEqual(
            tuple(outputs.image_hidden_states.shape),
            (
                self.model_tester.batch_size,
                self.model_tester.image_seq_length,
                self.model_tester.hidden_size,
            ),
        )

    def test_text_only_input_ids_and_inputs_embeds_match(self):
        config = self.model_tester.get_config()
        input_ids, attention_mask, _ = self.model_tester.prepare_inputs()
        input_ids = paddle.where(
            input_ids == self.model_tester.image_token_id,
            paddle.full_like(input_ids, 7),
            input_ids,
        )
        model = Ovis2Model(config)
        model.eval()
        inputs_embeds = model.get_input_embeddings()(input_ids)

        from_ids = model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        from_embeds = model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state

        self.assertTrue(paddle.allclose(from_ids, from_embeds, atol=1e-6))

    def test_image_token_count_mismatch_raises(self):
        config = self.model_tester.get_config()
        input_ids, attention_mask, pixel_values = self.model_tester.prepare_inputs()
        input_ids[:, 1] = 7
        model = Ovis2ForConditionalGeneration(config)

        with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )

    def test_exactly_one_text_input_representation_is_required(self):
        config = self.model_tester.get_config()
        input_ids, attention_mask, _ = self.model_tester.prepare_inputs()
        model = Ovis2Model(config)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(attention_mask=attention_mask)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

    def test_loss_uses_pre_shifted_labels(self):
        config = self.model_tester.get_config()
        input_ids, attention_mask, pixel_values = self.model_tester.prepare_inputs()
        labels = paddle.concat(
            [input_ids[:, 1:], paddle.full([self.model_tester.batch_size, 1], -100, dtype="int64")],
            axis=1,
        )
        labels[:, : 1 + self.model_tester.image_seq_length] = -100
        model = Ovis2ForConditionalGeneration(config)
        model.eval()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
        )
        expected = F.cross_entropy(
            outputs.logits.astype("float32").reshape([-1, self.model_tester.vocab_size]),
            labels.reshape([-1]),
            ignore_index=-100,
        )

        self.assertTrue(paddle.allclose(outputs.loss, expected, atol=1e-7))

    def test_backward_reaches_vision_and_language_parameters(self):
        config = self.model_tester.get_config()
        input_ids, attention_mask, pixel_values = self.model_tester.prepare_inputs()
        labels = paddle.randint(
            0,
            self.model_tester.vocab_size,
            shape=[self.model_tester.batch_size, self.model_tester.seq_length],
            dtype="int64",
        )
        labels[:, : 1 + self.model_tester.image_seq_length] = -100
        model = Ovis2ForConditionalGeneration(config)
        model.train()

        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
        ).loss
        loss.backward()

        self.assertIsNotNone(model.model.vision_tower.head_linear.weight.grad)
        self.assertIsNotNone(model.model.language_model.embed_tokens.weight.grad)

    def test_logits_to_keep(self):
        config = self.model_tester.get_config()
        input_ids, attention_mask, pixel_values = self.model_tester.prepare_inputs()
        model = Ovis2ForConditionalGeneration(config)
        model.eval()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            logits_to_keep=3,
            return_dict=True,
        )

        self.assertEqual(
            tuple(outputs.logits.shape),
            (self.model_tester.batch_size, 3, self.model_tester.vocab_size),
        )

    def test_generation_drops_pixel_values_after_prefill(self):
        model = Ovis2ForConditionalGeneration(self.model_tester.get_config())
        pixel_values = paddle.randn([1, 3, self.model_tester.image_size, self.model_tester.image_size])

        with mock.patch(
            "paddleformers.generation.utils.GenerationMixin.prepare_inputs_for_generation",
            return_value={},
        ):
            model_inputs = model.prepare_inputs_for_generation(
                paddle.ones([1, 1], dtype="int64"),
                past_key_values=object(),
                pixel_values=pixel_values,
                use_cache=True,
                is_first_iteration=False,
            )

        self.assertIsNone(model_inputs["pixel_values"])

    def test_embedding_and_lm_head_weights_are_tied(self):
        model = Ovis2ForConditionalGeneration(self.model_tester.get_config())

        self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)

    def test_auto_model_registration(self):
        config = self.model_tester.get_config()

        base_model = AutoModel.from_config(config)
        conditional_model = AutoModelForConditionalGeneration.from_config(config)

        self.assertIsInstance(base_model, Ovis2Model)
        self.assertIsInstance(conditional_model, Ovis2ForConditionalGeneration)


if __name__ == "__main__":
    unittest.main()
