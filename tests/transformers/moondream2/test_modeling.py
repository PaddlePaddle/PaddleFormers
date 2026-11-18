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

import paddle

from paddleformers.transformers import (
    Moondream2Config,
    Moondream2ForConditionalGeneration,
    Moondream2RegionConfig,
    Moondream2TextConfig,
    Moondream2VisionConfig,
    Moondream2VisionModel,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import ModelTesterMixin


class Moondream2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=5,
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        crop_size=8,
        patch_size=2,
        num_channels=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.grid_size = crop_size // patch_size
        self.num_image_tokens = self.grid_size**2
        self.image_prefix_length = 1 + self.num_image_tokens
        self.is_training = False

    def get_config(self):
        vision_config = Moondream2VisionConfig(
            hidden_size=24,
            patch_size=self.patch_size,
            num_hidden_layers=1,
            intermediate_size=48,
            num_attention_heads=4,
            projection_dim=self.hidden_size,
            crop_size=self.crop_size,
            num_channels=self.num_channels,
            max_crops=2,
            overlap_margin=1,
            projection_inner_dim=64,
            layer_norm_eps=1e-5,
        )
        text_config = Moondream2TextConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            prefix_attn=self.image_prefix_length,
            rotary_dim=8,
            rope_theta=10000.0,
            layer_norm_eps=1e-5,
        )
        region_config = Moondream2RegionConfig(
            hidden_size=self.hidden_size,
            coordinate_feature_dim=8,
            coordinate_output_dim=16,
            size_feature_dim=8,
            size_output_dim=16,
            intermediate_size=64,
        )
        return Moondream2Config(
            vision_config=vision_config,
            text_config=text_config,
            region_config=region_config,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            use_cache=True,
            _attn_implementation="eager",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = paddle.randint(
            low=3,
            high=self.vocab_size,
            shape=[self.batch_size, self.seq_length],
            dtype="int64",
        )
        attention_mask = paddle.ones([self.batch_size, self.seq_length], dtype="int64")
        pixel_values = paddle.randn(
            [self.batch_size, 1, self.num_channels, self.crop_size, self.crop_size],
            dtype="float32",
        )
        image_sizes = paddle.ones([self.batch_size, 2], dtype="int64")
        crop_counts = paddle.ones([self.batch_size], dtype="int64")
        labels = input_ids.clone()
        labels[:, :2] = -100
        return config, input_ids, attention_mask, pixel_values, image_sizes, crop_counts, labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, _, _, _, _ = self.prepare_config_and_inputs()
        return config, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }


class Moondream2ModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = Moondream2ForConditionalGeneration
    all_model_classes = (Moondream2ForConditionalGeneration,)
    test_resize_embeddings = False
    has_attentions = False

    @gpu_device_initializer(log_prefix="Moondream2ModelTest")
    def setUp(self):
        super().setUp()
        paddle.seed(42)
        self.model_tester = Moondream2ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Moondream2Config,
            has_text_modality=False,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def _get_model_and_inputs(self):
        inputs = self.model_tester.prepare_config_and_inputs()
        config = inputs[0]
        model = Moondream2ForConditionalGeneration(config)
        model.eval()
        return model, inputs

    def test_model_construction(self):
        model, inputs = self._get_model_and_inputs()
        config = inputs[0]
        self.assertIsInstance(model.vision_model, Moondream2VisionModel)
        self.assertEqual(model.get_input_embeddings().weight.shape, [self.model_tester.vocab_size, 32])
        self.assertEqual(model.get_output_embeddings().weight.shape, [32, self.model_tester.vocab_size])
        self.assertIs(model.get_decoder(), model.model)
        self.assertEqual(config.text_config.prefix_attn, self.model_tester.image_prefix_length)

    def test_vision_model_output_shape(self):
        model, inputs = self._get_model_and_inputs()
        pixel_values = inputs[3][:, 0]
        with paddle.no_grad():
            hidden_states = model.vision_model(pixel_values)
        self.assertEqual(
            hidden_states.shape,
            [self.model_tester.batch_size, self.model_tester.num_image_tokens, 24],
        )

    def test_vision_model_rejects_invalid_crop_size(self):
        model, _ = self._get_model_and_inputs()
        invalid_pixels = paddle.randn(
            [self.model_tester.batch_size, 3, self.model_tester.crop_size, self.model_tester.crop_size + 2]
        )
        with self.assertRaisesRegex(ValueError, "expects 8x8 crops"):
            model.vision_model(invalid_pixels)

    def test_text_only_forward_and_gqa(self):
        model, inputs = self._get_model_and_inputs()
        _, input_ids, attention_mask, _, _, _, _ = inputs
        with paddle.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        self.assertEqual(
            outputs.logits.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size],
        )
        attention = model.model.layers[0].attn
        self.assertEqual(attention.num_heads, 4)
        self.assertEqual(attention.num_key_value_heads, 2)

    def test_multimodal_forward_shape(self):
        model, inputs = self._get_model_and_inputs()
        _, input_ids, attention_mask, pixel_values, image_sizes, crop_counts, _ = inputs
        with paddle.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                crop_counts=crop_counts,
                return_dict=True,
            )
        expected_length = self.model_tester.seq_length + self.model_tester.image_prefix_length
        self.assertEqual(
            outputs.logits.shape,
            [self.model_tester.batch_size, expected_length, self.model_tester.vocab_size],
        )

    def test_multimodal_forward_accepts_four_dimensional_collator_mask(self):
        model, inputs = self._get_model_and_inputs()
        _, input_ids, attention_mask, pixel_values, image_sizes, crop_counts, _ = inputs
        mask_4d = attention_mask[:, None, None, :].expand(
            [-1, 1, self.model_tester.seq_length, -1]
        )
        with paddle.no_grad():
            outputs_2d = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                crop_counts=crop_counts,
                return_dict=True,
            )
            outputs_4d = model(
                input_ids=input_ids,
                attention_mask=mask_4d,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                crop_counts=crop_counts,
                return_dict=True,
            )
        self.assertTrue(paddle.allclose(outputs_2d.logits, outputs_4d.logits, atol=1e-6, rtol=1e-6))

    def test_multimodal_loss_ignores_prefix_and_masked_labels(self):
        model, inputs = self._get_model_and_inputs()
        _, input_ids, attention_mask, pixel_values, image_sizes, crop_counts, labels = inputs
        with paddle.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                crop_counts=crop_counts,
                labels=labels,
                return_dict=True,
            )
        self.assertEqual(outputs.loss.shape, [])
        self.assertTrue(bool(paddle.isfinite(outputs.loss).item()))
        self.assertGreater(float(outputs.loss), 0.0)

    def test_return_dict_false(self):
        model, inputs = self._get_model_and_inputs()
        _, input_ids, attention_mask, _, _, _, labels = inputs
        with paddle.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=True,
                return_dict=False,
            )
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, [])
        self.assertEqual(outputs[1].shape[-1], self.model_tester.vocab_size)
        self.assertIsNotNone(outputs[2])

    def test_cache_matches_full_text_decode(self):
        model, inputs = self._get_model_and_inputs()
        _, input_ids, attention_mask, _, _, _, _ = inputs
        prefix_ids = input_ids[:, :-1]
        next_id = input_ids[:, -1:]
        with paddle.no_grad():
            prefix_outputs = model(
                input_ids=prefix_ids,
                attention_mask=attention_mask[:, :-1],
                use_cache=True,
                return_dict=True,
            )
            cached_outputs = model(
                input_ids=next_id,
                attention_mask=attention_mask,
                past_key_values=prefix_outputs.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            full_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        self.assertEqual(cached_outputs.past_key_values.get_seq_length(), self.model_tester.seq_length)
        self.assertTrue(
            paddle.allclose(
                cached_outputs.logits[:, -1],
                full_outputs.logits[:, -1],
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_prepare_inputs_for_generation_drops_image_after_prefill(self):
        model, inputs = self._get_model_and_inputs()
        _, input_ids, attention_mask, pixel_values, image_sizes, crop_counts, _ = inputs
        with paddle.no_grad():
            prefill = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                crop_counts=crop_counts,
                use_cache=True,
                return_dict=True,
            )
        prepared = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=prefill.past_key_values,
            attention_mask=paddle.ones(
                [self.model_tester.batch_size, prefill.past_key_values.get_seq_length() + 1], dtype="int64"
            ),
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            crop_counts=crop_counts,
        )
        self.assertEqual(prepared["input_ids"].shape, [self.model_tester.batch_size, 1])
        self.assertEqual(
            prepared["attention_mask"].shape[-1],
            prefill.past_key_values.get_seq_length() + 1,
        )
        self.assertIsNone(prepared["pixel_values"])
        self.assertIsNone(prepared["image_sizes"])
        self.assertIsNone(prepared["crop_counts"])

    def test_generate_text_only_and_multimodal(self):
        model, inputs = self._get_model_and_inputs()
        _, input_ids, attention_mask, pixel_values, image_sizes, crop_counts, _ = inputs
        generate_kwargs = {
            "max_new_tokens": 2,
            "decode_strategy": "greedy_search",
            "use_cache": True,
        }
        with paddle.no_grad():
            text_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )[0]
            image_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                crop_counts=crop_counts,
                **generate_kwargs,
            )[0]
        self.assertEqual(text_sequences.shape[0], self.model_tester.batch_size)
        self.assertEqual(image_sequences.shape[0], self.model_tester.batch_size)
        self.assertLessEqual(text_sequences.shape[1], self.model_tester.seq_length + 2)
        self.assertLessEqual(image_sequences.shape[1], self.model_tester.seq_length + 2)

    def test_local_from_pretrained_round_trip(self):
        model, inputs = self._get_model_and_inputs()
        _, input_ids, attention_mask, pixel_values, image_sizes, crop_counts, _ = inputs
        with paddle.no_grad():
            expected = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                crop_counts=crop_counts,
                use_cache=False,
                return_dict=True,
            ).logits
        with tempfile.TemporaryDirectory() as tempdir:
            model.save_pretrained(tempdir, save_to_hf=False, save_checkpoint_format="")
            loaded = Moondream2ForConditionalGeneration.from_pretrained(
                tempdir,
                convert_from_hf=False,
                load_checkpoint_format="",
            )
            loaded.eval()
            with paddle.no_grad():
                actual = loaded(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    crop_counts=crop_counts,
                    use_cache=False,
                    return_dict=True,
                ).logits
        self.assertEqual(loaded.config.text_config.hidden_size, self.model_tester.hidden_size)
        self.assertTrue(paddle.allclose(expected, actual, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
