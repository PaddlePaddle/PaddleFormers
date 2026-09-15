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

import copy
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    AyaVisionConfig,
    AyaVisionForConditionalGeneration,
    AyaVisionModel,
    AyaVisionProcessor,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


class AyaVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=12,
        vocab_size=101,
        image_token_index=99,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vision_hidden_size=16,
        vision_intermediate_size=32,
        vision_num_hidden_layers=2,
        vision_num_attention_heads=4,
        image_size=16,
        patch_size=4,
        downsample_factor=2,
        alignment_intermediate_size=64,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.image_token_index = image_token_index
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.alignment_intermediate_size = alignment_intermediate_size
        self.num_channels = 3
        patch_grid = self.image_size // self.patch_size
        self.num_image_tokens = (patch_grid // self.downsample_factor) ** 2

    def get_config(self):
        text_config = {
            "model_type": "cohere2",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.hidden_size // self.num_attention_heads,
            "hidden_act": "silu",
            "max_position_embeddings": 128,
            "model_max_length": 128,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "tie_word_embeddings": True,
            "attention_bias": False,
            "use_qk_norm": False,
            "layer_types": ["full_attention"] * self.num_hidden_layers,
        }
        vision_config = {
            "model_type": "siglip_vision_model",
            "hidden_size": self.vision_hidden_size,
            "intermediate_size": self.vision_intermediate_size,
            "num_hidden_layers": self.vision_num_hidden_layers,
            "num_attention_heads": self.vision_num_attention_heads,
            "num_channels": self.num_channels,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "hidden_act": "gelu_pytorch_tanh",
            "vision_use_head": False,
        }
        return AyaVisionConfig(
            text_config=text_config,
            vision_config=vision_config,
            vision_feature_select_strategy="full",
            vision_feature_layer=-1,
            downsample_factor=self.downsample_factor,
            image_token_index=self.image_token_index,
            alignment_intermediate_size=self.alignment_intermediate_size,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor(
            [self.batch_size, self.num_channels, self.image_size, self.image_size],
        )
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        attention_mask = paddle.ones(input_ids.shape, dtype=paddle.int64)
        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, 0] = self.bos_token_id
        input_ids[:, -1] = self.eos_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_index
        labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        labels[:, : 1 + self.num_image_tokens] = -100
        return config, input_ids, attention_mask, pixel_values, labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, pixel_values, labels = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
        return config, inputs_dict


class AyaVisionModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = AyaVisionModel
    all_model_classes = (AyaVisionModel, AyaVisionForConditionalGeneration)
    all_generative_model_classes = {AyaVisionForConditionalGeneration: {AyaVisionModel, "aya_vision"}}
    max_new_tokens = 3
    test_resize_embeddings = False

    @gpu_device_initializer(log_prefix="AyaVisionModelTest")
    def setUp(self):
        self.model_tester = AyaVisionModelTester(self)

    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class)
        if model_class is AyaVisionModel:
            inputs_dict.pop("labels", None)
        return inputs_dict

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {
            "bad_words_ids": [[self.model_tester.image_token_index]],
            "repetition_penalty": 1.2,
            "remove_invalid_values": True,
        }
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )
        return logits_processor_kwargs

    def _greedy_generate(
        self,
        model,
        inputs_dict,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        return model.generate(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,
            **self._get_logits_processor_kwargs(do_sample=False, config=model.config),
            **inputs_dict,
        )

    def _sample_generate(
        self,
        model,
        inputs_dict,
        num_return_sequences,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        paddle.seed(0)
        return model.generate(
            do_sample=True,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,
            **self._get_logits_processor_kwargs(do_sample=True, config=model.config),
            **inputs_dict,
        )

    def _beam_search_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        return model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,
            **beam_kwargs,
            **self._get_logits_processor_kwargs(do_sample=False, config=model.config),
            **inputs_dict,
        )

    def prepare_config_and_inputs_for_generate(self, batch_size=1):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            key: value[:batch_size, ...] if isinstance(value, paddle.Tensor) else value
            for key, value in inputs_dict.items()
            if key != "labels"
        }
        inputs_dict["logits_to_keep"] = 1
        config.text_config.eos_token_id = None
        config.text_config.forced_eos_token_id = None
        return config, inputs_dict

    def test_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        self.assertEqual(config.model_type, "aya_vision")
        self.assertEqual(config.text_config.model_type, "cohere2")
        self.assertEqual(config.vision_config.model_type, "siglip_vision_model")
        self.assertEqual(config.image_token_index, self.model_tester.image_token_index)
        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            loaded_config = AyaVisionConfig.from_pretrained(tmpdirname)
        self.assertEqual(loaded_config.to_dict(), config.to_dict())

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, save_checkpoint_format="flex_checkpoint")
                hf_loaded_model = model_class.from_pretrained(
                    tmpdirname, convert_from_hf=True, load_checkpoint_format=""
                ).eval()
                flex_loaded_model = model_class.from_pretrained(
                    tmpdirname, load_checkpoint_format="flex_checkpoint"
                ).eval()

                hf_state_dict = hf_loaded_model.state_dict()
                flex_state_dict = flex_loaded_model.state_dict()
                self.assertEqual(set(hf_state_dict.keys()), set(flex_state_dict.keys()))
                for name, tensor in hf_state_dict.items():
                    self.assertEqual(tensor._md5sum(), flex_state_dict[name]._md5sum(), msg=name)

                with paddle.no_grad():
                    expected = hf_loaded_model(**inputs, return_dict=False)[0]
                    actual = flex_loaded_model(**inputs, return_dict=False)[0]
                self.assertLessEqual(float(paddle.max(paddle.abs(expected - actual)).item()), 1e-5)

    def test_model_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = AyaVisionModel(config)
        model.eval()
        outputs = model(**self._prepare_for_class(inputs_dict, AyaVisionModel), return_dict=True)
        self.assertEqual(
            outputs.last_hidden_state.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size],
        )
        self.assertIsNotNone(outputs.image_hidden_states)
        self.assertEqual(
            outputs.image_hidden_states.shape[1] * outputs.image_hidden_states.shape[2],
            self.model_tester.num_image_tokens,
        )

    def test_conditional_generation_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = AyaVisionForConditionalGeneration(config)
        model.eval()
        outputs = model(**inputs_dict, return_dict=True)
        self.assertEqual(
            outputs.logits.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size],
        )
        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.image_hidden_states)

    def test_mismatching_num_image_tokens(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = AyaVisionForConditionalGeneration(config)
        model.eval()
        _ = model(**inputs_dict)

        too_few_tokens = copy.deepcopy(inputs_dict)
        too_few_tokens["input_ids"][:, 1] = self.model_tester.pad_token_id
        with self.assertRaises(ValueError):
            _ = model(**too_few_tokens)

        too_many_tokens = copy.deepcopy(inputs_dict)
        too_many_tokens["input_ids"][:, 1 + self.model_tester.num_image_tokens] = self.model_tester.image_token_index
        with self.assertRaises(ValueError):
            _ = model(**too_many_tokens)

    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)
            self.assertEqual(output_generate[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)
            self.assertEqual(output_generate[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            beam_kwargs, _ = self._get_beam_scorer_and_kwargs(1, 1)
            output_generate = self._beam_search_generate(model=model, inputs_dict=inputs_dict, beam_kwargs=beam_kwargs)
            self.assertEqual(output_generate[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)

    def test_expand_inputs_for_generation_expands_pixel_values_by_image_tiles(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = AyaVisionForConditionalGeneration(config).eval()
        image_tokens_per_tile = self.model_tester.num_image_tokens
        input_ids = inputs_dict["input_ids"].clone()
        input_ids[1, 1 : 1 + image_tokens_per_tile * 2] = self.model_tester.image_token_index
        pixel_values = paddle.arange(
            3 * self.model_tester.num_channels * self.model_tester.image_size * self.model_tester.image_size,
            dtype="float32",
        ).reshape([3, self.model_tester.num_channels, self.model_tester.image_size, self.model_tester.image_size])

        expanded_input_ids, expanded_kwargs = model.expand_inputs_for_generation(
            input_ids,
            expand_size=2,
            attention_mask=inputs_dict["attention_mask"],
            pixel_values=pixel_values,
        )

        self.assertEqual(expanded_input_ids.shape[0], input_ids.shape[0] * 2)
        self.assertEqual(expanded_kwargs["pixel_values"].shape[0], 6)
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][0], pixel_values[0]))
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][1], pixel_values[0]))
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][2:4], pixel_values[1:3]))
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][4:6], pixel_values[1:3]))

    @unittest.skip("AyaVision generation requires input_ids and pixel_values on the first generation step.")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip("Group beam search is not compatible with current VLM implementation")
    def test_group_beam_search_generate(self):
        pass


class AyaVisionProcessorTest(unittest.TestCase):
    class DummyImageProcessor:
        model_input_names = ["pixel_values"]

        def __call__(self, images, **kwargs):
            self.kwargs = kwargs
            return {"pixel_values": [[0.0]] * 3, "num_patches": [1, 2]}

        def get_number_of_image_patches(self, height, width, images_kwargs=None):
            return 2

    class DummyTokenizer:
        init_kwargs = {}
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self):
            self.vocab = {
                "<|IMG_PATCH|>": 99,
                "TILE": 100,
                "TILE_GLOBAL": 101,
                "<|START_OF_IMG|>": 102,
                "<|END_OF_IMG|>": 103,
                "TILE_1": 104,
                "compare": 105,
            }

        def convert_tokens_to_ids(self, tokens):
            if isinstance(tokens, list):
                return [self.convert_tokens_to_ids(token) for token in tokens]
            return self.vocab[tokens]

        def __call__(self, text, **kwargs):
            del kwargs
            input_ids = []
            for sample in text:
                sample_ids = []
                index = 0
                special_tokens = sorted(self.vocab, key=len, reverse=True)
                while index < len(sample):
                    for token in special_tokens:
                        if sample.startswith(token, index):
                            sample_ids.append(self.vocab[token])
                            index += len(token)
                            break
                    else:
                        index += 1
                input_ids.append(sample_ids)
            return {"input_ids": input_ids, "attention_mask": [[1] * len(ids) for ids in input_ids]}

    def get_processor(self):
        processor = object.__new__(AyaVisionProcessor)
        processor.image_processor = self.DummyImageProcessor()
        processor.tokenizer = self.DummyTokenizer()
        processor.chat_template = None
        processor.image_token = "<image>"
        processor.patch_size = 28
        processor.img_size = 364
        processor.start_of_img_token = "<|START_OF_IMG|>"
        processor.end_of_img_token = "<|END_OF_IMG|>"
        processor.img_patch_token = "<|IMG_PATCH|>"
        processor.img_line_break_token = "<|IMG_LINE_BREAK|>"
        processor.tile_token = "TILE"
        processor.tile_global_token = "TILE_GLOBAL"
        processor.image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.img_patch_token)
        processor._set_image_token_ids(max_num_patches=1)
        return processor

    def test_processor_expands_image_placeholders(self):
        processor = self.get_processor()
        image = np.zeros((16, 16, 3), dtype=np.uint8)

        outputs = processor(text="<image> compare <image>", images=[image, image])

        self.assertEqual(outputs["input_ids"][0].count(processor.image_token_id), 169 + 338)
        self.assertNotIn("num_patches", outputs)
        self.assertTrue(processor.image_processor.kwargs["crop_to_patches"])

    def test_processor_returns_mm_token_type_ids_for_all_image_tokens(self):
        processor = self.get_processor()
        image = np.zeros((16, 16, 3), dtype=np.uint8)

        outputs = processor(text="<image> compare <image>", images=[image, image], return_mm_token_type_ids=True)

        self.assertIn("mm_token_type_ids", outputs)
        expected_token_type_ids = [
            1 if token_id in processor.image_token_ids else 0 for token_id in outputs["input_ids"][0]
        ]
        self.assertEqual(outputs["mm_token_type_ids"][0], expected_token_type_ids)
        self.assertIn(0, outputs["mm_token_type_ids"][0])
        self.assertIn(processor.tokenizer.convert_tokens_to_ids(processor.start_of_img_token), outputs["input_ids"][0])
        self.assertIn(processor.tokenizer.convert_tokens_to_ids("TILE_1"), outputs["input_ids"][0])
        self.assertIn(processor.tokenizer.convert_tokens_to_ids(processor.tile_global_token), outputs["input_ids"][0])
        self.assertIn(processor.tokenizer.convert_tokens_to_ids(processor.end_of_img_token), outputs["input_ids"][0])

    def test_processor_mismatching_image_placeholders(self):
        processor = self.get_processor()
        image = np.zeros((16, 16, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            processor(text="no image token", images=[image])


if __name__ == "__main__":
    unittest.main()
