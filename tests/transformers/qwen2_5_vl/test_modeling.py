# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
from parameterized import parameterized

from paddleformers.transformers import (
    AutoProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    process_vision_info,
)
from paddleformers.transformers.video_utils import load_video
from paddleformers.utils.log import logger
from tests.testing_utils import require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


class Qwen2_5_VLVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=14,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        hidden_act="silu",
        hidden_size=32,
        vocab_size=99,
        intermediate_size=37,
        max_position_embeddings=512,
        max_window_layers=3,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rope_theta=10000,
        tie_word_embeddings=True,
        is_training=True,
        vision_config=None,
        vision_start_token_id=3,
        image_token_id=4,
        video_token_id=5,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.vision_start_token_id = vision_start_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens
        # Default vision config is None to avoid a mutable default argument
        if vision_config is None:
            vision_config = {
                "depth": 2,
                "in_chans": 3,
                "hidden_act": "silu",
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "hidden_size": 32,
                "num_heads": 4,
                "patch_size": 14,
                "spatial_patch_size": 14,
                "spatial_merge_size": 1,
                "temporal_patch_size": 2,
            }
        self.vision_config = vision_config
        self.text_config = {
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "hidden_act": hidden_act,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "max_window_layers": max_window_layers,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_key_value_heads,
            "rope_theta": rope_theta,
            "tie_word_embeddings": tie_word_embeddings,
            "vocab_size": vocab_size,
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 1, 1]},
        }

    def get_config(self):
        return Qwen2_5_VLConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            vision_start_token_id=self.vision_start_token_id,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).astype(paddle.int64)
        attention_mask = paddle.ones(input_ids.shape, dtype=paddle.int64)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": paddle.to_tensor([[1, 1, 1]] * self.batch_size),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class Qwen2_5_VLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen2_5_VLForConditionalGeneration`.
    """

    all_model_classes = (Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration)
    all_generative_model_classes = {Qwen2_5_VLForConditionalGeneration: {Qwen2_5_VLModel, "qwen2_5_vl"}}
    max_new_tokens = 3

    def setUp(self):
        self.model_tester = Qwen2_5_VLVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen2_5_VLConfig, has_text_modality=False)

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {
            "bad_words_ids": [[1, 0]],
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
        if config is not None:
            for key in [
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            ]:
                token_index = getattr(config, key, None)
                if token_index is None and hasattr(self, "model_tester"):
                    token_index = getattr(self.model_tester, key, None)
                if token_index is not None and token_index < config.get_text_config().vocab_size:
                    logits_processor_kwargs["bad_words_ids"].append([token_index])

        return logits_processor_kwargs

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
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,  # Do not truncate the inputs from output sequences
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

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
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
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
            trunc_input=False,  # Do not truncate the inputs from output sequences
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

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
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)
        output_generate = model.generate(
            do_sample=True,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,  # Do not truncate the inputs from output sequences
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        # Prepare inputs and config specifically for VLM models, handling text generation settings
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        filtered_inputs_dict = {
            k: v[:batch_size, ...] if isinstance(v, paddle.Tensor) else v for k, v in inputs_dict.items()
        }

        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_text_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_config_dict = config.to_dict()
        base_config = Qwen2_5_VLConfig(**base_config_dict)

        # Trying to get or set text related attributes happens via text config
        vocab_size = base_config.vocab_size
        text_vocab_size = base_config.text_config.vocab_size
        self.assertEqual(vocab_size, text_vocab_size)

        base_config.vocab_size = 55
        self.assertEqual(base_config.vocab_size, 55)
        self.assertEqual(base_config.text_config.vocab_size, 55)

        # We can still initialize config from old-format json, i.e. flat structure
        text_config_dict = base_config_dict.pop("text_config")
        flat_config_dict = {**text_config_dict, **base_config_dict}
        config_from_flat_dict = Qwen2_5_VLConfig(**flat_config_dict)
        config_from_flat_dict.vocab_size = 78
        self.assertEqual(config_from_flat_dict.vocab_size, 78)
        self.assertEqual(config_from_flat_dict.text_config.vocab_size, 78)

        # Vision config attributes are NOT force-set via vision config
        base_config.patch_size = 8
        self.assertEqual(base_config.patch_size, 8)
        self.assertNotEqual(base_config.vision_config.patch_size, 8)

        # Test for making sure config save and load preserves correct model type
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        self.assertEqual(config.model_type, "qwen2_5_vl")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)

            loaded_config = Qwen2_5_VLConfig.from_pretrained(tmp_dir)
            self.assertEqual(loaded_config.model_type, "qwen2_5_vl")

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # remove one image but leave the image token in text
            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            input_ids = paddle.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

            # two images and two image tokens don't raise an error
            pixel_values = paddle.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = paddle.cat([image_grid_thw, image_grid_thw], dim=0)
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_chans
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)

        F = 4
        patch_H = self.model_tester.image_size // P
        patch_W = self.model_tester.image_size // P
        patch_T = F // T
        patches_per_video = patch_T * patch_H * patch_W
        pixel_values_videos = floats_tensor(
            [
                # first dim: batch_size * num_patches
                B * patches_per_video,
                # second dim: in_channels * temporal_patch_size * patch_size^2
                C * T * (P**2),
            ]
        )
        video_grid_thw = paddle.to_tensor([[patch_T, patch_H, patch_W]] * B)

        # sanity check
        assert pixel_values_videos.shape[0] == video_grid_thw.prod(dim=1).sum().item()

        # Insert video token sequence
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[:, self.model_tester.num_image_tokens] = self.model_tester.video_token_id

        insertion_point = self.model_tester.num_image_tokens

        assert (B * patches_per_video) + insertion_point <= self.model_tester.seq_length
        for b in range(B):
            input_ids[b, insertion_point - 1] = self.model_tester.vision_start_token_id
            input_ids[b, insertion_point : insertion_point + patches_per_video] = self.model_tester.video_token_id

        for model_class in self.all_model_classes:
            second_per_grid_ts = paddle.to_tensor([1.0] * B)
            model = model_class(config)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )
            self.assertIsNotNone(outputs)

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).eval()
            beam_kwargs, _ = self._get_beam_scorer_and_kwargs(1, 1)
            output_generate = self._beam_search_generate(model=model, inputs_dict=inputs_dict, beam_kwargs=beam_kwargs)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])

    @unittest.skip("Group beam search is not compatible with current VLM implementation")
    def test_group_beam_search_generate(self):
        pass

    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).eval()
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])

    def test_save_load_flex_checkpoint(self):
        for model_class in self.all_model_classes:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tiny_vision_config = {
                    "depth": 4,
                    "intermediate_size": 64,
                    "hidden_size": 64,
                    "out_hidden_size": 128,
                    "fullatt_block_indexes": [1],
                }
                config = Qwen2_5_VLConfig(
                    num_hidden_layers=4,
                    intermediate_size=256,
                    hidden_size=128,
                    tie_word_embedding=False,
                    vision_config=tiny_vision_config,
                )
                model = model_class(config)
                model.save_pretrained(tmpdirname, save_checkpoint_format="flex_checkpoint")

                model1 = model_class.from_pretrained(tmpdirname, convert_from_hf=True)

                model2 = model_class.from_pretrained(tmpdirname, load_checkpoint_format="flex_checkpoint")

                model_state_1 = model1.state_dict()
                model_state_2 = model2.state_dict()

                for k, v in model_state_1.items():
                    md51 = v._md5sum()
                    md52 = model_state_2[k]._md5sum()
                    assert md51 == md52


class Qwen2_5_VLIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "PaddleFormers/tiny-random-qwen2.5vl", convert_from_hf=True
        )

        self.processor = AutoProcessor.from_pretrained("PaddleFormers/tiny-random-qwen2.5vl")
        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        self.image, _ = process_vision_info(self.messages)

    def test_model_tiny_logits(self):
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text], images=self.image, return_tensors="pd")

        EXPECTED_INPUT_IDS = paddle.to_tensor(
            [
                151644,
                8948,
                198,
                2610,
                525,
                264,
                10950,
                17847,
                13,
                151645,
                198,
                151644,
                872,
                198,
                151652,
                151655,
                151655,
            ]
        )
        self.assertTrue(paddle.allclose(EXPECTED_INPUT_IDS, inputs.input_ids[0][:17]))

        EXPECTED_PIXEL_SLICE = paddle.to_tensor(
            [
                [-0.80660772, -0.92666984, -1.04673207],
                [-1.01671648, -1.03172433, -1.00170875],
                [-1.04673207, -1.00170875, -0.97169322],
                [-0.85163099, -0.74657661, -0.88164657],
                [-0.98670095, -0.74657661, -0.88164657],
                [-0.71656108, -0.77659214, -0.79159987],
            ],
        )
        self.assertTrue(
            paddle.allclose(EXPECTED_PIXEL_SLICE, inputs.pixel_values[3000:3006, 650:653], atol=5e-4, rtol=1e-5)
        )

        output = self.model(**inputs)["logits"].astype(paddle.float32)
        EXPECTED_SLICE = paddle.to_tensor(
            [
                0.00190364,
                0.03592115,
                -0.04341580,
                0.01197994,
                -0.00921174,
                0.03796646,
                -0.02577838,
                -0.00522765,
                -0.04814868,
                -0.00298906,
                -0.01993857,
                0.00585408,
                -0.07494697,
                0.03194539,
                0.00491890,
                -0.02527998,
                -0.05927889,
                0.00743866,
                0.05651516,
                -0.02403277,
                -0.03493933,
                -0.00822193,
                0.02282976,
                -0.00512971,
                -0.05163056,
                -0.05414432,
                -0.03009898,
                -0.06347255,
                -0.04479685,
                0.0384893,
            ]
        )
        logger.info(f"Output logits slice1:\n{output[0, 0, :30]}")
        self.assertTrue(paddle.allclose(output[0, 0, :30], EXPECTED_SLICE, atol=5e-4, rtol=1e-5))

    def test_model_tiny_logits_batch(self):
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pd")

        output = self.model(**inputs)["logits"].astype(paddle.float32)
        EXPECTED_SLICE = paddle.to_tensor(
            [
                0.00190364,
                0.03592116,
                -0.04341583,
                0.01197995,
                -0.00921174,
                0.03796645,
                -0.02577836,
                -0.00522765,
                -0.04814871,
                -0.00298906,
                -0.01993856,
                0.00585408,
                -0.07494697,
                0.03194538,
                0.00491889,
                -0.02527997,
                -0.05927888,
                0.00743864,
                0.05651516,
                -0.02403277,
                -0.03493932,
                -0.00822193,
                0.02282976,
                -0.00512972,
                -0.05163061,
                -0.05414433,
                -0.03009899,
                -0.06347256,
                -0.04479686,
                0.03848937,
            ]
        )
        logger.info(f"Output logits slice2:\n{output[0, 0, :30]}")
        self.assertTrue(paddle.allclose(output[0, 0, :30], EXPECTED_SLICE, atol=1e-3, rtol=1e-3))
        self.assertTrue(paddle.allclose(output[1, 0, :30], EXPECTED_SLICE, atol=1e-3, rtol=1e-3))

    def test_model_tiny_logits_batch_wo_image(self):
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        messages2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
        ]
        text2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text2], images=[self.image], padding=True, return_tensors="pd")

        output = self.model(**inputs)["logits"].astype(paddle.float32)
        EXPECTED_SLICE_1 = paddle.to_tensor(
            [
                -0.00971607,
                0.05542037,
                0.04857498,
                -0.02091954,
                0.06864400,
                -0.03543113,
                0.01865600,
                0.00045855,
                -0.03756892,
                -0.04998226,
                -0.03923185,
                0.03839054,
                -0.01160016,
                0.06611865,
                -0.01678847,
                0.02508730,
                0.02074671,
                0.03335522,
                -0.01111538,
                -0.01246256,
                0.04928322,
                0.04211595,
                0.06305327,
                -0.05337471,
                -0.03865834,
                0.01014945,
                0.05330236,
                -0.01990593,
                0.11149905,
                -0.01333437,
            ]
        )
        EXPECTED_SLICE_2 = paddle.to_tensor(
            [
                0.02815475,
                -0.01576233,
                0.00450739,
                -0.02225647,
                0.00621553,
                0.03126645,
                0.07959020,
                0.06999812,
                0.04361036,
                -0.02026659,
                0.02590707,
                0.00297066,
                0.04389704,
                -0.00458416,
                0.06226203,
                0.04485300,
                -0.01334734,
                0.01592915,
                -0.00583891,
                0.05962470,
                -0.03245433,
                0.01494119,
                -0.00826432,
                -0.00449095,
                0.06402205,
                -0.00996663,
                0.01484825,
                -0.02389973,
                -0.00386393,
                0.05480766,
            ]
        )
        logger.info(f"Output logits slice3:\n{output[0, 1000, 10000:10030]}")
        logger.info(f"Output logits slice4:\n{output[1, 1000, 10000:10030]}")
        self.assertTrue(paddle.allclose(output[0, 1000, 10000:10030], EXPECTED_SLICE_1, atol=1e-3, rtol=1e-3))
        self.assertTrue(paddle.allclose(output[1, 1000, 10000:10030], EXPECTED_SLICE_2, atol=1e-3, rtol=1e-3))

    def test_model_tiny_logits_with_video(self):
        video_url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"
        messages2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        video = load_video(video_url)[0][:3, ::4, ::4]  # Only the first 3 frames for testing

        inputs = self.processor(
            text=[text], videos=video, return_tensors="pd", do_normalize=False
        )  # Disable normalize to avoid unit test issue

        output = self.model(**inputs)["logits"].astype(paddle.float32)
        EXPECTED_SLICE = paddle.to_tensor(
            [
                -0.02633137,
                -0.04612442,
                0.00469637,
                -0.01361415,
                0.03280038,
                -0.03935526,
                0.07284112,
                0.05291817,
                -0.07852594,
                -0.04179034,
                -0.04760805,
                -0.00471714,
                -0.00523766,
                0.00449952,
                0.06172895,
                -0.02328227,
                0.02816464,
                0.04354570,
                0.01082543,
                0.01850530,
                0.02449479,
                0.04358618,
                -0.02710543,
                0.01480347,
                0.02714857,
                0.02346411,
                0.04314926,
                0.03883061,
                0.10793331,
                -0.06213767,
            ]
        )
        logger.info(f"Output logits slice5:\n{output[0, 150, 10000:10030]}")
        self.assertTrue(paddle.allclose(output[0, 150, 10000:10030], EXPECTED_SLICE, atol=1e-3, rtol=1e-3))


class Qwen2_5_VLCompatibilityTest(unittest.TestCase):
    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        tiny_vision_config = {
            "depth": 4,
            "intermediate_size": 95,
            "hidden_size": 64,
            "out_hidden_size": 128,
            "fullatt_block_indexes": [1, 3],
        }
        tiny_rope_scaling = {"type": "mrope", "mrope_section": [1]}
        config = Qwen2_5_VLConfig(
            hidden_size=64,
            intermediate_size=344,
            num_hidden_layers=2,
            vision_config=tiny_vision_config,
            rope_scaling=tiny_rope_scaling,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
            image_token_id=151655,
        )

        input_ids = np.random.randint(0, 200, [1, 20]).astype("int64")
        visual_token_ids = (
            [config.vision_start_token_id] + [config.image_token_id] * 2 + [config.vision_start_token_id]
        )
        input_ids[:, 10 : 10 + len(visual_token_ids)] = visual_token_ids

        attention_mask = np.ones([1, 20], dtype="int64")
        pixel_values = np.random.randn(2 * 2, 1176).astype("float32")
        image_grid_thw = np.array([[1, 2, 2]], dtype="int64")
        cls.inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "attention_mask": attention_mask,
        }

        model = Qwen2_5_VLForConditionalGeneration(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_Qwen2_5_VL_converter(self):

        # 1. forward the paddle model
        from paddleformers.transformers import Qwen2_5_VLModel

        paddle_inputs = {k: paddle.to_tensor(v) for k, v in self.inputs.items()}
        paddle_model = Qwen2_5_VLModel.from_pretrained(
            self.torch_model_path, convert_from_hf=True, dtype="float32"
        ).eval()
        paddle_logit = paddle_model(**paddle_inputs)[0]

        # 2. forward the torch  model
        import torch
        from transformers import Qwen2_5_VLModel

        torch_inputs = {k: torch.tensor(v) for k, v in self.inputs.items()}
        torch_model = Qwen2_5_VLModel.from_pretrained(self.torch_model_path, torch_dtype=torch.float32).eval()
        torch_logit = torch_model(**torch_inputs)[0]

        # 3. compare the result between paddle and torch
        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @require_package("transformers", "torch")
    def test_Qwen2_5_VL_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. forward the torch  model
            import torch
            from transformers import Qwen2_5_VLModel

            torch_inputs = {k: torch.tensor(v) for k, v in self.inputs.items()}
            torch_model = Qwen2_5_VLModel.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(**torch_inputs)[0]

            # 2. forward the paddle model
            from paddleformers.transformers import Qwen2_5_VLModel

            paddle_inputs = {k: paddle.to_tensor(v) for k, v in self.inputs.items()}
            paddle_model = Qwen2_5_VLModel.from_pretrained(tempdir, convert_from_hf=True, dtype="float32")
            paddle_model.eval()
            paddle_logit = paddle_model(**paddle_inputs)[0]

            # 3. compare the result between paddle and torch
            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            )

    @parameterized.expand([("Qwen2_5_VLModel",), ("Qwen2_5_VLForConditionalGeneration",)])
    @require_package("transformers", "torch")
    def test_Qwen2_5_VL_classes_from_local_dir(self, class_name, pytorch_class_name: str | None = None):
        pytorch_class_name = pytorch_class_name or class_name
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. forward the torch model
            import torch
            import transformers

            torch_inputs = {k: torch.tensor(v) for k, v in self.inputs.items()}
            torch_model_class = getattr(transformers, pytorch_class_name)
            torch_model = torch_model_class.from_pretrained(self.torch_model_path, torch_dtype=torch.float32).eval()

            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(**torch_inputs)[0]

            # 2. forward the paddle model
            from paddleformers import transformers

            paddle_inputs = {k: paddle.to_tensor(v) for k, v in self.inputs.items()}
            paddle_model_class = getattr(transformers, class_name)
            paddle_model = paddle_model_class.from_pretrained(tempdir, convert_from_hf=True, dtype="float32").eval()
            paddle_model_fused = paddle_model_class.from_pretrained(
                tempdir,
                convert_from_hf=True,
                dtype="float32",
                fuse_attention_qkv=True,
                fuse_attention_ffn=True,
                load_checkpoint_format="flex_checkpoint",
            ).eval()

            if class_name == "Qwen2_5_VLModel":
                paddle_logit = paddle_model(**paddle_inputs)[0]
                paddle_fused_logit = paddle_model_fused(**paddle_inputs)[0]
            else:
                paddle_logit = paddle_model(**paddle_inputs)["logits"]
                paddle_fused_logit = paddle_model_fused(**paddle_inputs)["logits"]

            # 3. compare the result between paddle and torch
            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            )

            # 4.compare the result between paddle and paddle_fused
            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    paddle_fused_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            )
