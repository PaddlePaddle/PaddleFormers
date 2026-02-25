# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import gc
import shutil
import tempfile
import unittest
from io import BytesIO

import paddle
import requests
from PIL import Image

from paddleformers.transformers import (
    AutoProcessor,
    PaddleOCRVLConfig,
    PaddleOCRVLForConditionalGeneration,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import ModelTesterMixin, floats_tensor


class PaddleOCRVLModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=10,
        is_training=True,
        head_dim=64,
        hidden_size=128,
        image_token_id=100295,
        intermediate_size=384,
        max_position_embeddings=131072,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-05,
        rope_scaling={
            "mrope_section": [8, 12, 12],
            "rope_type": "default",
            "type": "default",
        },
        rope_theta=500000,
        tie_word_embeddings=False,
        _attn_implementation="eager",
        video_token_id=101307,
        vision_config={
            "hidden_size": 144,
            "image_size": 28,
            "intermediate_size": 269,
            "layer_norm_eps": 1e-06,
            "num_attention_heads": 4,
            "num_channels": 3,
            "num_hidden_layers": 2,
            "pad_token_id": 0,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "tokens_per_second": 2,
        },
        vision_start_token_id=101305,
        vision_end_token_id=101306,
        vocab_size=103424,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.num_image_tokens = 4
        self.seq_length = seq_length + self.num_image_tokens

        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.image_token_id = image_token_id
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self._attn_implementation = _attn_implementation
        self.video_token_id = video_token_id
        self.vision_config = vision_config
        self.vision_start_token_id = vision_start_token_id
        self.vocab_size = vocab_size

    def get_config(self) -> PaddleOCRVLConfig:
        return PaddleOCRVLConfig(
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            image_token_id=self.image_token_id,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            num_key_value_heads=self.num_key_value_heads,
            rms_norm_eps=self.rms_norm_eps,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            tie_word_embeddings=self.tie_word_embeddings,
            _attn_implementation=self._attn_implementation,
            video_token_id=self.video_token_id,
            vision_config=self.vision_config,
            vision_start_token_id=self.vision_start_token_id,
            vocab_size=self.vocab_size,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        num_channels = config.vision_config.num_channels
        # only images are supported for now
        pixel_values_len = self.num_image_tokens * (config.vision_config.spatial_merge_size**2)
        pixel_values = floats_tensor(
            [
                self.batch_size * pixel_values_len,
                num_channels,
                patch_size,
                patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs

        # input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).astype(paddle.int64)
        input_ids = paddle.to_tensor(
            [100273, 2969, 93963, 93919, 101305] + ([100295] * self.num_image_tokens) + [101306, 23, 351, 93951, 8],
            dtype="int64",
        ).expand([self.batch_size, -1])
        labels = paddle.to_tensor(
            [-100, -100, -100, -100, -100] + ([-100] * self.num_image_tokens) + [-100, 23, 351, 93951, 8],
            dtype="int64",
        ).expand([self.batch_size, -1])
        attention_mask = paddle.ones(input_ids.shape, dtype=paddle.int64)
        # attention_mask = paddle.tril(attention_mask)
        position_ids = (
            paddle.arange(self.seq_length, dtype="int32")
            .unsqueeze(1)
            .expand([self.batch_size, -1, 3])
            .transpose([2, 0, 1])
        )

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": paddle.to_tensor([[1, 2, 4], [1, 4, 2]] * self.batch_size),
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class PaddleOCRVLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `PaddleOCRVLForConditionalGeneration`.
    """

    all_model_classes = (PaddleOCRVLForConditionalGeneration,)
    all_generative_model_classes = {
        PaddleOCRVLForConditionalGeneration: {PaddleOCRVLForConditionalGeneration, "paddleocr_vl"}
    }
    max_new_tokens = 3

    @gpu_device_initializer(log_prefix="PaddleOCRVLModelTest")
    def setUp(self):
        self.model_tester = PaddleOCRVLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PaddleOCRVLConfig)

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {
            "bad_words_ids": [[1, 2]],
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
                if token_index is not None and token_index < config.vocab_size:
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

        return config, inputs_dict

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # remove one image but leave the image token in text
            remove_image_grid_thw = curr_input_dict["image_grid_thw"][-1:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][:-1, ...]
            remove_img_length = remove_image_grid_thw.prod(axis=1)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][:-remove_img_length, ...]

            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs
            curr_input_dict = copy.deepcopy(input_dict)
            input_ids = curr_input_dict["input_ids"]
            pixel_values = curr_input_dict["pixel_values"]
            image_grid_thw = curr_input_dict["image_grid_thw"]
            input_ids = paddle.cat([input_ids, input_ids], axis=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

            # two images and two image tokens don't raise an error
            pixel_values = paddle.cat([pixel_values, pixel_values], axis=0)
            image_grid_thw = paddle.cat([image_grid_thw, image_grid_thw], axis=0)
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

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

    @unittest.skip(
        "PaddleOCR-VL uses non-tied weights (tie_word_embeddings=False), so lm_head dimensions are not updated when resize_token_embeddings is called"
    )
    def test_resize_tokens_embeddings(self):
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

    @unittest.skip("TODO: Temporarily skipped")
    def test_save_load_flex_checkpoint(self):
        for model_class in self.all_model_classes:
            tmpdirname = tempfile.mkdtemp()
            try:
                tiny_vision_config = {
                    "hidden_size": 144,
                    "num_attention_heads": 4,
                    "intermediate_size": 269,
                    "num_hidden_layers": 2,
                    "patch_size": 14,
                    "image_size": 28,
                }
                config = PaddleOCRVLConfig(
                    head_dim=64,
                    num_attention_heads=4,
                    hidden_size=128,
                    intermediate_size=384,
                    num_hidden_layers=2,
                    num_key_value_heads=2,
                    pixel_hidden_size=144,
                    rope_scaling={
                        "mrope_section": [8, 12, 12],
                        "rope_type": "default",
                    },
                    vision_config=tiny_vision_config,
                )

                model = model_class(config)
                model.save_pretrained(tmpdirname, save_checkpoint_format="flex_checkpoint")

                model = None
                gc.collect()

                model1 = model_class.from_pretrained(tmpdirname, convert_from_hf=True, load_checkpoint_format="")

                model2 = model_class.from_pretrained(tmpdirname, load_checkpoint_format="flex_checkpoint")

                model_state_1 = model1.state_dict()
                model_state_2 = model2.state_dict()

                for k, v in model_state_1.items():
                    md51 = v._md5sum()
                    md52 = model_state_2[k]._md5sum()
                    assert md51 == md52

            finally:
                shutil.rmtree(tmpdirname, ignore_errors=True)


class PaddleOCRVLIntegrationTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="PaddleOCRVLIntegrationTest")
    def setUp(self):
        self.model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            "PaddleFormers/tiny-random-paddleocr-vl-bf16",
            dtype="float32",
            load_checkpoint_format="flex_checkpoint",
        )

        self.processor = AutoProcessor.from_pretrained("PaddleFormers/tiny-random-paddleocr-vl-bf16")
        image_path = (
            "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/exam_paper_0829/part_0000/img_000040676.png"
        )
        image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]

    def test_model_tiny_logits(self):
        inputs = self.processor.apply_chat_template(
            self.messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd"
        )

        EXPECTED_INPUT_IDS = paddle.to_tensor(
            [
                100273,
                2969,
                93963,
                93919,
                101305,
                100295,
                100295,
                100295,
                100295,
                100295,
                100295,
                100295,
                100295,
                100295,
                100295,
                100295,
                100295,
            ]
        )
        self.assertTrue(paddle.allclose(EXPECTED_INPUT_IDS, inputs.input_ids[0][:17]))

        EXPECTED_PIXEL_SLICE = paddle.to_tensor(
            [
                1.0,
                1.0,
                0.99215686,
                0.96862745,
                1.0,
                1.0,
                0.98431373,
                0.82745099,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.96862745,
                1.0,
                1.0,
                0.99215686,
                0.98431373,
                1.0,
                1.0,
                1.0,
                0.98431373,
                1.0,
                1.0,
                0.97647059,
            ]
        )

        self.assertTrue(
            paddle.allclose(
                EXPECTED_PIXEL_SLICE,
                inputs.pixel_values[420, :, :, :].transpose(1, 2, 0).flatten()[::24],
                atol=5e-4,
                rtol=1e-5,
            )
        )

        output = self.model(**inputs, return_dict=True)["logits"].astype(paddle.float32)
        EXPECTED_SLICE = paddle.to_tensor(
            [
                -1.33352613,
                0.75701588,
                2.13876581,
                0.96507418,
                1.44392681,
                -0.25195584,
                -0.78855759,
                -0.72871935,
                -0.40182495,
                1.30361509,
                1.37958324,
                2.72390699,
                -0.98425049,
                -1.13070810,
                1.62200963,
                1.53091741,
                -0.16910096,
                -3.02286720,
                -0.54723328,
                0.26272354,
                -0.48064074,
                2.56522560,
                -0.56610370,
                1.64531112,
                0.64056832,
                -2.33065510,
                0.93232709,
                0.57917541,
                2.06145167,
                -1.49998081,
            ]
        )
        self.assertTrue(paddle.allclose(EXPECTED_SLICE, output[0, 0, :30], atol=5e-4, rtol=1e-5))
