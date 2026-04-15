# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import paddle
from parameterized import parameterized

from paddleformers.transformers import (
    AutoProcessor,
    Glm4vMoeConfig,
    Glm4vMoeForConditionalGeneration,
    Glm4vMoeModel,
    process_vision_info,
)
from paddleformers.transformers.video_utils import load_video
from paddleformers.utils.log import logger
from tests.testing_utils import gpu_device_initializer, require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


class Glm4vMoeModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        num_channels=3,
        # config
        text_config=None,
        vision_config=None,
        image_token_id=4,
        video_token_id=5,
        image_start_token_id=6,
        image_end_token_id=151340,
        video_start_token_id=3,
        video_end_token_id=151342,
        # text/video config
        vocab_size=151552,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=65536,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=1,
        moe_intermediate_size=32,
        n_group=1,
        n_routed_experts=2,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_experts_per_tok=1,
        partial_rotary_factor=0.5,
        routed_scaling_factor=1.0,
        router_aux_loss_coef=0.0001,
        topk_group=1,
        use_qk_norm=False,
    ):
        self.parent: Glm4vMoeModelTest = parent
        self.batch_size = batch_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.num_channels = num_channels
        # config
        self.image_token_id = image_token_id
        self.video_token_id = (video_token_id,)
        self.image_start_token_id = (image_start_token_id,)
        self.image_end_token_id = (image_end_token_id,)
        self.video_start_token_id = (video_start_token_id,)
        self.video_end_token_id = (video_end_token_id,)
        # text/video config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Default vision config is None to avoid a mutable default argument
        if vision_config is None:
            vision_config = {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "depth": 2,
                "hidden_act": "silu",
                "hidden_size": 16,
                "image_size": 32,
                "in_channels": 3,
                "initializer_range": 0.02,
                "intermediate_size": 10944,
                "model_type": "glm4v_moe",
                "num_heads": 2,
                "out_hidden_size": 32,
                "patch_size": 4,
                "rms_norm_eps": 1e-05,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            }
        self.vision_config = vision_config
        if text_config is None:
            self.text_config = {
                "attention_bias": attention_bias,
                "attention_dropout": attention_dropout,
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
                "first_k_dense_replace": first_k_dense_replace,
                "vocab_size": vocab_size,
                "head_dim": head_dim,
                "hidden_act": hidden_act,
                "hidden_size": hidden_size,
                "image_end_token_id": image_end_token_id,
                "image_start_token_id": image_start_token_id,
                "image_token_id": image_token_id,
                "initializer_range": initializer_range,
                "intermediate_size": intermediate_size,
                "max_position_embeddings": max_position_embeddings,
                "model_type": "Glm4vMoe_text",
                "moe_intermediate_size": moe_intermediate_size,
                "n_group": n_group,
                "n_routed_experts": n_routed_experts,
                "n_shared_experts": n_shared_experts,
                "norm_topk_prob": norm_topk_prob,
                "num_attention_heads": num_attention_heads,
                "num_experts_per_tok": num_experts_per_tok,
                "num_hidden_layers": num_hidden_layers,
                "num_key_value_heads": num_key_value_heads,
                "partial_rotary_factor": partial_rotary_factor,
                "rms_norm_eps": rms_norm_eps,
                "rope_theta": rope_theta,
                "routed_scaling_factor": routed_scaling_factor,
                "router_aux_loss_coef": router_aux_loss_coef,
                "topk_group": topk_group,
                "use_cache": use_cache,
                "use_qk_norm": use_qk_norm,
                "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
            }

    def get_config(self) -> Glm4vMoeConfig:
        return Glm4vMoeConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        image_size = config.vision_config.image_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        # attention mask
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        return config, input_ids, input_mask, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask, pixel_values = config_and_inputs

        # inputs_dict
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_end_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = paddle.to_tensor(self.video_end_token_id, dtype="int64")
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": paddle.to_tensor([[1, 1, 1]] * self.batch_size),
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }

        config = self.get_config()
        return config, inputs_dict

    def create_and_check_model(self, config: Glm4vMoeConfig, input_ids, input_mask, *args):
        model = Glm4vMoeModel(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config: Glm4vMoeConfig, input_ids, input_mask, *args):
        model = Glm4vMoeModel(config)
        model.eval()
        attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length])
        result_2d = model(input_ids, attention_mask=attn_mask_2d)[0]
        result_no_attention_mask = model(input_ids, attention_mask=None)[0]
        # Assert non-padding tokens have the same logits with different attention_mask shape
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_no_attention_mask[attn_mask_2d]).all())

    def create_and_check_model_past_large_inputs(self, config: Glm4vMoeConfig, input_ids, input_mask, *args):
        model = Glm4vMoeModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True, return_dict=self.return_dict)
        past_key_values = outputs.past_key_values if self.return_dict else outputs[2]

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), self.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = paddle.cat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.cat([input_mask, next_mask], axis=-1)

        outputs = model(
            next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True, return_dict=self.return_dict
        )

        output_from_no_past = outputs[2][0]

        outputs = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=self.return_dict,
        )

        output_from_past = outputs[2][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
        model = Glm4vMoeForConditionalGeneration(config)
        model.eval()

        result = model(
            input_ids,
            use_cache=True,
            labels=None,
            return_dict=self.parent.return_dict,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, input_mask, *args):
        model = Glm4vMoeForConditionalGeneration(config)
        model.eval()

        result_no_position_id = model(
            input_ids,
            labels=None,
            return_dict=self.parent.return_dict,
        )
        batch_size, seq_len = input_ids.shape
        position_ids = paddle.arange(seq_len).expand((batch_size, seq_len))
        result_position_id = model(
            input_ids,
            position_ids=position_ids,
            labels=None,
            return_dict=self.parent.return_dict,
        )
        self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())

    def create_and_check_gqa_model(self, config, input_ids, input_mask, *args):
        model = Glm4vMoeForConditionalGeneration(config)
        config.num_key_value_heads = 8  # gqa
        config.apply_rope_fusion = True
        model.eval()

        result = model(
            input_ids,
            use_cache=True,
            labels=None,
            return_dict=self.parent.return_dict,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_tp_failed(self, config, input_ids, input_mask, *args):
        config.text_config.tensor_model_parallel_size = 2

        # check num_key_value_heads
        config.text_config.num_key_value_heads = 1
        with self.parent.assertRaises(AssertionError):
            Glm4vMoeForConditionalGeneration(config)

        # check num_attention_heads
        config.text_config.num_key_value_heads = 4
        config.text_config.num_attention_heads = 1
        with self.parent.assertRaises(AssertionError):
            Glm4vMoeForConditionalGeneration(config)

    def create_and_check_fuse_attn(self, config, input_ids, input_mask, *args):
        model = Glm4vMoeForConditionalGeneration(config)
        model.eval()

        result = model(
            input_ids,
            use_cache=True,
            labels=None,
            return_dict=self.parent.return_dict,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])


class Glm4vMoeModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Glm4vMoeModel
    return_dict = False
    use_labels = False

    all_model_classes = (Glm4vMoeModel, Glm4vMoeForConditionalGeneration)
    all_generative_model_classes = {Glm4vMoeForConditionalGeneration: {Glm4vMoeModel, "glm4v_moe"}}

    @gpu_device_initializer(log_prefix="Glm4vMoeModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = Glm4vMoeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Glm4vMoeConfig, vocab_size=256, hidden_size=24)

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict[self.input_name]
        attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)

        max_batch_size = 2
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:max_batch_size, :sequence_length]
        attention_mask = attention_mask[:max_batch_size, :sequence_length]
        max_length = 3

        return config, input_ids, attention_mask, max_length

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_attention_mask(*config_and_inputs)

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_generate_without_input_ids(self):
        # this requires 4-D attention mask logic, which is not supported yet
        pass

    def test_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_gqa_model(self):
        pass

    def test_attention_outputs(self):
        pass

    def test_beam_search_generate(self):
        pass

    def test_greedy_generate(self):
        pass

    def test_group_beam_search_generate(self):
        pass

    def test_resize_tokens_embeddings(self):
        pass

    def test_sample_generate(self):
        pass

    def test_determinism(self):
        pass

    def test_model_name_list(self):
        pass

    def test_save_load(self):
        pass

    def test_hidden_states_output(self):
        pass

    def test_tp_failed(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_tp_failed(*config_and_inputs)

    def test_fuse_attn(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_fuse_attn(*config_and_inputs)

    def test_generate(self):
        config = self.model_tester.get_config()
        model = Glm4vMoeForConditionalGeneration(config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
        )
        assert output[0].shape == [1, 2]


class Glm4vMoeIntegrationTest(unittest.TestCase):
    base_model_class = Glm4vMoeModel
    test_dtype = "float32"  # "bfloat16"

    model_path = "PaddleFormers/tiny-random-glm4vmoe-bf16"
    image_url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
    video_url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"

    @gpu_device_initializer(log_prefix="Glm4vMoeIntegrationTest")
    def setUp(self):
        pass

    @classmethod
    def setUpClass(self):
        self.model = Glm4vMoeForConditionalGeneration.from_pretrained(
            self.model_path, download_hub="aistudio", convert_from_hf=True, dtype=self.test_dtype
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.image_url,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        self.image, _ = process_vision_info(self.messages)
        self.messages_with_video = [
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
        # load_video with paddlecodec returns [N, C, H, W]; take first 3 frames
        # and spatially downsample H/W dims (dim 2 and 3), keeping all 3 channels
        self.video = load_video(self.video_url, video_backend="paddlecodec")[0][
            :3, :, ::4, ::4
        ]  # Only the first 3 frames for testing

    def test_inference_no_attention(self):
        self.model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = self.model(input_ids)[0]
        expected_shape = [1, 11, 151552]
        self.assertEqual(output.shape, expected_shape)
        expected_slice_bf16 = paddle.to_tensor(
            [
                [
                    [-0.05786133, 0.05249023, 0.05346680],
                    [-0.07421875, 0.10742188, 0.14648438],
                    [-0.17773438, -0.14453125, 0.11621094],
                ]
            ]
        )
        expected_slice_fp32 = paddle.to_tensor(
            [
                [
                    [-0.05845977, 0.05381991, 0.05317536],
                    [-0.07495875, 0.10874964, 0.14657366],
                    [-0.17811675, -0.14373989, 0.11646465],
                ]
            ]
        )
        expected_slice = expected_slice_fp32 if self.test_dtype == "float32" else expected_slice_bf16
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4].cast(paddle.float32), expected_slice, atol=1e-4))

    def test_inference_with_attention(self):
        self.model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = self.model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 151552]
        self.assertEqual(output.shape, expected_shape)
        expected_slice_bf16 = paddle.to_tensor(
            [
                [
                    [-0.00349426, 0.03417969, 0.02490234],
                    [-0.06201172, 0.03320312, 0.11132812],
                    [-0.17480469, -0.21386719, 0.08056641],
                ]
            ]
        )
        expected_slice_fp32 = paddle.to_tensor(
            [
                [
                    [-0.00362905, 0.03416932, 0.02502437],
                    [-0.06238038, 0.03297716, 0.11128731],
                    [-0.17635292, -0.21400933, 0.08005644],
                ]
            ]
        )
        expected_slice = expected_slice_fp32 if self.test_dtype == "float32" else expected_slice_bf16
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4].cast(paddle.float32), expected_slice, atol=1e-4))

    def test_model_logits(self):
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=self.image, return_tensors="pd")

        EXPECTED_INPUT_IDS = paddle.to_tensor(
            [151331, 151333, 151336, 198, 151339, 151363, 151340, 74198, 419, 2168, 13, 151337, 198]
        )
        self.assertTrue(paddle.allclose(EXPECTED_INPUT_IDS, inputs.input_ids[0][:13]))

        EXPECTED_PIXEL_SLICE = paddle.to_tensor(
            [
                [0.52908373, -0.02620377, -0.44642124, -0.11625037],
                [0.40902159, 0.42402935, 0.43903711, 0.48406041],
                [0.48406041, 0.48406041, 0.48406041, 0.48406041],
                [0.48406041, 0.48406041, 0.48406041, 0.48406041],
            ],
        )
        self.assertTrue(paddle.allclose(EXPECTED_PIXEL_SLICE, inputs.pixel_values[:, 60:64], atol=5e-4, rtol=1e-5))

        output = self.model(**inputs)["logits"].astype(paddle.float32)
        EXPECTED_SLICE_BF16 = paddle.to_tensor(
            [
                0.05664062,
                -0.00021839,
                0.10888672,
                -0.18847656,
                -0.07958984,
                0.04907227,
                -0.16015625,
                0.08593750,
                0.04394531,
                0.00448608,
                0.06591797,
                -0.15722656,
                0.06079102,
                -0.07812500,
                -0.09423828,
                -0.10302734,
                -0.12500000,
                -0.04931641,
                0.07373047,
                0.04199219,
                -0.01684570,
                -0.16308594,
                0.21289062,
                0.05541992,
                -0.04956055,
                -0.04248047,
                0.12500000,
                0.03686523,
                0.00512695,
                0.13867188,
            ]
        )
        EXPECTED_SLICE_FP32 = paddle.to_tensor(
            [
                0.05710627,
                -0.00003795,
                0.10857674,
                -0.18881349,
                -0.07993484,
                0.04967834,
                -0.16068102,
                0.08517338,
                0.04399901,
                0.00421128,
                0.06672034,
                -0.15796107,
                0.06112768,
                -0.07810304,
                -0.09384070,
                -0.10282400,
                -0.12504721,
                -0.04882925,
                0.07342925,
                0.04192815,
                -0.01702908,
                -0.16371144,
                0.21237904,
                0.05544173,
                -0.04852030,
                -0.04287795,
                0.12538818,
                0.03697439,
                0.00572346,
                0.13874394,
            ]
        )
        logger.info(f"Output logits slice1:\n{output[0, 0, :30]}")
        EXPECTED_SLICE = EXPECTED_SLICE_FP32 if self.test_dtype == "float32" else EXPECTED_SLICE_BF16
        self.assertTrue(paddle.allclose(output[0, 0, :30], EXPECTED_SLICE, atol=5e-4, rtol=1e-5))

    def test_model_logits_with_video(self):
        text = self.processor.apply_chat_template(self.messages_with_video, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], videos=self.video, return_tensors="pd", do_normalize=False, do_sample_frames=False
        )  # Disable normalize and frame sampling to avoid unit test issues

        output = self.model(**inputs).logits.astype(paddle.float32)
        EXPECTED_SLICE_BF16 = paddle.to_tensor(
            [
                -0.15722656,
                0.19238281,
                -0.04833984,
                0.12207031,
                0.05688477,
                0.01855469,
                -0.08642578,
                0.00415039,
                -0.00364685,
                -0.19824219,
                0.13085938,
                -0.02600098,
                -0.06542969,
                0.09619141,
                -0.04638672,
                0.02661133,
                0.04492188,
                -0.18261719,
                -0.12695312,
                0.09228516,
                -0.09472656,
                0.00127411,
                -0.18847656,
                0.02722168,
                -0.15527344,
                -0.09863281,
                -0.09521484,
                0.15625000,
                0.07812500,
                0.06738281,
            ]
        )
        EXPECTED_SLICE_FP32 = paddle.to_tensor(
            [
                -0.15698256,
                0.19180177,
                -0.04769697,
                0.12118968,
                0.05780649,
                0.01856295,
                -0.08716636,
                0.00408011,
                -0.00296549,
                -0.19909000,
                0.12997383,
                -0.02576618,
                -0.06694337,
                0.09707671,
                -0.04754643,
                0.02669407,
                0.04508439,
                -0.18298836,
                -0.12740049,
                0.09185405,
                -0.09497501,
                0.00080160,
                -0.18773033,
                0.02641932,
                -0.15480673,
                -0.09833588,
                -0.09502551,
                0.15664379,
                0.07882637,
                0.06690788,
            ]
        )
        EXPECTED_SLICE = EXPECTED_SLICE_FP32 if self.test_dtype == "float32" else EXPECTED_SLICE_BF16
        logger.info(f"Output logits slice5:\n{output[0, 10, 50000:50030]}")
        self.assertTrue(paddle.allclose(output[0, 10, 50000:50030], EXPECTED_SLICE, atol=1e-3, rtol=1e-3))


class Glm4vMoeCompatibilityTest(unittest.TestCase):
    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import Glm4vMoeConfig, Glm4vMoeForConditionalGeneration

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        tiny_vision_config = {
            "depth": 4,
            "intermediate_size": 64,
            "hidden_size": 48,
            "out_hidden_size": 32,
            "num_heads": 12,
            "head_dim": 4,
        }
        tiny_rope_scaling = {
            "mrope_section": [1, 1],
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000.0,
            "rope_type": "default",
        }
        config = Glm4vMoeConfig(
            hidden_size=32,
            intermediate_size=344,
            num_hidden_layers=2,
            vision_config=tiny_vision_config,
            image_token_id=151363,
            video_token_id=151364,
            image_start_token_id=151339,
            image_end_token_id=151340,
            video_start_token_id=151341,
            video_end_token_id=151342,
            rope_scaling=tiny_rope_scaling,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )

        input_ids = np.random.randint(0, 200, [1, 20]).astype("int64")
        visual_token_ids = [config.video_end_token_id] + [config.image_token_id] * 2 + [config.video_end_token_id]
        input_ids[:, 10 : 10 + len(visual_token_ids)] = visual_token_ids

        attention_mask = np.ones([1, 20], dtype="int64")
        pixel_values = np.random.randn(2 * 2 * 2, 1176).astype("float32")
        image_grid_thw = np.array([[1, 2 * 2, 2]], dtype="int64")
        cls.inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "attention_mask": attention_mask,
        }

        model = Glm4vMoeForConditionalGeneration(config)
        # lm_head.weight will not be saved although 'tie_word_embeddings' is False
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_Glm4vMoe_converter(self):
        # 1. forward the paddle model
        from paddleformers.transformers import Glm4vMoeForConditionalGeneration

        paddle_inputs = {k: paddle.to_tensor(v) for k, v in self.inputs.items()}
        paddle_model = Glm4vMoeForConditionalGeneration.from_pretrained(
            self.torch_model_path, convert_from_hf=True, dtype="float32"
        ).model
        paddle_model.eval()
        paddle_logit = paddle_model(**paddle_inputs)[0]

        # 2. forward the torch model
        import torch
        from transformers import Glm4vMoeForConditionalGeneration

        torch_inputs = {k: torch.tensor(v) for k, v in self.inputs.items()}
        torch_model = Glm4vMoeForConditionalGeneration.from_pretrained(
            self.torch_model_path, torch_dtype=torch.float32
        ).model
        torch_model.eval()
        torch_logit = torch_model(**torch_inputs)[0]

        # 3. compare the result between paddle and torch
        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @require_package("transformers", "torch")
    def test_Glm4vMoe_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. forward the torch model
            import torch
            from transformers import Glm4vMoeModel

            torch_inputs = {k: torch.tensor(v) for k, v in self.inputs.items()}
            torch_model = Glm4vMoeModel.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(**torch_inputs)[0]

            # 2. forward the paddle model
            from paddleformers.transformers import Glm4vMoeModel

            paddle_inputs = {k: paddle.to_tensor(v) for k, v in self.inputs.items()}
            paddle_model = Glm4vMoeModel.from_pretrained(tempdir, convert_from_hf=True, dtype="float32")
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

    @parameterized.expand([("Glm4vMoeModel",), ("Glm4vMoeForConditionalGeneration",)])
    @require_package("transformers", "torch")
    def test_Glm4vMoe_classes_from_local_dir(self, class_name, pytorch_class_name: str | None = None):
        pytorch_class_name = pytorch_class_name or class_name
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. forward the torch model
            import torch
            import transformers

            torch_inputs = {k: torch.tensor(v) for k, v in self.inputs.items()}
            torch_model_class = getattr(transformers, pytorch_class_name)
            torch_model = torch_model_class.from_pretrained(self.torch_model_path, torch_dtype=torch.float32).eval()
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(**torch_inputs)[0]

            # 3. forward the paddle model
            from paddleformers import transformers

            paddle_model_class = getattr(transformers, class_name)
            paddle_model = paddle_model_class.from_pretrained(tempdir, convert_from_hf=True, dtype="float32")
            paddle_model.eval()

            paddle_inputs = {k: paddle.to_tensor(v) for k, v in self.inputs.items()}
            if class_name == "Glm4vMoeModel":
                paddle_logit = paddle_model(**paddle_inputs)[0]
            else:
                paddle_logit = paddle_model(**paddle_inputs).logits

            # 3. compare the result between paddle and torch
            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            )


if __name__ == "__main__":
    unittest.main()
