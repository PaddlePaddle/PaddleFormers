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
import unittest
from io import BytesIO

import paddle
import requests
from PIL import Image

from paddleformers.transformers import (
    AutoProcessor,
    GlmOcrConfig,
    GlmOcrForConditionalGeneration,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import ModelTesterMixin, floats_tensor


class GlmOcrModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=10,
        is_training=True,
        # ---------- text config ----------
        # hidden_size / num_attention_heads determines head_dim = hidden_size // num_attention_heads
        # GlmOcrTextConfig does not have an independent head_dim parameter
        hidden_size=128,
        intermediate_size=384,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        vocab_size=65024,
        pad_token_id=0,
        image_token_id=59280,
        image_start_token_id=59256,
        image_end_token_id=59257,
        video_start_token_id=59258,
        video_end_token_id=59259,
        video_token_id=59281,
        _attn_implementation="eager",
        # ---------- vision config ----------
        # spatial_merge_size=2 → num_image_tokens per image = spatial_merge_size^2 = 4
        vision_hidden_size=128,
        vision_out_hidden_size=128,
        vision_num_heads=4,
        vision_depth=2,
        vision_patch_size=14,
        vision_temporal_patch_size=2,
        vision_in_channels=3,
        vision_spatial_merge_size=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training

        self.vision_spatial_merge_size = vision_spatial_merge_size
        # Number of image tokens per image after expansion = spatial_merge_size^2
        self.num_image_tokens = vision_spatial_merge_size**2
        self.seq_length = seq_length + self.num_image_tokens

        # text
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_token_id = video_token_id
        self._attn_implementation = _attn_implementation

        # vision
        self.vision_hidden_size = vision_hidden_size
        self.vision_out_hidden_size = vision_out_hidden_size
        self.vision_num_heads = vision_num_heads
        self.vision_depth = vision_depth
        self.vision_patch_size = vision_patch_size
        self.vision_temporal_patch_size = vision_temporal_patch_size
        self.vision_in_channels = vision_in_channels

    def get_config(self) -> GlmOcrConfig:
        from paddleformers.transformers import GlmOcrTextConfig, GlmOcrVisionConfig

        # GlmOcrTextConfig accepts rope_theta directly as a parameter,
        # standardize_rope_params writes it to rope_parameters["rope_theta"]
        text_config = GlmOcrTextConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            mrope_section=[8, 12, 12],
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
        )

        vision_config = GlmOcrVisionConfig(
            hidden_size=self.vision_hidden_size,
            out_hidden_size=self.vision_out_hidden_size,
            num_heads=self.vision_num_heads,
            depth=self.vision_depth,
            patch_size=self.vision_patch_size,
            temporal_patch_size=self.vision_temporal_patch_size,
            in_channels=self.vision_in_channels,
            spatial_merge_size=self.vision_spatial_merge_size,
            hidden_act="silu",
            attention_bias=True,
            attention_dropout=0.0,
            rms_norm_eps=1e-6,
            # intermediate_size controls vision MLP, keep it in the same order of magnitude as hidden_size
            intermediate_size=self.vision_hidden_size * 3,
        )

        return GlmOcrConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=self.image_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
            video_token_id=self.video_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        in_channels = config.vision_config.in_channels
        temporal_patch_size = config.vision_config.temporal_patch_size

        # In GlmOcrVisionPatchEmbed.forward:
        #   hidden_states.reshape([-1, C, temporal_patch_size, patch_size, patch_size])
        # Therefore the 0th dimension of pixel_values is total_raw_patches,
        # and the number of patches per image = t * h * w (prod of grid_thw).
        #
        # We use 1 image with grid_thw = [1, 2, 2]:
        #   t=1, h=2, w=2 → raw patches = 1*2*2 = 4
        # After downsample (stride=spatial_merge_size=2):
        #   merged tokens = h/2 * w/2 * t = 1*1*1 = 1  ← too few, use [1, 4, 4] instead
        #
        # To get num_image_tokens = spatial_merge_size^2 = 4, use grid_thw = [1, 4, 4]:
        #   raw patches = 1*4*4 = 16
        #   merged tokens = (4/2) * (4/2) * 1 = 4 ✓
        #
        # pixel_values shape: [batch_size * 16, C, temporal_patch_size, patch_size, patch_size]
        self.grid_thw = [1, 4, 4]  # t, h, w (for prepare_config_and_inputs_for_common)
        t, h, w = self.grid_thw
        total_raw_patches = t * h * w  # = 16
        pixel_values = floats_tensor(
            [
                self.batch_size * total_raw_patches,
                in_channels,
                temporal_patch_size,
                patch_size,
                patch_size,
            ]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()

        # image_token_id uses the value from config (default 59280)
        # prefix token ids must be < vocab_size and not conflict with special tokens
        prefix = [1, 100, 101, 102]
        # Each image expands to num_image_tokens image tokens
        image_tokens = [self.image_token_id] * self.num_image_tokens  # 4 tokens
        suffix = [200, 201, 202, 203, 204]

        ids_list = prefix + image_tokens + suffix
        input_ids = paddle.to_tensor(ids_list, dtype="int64").expand([self.batch_size, -1])

        # labels: mask image region and prefix with -100, only compute loss on suffix
        labels_list = ([-100] * (len(prefix) + self.num_image_tokens)) + suffix
        labels = paddle.to_tensor(labels_list, dtype="int64").expand([self.batch_size, -1])

        attention_mask = paddle.ones(input_ids.shape, dtype="int64")

        # image_grid_thw: [batch_size * num_images_per_sample, 3]

        t, h, w = self.grid_thw
        image_grid_thw = paddle.to_tensor([[t, h, w]] * self.batch_size, dtype="int64")

        inputs_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        return config, inputs_dict


class GlmOcrModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `GlmOcrForConditionalGeneration`.
    """

    all_model_classes = (GlmOcrForConditionalGeneration,)
    all_generative_model_classes = {GlmOcrForConditionalGeneration: {GlmOcrForConditionalGeneration, "glmocr"}}
    max_new_tokens = 3

    @gpu_device_initializer(log_prefix="GlmOcrModelTest")
    def setUp(self):
        self.model_tester = GlmOcrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GlmOcrConfig)

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
            for key in ["image_token_id", "video_start_token_id", "video_end_token_id"]:
                token_index = getattr(config, key, None)
                if token_index is None and hasattr(self, "model_tester"):
                    token_index = getattr(self.model_tester, key, None)
                if token_index is not None and token_index < config.vocab_size:
                    logits_processor_kwargs["bad_words_ids"].append([token_index])

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
            trunc_input=False,
            **logits_processor_kwargs,
            **inputs_dict,
        )
        return output_generate

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
            trunc_input=False,
            **beam_kwargs,
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
            trunc_input=False,
            **logits_processor_kwargs,
            **inputs_dict,
        )
        return output_generate

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        return config, inputs_dict

    # ------------------------------------------------------------------ #
    #  Forward / Loss                                                      #
    # ------------------------------------------------------------------ #

    def test_forward_pass(self):
        """Basic forward pass, verify logits shape is correct."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            outputs = model(**inputs_dict)
            bsz, seq_len = inputs_dict["input_ids"].shape
            self.assertEqual(
                outputs.logits.shape,
                [bsz, seq_len, config.vocab_size],
            )

    def test_forward_with_labels(self):
        """When labels are provided, should return non-None loss."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            outputs = model(**inputs_dict)
            self.assertIsNotNone(outputs.loss)
            self.assertEqual(outputs.loss.shape, [])  # scalar

    def test_forward_without_images(self):
        """Pure text input (no pixel_values / image_grid_thw) should not raise error."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        text_only = {k: v for k, v in inputs_dict.items() if k not in ("pixel_values", "image_grid_thw")}
        # Replace image tokens with padding so shapes stay consistent
        input_ids = text_only["input_ids"].clone()
        image_tok = self.model_tester.image_token_id
        pad_id = config.text_config.pad_token_id or 0
        input_ids[input_ids == image_tok] = pad_id
        text_only["input_ids"] = input_ids
        text_only.pop("labels", None)

        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            outputs = model(**text_only)
            self.assertIsNotNone(outputs.logits)

    # ------------------------------------------------------------------ #
    #  Image-token mismatch validation                                    #
    # ------------------------------------------------------------------ #

    def test_mismatching_num_image_tokens(self):
        """Should raise ValueError when pixel_values and image_grid_thw describe mismatched token counts."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            _ = model(**inputs_dict)  # baseline: no error

            # Remove image_grid_thw but keep image tokens in input_ids → count mismatch → error
            curr = copy.deepcopy(inputs_dict)
            curr["image_grid_thw"] = curr["image_grid_thw"][:0]  # empty
            curr["pixel_values"] = curr["pixel_values"][:0]

            with self.assertRaises(ValueError):
                _ = model(**curr)

    # ------------------------------------------------------------------ #
    #  Generate                                                           #
    # ------------------------------------------------------------------ #

    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])

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

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])

    # ------------------------------------------------------------------ #
    #  KV Cache                                                           #
    # ------------------------------------------------------------------ #

    def test_use_cache_consistency(self):
        """Logits with use_cache=True and use_cache=False should be consistent (prefill stage)."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        no_label_inputs = {k: v for k, v in inputs_dict.items() if k != "labels"}

        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            with paddle.no_grad():
                out_cache = model(**no_label_inputs, use_cache=True)
                out_no_cache = model(**no_label_inputs, use_cache=False)

            self.assertTrue(
                paddle.allclose(
                    out_cache.logits.astype("float32"),
                    out_no_cache.logits.astype("float32"),
                    atol=1e-4,
                ),
                "Logits with and without KV cache should match on prefill.",
            )

    def test_past_key_values_not_none_with_cache(self):
        """past_key_values should not be None when use_cache=True."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        no_label_inputs = {k: v for k, v in inputs_dict.items() if k != "labels"}

        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            with paddle.no_grad():
                outputs = model(**no_label_inputs, use_cache=True)
            self.assertIsNotNone(outputs.past_key_values)

    # ------------------------------------------------------------------ #
    #  rope_deltas                                                        #
    # ------------------------------------------------------------------ #

    def test_rope_deltas_not_none_after_forward(self):
        """After forward pass, rope_deltas should be computed and cached to model.model.rope_deltas."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        no_label_inputs = {k: v for k, v in inputs_dict.items() if k != "labels"}

        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            with paddle.no_grad():
                outputs = model(**no_label_inputs)
            self.assertIsNotNone(outputs.rope_deltas)

    # ------------------------------------------------------------------ #
    #  prepare_inputs_for_generation                                      #
    # ------------------------------------------------------------------ #

    def test_prepare_inputs_for_generation_prefill(self):
        """In prefill stage (past_key_values=None), pixel_values should be preserved."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GlmOcrForConditionalGeneration(config).eval()

        model_inputs = model.prepare_inputs_for_generation(
            input_ids=inputs_dict["input_ids"],
            past_key_values=None,
            attention_mask=inputs_dict["attention_mask"],
            pixel_values=inputs_dict["pixel_values"],
            image_grid_thw=inputs_dict["image_grid_thw"],
        )
        self.assertIsNotNone(model_inputs["pixel_values"])
        self.assertIsNotNone(model_inputs["image_grid_thw"])

    def test_prepare_inputs_for_generation_decode(self):
        """In decode stage (past_key_values is not None), pixel_values should be cleared."""
        from paddleformers.transformers.cache_utils import DynamicCache

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = GlmOcrForConditionalGeneration(config).eval()

        dummy_cache = DynamicCache(config=config.text_config)

        model_inputs = model.prepare_inputs_for_generation(
            input_ids=inputs_dict["input_ids"],
            past_key_values=dummy_cache,
            attention_mask=inputs_dict["attention_mask"],
            pixel_values=inputs_dict["pixel_values"],
            image_grid_thw=inputs_dict["image_grid_thw"],
        )
        self.assertIsNone(model_inputs["pixel_values"])
        self.assertIsNone(model_inputs["image_grid_thw"])

    # ------------------------------------------------------------------ #
    #  return_dict=False                                                  #
    # ------------------------------------------------------------------ #

    def test_return_dict_false(self):
        """When return_dict=False, should return a tuple with logits as the first element."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        no_label_inputs = {k: v for k, v in inputs_dict.items() if k != "labels"}

        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            outputs = model(**no_label_inputs, return_dict=False)
            self.assertIsInstance(outputs, tuple)
            # first element should be logits tensor
            self.assertIsInstance(outputs[0], paddle.Tensor)

    # ------------------------------------------------------------------ #
    #  Skipped tests                                                      #
    # ------------------------------------------------------------------ #

    @unittest.skip("Group beam search is not compatible with current GlmOcr implementation")
    def test_group_beam_search_generate(self):
        pass

    @unittest.skip(
        "GlmOcr uses non-tied weights (tie_word_embeddings=False), "
        "lm_head dimensions are not updated when resize_token_embeddings is called"
    )
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip("GlmOcr currently does not support flex checkpoint save and load")
    def test_save_load_flex_checkpoint(self):
        pass

    @unittest.skip("GlmOcr currently does not support checkpoint save and load")
    def test_save_load(self):
        pass


# ------------------------------------------------------------------ #
#  Integration test                                                    #
# ------------------------------------------------------------------ #


class GlmOcrIntegrationTest(unittest.TestCase):
    """End-to-end test using a tiny pretrained checkpoint."""

    @gpu_device_initializer(log_prefix="GlmOcrIntegrationTest")
    def setUp(self):
        pass

    @classmethod
    def setUpClass(self):
        # NOTE: replace with actual tiny checkpoint path once available
        self.model = GlmOcrForConditionalGeneration.from_pretrained(
            "PaddleFormers/tiny-random-glmocr",
            dtype="float32",
            load_checkpoint_format="flex_checkpoint",
        )
        self.processor = AutoProcessor.from_pretrained("PaddleFormers/tiny-random-glmocr")

        image_path = (
            "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/exam_paper_0829/part_0000/img_000040676.png"
        )
        image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]

    def test_model_tiny_logits(self):
        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pd",
        )

        # -------------------------
        # 1) Hard-coded input_ids prefix (17)
        # -------------------------
        EXPECTED_INPUT_IDS = paddle.to_tensor(
            [
                59248,
                59250,
                59253,
                10,
                59280,
                59280,
                59280,
                59280,
                59280,
                59280,
                59280,
                59280,
                59280,
                59280,
                59280,
                59280,
                59280,
            ],
            dtype="int64",
        )
        self.assertTrue(
            paddle.allclose(EXPECTED_INPUT_IDS, inputs["input_ids"][0][:17]),
            msg=f"input_ids prefix mismatch. got={inputs['input_ids'][0][:17].cpu().numpy().tolist()}",
        )

        # -------------------------
        # 2) Hard-coded pixel_values slice: pixel_values[0,:25]
        # pixel_values is 2D: [256, 1176]
        # -------------------------
        EXPECTED_PIXEL_SLICE = paddle.to_tensor(
            [
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.901139497756958,
                1.9303361177444458,
                1.9303361177444458,
                1.8427457809448242,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
                1.9157379865646362,
                1.9303361177444458,
                1.9303361177444458,
                1.9303361177444458,
            ],
            dtype="float32",
        )
        self.assertTrue(
            paddle.allclose(
                EXPECTED_PIXEL_SLICE,
                inputs["pixel_values"][0, :25].astype("float32"),
                atol=5e-4,
                rtol=1e-5,
            ),
            msg="pixel_values slice mismatch on [0, :25].",
        )

        # -------------------------
        # 3) Forward logits hard-coded slice: logits[0,0,:30]
        # -------------------------
        with paddle.no_grad():
            output = self.model(**inputs, return_dict=True)

        logits = output.logits.astype("float32")
        EXPECTED_LOGITS_SLICE = paddle.to_tensor(
            [
                0.0739683136343956,
                -0.010490708984434605,
                -0.01828090474009514,
                -0.1487439125776291,
                0.09513020515441895,
                -0.10917816311120987,
                0.265825480222702,
                -0.09701524674892426,
                -0.08476560562849045,
                0.015599575825035572,
                -0.16371792554855347,
                0.07208860665559769,
                -0.11039764434099197,
                -0.04978354275226593,
                -0.0139118991792202,
                0.019330939278006554,
                -0.14393363893032074,
                -0.12189751863479614,
                -0.05219394713640213,
                -0.02151140756905079,
                0.11024126410484314,
                0.015893785282969475,
                0.08080123364925385,
                0.13594745099544525,
                -0.014004693366587162,
                -0.03796566277742386,
                -0.13894250988960266,
                0.1163255125284195,
                -0.03998023644089699,
                -0.04189044609665871,
            ],
            dtype="float32",
        )
        self.assertTrue(
            paddle.allclose(EXPECTED_LOGITS_SLICE, logits[0, 0, :30], atol=5e-4, rtol=1e-5),
            msg=f"logits slice mismatch. got={logits[0,0,:30].cpu().numpy().tolist()}",
        )
