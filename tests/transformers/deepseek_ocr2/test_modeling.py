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

import tempfile
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest import mock

import paddle
import requests
from PIL import Image

from paddleformers.transformers import (
    AutoModel,
    AutoTokenizer,
    DeepseekOCR2Config,
    DeepseekOCR2ForConditionalGeneration,
)
from paddleformers.transformers.deepseek_ocr2.modeling import (
    DeepseekOCR2Model,
    _parse_line_result,
    extract_coordinates_and_label,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


class DeepseekOCR2ModelTester:
    """Tester for DeepseekOCR2 models with tiny config matching tiny-random-deepseekocr2-bf16."""

    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=26,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        # LLM config (from tiny_model.py)
        vocab_size=129280,
        head_dim=32,
        hidden_size=320,
        intermediate_size=1712,
        max_position_embeddings=1024,
        moe_intermediate_size=224,
        n_routed_experts=64,
        n_shared_experts=2,
        num_attention_heads=10,
        num_key_value_heads=10,
        num_hidden_layers=2,  # reduced for faster testing
        num_experts_per_tok=6,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        # Vision config (from tiny_model.py)
        encoder_embed_dim=192,
        encoder_depth=5,
        encoder_num_heads=4,
        encoder_global_attn_indexes=[2, 4],
        prompt_embed_dim=256,
        image_size=1024,
        mlp_ratio=2,
        decoder_layer=4,
        vision_hidden_dimension=224,
        vision_num_attention_heads=4,
        vision_num_key_value_heads=2,
        vision_intermediate_size=1216,
        # Other
        use_mla=False,
        pad_token_id=0,
        # Image token
        image_token_id=128815,
        num_image_tokens=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels

        # LLM config
        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group

        # Vision config
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.encoder_global_attn_indexes = encoder_global_attn_indexes
        self.prompt_embed_dim = prompt_embed_dim
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.decoder_layer = decoder_layer
        self.vision_hidden_dimension = vision_hidden_dimension
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_num_key_value_heads = vision_num_key_value_heads
        self.vision_intermediate_size = vision_intermediate_size

        self.use_mla = use_mla
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens

    def get_config(self) -> DeepseekOCR2Config:
        vision_config = {
            "encoder_embed_dim": self.encoder_embed_dim,
            "encoder_depth": self.encoder_depth,
            "encoder_num_heads": self.encoder_num_heads,
            "encoder_global_attn_indexes": self.encoder_global_attn_indexes,
            "prompt_embed_dim": self.prompt_embed_dim,
            "image_size": self.image_size,
            "mlp_ratio": self.mlp_ratio,
            "decoder_layer": self.decoder_layer,
            "hidden_dimension": self.vision_hidden_dimension,
            "num_attention_heads": self.vision_num_attention_heads,
            "num_key_value_heads": self.vision_num_key_value_heads,
            "intermediate_size": self.vision_intermediate_size,
        }
        return DeepseekOCR2Config(
            vocab_size=self.vocab_size,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            moe_intermediate_size=self.moe_intermediate_size,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            first_k_dense_replace=self.first_k_dense_replace,
            n_group=self.n_group,
            topk_group=self.topk_group,
            use_mla=self.use_mla,
            vision_config=vision_config,
            pad_token_id=self.pad_token_id,
            # Disable MLA: kv_lora_rank must be None so that
            # DeepseekV3Attention._init_gqa is used instead of _init_mla.
            kv_lora_rank=None,
            q_lora_rank=None,
            qk_rope_head_dim=0,
            qk_nope_head_dim=0,
            v_head_dim=0,
            # MoE gate requires "greedy" (not default "gready")
            topk_method="greedy",
        )

    def _make_dummy_images(self, batch_size, all_zeros=False):
        """Create dummy image inputs.

        DeepseekOCR2Model.forward always accesses ``images[0][1]`` to check
        whether vision processing should run.  When *all_zeros* is True the
        global view sums to 0 so the vision branch is skipped (text-only).
        """
        images = []
        for _ in range(batch_size):
            if all_zeros:
                crop_patches = paddle.zeros([1, 3, self.image_size, self.image_size], dtype=paddle.float32)
                global_view = paddle.zeros([1, 3, self.image_size, self.image_size], dtype=paddle.float32)
            else:
                crop_patches = floats_tensor([1, 3, self.image_size, self.image_size])
                global_view = floats_tensor([1, 3, self.image_size, self.image_size])
            images.append((crop_patches, global_view))
        return images

    def prepare_config_and_inputs_text_only(self):
        """Prepare inputs for text-only path (vision branch skipped).

        DeepseekOCR2 requires:
        - ``position_ids``: parent DeepseekV3Model uses ``input_ids.shape``
          to build position_ids, but DeepseekOCR2Model passes
          ``input_ids=None`` to the parent so we must supply position_ids.
        - ``images``: forward always dereferences ``images[0][1]``, so we
          must provide all-zero images to skip the vision branch.
        - ``images_seq_mask`` / ``images_spatial_crop``: required by
          ``prepare_inputs_for_generation``.
        """
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        attention_mask = paddle.ones(input_ids.shape, dtype=paddle.int64)
        position_ids = paddle.arange(self.seq_length, dtype=paddle.int64).unsqueeze(0).expand([self.batch_size, -1])

        # All-zero images -> vision branch skipped
        images = self._make_dummy_images(self.batch_size, all_zeros=True)
        images_seq_mask = paddle.zeros([self.batch_size, self.seq_length], dtype=paddle.bool)
        images_spatial_crop = paddle.ones([self.batch_size, 2], dtype=paddle.int64)

        config.seq_length = self.seq_length

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "images": images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        """Prepare inputs with images for common model tests (VLM path).

        For VLM models, the standard ``prepare_config_and_inputs_for_common``
        should include image inputs so that parent class tests exercise the
        full vision-language pipeline.
        """
        config = self.get_config()

        # Derive text length from total seq_length
        text_len = self.seq_length - self.num_image_tokens
        prefix_len = text_len // 2
        suffix_len = text_len - prefix_len

        # Build input_ids: text prefix + image tokens + text suffix
        text_prefix = ids_tensor([self.batch_size, prefix_len], self.vocab_size, dtype=paddle.int64)
        image_tokens = paddle.full([self.batch_size, self.num_image_tokens], self.image_token_id, dtype=paddle.int64)
        text_suffix = ids_tensor([self.batch_size, suffix_len], self.vocab_size, dtype=paddle.int64)
        input_ids = paddle.concat([text_prefix, image_tokens, text_suffix], axis=1)

        attention_mask = paddle.ones([self.batch_size, self.seq_length], dtype=paddle.int64)
        position_ids = paddle.arange(self.seq_length, dtype=paddle.int64).unsqueeze(0).expand([self.batch_size, -1])

        # Non-zero images -> vision branch active
        images = self._make_dummy_images(self.batch_size)

        images_seq_mask = paddle.zeros([self.batch_size, self.seq_length], dtype=paddle.bool)
        images_seq_mask[:, prefix_len : prefix_len + self.num_image_tokens] = True

        images_spatial_crop = paddle.ones([self.batch_size, 2], dtype=paddle.int64)

        labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        labels[:, : prefix_len + self.num_image_tokens] = -100

        config.seq_length = self.seq_length

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "images": images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "labels": labels,
        }
        return config, inputs_dict


class DeepseekOCR2ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for ``DeepseekOCR2ForConditionalGeneration``.
    """

    all_model_classes = (DeepseekOCR2ForConditionalGeneration,)
    all_generative_model_classes = {
        DeepseekOCR2ForConditionalGeneration: {DeepseekOCR2ForConditionalGeneration, "deepseek_ocr2"}
    }
    max_new_tokens = 3

    @gpu_device_initializer(log_prefix="DeepseekOCR2ModelTest")
    def setUp(self):
        self.model_tester = DeepseekOCR2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekOCR2Config)

    # ------------------------------------------------------------------ #
    #  Config tests                                                       #
    # ------------------------------------------------------------------ #
    def test_config(self):
        self.config_tester.run_common_tests()

    # ------------------------------------------------------------------ #
    #  Forward tests                                                      #
    # ------------------------------------------------------------------ #
    def test_model_forward_text_only(self):
        """Test forward pass with text-only inputs (vision branch skipped)."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_text_only()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()
            with paddle.no_grad():
                result = model(return_dict=True, **inputs_dict)
            self.assertIsNotNone(result.logits)
            self.assertEqual(
                result.logits.shape,
                [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size],
            )

    def test_model_forward_text_only_without_images(self):
        """Pure-text SFT must not require dummy image tensors."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_text_only()
        for key in ("images", "images_seq_mask", "images_spatial_crop", "position_ids"):
            inputs_dict.pop(key)
        model = DeepseekOCR2ForConditionalGeneration(config)
        model.eval()
        with paddle.no_grad():
            result = model(return_dict=True, **inputs_dict)
        self.assertEqual(
            result.logits.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size],
        )

    def test_auto_model_from_config(self):
        model = AutoModel.from_config(self.model_tester.get_config())
        self.assertIsInstance(model, DeepseekOCR2Model)

    def test_auto_model_flex_checkpoint_round_trip(self):
        model = DeepseekOCR2Model(self.model_tester.get_config())
        with tempfile.TemporaryDirectory() as save_dir:
            model.save_pretrained(save_dir, save_checkpoint_format="flex_checkpoint")
            loaded = AutoModel.from_pretrained(save_dir, load_checkpoint_format="flex_checkpoint")
        self.assertIsInstance(loaded, DeepseekOCR2Model)
        for name, tensor in model.state_dict().items():
            reloaded = loaded.state_dict()[name]
            if name.endswith("mlp.gate.weight"):
                self.assertTrue(paddle.allclose(tensor, reloaded, rtol=1e-2, atol=1e-2), name)
            else:
                self.assertEqual(tensor._md5sum(), reloaded._md5sum(), name)

    def test_detection_parser_rejects_code_and_invalid_coordinates(self):
        malicious = ("", "text", "__import__('os').system('false')")
        self.assertIsNone(extract_coordinates_and_label(malicious, 100, 100))
        invalid_box = ("", "text", "[[-1, 0, 10, 10]]")
        self.assertIsNone(extract_coordinates_and_label(invalid_box, 100, 100))
        valid_box = ("", "text", "[[0, 1, 998, 999]]")
        self.assertEqual(extract_coordinates_and_label(valid_box, 100, 100)[1], [(0.0, 1.0, 998.0, 999.0)])
        with self.assertRaises((SyntaxError, ValueError)):
            _parse_line_result("__import__('os').system('false')")

    def test_prepare_inputs_for_infer_text_only(self):
        config = self.model_tester.get_config()
        model = DeepseekOCR2ForConditionalGeneration(config)
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [11, 12]
        conversation = model._build_conversation("hello", image_file="")
        inputs = model.prepare_inputs_for_infer(
            tokenizer,
            conversation,
            base_size=config.vision_config.image_size,
            image_size=64,
            crop_mode=False,
        )
        self.assertEqual(inputs["images_seq_mask"].astype("int64").sum().item(), 0)
        self.assertIsNone(inputs["image_draw"])

    def test_text_only_dataset_sequence_collation(self):
        from paddleformers.datasets.collate import mm_collate_fn_ds_ocr2
        from paddleformers.datasets.SFTDataset import Sequence

        sequence = Sequence(
            token_ids=[3, 4, 5],
            position_ids=[0, 1, 2],
            labels=[-100, 4, 5],
            num_examples=1,
            mm_inputs={},
        )
        tokenizer = mock.Mock(pad_token_id=0)
        tokenizer.encode.return_value = [128815]
        result = mm_collate_fn_ds_ocr2(
            [[sequence]],
            SimpleNamespace(mm_plugin=SimpleNamespace(image_token="<image>")),
            processor=None,
            tokenizer=tokenizer,
            training_args=SimpleNamespace(
                num_nextn_predict_layers=0,
                context_parallel_size=1,
                tensor_model_parallel_size=1,
                sequence_parallel=False,
                fp8=False,
            ),
            model_args=SimpleNamespace(use_attn_mask_startend_row_indices=False, use_global_causal_attn=False),
            max_seq_len=8,
            padding_free=False,
            model=mock.Mock(),
        )
        self.assertNotIn("images", result)
        self.assertNotIn("images_spatial_crop", result)
        self.assertEqual(result["input_ids"].shape, [1, 8])

    def test_model_forward_with_images(self):
        """Test forward pass with image inputs (vision branch active)."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()
            with paddle.no_grad():
                result = model(return_dict=True, **inputs_dict)
            self.assertIsNotNone(result.logits)
            total_len = inputs_dict["input_ids"].shape[1]
            self.assertEqual(
                result.logits.shape,
                [self.model_tester.batch_size, total_len, self.model_tester.vocab_size],
            )

    def test_model_forward_with_labels(self):
        """Test forward pass with labels to compute loss."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()
            with paddle.no_grad():
                result = model(return_dict=True, **inputs_dict)
            self.assertIsNotNone(result.loss)
            self.assertIsNotNone(result.logits)

    # ------------------------------------------------------------------ #
    #  Generation helpers                                                 #
    # ------------------------------------------------------------------ #
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

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        # Do not pass position_ids to generate() — matches the real infer()
        # usage. prepare_inputs_for_generation auto-generates position_ids
        # from attention_mask and correctly truncates it for decode steps.
        inputs_dict.pop("position_ids", None)
        return config, inputs_dict

    # ------------------------------------------------------------------ #
    #  Generation tests                                                   #
    # ------------------------------------------------------------------ #
    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)
            self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)
            self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            beam_kwargs, _ = self._get_beam_scorer_and_kwargs(1, 1)
            output_generate = self._beam_search_generate(model=model, inputs_dict=inputs_dict, beam_kwargs=beam_kwargs)
            self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])

    # ------------------------------------------------------------------ #
    #  Skipped / no-op tests                                              #
    # ------------------------------------------------------------------ #
    @unittest.skip("Group beam search is not compatible with current VLM implementation")
    def test_group_beam_search_generate(self):
        pass

    @unittest.skip(
        "DeepseekOCR2 uses non-tied weights (tie_word_embeddings=False), so lm_head dimensions are not updated"
    )
    def test_resize_tokens_embeddings(self):
        pass

    def test_save_load_flex_checkpoint(self):
        for model_class in self.all_model_classes:
            with tempfile.TemporaryDirectory() as tmpdirname:
                config = self.model_tester.get_config()
                model = model_class(config)
                model.save_pretrained(tmpdirname, save_checkpoint_format="flex_checkpoint")

                # model1: load from HF-format keys via AOA (default load_checkpoint_format="flex_checkpoint")
                model1 = model_class.from_pretrained(tmpdirname, convert_from_hf=True)
                # model2: load directly from flex_checkpoint
                model2 = model_class.from_pretrained(tmpdirname, load_checkpoint_format="flex_checkpoint")

                model_state_1 = model1.state_dict()
                model_state_2 = model2.state_dict()

                for k, v in model_state_1.items():
                    md51 = v._md5sum()
                    md52 = model_state_2[k]._md5sum()
                    assert md51 == md52, f"State dict mismatch for key: {k}"

    @unittest.skip("DeepseekOCR2 does not support generate without input_ids")
    def test_generate_without_input_ids(self):
        pass


class DeepseekOCR2IntegrationTest(unittest.TestCase):
    """Integration tests using the pre-built tiny model."""

    MODEL_PATH = "PaddleFormers/tiny-random-deepseekocr2-bf16"
    IMAGE_URL = (
        "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/exam_paper_0829/part_0000/img_000040676.png"
    )
    PROMPT = "<image>\nFree OCR."

    @gpu_device_initializer(log_prefix="DeepseekOCR2IntegrationTest")
    def setUp(self):
        self.model = DeepseekOCR2ForConditionalGeneration.from_pretrained(
            self.MODEL_PATH,
            dtype="float32",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        raw = Image.open(BytesIO(requests.get(self.IMAGE_URL).content)).convert("RGB")
        w, h = raw.size
        tiled = Image.new("RGB", (w * 2, h * 2))
        for r in range(2):
            for c in range(2):
                tiled.paste(raw, (c * w, r * h))
        self.image_file = tiled

    def _build_inputs(self):
        """Build model inputs from the 2x2 tiled PIL image and prompt."""
        conversation = [
            {
                "role": "<|User|>",
                "content": self.PROMPT,
                "images": [self.image_file],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        inputs = self.model.prepare_inputs_for_infer(
            self.tokenizer,
            conversation,
            base_size=self.model.config.vision_config.image_size,
            image_size=768,
            crop_mode=True,
        )
        return inputs

    def test_model_tiny_image_logits(self):
        """Test tiny model forward pass with a real image input."""
        inputs = self._build_inputs()
        input_ids = inputs["input_ids"].unsqueeze(0)
        images_seq_mask = inputs["images_seq_mask"].unsqueeze(0)
        images_spatial_crop = inputs["images_spatial_crop"]
        images_crop = inputs["images_crop"]
        images_ori = inputs["images_ori"]

        EXPECTED_INPUT_IDS = paddle.to_tensor(
            [
                128815,
                128815,
                128815,
                128815,
                128815,
                128815,
                128815,
                128815,
                128815,
                128815,
                128815,
                128815,
                128815,
                201,
                21431,
                126041,
                16,
            ]
        )
        self.assertTrue(paddle.equal_all(EXPECTED_INPUT_IDS, input_ids[0, -17:]))

        EXPECTED_PIXEL_SLICE = paddle.to_tensor(
            [
                1.0,
                1.0,
                0.97647071,
                0.99215698,
                0.07450986,
                1.0,
                0.99215698,
                0.78823543,
                1.0,
                0.99215698,
                0.99215698,
                0.33333337,
                1.0,
                1.0,
                1.0,
                0.97647071,
                0.67058837,
                1.0,
                1.0,
                1.0,
                1.0,
                0.99215698,
                1.0,
                0.97647071,
                0.95294130,
                0.99215698,
            ]
        )
        self.assertTrue(
            paddle.allclose(
                EXPECTED_PIXEL_SLICE,
                images_ori[0, 0, 400, ::40],
                atol=5e-4,
                rtol=1e-5,
            )
        )

        self.model.config.seq_length = input_ids.shape[1]
        seq_len = input_ids.shape[1]
        position_ids = paddle.arange(seq_len, dtype=paddle.int64).unsqueeze(0)
        with paddle.no_grad():
            output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                images=[(images_crop, images_ori)],
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
                return_dict=True,
            )
        logits = output.logits.astype(paddle.float32)

        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], input_ids.shape[1])
        self.assertEqual(logits.shape[2], self.model.config.vocab_size)
        self.assertTrue(paddle.isfinite(logits).all().item())

        EXPECTED_SLICE = paddle.to_tensor(
            [
                3.23286867,
                -0.59275615,
                -0.90195876,
                -0.13619526,
                0.69505769,
                -0.78623712,
                1.44161093,
                -2.74753880,
                1.70963466,
                -0.28738150,
                -0.73505950,
                -1.96136701,
                -2.23667574,
                -1.10724699,
                0.69466162,
                2.09361839,
                1.23767567,
                -0.74303693,
                1.78987753,
                0.10986544,
                0.52748066,
                -1.44185197,
                0.93142855,
                2.17866540,
                -0.38639364,
                1.25585449,
                -0.56216007,
                0.67937303,
                0.00060895,
                0.82671565,
            ]
        )
        self.assertTrue(paddle.allclose(EXPECTED_SLICE, logits[0, 0, :30], atol=5e-4, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
