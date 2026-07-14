# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import json
import os
import tempfile
import unittest

import paddle

from paddleformers.transformers import AutoConfig, AutoModelForConditionalGeneration
from paddleformers.transformers.paligemma2.configuration import (
    PaliGemma2Config,
    SiglipVisionConfig,
    Gemma2TextConfig,
)
from paddleformers.transformers.paligemma2.modeling import (
    Gemma2RMSNorm,
    PaliGemma2ForCausalLM,
    PaliGemma2ForConditionalGeneration,
    PaliGemma2PreTrainedModel,
)


class PaliGemma2ModelTester:
    """Test configuration for PaliGemma2 model unit tests."""

    def __init__(self, parent):
        self.parent = parent
        # Use small dimensions for fast testing
        self.vision_hidden_size = 32
        self.vision_image_size = 56  # Must be divisible by patch_size
        self.vision_intermediate_size = 64
        self.vision_num_hidden_layers = 2
        self.vision_num_attention_heads = 4
        self.vision_patch_size = 14
        self.vision_projection_dim = 32

        self.text_vocab_size = 128
        self.text_hidden_size = 32
        self.text_intermediate_size = 64
        self.text_num_hidden_layers = 2
        self.text_num_attention_heads = 4
        self.text_num_key_value_heads = 2
        self.text_head_dim = 8

        self.batch_size = 2
        self.seq_length = 16
        self.image_token_index = 126

    def get_vision_config(self):
        return SiglipVisionConfig(
            hidden_size=self.vision_hidden_size,
            image_size=self.vision_image_size,
            intermediate_size=self.vision_intermediate_size,
            num_hidden_layers=self.vision_num_hidden_layers,
            num_attention_heads=self.vision_num_attention_heads,
            patch_size=self.vision_patch_size,
            projection_dim=self.vision_projection_dim,
        )

    def get_text_config(self):
        return Gemma2TextConfig(
            vocab_size=self.text_vocab_size,
            hidden_size=self.text_hidden_size,
            intermediate_size=self.text_intermediate_size,
            num_hidden_layers=self.text_num_hidden_layers,
            num_attention_heads=self.text_num_attention_heads,
            num_key_value_heads=self.text_num_key_value_heads,
            head_dim=self.text_head_dim,
        )

    def get_config(self):
        return PaliGemma2Config(
            vision_config=self.get_vision_config(),
            text_config=self.get_text_config(),
            image_token_index=self.image_token_index,
            projection_dim=self.vision_projection_dim,
            hidden_size=self.text_hidden_size,
            vocab_size=self.text_vocab_size,
        )

    def prepare_inputs(self):
        input_ids = paddle.randint(0, self.text_vocab_size, (self.batch_size, self.seq_length))
        # Insert image tokens at the beginning to match vision tower output
        # num_patches = (image_size // patch_size) ** 2 = (56 // 14) ** 2 = 16
        num_image_tokens = (self.vision_image_size // self.vision_patch_size) ** 2
        num_image_tokens = min(num_image_tokens, self.seq_length)
        input_ids[:, :num_image_tokens] = self.image_token_index
        attention_mask = paddle.ones((self.batch_size, self.seq_length))
        pixel_values = paddle.randn((self.batch_size, 3, self.vision_image_size, self.vision_image_size))
        return input_ids, attention_mask, pixel_values

    def check_conditional_generation(self):
        config = self.get_config()
        input_ids, attention_mask, pixel_values = self.prepare_inputs()
        model = PaliGemma2ForConditionalGeneration(config)
        model.eval()
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        self.parent.assertIsNotNone(output)
        self.parent.assertIsNotNone(output.logits)
        self.parent.assertEqual(
            output.logits.shape,
            [self.batch_size, self.seq_length, self.text_vocab_size],
        )

    def check_causal_lm(self):
        config = self.get_config()
        input_ids = paddle.randint(0, self.text_vocab_size, (self.batch_size, self.seq_length))
        model = PaliGemma2ForCausalLM(config)
        model.eval()
        output = model(input_ids=input_ids)
        self.parent.assertIsNotNone(output)
        self.parent.assertIsNotNone(output.logits)

    def check_loss(self):
        config = self.get_config()
        input_ids, attention_mask, pixel_values = self.prepare_inputs()
        model = PaliGemma2ForConditionalGeneration(config)
        model.train()
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        self.parent.assertIsNotNone(output.loss)
        self.parent.assertIsInstance(output.loss.item(), float)

    def check_shifted_labels_loss(self):
        config = self.get_config()
        input_ids, attention_mask, pixel_values = self.prepare_inputs()
        labels = paddle.concat([input_ids[:, 1:], paddle.full([self.batch_size, 1], -100, dtype="int64")], axis=1)
        labels[:, :2] = -100
        model = PaliGemma2ForConditionalGeneration(config)
        model.eval()
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )
        valid = labels != -100
        safe_labels = paddle.where(valid, labels, paddle.zeros_like(labels))
        token_loss = paddle.nn.functional.cross_entropy(
            output.logits.reshape([-1, self.text_vocab_size]),
            safe_labels.reshape([-1]),
            reduction="none",
        )
        expected_loss = (token_loss * valid.reshape([-1]).astype(token_loss.dtype)).sum() / valid.astype("float32").sum()
        self.parent.assertAlmostEqual(output.loss.item(), expected_loss.item(), places=6)

    def check_backward(self):
        config = self.get_config()
        input_ids, attention_mask, pixel_values = self.prepare_inputs()
        model = PaliGemma2ForConditionalGeneration(config)
        model.train()
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        output.loss.backward()
        # Check at least one parameter has gradient
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                break
        self.parent.assertTrue(has_grad, "No parameter has gradient after backward")


class PaliGemma2Test(unittest.TestCase):
    """Unit tests for PaliGemma2 model."""

    def setUp(self):
        self.tester = PaliGemma2ModelTester(self)

    def test_conditional_generation_forward(self):
        """Test forward pass of PaliGemma2ForConditionalGeneration."""
        self.tester.check_conditional_generation()

    def test_causal_lm_forward(self):
        """Test forward pass of PaliGemma2ForCausalLM."""
        self.tester.check_causal_lm()

    def test_loss_computation(self):
        """Test loss computation with labels."""
        self.tester.check_loss()

    def test_shifted_labels_loss(self):
        """Test PaddleFormers pre-shifted label loss."""
        self.tester.check_shifted_labels_loss()

    def test_backward_pass(self):
        """Test backward pass for gradient computation."""
        self.tester.check_backward()

    def test_rms_norm_default_is_unit_scaling(self):
        norm = Gemma2RMSNorm(4)
        inputs = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]])
        expected = inputs * paddle.rsqrt(paddle.mean(inputs * inputs, axis=-1, keepdim=True) + norm.eps)
        self.assertTrue(paddle.allclose(norm(inputs), expected).item())
        self.assertTrue(paddle.allclose(norm.weight, paddle.zeros_like(norm.weight)).item())

    def test_causal_lm_accepts_batched_padding_mask(self):
        config = self.tester.get_config()
        model = PaliGemma2ForCausalLM(config)
        input_ids = paddle.randint(0, self.tester.text_vocab_size, [2, self.tester.seq_length])
        attention_mask = paddle.to_tensor(
            [[1] * self.tester.seq_length, [1] * (self.tester.seq_length - 3) + [0] * 3], dtype="int64"
        )
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual(output.logits.shape, [2, self.tester.seq_length, self.tester.text_vocab_size])

    def test_embedding_and_lm_head_weights_are_tied(self):
        config = self.tester.get_config()
        conditional_model = PaliGemma2ForConditionalGeneration(config)
        causal_model = PaliGemma2ForCausalLM(config)
        self.assertIs(conditional_model.get_input_embeddings().weight, conditional_model.get_output_embeddings().weight)
        self.assertIs(causal_model.get_input_embeddings().weight, causal_model.get_output_embeddings().weight)

    def test_conditional_generation_applies_final_logit_softcapping(self):
        config = self.tester.get_config()
        config.text_config.final_logit_softcapping = 1.0
        model = PaliGemma2ForConditionalGeneration(config)
        model.lm_head.weight.set_value(paddle.full_like(model.lm_head.weight, 100.0))
        input_ids = paddle.randint(0, self.tester.text_vocab_size, [1, self.tester.seq_length])
        logits = model(input_ids=input_ids).logits
        self.assertLessEqual(float(paddle.abs(logits).max()), 1.0)
        self.assertGreater(float(paddle.abs(logits).max()), 0.9)

    def test_hf_paligemma_auto_loading_alias(self):
        config_dict = self.tester.get_config().to_dict()
        config_dict.update(
            {
                "model_type": "paligemma",
                "architectures": ["PaliGemmaForConditionalGeneration"],
            }
        )
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
            config = AutoConfig.from_pretrained(tempdir)
            self.assertIsInstance(config, PaliGemma2Config)
            model_class = AutoModelForConditionalGeneration._get_model_class_from_config(
                tempdir, os.path.join(tempdir, "config.json")
            )
            self.assertIs(model_class, PaliGemma2ForConditionalGeneration)


if __name__ == "__main__":
    unittest.main()
