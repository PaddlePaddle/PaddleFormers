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
from paddle import nn

from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForConditionalGeneration,
    Mistral3Config,
    Mistral3ForCausalLM,
    Mistral3ForConditionalGeneration,
    Mistral3Model,
)


class Mistral3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        image_seq_length=4,
        image_size=24,
        patch_size=6,
        spatial_merge_size=2,
        num_channels=3,
        hidden_size=32,
        vocab_size=99,
        image_token_index=1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length + image_seq_length
        self.image_seq_length = image_seq_length
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.image_token_index = image_token_index

    def get_config(self):
        return Mistral3Config(
            text_config={
                "model_type": "mistral",
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": 37,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 512,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000000000.0,
                "bos_token_id": 2,
                "eos_token_id": 3,
                "pad_token_id": 4,
            },
            vision_config={
                "model_type": "pixtral",
                "hidden_size": self.hidden_size,
                "intermediate_size": 37,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_channels": self.num_channels,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_act": "gelu",
            },
            image_token_index=self.image_token_index,
            spatial_merge_size=self.spatial_merge_size,
            vision_feature_layer=-1,
        )

    def prepare_inputs(self):
        input_ids = paddle.randint(5, self.vocab_size, [self.batch_size, self.seq_length], dtype="int64")
        input_ids[:, : self.image_seq_length] = self.image_token_index
        pixel_values = paddle.randn(
            [self.batch_size, self.num_channels, self.image_size, self.image_size],
            dtype="float32",
        )
        image_sizes = paddle.to_tensor([[self.image_size, self.image_size]] * self.batch_size, dtype="int64")
        attention_mask = paddle.ones(input_ids.shape, dtype="int64")
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "attention_mask": attention_mask,
        }


class Mistral3ModelTest(unittest.TestCase):
    def setUp(self):
        paddle.seed(1234)
        self.model_tester = Mistral3ModelTester(self)

    def test_config_save_load(self):
        config = self.model_tester.get_config()
        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            loaded_config = AutoConfig.from_pretrained(tmpdirname)

        self.assertIsInstance(loaded_config, Mistral3Config)
        self.assertEqual(loaded_config.model_type, "mistral3")
        self.assertEqual(loaded_config.image_token_index, config.image_token_index)
        self.assertEqual(loaded_config.spatial_merge_size, config.spatial_merge_size)
        self.assertEqual(loaded_config.vision_config.model_type, "pixtral")

    def test_model_forward(self):
        config = self.model_tester.get_config()
        model = Mistral3Model(config).eval()
        inputs = self.model_tester.prepare_inputs()
        with paddle.no_grad():
            outputs = model(**inputs)

        self.assertEqual(
            outputs.last_hidden_state.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size],
        )
        self.assertEqual(
            outputs.image_hidden_states.shape,
            [self.model_tester.batch_size * self.model_tester.image_seq_length, self.model_tester.hidden_size],
        )

    def test_for_conditional_generation_forward(self):
        config = self.model_tester.get_config()
        model = Mistral3ForConditionalGeneration(config).eval()
        inputs = self.model_tester.prepare_inputs()
        labels = paddle.randint(0, self.model_tester.vocab_size, inputs["input_ids"].shape, dtype="int64")
        with paddle.no_grad():
            outputs = model(**inputs, labels=labels)

        self.assertEqual(
            outputs.logits.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size],
        )
        self.assertIsNotNone(outputs.loss)

    def test_for_causal_lm_alias_forward(self):
        config = self.model_tester.get_config()
        model = Mistral3ForCausalLM(config).eval()
        inputs = self.model_tester.prepare_inputs()

        with paddle.no_grad():
            outputs = model(**inputs)

        self.assertEqual(
            outputs.logits.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size],
        )

    def test_loss_uses_pre_shifted_labels(self):
        config = self.model_tester.get_config()
        model = Mistral3ForConditionalGeneration(config).eval()
        inputs = self.model_tester.prepare_inputs()
        labels = paddle.randint(0, self.model_tester.vocab_size, inputs["input_ids"].shape, dtype="int64")

        with paddle.no_grad():
            outputs = model(**inputs, labels=labels)

        loss_fct = nn.CrossEntropyLoss()
        expected_loss = loss_fct(outputs.logits.reshape([-1, outputs.logits.shape[-1]]), labels.reshape([-1]))
        self.assertTrue(paddle.allclose(outputs.loss, expected_loss))

    def test_image_token_mismatch_raises(self):
        config = self.model_tester.get_config()
        model = Mistral3Model(config).eval()
        inputs = self.model_tester.prepare_inputs()
        inputs["input_ids"][:, 0] = 5

        with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
            with paddle.no_grad():
                model(**inputs)

    def test_text_head_dim_can_differ_from_hidden_per_head(self):
        config = self.model_tester.get_config()
        config.text_config.hidden_size = 20
        config.text_config.num_attention_heads = 4
        config.text_config.num_key_value_heads = 2
        config.text_config.head_dim = 4
        config.text_config.intermediate_size = 32
        model = Mistral3Model(config).eval()
        input_ids = paddle.randint(5, self.model_tester.vocab_size, [1, 5], dtype="int64")

        with paddle.no_grad():
            outputs = model(input_ids=input_ids)

        self.assertEqual(outputs.last_hidden_state.shape, [1, 5, 20])

    def test_aoa_config(self):
        config = self.model_tester.get_config()
        config.tie_word_embeddings = False
        aoa_config = Mistral3ForConditionalGeneration._gen_aoa_config(config)
        statements = aoa_config["aoa_statements"]

        self.assertIn(
            "language_model.model.embed_tokens.weight -> model.language_model.embed_tokens.weight",
            statements,
        )
        self.assertIn("language_model.lm_head.weight -> lm_head.weight", statements)
        self.assertIn("vision_tower.patch_conv.weight -> model.vision_tower.patch_conv.weight", statements)
        self.assertNotIn("lm_head", Mistral3ForConditionalGeneration.transpose_weight_keys)

        inv_statements = Mistral3ForConditionalGeneration._gen_inv_aoa_config(config)["aoa_statements"]
        self.assertIn(
            "model.language_model.layers.$LAYER_ID.self_attn.q_proj.weight^T -> "
            "language_model.model.layers.$LAYER_ID.self_attn.q_proj.weight",
            inv_statements,
        )
        self.assertFalse(any(statement.split(" -> ")[1].endswith("^T") for statement in inv_statements))

    def test_auto_model_mapping(self):
        config = self.model_tester.get_config()
        model = Mistral3ForConditionalGeneration(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, save_checkpoint_format="flex_checkpoint")
            auto_model = AutoModel.from_pretrained(tmpdirname, load_checkpoint_format="flex_checkpoint")
            auto_causal_lm = AutoModelForCausalLM.from_pretrained(tmpdirname, load_checkpoint_format="flex_checkpoint")
            auto_conditional_generation = AutoModelForConditionalGeneration.from_pretrained(
                tmpdirname,
                load_checkpoint_format="flex_checkpoint",
            )

        self.assertIsInstance(auto_model, Mistral3Model)
        self.assertIsInstance(auto_causal_lm, Mistral3ForConditionalGeneration)
        self.assertIsInstance(auto_conditional_generation, Mistral3ForConditionalGeneration)


if __name__ == "__main__":
    unittest.main()
