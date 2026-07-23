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
    ShieldGemma2Config,
    ShieldGemma2ForImageClassification,
)
from tests.transformers.test_configuration_common import ConfigTester


class ShieldGemma2ModelTester:
    def __init__(self, parent, batch_size=2, seq_length=8, vocab_size=64, hidden_size=32):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.image_token_index = 5
        self.mm_tokens_per_image = 4

    def get_config(self):
        return ShieldGemma2Config(
            text_config={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "query_pre_attn_scalar": 8,
                "sliding_window": 16,
            },
            vision_config={"hidden_size": 32, "image_size": 16, "patch_size": 4},
            image_token_index=self.image_token_index,
            mm_tokens_per_image=self.mm_tokens_per_image,
            yes_token_index=3,
            no_token_index=7,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = paddle.randint(0, self.vocab_size, shape=[self.batch_size, self.seq_length], dtype="int64")
        input_ids[:, 1 : 1 + self.mm_tokens_per_image] = self.image_token_index
        pixel_values = paddle.randn([self.batch_size, 3, 16, 16], dtype="float32")
        return config, input_ids, pixel_values


class ShieldGemma2ModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = ShieldGemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ShieldGemma2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_round_trip(self):
        config = self.model_tester.get_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded_config = ShieldGemma2Config.from_pretrained(tmpdir)
        self.assertIsInstance(loaded_config, ShieldGemma2Config)
        self.assertEqual(loaded_config.text_config.model_type, "gemma3_text")
        self.assertEqual(loaded_config.vision_config.model_type, "shieldgemma2_vision")

    def test_model_forward(self):
        config, input_ids, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = ShieldGemma2ForImageClassification(config)
        model.eval()

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)

        self.assertEqual(outputs.logits.shape, [self.model_tester.batch_size, 2])
        self.assertEqual(outputs.probabilities.shape, [self.model_tester.batch_size, 2])
        self.assertTrue(
            paddle.allclose(
                outputs.probabilities.sum(axis=-1),
                paddle.ones([self.model_tester.batch_size], dtype=outputs.probabilities.dtype),
            ).item()
        )

    def test_pixel_values_affect_outputs(self):
        paddle.seed(1234)
        config, input_ids, _ = self.model_tester.prepare_config_and_inputs()
        model = ShieldGemma2ForImageClassification(config)
        model.eval()

        dark_pixels = paddle.zeros([self.model_tester.batch_size, 3, 16, 16], dtype="float32")
        bright_pixels = paddle.ones([self.model_tester.batch_size, 3, 16, 16], dtype="float32")

        dark_outputs = model(input_ids=input_ids, pixel_values=dark_pixels, return_dict=True)
        bright_outputs = model(input_ids=input_ids, pixel_values=bright_pixels, return_dict=True)

        self.assertFalse(paddle.allclose(dark_outputs.logits, bright_outputs.logits).item())

    def test_image_token_mismatch_raises(self):
        config = self.model_tester.get_config()
        model = ShieldGemma2ForImageClassification(config)
        input_ids = paddle.randint(
            0,
            self.model_tester.vocab_size,
            shape=[self.model_tester.batch_size, self.model_tester.seq_length],
            dtype="int64",
        )
        input_ids[:, 1:3] = self.model_tester.image_token_index
        pixel_values = paddle.randn([self.model_tester.batch_size, 3, 16, 16], dtype="float32")

        with self.assertRaises(ValueError):
            model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)

    def test_embedding_accessors(self):
        config = self.model_tester.get_config()
        model = ShieldGemma2ForImageClassification(config)
        self.assertIsNotNone(model.get_input_embeddings())
        self.assertIsNotNone(model.get_output_embeddings())
