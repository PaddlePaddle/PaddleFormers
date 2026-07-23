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
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import paddle

import paddleformers.transformers as transformers_module
from paddleformers.transformers import (
    AutoModel,
    AutoModelForConditionalGeneration,
    Gemma3Config,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3TextModel,
)
from paddleformers.transformers.gemma3_text.configuration import Gemma3TextConfig
from paddleformers.transformers.gemma3.modeling import (
    _convert_hf_vision_tensor,
    _use_high_precision_cublas_for_fp32,
)
from tests.transformers.test_configuration_common import ConfigTester


class Gemma3ModelTester:
    def __init__(self, parent, batch_size=2, seq_length=8, vocab_size=64, hidden_size=32):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.image_token_index = 5
        self.mm_tokens_per_image = 4

    def get_config(self):
        return Gemma3Config(
            text_config={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 64,
                "query_pre_attn_scalar": 8,
                "sliding_window": 16,
                "pad_token_id": 0,
            },
            vision_config={
                "hidden_size": self.hidden_size,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 16,
                "patch_size": 4,
            },
            image_token_index=self.image_token_index,
            mm_tokens_per_image=self.mm_tokens_per_image,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = paddle.randint(6, self.vocab_size, shape=[self.batch_size, self.seq_length], dtype="int64")
        input_ids[:, 1 : 1 + self.mm_tokens_per_image] = self.image_token_index
        token_type_ids = paddle.zeros_like(input_ids)
        token_type_ids[:, 1 : 1 + self.mm_tokens_per_image] = 1
        pixel_values = paddle.randn([self.batch_size, 3, 16, 16], dtype="float32")
        return config, input_ids, token_type_ids, pixel_values


class Gemma3ModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Gemma3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma3Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_round_trip(self):
        config = self.model_tester.get_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded_config = Gemma3Config.from_pretrained(tmpdir)

        self.assertIsInstance(loaded_config, Gemma3Config)
        self.assertEqual(loaded_config.text_config.model_type, "gemma3_text")
        self.assertEqual(loaded_config.vision_config.model_type, "siglip_vision_model")

    def test_model_forward(self):
        config, input_ids, token_type_ids, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = Gemma3ForConditionalGeneration(config)
        model.eval()

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            return_dict=True,
        )

        self.assertEqual(
            tuple(outputs.logits.shape),
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size),
        )
        self.assertEqual(
            tuple(outputs.image_hidden_states.shape),
            (self.model_tester.batch_size, self.model_tester.mm_tokens_per_image, self.model_tester.hidden_size),
        )

    def test_logits_to_keep_tensor_selects_requested_tokens(self):
        config, input_ids, token_type_ids, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = Gemma3ForConditionalGeneration(config)
        model.eval()

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            logits_to_keep=paddle.to_tensor([0, 3, 7], dtype="int64"),
            return_dict=True,
        )

        self.assertEqual(
            tuple(outputs.logits.shape),
            (self.model_tester.batch_size, 3, self.model_tester.vocab_size),
        )

    def test_generation_drops_pixel_values_after_prefill(self):
        model = Gemma3ForConditionalGeneration(self.model_tester.get_config())
        pixel_values = paddle.randn([1, 3, 16, 16], dtype="float32")

        with mock.patch(
            "paddleformers.generation.utils.GenerationMixin.prepare_inputs_for_generation",
            return_value={"pixel_values": pixel_values},
        ):
            model_inputs = model.prepare_inputs_for_generation(
                paddle.ones([1, 1], dtype="int64"),
                past_key_values=object(),
                pixel_values=pixel_values,
                use_cache=True,
                is_first_iteration=False,
            )

        self.assertIsNone(model_inputs["pixel_values"])

    def test_auto_model_registration(self):
        config = self.model_tester.get_config()

        auto_model = AutoModel.from_config(config)
        auto_conditional_model = AutoModelForConditionalGeneration.from_config(config)

        self.assertEqual(type(auto_model).__name__, "Gemma3Model")
        self.assertIsInstance(auto_conditional_model, Gemma3ForConditionalGeneration)

    def test_gemma3_text_auto_model_compatibility(self):
        text_config = Gemma3TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
        )

        text_model = AutoModel.from_config(text_config)

        self.assertEqual(type(text_model).__name__, "Gemma3TextModel")

    def test_top_level_exports_resolve_to_upstream_gemma3_text(self):
        self.assertEqual(transformers_module._class_to_module["Gemma3Config"], "gemma3.configuration")
        self.assertEqual(transformers_module._class_to_module["Gemma3TextConfig"], "gemma3_text.configuration")
        self.assertEqual(transformers_module._class_to_module["Gemma3TextModel"], "gemma3_text.modeling")
        self.assertEqual(transformers_module._class_to_module["Gemma3ForCausalLM"], "gemma3_text.modeling")
        self.assertEqual(Gemma3Config.__module__, "paddleformers.transformers.gemma3.configuration")
        self.assertEqual(Gemma3ForCausalLM.__module__, "paddleformers.transformers.gemma3_text.modeling")

    def test_multimodal_uses_private_text_backbone(self):
        model = Gemma3ForConditionalGeneration(self.model_tester.get_config())
        self.assertEqual(Gemma3TextModel.__module__, "paddleformers.transformers.gemma3_text.modeling")
        self.assertEqual(
            model.model.language_model.__class__.__module__,
            "paddleformers.transformers.gemma3.multimodal_text_modeling",
        )

    def test_hf_vision_linear_weight_conversion(self):
        tensor = paddle.arange(6, dtype="float32").reshape([2, 3])

        converted = _convert_hf_vision_tensor(
            "model.vision_tower.encoder.layers.0.mlp.fc1.weight", tensor
        )
        unchanged = _convert_hf_vision_tensor(
            "model.vision_tower.embeddings.position_embedding.weight", tensor
        )

        self.assertTrue(paddle.equal_all(converted, tensor.transpose([1, 0])))
        self.assertIs(unchanged, tensor)

    def test_fp32_gpu_disables_tf32_cublas(self):
        fake_tensor = SimpleNamespace(
            dtype=paddle.float32,
            place=SimpleNamespace(is_gpu_place=lambda: True),
        )
        with (
            mock.patch("paddle.base.core.get_cublas_switch", return_value=True),
            mock.patch("paddle.base.core.set_cublas_switch") as set_cublas_switch,
        ):
            _use_high_precision_cublas_for_fp32(fake_tensor)

        set_cublas_switch.assert_called_once_with(False)

    def test_image_token_mismatch_raises(self):
        config = self.model_tester.get_config()
        model = Gemma3ForConditionalGeneration(config)
        input_ids = paddle.randint(
            6,
            self.model_tester.vocab_size,
            shape=[self.model_tester.batch_size, self.model_tester.seq_length],
            dtype="int64",
        )
        input_ids[:, 1:3] = self.model_tester.image_token_index
        token_type_ids = paddle.zeros_like(input_ids)
        token_type_ids[:, 1:3] = 1
        pixel_values = paddle.randn([self.model_tester.batch_size, 3, 16, 16], dtype="float32")

        with self.assertRaises(ValueError):
            model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                pixel_values=pixel_values,
                return_dict=True,
            )

    def test_embedding_accessors(self):
        config = self.model_tester.get_config()
        model = Gemma3ForConditionalGeneration(config)

        self.assertIsNotNone(model.get_input_embeddings())
        self.assertIsNotNone(model.get_output_embeddings())

    def test_paddle_checkpoint_round_trip_honors_dtype(self):
        model = Gemma3ForConditionalGeneration(self.model_tester.get_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            model.config.save_pretrained(tmpdir)
            paddle.save(model.state_dict(), str(Path(tmpdir) / "model_state.pdparams"))
            loaded = Gemma3ForConditionalGeneration.from_pretrained(tmpdir, dtype="bfloat16")

        parameter_dtypes = {parameter.dtype for parameter in loaded.parameters()}
        self.assertEqual(parameter_dtypes, {paddle.bfloat16})
