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

import unittest
from unittest import mock

import paddle

from paddleformers.transformers import (
    AutoModel,
    Llama4Config,
    Llama4ForConditionalGeneration,
    Llama4VisionModel,
)
from paddleformers.transformers.cache_utils import DynamicCache
from tests.testing_utils import gpu_device_initializer


def get_tiny_config():
    return Llama4Config(
        image_token_index=100,
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "intermediate_size_mlp": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_local_experts": 0,
            "moe_layers": [],
            "interleave_moe_layer_step": 0,
            "no_rope_layers": [1, 1],
            "attention_chunk_size": None,
            "max_position_embeddings": 128,
        },
        vision_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "image_size": 28,
            "patch_size": 14,
            "projector_input_dim": 64,
            "projector_output_dim": 64,
            "vision_output_dim": 64,
            "pixel_shuffle_ratio": 0.5,
        },
    )


class Llama4MultimodalModelTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="Llama4MultimodalModelTest")
    def setUp(self):
        paddle.seed(2026)
        self.config = get_tiny_config()
        self.pixel_values = paddle.randn([1, 3, 28, 28])

    def test_vision_forward(self):
        model = Llama4VisionModel(self.config.vision_config)
        model.eval()
        with paddle.no_grad():
            output = model(self.pixel_values, return_dict=True)
        self.assertEqual(output.last_hidden_state.shape, [1, 1, 64])

    def test_auto_model_from_multimodal_config(self):
        model = AutoModel.from_config(self.config)
        self.assertIsInstance(model, Llama4ForConditionalGeneration)

    def test_multimodal_forward_and_loss(self):
        model = Llama4ForConditionalGeneration(self.config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 100, 2, 3]], dtype="int64")
        with paddle.no_grad():
            output = model(input_ids=input_ids, pixel_values=self.pixel_values, labels=input_ids)
        self.assertEqual(output.logits.shape, [1, 4, 128])
        self.assertEqual(output.image_hidden_states.shape, [1, 1, 64])
        self.assertIsNotNone(output.loss)

    def test_multimodal_uses_text_config_output_hidden_states(self):
        self.config.text_config.output_hidden_states = True
        model = Llama4ForConditionalGeneration(self.config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")

        with paddle.no_grad():
            output = model(input_ids=input_ids)

        self.assertEqual(len(output.hidden_states), self.config.text_config.num_hidden_layers + 1)

    def test_image_token_count_validation(self):
        model = Llama4ForConditionalGeneration(self.config)
        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
            model(input_ids=input_ids, pixel_values=self.pixel_values)

    def test_determinism(self):
        model = Llama4ForConditionalGeneration(self.config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 100, 2, 3]], dtype="int64")
        with paddle.no_grad():
            first = model(input_ids=input_ids, pixel_values=self.pixel_values).logits
            second = model(input_ids=input_ids, pixel_values=self.pixel_values).logits
        self.assertLessEqual(float(paddle.max(paddle.abs(first - second))), 1e-6)

    def test_generation_without_attention_chunks(self):
        self.assertEqual(self.config.text_config.layer_types, ["full_attention", "full_attention"])
        model = Llama4ForConditionalGeneration(self.config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        with paddle.no_grad():
            generated = model.generate(input_ids=input_ids, max_new_tokens=2)[0]
        self.assertEqual(generated.shape[-1], 2)

    def test_multimodal_generation_advances_cache_without_reencoding_image(self):
        model = Llama4ForConditionalGeneration(self.config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 100, 2]], dtype="int64")
        with mock.patch.object(model, "get_image_features", wraps=model.get_image_features) as encode_image:
            with paddle.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    pixel_values=self.pixel_values,
                    max_new_tokens=2,
                    use_cache=True,
                )[0]
        self.assertEqual(generated.shape[-1], 2)
        self.assertEqual(encode_image.call_count, 1)

    def test_generation_preserves_inputs_with_empty_cache(self):
        model = Llama4ForConditionalGeneration(self.config)
        input_ids = paddle.to_tensor([[1, 100, 2]], dtype="int64")
        position_ids = paddle.to_tensor([[4, 5, 6]], dtype="int64")
        empty_cache = DynamicCache(config=self.config.text_config)

        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=empty_cache,
            pixel_values=self.pixel_values,
            position_ids=position_ids,
            use_cache=True,
        )

        self.assertTrue(paddle.equal_all(model_inputs["input_ids"], input_ids))
        self.assertTrue(paddle.equal_all(model_inputs["position_ids"], position_ids))
        self.assertIs(model_inputs["pixel_values"], self.pixel_values)

    def test_generation_slices_inputs_only_after_cache_advances(self):
        model = Llama4ForConditionalGeneration(self.config)
        input_ids = paddle.to_tensor([[1, 100, 2, 3]], dtype="int64")
        position_ids = paddle.to_tensor([[4, 5, 6, 7]], dtype="int64")
        cache = DynamicCache(config=self.config.text_config)
        cached_states = paddle.zeros([1, 2, 3, 16], dtype="float32")
        cache.update(cached_states, cached_states, 0)

        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=cache,
            pixel_values=self.pixel_values,
            position_ids=position_ids,
            use_cache=True,
        )

        self.assertTrue(paddle.equal_all(model_inputs["input_ids"], input_ids[:, -1:]))
        self.assertTrue(paddle.equal_all(model_inputs["position_ids"], position_ids[:, -1:]))
        self.assertIsNone(model_inputs["pixel_values"])


if __name__ == "__main__":
    unittest.main()
