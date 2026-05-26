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

import tempfile
import unittest

import paddle

from paddleformers.transformers import (
    AutoModelForConditionalGeneration,
    Llavaonevision1_5Config,
    LLaVAOneVision1_5ForConditionalGeneration,
    LLaVAOneVision1_5Model,
    LLaVAOneVision1_5TextModel,
    RiceConfig,
    RiceTransformerPretrainedModel,
)
from tests.testing_utils import gpu_device_initializer


class RiceTransformerModelTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="RiceTransformerModelTest", gpu_id=0)
    def setUp(self):
        self.config = RiceConfig(
            depth=2,
            hidden_size=32,
            embed_dim=32,
            intermediate_size=64,
            num_heads=4,
            in_channels=3,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=1,
            text_hidden_size=48,
            layer_norm_eps=1e-5,
            _attn_implementation="eager",
        )

    def test_forward_shape(self):
        model = RiceTransformerPretrainedModel(self.config)
        model.eval()
        pixel_values = paddle.randn([4, 3 * 14 * 14], dtype="float32")
        grid_thw = paddle.to_tensor([[1, 2, 2]], dtype="int64")

        with paddle.no_grad():
            output = model(pixel_values, grid_thw)

        self.assertEqual(output.shape, [1, 48])

    def test_verify_forward_shape_before_merger(self):
        model = RiceTransformerPretrainedModel(self.config)
        model.eval()
        pixel_values = paddle.randn([4, 3 * 14 * 14], dtype="float32")
        grid_thw = paddle.to_tensor([[1, 2, 2]], dtype="int64")

        with paddle.no_grad():
            output = model(pixel_values, grid_thw, is_verifying=True)

        self.assertEqual(output.shape, [4, 32])


class LLaVAOneVision1_5TextModelTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="LLaVAOneVision1_5TextModelTest", gpu_id=0)
    def setUp(self):
        self.config = Llavaonevision1_5Config(
            text_config={
                "vocab_size": 99,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 64,
                "max_window_layers": 2,
                "rope_theta": 10000.0,
                "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
                "tie_word_embeddings": False,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 32,
                "embed_dim": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "in_channels": 3,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "text_hidden_size": 32,
            },
            vocab_size=99,
            image_token_id=95,
            video_token_id=96,
            vision_start_token_id=94,
        )

    def test_text_model_forward_shape(self):
        model = LLaVAOneVision1_5TextModel(self.config.text_config)
        model.eval()
        input_ids = paddle.randint(0, 90, [2, 5], dtype="int64")

        with paddle.no_grad():
            output = model(input_ids=input_ids, return_dict=True)

        self.assertEqual(output.last_hidden_state.shape, [2, 5, 32])

    def test_conditional_generation_text_only_shape(self):
        model = LLaVAOneVision1_5ForConditionalGeneration(self.config)
        model.eval()
        input_ids = paddle.randint(0, 90, [2, 5], dtype="int64")

        with paddle.no_grad():
            output = model(input_ids=input_ids)

        self.assertEqual(output.logits.shape, [2, 5, 99])

    def test_conditional_generation_image_shape(self):
        model = LLaVAOneVision1_5ForConditionalGeneration(self.config)
        model.eval()
        input_ids = paddle.to_tensor(
            [[1, self.config.vision_start_token_id, self.config.image_token_id, 2, 3]], dtype="int64"
        )
        pixel_values = paddle.randn([4, 3 * 14 * 14], dtype="float32")
        image_grid_thw = paddle.to_tensor([[1, 2, 2]], dtype="int64")

        with paddle.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

        self.assertEqual(output.logits.shape, [1, 5, 99])

    def test_auto_model_for_conditional_generation_from_config(self):
        model = AutoModelForConditionalGeneration.from_config(self.config)
        self.assertIsInstance(model, LLaVAOneVision1_5Model)

    def test_auto_model_for_conditional_generation_from_pretrained(self):
        model = LLaVAOneVision1_5ForConditionalGeneration(self.config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, save_to_hf=False, save_checkpoint_format="")
            reloaded = AutoModelForConditionalGeneration.from_pretrained(
                tmp_dir,
                convert_from_hf=False,
                load_checkpoint_format="",
            )
        self.assertIsInstance(reloaded, LLaVAOneVision1_5ForConditionalGeneration)


if __name__ == "__main__":
    unittest.main()
