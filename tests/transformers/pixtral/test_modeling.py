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

from paddleformers.transformers import AutoModel
from paddleformers.transformers.pixtral import PixtralVisionConfig, PixtralVisionModel


class PixtralVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=32,
        patch_size=4,
        num_channels=3,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_act="gelu",
        attention_dropout=0.0,
        rope_theta=10000.0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.num_patches = (image_size // patch_size) ** 2

    def get_config(self):
        return PixtralVisionConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_channels=self.num_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_act=self.hidden_act,
            attention_dropout=self.attention_dropout,
            rope_theta=self.rope_theta,
        )

    def get_pixel_values(self):
        return paddle.randn(
            [self.batch_size, self.num_channels, self.image_size, self.image_size],
            dtype=paddle.float32,
        )

    def create_and_check_vision_model(self):
        config = self.get_config()
        pixel_values = self.get_pixel_values()
        model = PixtralVisionModel(config)
        model.eval()

        with paddle.no_grad():
            result = model(pixel_values)

        # Pixtral concatenates all images into a single sequence with batch=1
        total_patches = self.batch_size * self.num_patches
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            [1, total_patches, self.hidden_size],
        )

    def create_and_check_vision_model_output_hidden_states(self):
        config = self.get_config()
        pixel_values = self.get_pixel_values()
        model = PixtralVisionModel(config)
        model.eval()

        with paddle.no_grad():
            result = model(pixel_values, output_hidden_states=True)

        total_patches = self.batch_size * self.num_patches
        # num_hidden_layers + 1 (input embedding)
        self.parent.assertEqual(len(result.hidden_states), self.num_hidden_layers + 1)
        for hs in result.hidden_states:
            self.parent.assertEqual(hs.shape, [1, total_patches, self.hidden_size])

    def create_and_check_vision_model_single_image(self):
        """Test vision model with a single image."""
        config = self.get_config()
        model = PixtralVisionModel(config)
        model.eval()

        pixel_values = paddle.randn([1, self.num_channels, self.image_size, self.image_size])
        with paddle.no_grad():
            result = model(pixel_values)

        expected_patches = self.num_patches
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            [1, expected_patches, self.hidden_size],
        )

    def create_and_check_vision_model_deterministic(self):
        """Verify deterministic outputs."""
        config = self.get_config()
        pixel_values = self.get_pixel_values()
        model = PixtralVisionModel(config)
        model.eval()

        with paddle.no_grad():
            out1 = model(pixel_values).last_hidden_state
            out2 = model(pixel_values).last_hidden_state

        self.parent.assertTrue(paddle.allclose(out1, out2))


class PixtralModelTest(unittest.TestCase):
    def setUp(self):
        paddle.seed(1234)
        self.model_tester = PixtralVisionModelTester(self)

    def test_config_save_load(self):
        config = self.model_tester.get_config()

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            loaded_config = PixtralVisionConfig.from_pretrained(tmpdirname)

        self.assertEqual(loaded_config.model_type, "pixtral")
        self.assertEqual(loaded_config.hidden_size, config.hidden_size)
        self.assertEqual(loaded_config.num_hidden_layers, config.num_hidden_layers)
        self.assertEqual(loaded_config.num_attention_heads, config.num_attention_heads)
        self.assertEqual(loaded_config.patch_size, config.patch_size)
        self.assertEqual(loaded_config.image_size, config.image_size)
        self.assertEqual(loaded_config.rope_theta, config.rope_theta)

    def test_vision_model(self):
        self.model_tester.create_and_check_vision_model()

    def test_vision_model_output_hidden_states(self):
        self.model_tester.create_and_check_vision_model_output_hidden_states()

    def test_vision_model_uses_config_output_hidden_states(self):
        config = self.model_tester.get_config()
        config.output_hidden_states = True
        model = PixtralVisionModel(config)
        model.eval()

        with paddle.no_grad():
            result = model(self.model_tester.get_pixel_values())

        self.assertEqual(len(result.hidden_states), config.num_hidden_layers + 1)

    def test_vision_model_recompute(self):
        config = self.model_tester.get_config()
        config.recompute_granularity = "full"
        config.recompute_use_reentrant = False
        model = PixtralVisionModel(config)
        model.train()

        result = model(self.model_tester.get_pixel_values())
        result.last_hidden_state.square().mean().backward()

        self.assertIsNotNone(model.transformer.layers[0].attention.q_proj.weight.grad)

    def test_vision_model_single_image(self):
        self.model_tester.create_and_check_vision_model_single_image()

    def test_vision_model_deterministic(self):
        self.model_tester.create_and_check_vision_model_deterministic()

    def test_aoa_config_format(self):
        config = self.model_tester.get_config()
        aoa_config = PixtralVisionModel._gen_aoa_config(config)
        inv_aoa_config = PixtralVisionModel._gen_inv_aoa_config(config)

        self.assertIn("aoa_statements", aoa_config)
        self.assertIn("aoa_statements", inv_aoa_config)
        self.assertIn(
            "vision_encoder.patch_conv.weight -> patch_conv.weight",
            aoa_config["aoa_statements"],
        )
        self.assertIn(
            "patch_conv.weight -> vision_encoder.patch_conv.weight",
            inv_aoa_config["aoa_statements"],
        )
        self.assertIn(
            "transformer.layers.$LAYER_ID.attention.q_proj.weight^T -> "
            "vision_encoder.transformer.layers.$LAYER_ID.attention.q_proj.weight",
            inv_aoa_config["aoa_statements"],
        )

    def test_auto_model_save_load(self):
        model = PixtralVisionModel(self.model_tester.get_config())
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            loaded_model = AutoModel.from_pretrained(tmpdirname)

        self.assertIsInstance(loaded_model, PixtralVisionModel)
        loaded_state_dict = loaded_model.state_dict()
        for name, value in model.state_dict().items():
            self.assertIn(name, loaded_state_dict)
            paddle.testing.assert_close(value, loaded_state_dict[name])


if __name__ == "__main__":
    unittest.main()
