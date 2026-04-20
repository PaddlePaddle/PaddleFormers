# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.transformers import PixtralVisionConfig, PixtralVisionModel
from tests.transformers.test_configuration_common import ConfigTester


class PixtralVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=16,
        patch_size=4,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

    def get_config(self):
        return PixtralVisionConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_channels=self.num_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = paddle.randn([self.batch_size, self.num_channels, self.image_size, self.image_size])
        return config, pixel_values

    def create_and_check_model(self, config, pixel_values):
        model = PixtralVisionModel(config)
        model.eval()
        outputs = model(pixel_values, return_dict=True)
        seq_len = (self.image_size // self.patch_size) ** 2 * self.batch_size
        self.parent.assertEqual(outputs.last_hidden_state.shape, [1, seq_len, self.hidden_size])


class PixtralVisionModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def setUp(self):
        self.model_tester = PixtralVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PixtralVisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, pixel_values)

    def test_save_load(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = PixtralVisionModel(config)
        model.eval()

        with paddle.no_grad():
            original = model(pixel_values, return_dict=True).last_hidden_state

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, save_to_hf=False, save_checkpoint_format="")
            loaded = PixtralVisionModel.from_pretrained(tmp_dir, convert_from_hf=False, load_checkpoint_format="")
            loaded.eval()

            with paddle.no_grad():
                reloaded = loaded(pixel_values, return_dict=True).last_hidden_state

        self.assertTrue(paddle.allclose(original, reloaded))
