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

from paddleformers.transformers import LLaVAOneVision1_5TextConfig, Llavaonevision1_5Config, RiceConfig
from tests.transformers.test_configuration_common import ConfigTester


class Llavaonevision1_5ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_tester = ConfigTester(
            self,
            config_class=Llavaonevision1_5Config,
            has_text_modality=True,
            common_properties=["vocab_size"],
            text_config={
                "vocab_size": 99,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_window_layers": 2,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "text_hidden_size": 32,
            },
            image_token_id=95,
            video_token_id=96,
            vocab_size=99,
        )

    def test_config_common_properties(self):
        self.config_tester.create_and_test_config_common_properties()

    def test_config_to_json_string(self):
        self.config_tester.create_and_test_config_to_json_string()

    def test_config_to_json_file(self):
        self.config_tester.create_and_test_config_to_json_file()

    def test_sub_configs(self):
        config = Llavaonevision1_5Config()
        self.assertIsInstance(config.text_config, LLaVAOneVision1_5TextConfig)
        self.assertIsInstance(config.vision_config, RiceConfig)
        self.assertEqual(config.model_type, "llavaonevision1_5")
        self.assertEqual(config.text_config.model_type, "llavaonevision1_5_text")
        self.assertEqual(config.vision_config.model_type, "rice_vit")


if __name__ == "__main__":
    unittest.main()
