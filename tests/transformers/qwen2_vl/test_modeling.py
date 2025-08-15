# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

import unittest

import numpy as np
import paddle
import requests
from PIL import Image

from paddleformers.transformers import (
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import ModelTesterMixin


class Qwen2VLModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_name_or_path, from_modelscope=True)

    def get_config(self):
        test_config = {
            "_name_or_path": "./",
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "video_token_id": 151656,
            "hidden_act": "silu",
            "hidden_size": 1536,
            "initializer_range": 0.02,
            "intermediate_size": 8960,
            "max_position_embeddings": 32768,
            "max_window_layers": 28,
            "model_type": "qwen2_vl",
            "num_attention_heads": 12,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "tie_word_embeddings": True,
            "dtype": "bfloat16",
            "use_cache": True,
            "use_sliding_window": False,
            "vision_config": {
                "depth": 32,
                "embed_dim": 1280,
                "mlp_ratio": 4,
                "num_heads": 16,
                "in_chans": 3,
                "hidden_size": 1536,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "temporal_patch_size": 2,
            },
            "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
            "vocab_size": 151936,
        }
        return Qwen2VLConfig(**test_config)

    def prepare_config_and_inputs(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        image_url = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.processor(
            text=[text],
            images=image,
            padding=True,
            return_tensors="pd",
        )
        config = self.get_config()
        return config, inputs

    def prepare_config_and_inputs_for_common(self):
        config, inputs = self.prepare_config_and_inputs()
        return config, inputs

    def create_and_check_model(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        config = self.get_config()
        model = Qwen2VLForConditionalGeneration(config)
        model.eval()
        with paddle.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        self.parent.assertIsNotNone(result)


class Qwen2VLModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Qwen2VLForConditionalGeneration,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        # model tester instance
        self.model_tester = Qwen2VLModelTester(self)

        self.config_tester = ConfigTester(
            self,
            config_class=Qwen2VLConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            # Handle both tuple outputs and model output objects
            if hasattr(first, "logits"):
                first = first.logits
                second = second.logits
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 5e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                first = model(**inputs_dict)
                second = model(**inputs_dict)

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(**inputs_dict)

    def test_model_from_pretrained(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_tester.model_name_or_path)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
