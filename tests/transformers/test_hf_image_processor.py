# coding=utf-8
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

import unittest

import paddle

from paddleformers.transformers import AutoImageProcessor
from tests.testing_utils import skip_for_none_ce_case


class TestHFMultiSourceImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import requests
        from PIL import Image

        IMAGE_URL = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
        response = requests.get(IMAGE_URL, stream=True)
        cls.image = Image.open(response.raw).convert("RGB")

    def preprocess(self, image_processor):
        inputs = image_processor(self.image, return_tensors="pd")
        EXPECTED_PIXEL_VALUES_MEAN = paddle.to_tensor(0.29226744)
        EXPECTED_PIXEL_VALUES_MAX = paddle.to_tensor(2.14589691)
        EXPECTED_IMAGE_GRID_THW = [[1, 62, 92]]
        self.assertIsInstance(inputs["pixel_values"], paddle.Tensor)
        self.assertIsInstance(inputs["image_grid_thw"], paddle.Tensor)
        self.assertEqual(inputs["pixel_values"].shape, [5704, 1176])
        self.assertEqual(inputs["pixel_values"].dtype, paddle.float32)
        self.assertEqual(inputs["image_grid_thw"].tolist(), EXPECTED_IMAGE_GRID_THW)
        self.assertTrue(
            paddle.allclose(
                paddle.to_tensor(paddle.mean(inputs["pixel_values"])),
                EXPECTED_PIXEL_VALUES_MEAN,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            paddle.allclose(
                paddle.to_tensor(paddle.max(inputs["pixel_values"])), EXPECTED_PIXEL_VALUES_MAX, atol=1e-5, rtol=1e-5
            )
        )

    # def test_ai_studio(self):
    #     image_processor = AutoImageProcessor.from_pretrained(
    #         "ModelHub/Qwen2.5-VL-3B-Instruct", download_hub="aistudio"
    #     )
    #     self.preprocess(image_processor)

    def test_model_scope(self):
        image_processor = AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", download_hub="modelscope")
        self.preprocess(image_processor)

    @skip_for_none_ce_case
    def test_hf_hub(self):
        image_processor = AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", download_hub="huggingface")
        self.preprocess(image_processor)

    @skip_for_none_ce_case
    def test_preprocess_consistency_with_hf(self):
        from transformers import AutoImageProcessor as AutoImageProcessor_hf

        image_processor_pd = AutoImageProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", download_hub="huggingface"
        )
        image_processor_hf = AutoImageProcessor_hf.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=False)
        inputs_pd = image_processor_pd(self.image, return_tensors="pd")
        inputs_hf = image_processor_hf(self.image, return_tensors="pt")

        self.assertTrue(
            paddle.allclose(
                paddle.to_tensor(inputs_hf["pixel_values"].numpy()), inputs_pd["pixel_values"], atol=1e-5, rtol=1e-5
            )
        )
