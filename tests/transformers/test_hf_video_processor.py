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

from paddleformers.transformers import AutoVideoProcessor
from tests.testing_utils import skip_for_none_ce_case


class TestHFMultiSourceVideoProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import requests
        from PIL import Image

        IMAGE_URL = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
        response = requests.get(IMAGE_URL, stream=True)
        cls.video = [Image.open(response.raw).convert("RGB")] * 5  # load by img_list

        VIDEO_URL = "http://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"
        cls.video_url = VIDEO_URL  # load by url (only for ce)

    def preprocess(self, video_processor):
        inputs = video_processor(self.video, return_tensors="pd")
        EXPECTED_PIXEL_VALUES_MEAN = paddle.to_tensor(0.29226735)
        EXPECTED_PIXEL_VALUES_MAX = paddle.to_tensor(2.14589715)
        EXPECTED_IMAGE_GRID_THW = [[3, 62, 92]]
        self.assertIsInstance(inputs["pixel_values_videos"], paddle.Tensor)
        self.assertIsInstance(inputs["video_grid_thw"], paddle.Tensor)
        self.assertEqual(inputs["pixel_values_videos"].shape, [17112, 1176])
        self.assertEqual(inputs["pixel_values_videos"].dtype, paddle.float32)
        self.assertEqual(inputs["video_grid_thw"].tolist(), EXPECTED_IMAGE_GRID_THW)
        self.assertTrue(
            paddle.allclose(
                paddle.to_tensor(paddle.mean(inputs["pixel_values_videos"])),
                EXPECTED_PIXEL_VALUES_MEAN,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            paddle.allclose(
                paddle.to_tensor(paddle.max(inputs["pixel_values_videos"])),
                EXPECTED_PIXEL_VALUES_MAX,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    # def test_ai_studio(self):
    #     video_processor = AutoVideoProcessor.from_pretrained(
    #         "ModelHub/Qwen2.5-VL-3B-Instruct", download_hub="aistudio"
    #     )
    #     self.preprocess(video_processor)

    @skip_for_none_ce_case
    def test_model_scope(self):
        video_processor = AutoVideoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", download_hub="modelscope")
        self.preprocess(video_processor)

    @skip_for_none_ce_case
    def test_hf_hub(self):
        video_processor = AutoVideoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", download_hub="huggingface")
        self.preprocess(video_processor)

    @skip_for_none_ce_case
    def test_preprocess_consistency_with_hf(self):
        from transformers import AutoVideoProcessor as AutoVideoProcessor_hf

        video_processor_pd = AutoVideoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", download_hub="huggingface"
        )
        video_processor_hf = AutoVideoProcessor_hf.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        inputs_pd = video_processor_pd(self.video_url, return_tensors="pd")
        inputs_hf = video_processor_hf(self.video_url, return_tensors="pt")

        self.assertTrue(
            paddle.allclose(
                paddle.to_tensor(inputs_hf["pixel_values_videos"].numpy()),
                inputs_pd["pixel_values_videos"],
                atol=3e-1,
                rtol=1e-5,
            )
        )
