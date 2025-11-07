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

from paddleformers.transformers import AutoProcessor
from paddleformers.transformers.qwen2_vl import process_vision_info
from tests.testing_utils import skip_for_none_ce_case


class TestHFMultiSourceProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import requests
        from PIL import Image

        IMAGE_URL = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
        response = requests.get(IMAGE_URL, stream=True)
        cls.image = Image.open(response.raw).convert("RGB")
        cls.video = [cls.image.copy()] * 5  # load by img_list

        cls.messages_with_image = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe this image."}, {"type": "image", "image": cls.image}],
            }
        ]

        cls.messages_with_video = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe this video."}, {"type": "video", "video": cls.video}],
            }
        ]

    def preprocess_image(self, processor):
        text = processor.apply_chat_template(self.messages_with_image, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(self.messages_with_image)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        EXPECTED_INPUT_IDS = paddle.to_tensor([151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645])
        EXPECTED_PIXEL_VALUES_MEAN = paddle.to_tensor(0.29226744)
        EXPECTED_PIXEL_VALUES_MAX = paddle.to_tensor(2.14589691)
        EXPECTED_IMAGE_GRID_THW = [[1, 62, 92]]
        self.assertIsInstance(inputs["input_ids"], paddle.Tensor)
        self.assertIsInstance(inputs["attention_mask"], paddle.Tensor)
        self.assertIsInstance(inputs["pixel_values"], paddle.Tensor)
        self.assertIsInstance(inputs["image_grid_thw"], paddle.Tensor)
        self.assertEqual(inputs["input_ids"].shape, [1, 1451])
        self.assertEqual(inputs["attention_mask"].shape, [1, 1451])
        self.assertEqual(inputs["pixel_values"].shape, [5704, 1176])
        self.assertEqual(inputs["pixel_values"].dtype, paddle.float32)
        self.assertEqual(inputs["image_grid_thw"].tolist(), EXPECTED_IMAGE_GRID_THW)
        self.assertTrue(
            paddle.allclose(
                inputs["input_ids"][0, :10],
                EXPECTED_INPUT_IDS,
            )
        )
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
                paddle.to_tensor(paddle.max(inputs["pixel_values"])),
                EXPECTED_PIXEL_VALUES_MAX,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def preprocess_video(self, processor):
        text = processor.apply_chat_template(self.messages_with_video, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(self.messages_with_video)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        EXPECTED_INPUT_IDS = paddle.to_tensor([151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645])
        EXPECTED_PIXEL_VALUES_MEAN = paddle.to_tensor(0.29228371)
        EXPECTED_PIXEL_VALUES_MAX = paddle.to_tensor(2.11745691)
        EXPECTED_VIDEO_GRID_THW = [[3, 46, 66]]
        EXPECTED_SECOND_PER_GRID_TS = [1]
        self.assertIsInstance(inputs["input_ids"], paddle.Tensor)
        self.assertIsInstance(inputs["attention_mask"], paddle.Tensor)
        self.assertIsInstance(inputs["pixel_values_videos"], paddle.Tensor)
        self.assertIsInstance(inputs["video_grid_thw"], paddle.Tensor)
        self.assertEqual(inputs["input_ids"].shape, [1, 2302])
        self.assertEqual(inputs["attention_mask"].shape, [1, 2302])
        self.assertEqual(inputs["pixel_values_videos"].shape, [9108, 1176])
        self.assertEqual(inputs["pixel_values_videos"].dtype, paddle.float32)
        self.assertEqual(inputs["video_grid_thw"].tolist(), EXPECTED_VIDEO_GRID_THW)
        self.assertEqual(inputs["second_per_grid_ts"].tolist(), EXPECTED_SECOND_PER_GRID_TS)
        self.assertTrue(
            paddle.allclose(
                inputs["input_ids"][0, :10],
                EXPECTED_INPUT_IDS,
            )
        )
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

    #     processor = AutoProcessor.from_pretrained(
    #         "ModelHub/Qwen2.5-VL-3B-Instruct", download_hub="aistudio"
    #     )
    #     self.preprocess_image(processor)
    #     self.preprocess_video(processor)

    @skip_for_none_ce_case
    def test_model_scope(self):
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", download_hub="modelscope")
        self.preprocess_image(processor)
        self.preprocess_video(processor)

    @skip_for_none_ce_case
    def test_hf_hub(self):
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", download_hub="huggingface")
        self.preprocess_image(processor)
        self.preprocess_video(processor)
