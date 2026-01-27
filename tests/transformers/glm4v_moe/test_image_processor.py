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

import os
import tempfile
import unittest

import paddle

from paddleformers.transformers import AutoImageProcessor
from paddleformers.utils.log import logger


class Glm4vImageProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import requests
        from PIL import Image

        IMAGE_URL = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
        response = requests.get(IMAGE_URL, stream=True)
        cls.image = Image.open(response.raw).convert("RGB")

        cls.patch_size = 14
        cls.size = {"longest_edge": 9633792, "shortest_edge": 12544}

    def test_slow_image_processor_consistency_with_hf(self):
        with tempfile.TemporaryDirectory() as tempdir:
            image_processor_pd = AutoImageProcessor.from_pretrained(
                "PaddleFormers/tiny-random-glm4vmoe-bf16", use_fast=False
            )
            image_processor_pd.save_pretrained(tempdir)

            from transformers import AutoImageProcessor as AutoImageProcessor_hf

            image_processor_hf = AutoImageProcessor_hf.from_pretrained(tempdir, use_fast=False)
            inputs_pd = image_processor_pd(self.image, return_tensors="pd")
            inputs_hf = image_processor_hf(self.image, return_tensors="pt")

            self.assertTrue(
                paddle.to_tensor(inputs_hf["pixel_values"].numpy())._md5sum() == inputs_pd["pixel_values"]._md5sum()
            )

    def test_fast_image_processor_consistency_with_hf(self):
        # Initialize device when GPU is needed by certain test case
        gpu_count = paddle.device.cuda.device_count()
        pid = os.getpid()

        if gpu_count > 0:
            paddle.set_device(f"gpu:{pid % gpu_count}")
        else:
            paddle.set_device("cpu")
            self.skipTest("No GPU currently available/allocated")
        logger.info(f"Glm4vImageProcessorTest [PID:{pid}] Device initialized: {paddle.get_device()}")

        with tempfile.TemporaryDirectory() as tempdir:
            image_processor_pd = AutoImageProcessor.from_pretrained(
                "PaddleFormers/tiny-random-glm4vmoe-bf16", patch_size=self.patch_size, size=self.size, use_fast=True
            )
            image_processor_pd.save_pretrained(tempdir)

            from transformers import AutoImageProcessor as AutoImageProcessor_hf

            image_processor_hf = AutoImageProcessor_hf.from_pretrained(tempdir, device="cuda", use_fast=True)
            inputs_pd = image_processor_pd(self.image, return_tensors="pd")
            inputs_hf = image_processor_hf(self.image, return_tensors="pt")

            # NOTE: Fallback to CPU leads to precision differences during resize.
            if inputs_hf["pixel_values"].device.type == "cuda":
                self.assertTrue(
                    paddle.to_tensor(inputs_hf["pixel_values"].cpu().numpy())._md5sum()
                    == inputs_pd["pixel_values"]._md5sum()
                )
            else:
                self.assertTrue(
                    paddle.allclose(
                        paddle.to_tensor(inputs_hf["pixel_values"].numpy()),
                        inputs_pd["pixel_values"],
                        rtol=1e-6,
                        atol=1e0,
                    )
                )
