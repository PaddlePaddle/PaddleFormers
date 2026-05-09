# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import tempfile
import unittest

import paddle

from paddleformers.transformers import AutoImageProcessor
from paddleformers.transformers.glm_ocr.image_processor import Glm46VImageProcessor


class GlmOcrImageProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import requests
        from PIL import Image

        IMAGE_URL = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
        response = requests.get(IMAGE_URL, stream=True)
        cls.image = Image.open(response.raw).convert("RGB")
        cls.model_path = "PaddleFormers/tiny-random-glmocr"

    def test_image_processor_output_keys(self):
        """Verify that output contains pixel_values and image_grid_thw"""
        processor = Glm46VImageProcessor.from_pretrained(self.model_path)
        inputs = processor(self.image, return_tensors="pd")

        self.assertIn("pixel_values", inputs)
        self.assertIn("image_grid_thw", inputs)

    def test_pixel_values_dtype(self):
        """Verify that pixel_values is float32"""
        processor = Glm46VImageProcessor.from_pretrained(self.model_path)
        inputs = processor(self.image, return_tensors="pd")

        self.assertEqual(inputs["pixel_values"].dtype, paddle.float32)

    def test_image_grid_thw_shape(self):
        """Verify that image_grid_thw shape is [1, 3]"""
        processor = Glm46VImageProcessor.from_pretrained(self.model_path)
        inputs = processor(self.image, return_tensors="pd")

        self.assertEqual(inputs["image_grid_thw"].shape[0], 1)
        self.assertEqual(inputs["image_grid_thw"].shape[1], 3)

    def test_pixel_values_patch_count(self):
        """Verify that pixel_values row count equals grid_t * grid_h * grid_w"""
        processor = Glm46VImageProcessor.from_pretrained(self.model_path)
        inputs = processor(self.image, return_tensors="pd")

        thw = inputs["image_grid_thw"][0]
        expected_patches = int(thw[0]) * int(thw[1]) * int(thw[2])
        self.assertEqual(inputs["pixel_values"].shape[0], expected_patches)

    def test_save_and_reload(self):
        """Verify that output remains consistent after saving and reloading"""
        processor = Glm46VImageProcessor.from_pretrained(self.model_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            processor2 = Glm46VImageProcessor.from_pretrained(tmpdir)

        inputs1 = processor(self.image, return_tensors="pd")
        inputs2 = processor2(self.image, return_tensors="pd")

        self.assertTrue(paddle.allclose(inputs1["pixel_values"], inputs2["pixel_values"]))

    def test_auto_image_processor(self):
        """Verify that AutoImageProcessor correctly loads Glm46VImageProcessor"""
        processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.assertIsInstance(processor, Glm46VImageProcessor)

    def test_batch_images(self):
        """Verify batch image input"""
        processor = Glm46VImageProcessor.from_pretrained(self.model_path)
        inputs = processor([self.image, self.image], return_tensors="pd")

        self.assertEqual(inputs["image_grid_thw"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
