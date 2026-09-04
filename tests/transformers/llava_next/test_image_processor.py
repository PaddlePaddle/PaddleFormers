# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import paddle
from PIL import Image

from paddleformers.transformers import AutoImageProcessor, LlavaNextImageProcessor


class LlavaNextImageProcessorTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8))
        self.image_processor = LlavaNextImageProcessor(
            image_grid_pinpoints=[[16, 16]],
            size={"shortest_edge": 16},
            crop_size={"height": 16, "width": 16},
        )

    def test_output_keys_and_shapes(self):
        inputs = self.image_processor(self.image, return_tensors="pd")

        self.assertIn("pixel_values", inputs)
        self.assertIn("image_sizes", inputs)
        self.assertEqual(inputs["pixel_values"].shape, [1, 2, 3, 16, 16])
        self.assertEqual(inputs["image_sizes"].shape, [1, 2])
        self.assertEqual(inputs["pixel_values"].dtype, paddle.float32)

    def test_batch_padding(self):
        wide_image = Image.fromarray(np.full((16, 32, 3), 255, dtype=np.uint8))
        image_processor = LlavaNextImageProcessor(
            image_grid_pinpoints=[[16, 16], [16, 32]],
            size={"shortest_edge": 16},
            crop_size={"height": 16, "width": 16},
        )

        inputs = image_processor([self.image, wide_image], return_tensors="pd")

        self.assertEqual(inputs["pixel_values"].shape[0], 2)
        self.assertEqual(inputs["pixel_values"].shape[1], 3)
        self.assertEqual(inputs["image_sizes"].tolist(), [[16, 16], [16, 32]])

    def test_rescale_factor(self):
        image_processor = LlavaNextImageProcessor(
            image_grid_pinpoints=[[16, 16]],
            size={"shortest_edge": 16},
            crop_size={"height": 16, "width": 16},
            do_normalize=False,
            rescale_factor=1.0,
        )

        inputs = image_processor(self.image, return_tensors="np")

        self.assertGreater(inputs["pixel_values"][0][0].max(), 1.0)

    def test_unsupported_resize_and_data_format_raise(self):
        with self.assertRaises(ValueError):
            self.image_processor(self.image, do_resize=False)

    def test_input_data_format_and_channels_last(self):
        chw = np.full((3, 16, 16), 127, dtype=np.uint8)
        inputs = self.image_processor(
            chw,
            input_data_format="channels_first",
            data_format="channels_last",
            return_tensors="np",
        )

        self.assertEqual(inputs["pixel_values"].shape[-1], 3)

    def test_do_center_crop_runs(self):
        inputs = self.image_processor(self.image, do_center_crop=True, return_tensors="pd")

        self.assertEqual(inputs["pixel_values"].shape, [1, 2, 3, 16, 16])

    def test_save_load_with_auto_image_processor(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.image_processor.save_pretrained(tmpdirname)
            reloaded = AutoImageProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(reloaded, LlavaNextImageProcessor)
        inputs = reloaded(self.image, return_tensors="pd")
        self.assertEqual(inputs["pixel_values"].shape, [1, 2, 3, 16, 16])


if __name__ == "__main__":
    unittest.main()
