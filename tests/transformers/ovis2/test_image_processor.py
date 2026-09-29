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

import tempfile
import unittest

import numpy as np
import paddle
from PIL import Image

from paddleformers.transformers import Ovis2ImageProcessor
from paddleformers.transformers.ovis2.image_processor import (
    get_all_supported_aspect_ratios,
    get_min_tile_covering_grid,
    split_image_into_grid,
)


class Ovis2ImageProcessorTest(unittest.TestCase):
    image_processor_dict = {
        "do_resize": True,
        "size": {"height": 16, "width": 16},
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "do_convert_rgb": True,
        "crop_to_patches": False,
        "min_patches": 1,
        "max_patches": 6,
        "use_covering_area_grid": True,
    }

    def setUp(self):
        self.processor = Ovis2ImageProcessor(**self.image_processor_dict)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.processor.save_pretrained(tmpdir)
            loaded = Ovis2ImageProcessor.from_pretrained(tmpdir)

        self.assertEqual(loaded.to_dict(), self.processor.to_dict())

    def test_preprocess_pil_batch_returns_paddle_tensors(self):
        images = [
            Image.fromarray(np.zeros((20, 24, 3), dtype=np.uint8)),
            Image.fromarray(np.full((32, 18, 3), 255, dtype=np.uint8)),
        ]

        inputs = self.processor(images, return_tensors="pd")

        self.assertIsInstance(inputs["pixel_values"], paddle.Tensor)
        self.assertEqual(tuple(inputs["pixel_values"].shape), (2, 3, 16, 16))
        self.assertEqual(inputs["grids"].tolist(), [[1, 1], [1, 1]])
        self.assertTrue(paddle.allclose(inputs["pixel_values"][0], paddle.full([3, 16, 16], -1.0)))
        self.assertTrue(paddle.allclose(inputs["pixel_values"][1], paddle.ones([3, 16, 16])))

    def test_crop_to_patches_adds_global_patch(self):
        processor = Ovis2ImageProcessor(
            **{**self.image_processor_dict, "crop_to_patches": True}
        )
        image = Image.fromarray(np.zeros((16, 32, 3), dtype=np.uint8))

        inputs = processor(image, return_tensors="np")

        # A 2:1 image uses a 2x1 grid and prepends one global thumbnail.
        self.assertEqual(inputs["grids"].tolist(), [[2, 1]])
        self.assertEqual(inputs["pixel_values"].shape, (3, 3, 16, 16))

    def test_crop_to_patches_keeps_single_patch_without_thumbnail(self):
        image = np.zeros((3, 16, 16), dtype=np.uint8)

        patches, grid = self.processor.crop_image_to_patches(
            image,
            min_patches=1,
            max_patches=6,
            patch_size={"height": 16, "width": 16},
            resample_filter=self.processor.resample,
        )

        self.assertEqual(grid, [1, 1])
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].shape, (3, 16, 16))

    def test_grid_helpers(self):
        self.assertEqual(get_all_supported_aspect_ratios(2, 3), [(1, 2), (2, 1), (1, 3), (3, 1)])
        self.assertEqual(get_min_tile_covering_grid((16, 32), 16, 6), (1, 2))
        self.assertEqual(
            split_image_into_grid(20, 30, (2, 3)),
            [
                (0, 0, 10, 10),
                (10, 0, 20, 10),
                (20, 0, 30, 10),
                (0, 10, 10, 20),
                (10, 10, 20, 20),
                (20, 10, 30, 20),
            ],
        )

    def test_invalid_image_raises(self):
        with self.assertRaisesRegex(ValueError, "flat list of images|Invalid image type"):
            self.processor(["not-an-image"])


if __name__ == "__main__":
    unittest.main()
