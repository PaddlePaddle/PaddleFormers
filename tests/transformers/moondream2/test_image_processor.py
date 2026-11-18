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

import numpy as np
from PIL import Image

from paddleformers.transformers.moondream2.image_processor import Moondream2ImageProcessor, select_tiling

CROP_WINDOW = 266


class Moondream2ImageProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = Moondream2ImageProcessor()

    def test_small_image_uses_single_tile(self):
        # Both sides fit one crop window after removing the overlap margin.
        self.assertEqual(select_tiling(300 - 112, 200 - 112, CROP_WINDOW, 12), (1, 1))

        output = self.processor.preprocess(images=[Image.new("RGB", (300, 200))])
        # One global crop plus a single local tile.
        self.assertEqual(int(output["crop_counts"][0]), 2)
        self.assertEqual(tuple(output["image_sizes"][0]), (1, 1))

    def test_wide_image_uses_aspect_ratio_grid(self):
        output = self.processor.preprocess(images=[Image.new("RGB", (1600, 400))])
        tiles_h, tiles_w = (int(value) for value in output["image_sizes"][0])

        self.assertGreaterEqual(tiles_w, tiles_h)
        self.assertEqual(int(output["crop_counts"][0]), tiles_h * tiles_w + 1)

    def test_large_image_is_capped_at_max_crops(self):
        output = self.processor.preprocess(images=[Image.new("RGB", (6000, 6000))])
        tiles_h, tiles_w = (int(value) for value in output["image_sizes"][0])

        self.assertLessEqual(tiles_h * tiles_w, self.processor.max_crops)
        self.assertLessEqual(int(output["crop_counts"][0]), self.processor.max_crops + 1)

    def test_batch_pads_to_largest_crop_count(self):
        images = [Image.new("RGB", (300, 200)), Image.new("RGB", (1600, 400))]
        output = self.processor.preprocess(images=images)

        crop_counts = [int(value) for value in output["crop_counts"]]
        self.assertNotEqual(crop_counts[0], crop_counts[1])
        self.assertEqual(output["pixel_values"].shape[0], 2)
        self.assertEqual(output["pixel_values"].shape[1], max(crop_counts))
        # Padded crops stay zero so they are visibly inactive.
        self.assertTrue(bool(np.all(output["pixel_values"][0, crop_counts[0] :] == 0)))
        self.assertTrue(bool(np.all(output["pixel_values"][1, crop_counts[1] :] == 0)))

    def test_normalization_matches_reference(self):
        image = Image.fromarray(np.full((300, 300, 3), 128, dtype=np.uint8))
        output = self.processor.preprocess(images=[image])

        # (x / 255 - 0.5) / 0.5 for x = 128
        expected = (128 / 255 - 0.5) / 0.5
        self.assertAlmostEqual(float(output["pixel_values"][0, 0, 0, 10, 10]), expected, places=5)


if __name__ == "__main__":
    unittest.main()
