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

from paddleformers.transformers import AutoImageProcessor, Idefics3ImageProcessor
from tests.testing_utils import gpu_device_initializer


class Idefics3ImageProcessorTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="Idefics3ImageProcessorTest", gpu_id=0)
    def setUp(self):
        self.image = Image.fromarray(np.full((8, 10, 3), 127, dtype=np.uint8))
        self.processor = Idefics3ImageProcessor(
            size={"longest_edge": 8},
            max_image_size={"longest_edge": 8},
            do_image_splitting=False,
        )

    def test_output_keys_and_shapes(self):
        inputs = self.processor(self.image, return_tensors="pd")

        self.assertIn("pixel_values", inputs)
        self.assertEqual(inputs["pixel_values"].shape, [1, 3, 8, 8])
        self.assertEqual(inputs["pixel_values"].dtype, paddle.float32)

    def test_return_row_col_info(self):
        inputs = self.processor(self.image, return_row_col_info=True)

        self.assertEqual(inputs["rows"], [[0]])
        self.assertEqual(inputs["cols"], [[0]])

    def test_batch_padding(self):
        large_image = Image.fromarray(np.full((8, 16, 3), 255, dtype=np.uint8))
        processor = Idefics3ImageProcessor(
            size={"longest_edge": 16},
            max_image_size={"longest_edge": 8},
            do_image_splitting=True,
        )

        inputs = processor([self.image, large_image], return_row_col_info=True, return_tensors="pd")

        self.assertGreater(inputs["pixel_values"].shape[0], 2)
        self.assertEqual(inputs["pixel_values"].shape[1], 3)
        self.assertEqual(inputs["rows"], [[2, 1]])
        self.assertEqual(inputs["cols"], [[2, 2]])

    def test_nested_samples_use_image_axis_for_concat(self):
        large_image = Image.fromarray(np.full((8, 16, 3), 255, dtype=np.uint8))
        processor = Idefics3ImageProcessor(
            size={"longest_edge": 16},
            max_image_size={"longest_edge": 8},
            do_image_splitting=True,
        )

        first = processor([[self.image]], return_tensors="pd")
        second = processor([[large_image]], return_tensors="pd")
        batch = processor([[self.image], [large_image]], return_tensors="pd")
        concatenated = paddle.concat([first["pixel_values"], second["pixel_values"]], axis=0)

        self.assertEqual(batch["pixel_values"].shape, concatenated.shape)

    def test_do_pad_false_keeps_flat_image_axis(self):
        inputs = self.processor(
            [[self.image], [self.image]],
            do_pad=False,
            return_tensors="pd",
        )

        self.assertEqual(inputs["pixel_values"].shape, [2, 3, 8, 8])

    def test_local_image_path_is_supported(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            self.image.save(image_file.name)
            inputs = self.processor(image_file.name, return_tensors="pd")

        self.assertEqual(inputs["pixel_values"].shape, [1, 3, 8, 8])

    def test_save_load_with_auto_image_processor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.processor.save_pretrained(tmpdir)
            reloaded = AutoImageProcessor.from_pretrained(tmpdir)

        self.assertIsInstance(reloaded, Idefics3ImageProcessor)
        inputs = reloaded(self.image, return_tensors="pd")
        self.assertEqual(inputs["pixel_values"].shape, [1, 3, 8, 8])


if __name__ == "__main__":
    unittest.main()
