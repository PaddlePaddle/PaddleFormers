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

import numpy as np

from paddleformers.transformers.image_processing_utils import PaddleImageProcessingMixin


class _LegacyImageProcessor:
    def __init__(self, **kwargs):
        pass

    def rescale(self, image, scale, **kwargs):
        return (image.astype(np.float64) * scale).astype(np.float32)

    def normalize(self, image, mean, std, **kwargs):
        mean = np.asarray(mean, dtype=image.dtype)
        std = np.asarray(std, dtype=image.dtype)
        return (image - mean) / std


class _TestImageProcessor(PaddleImageProcessingMixin, _LegacyImageProcessor):
    methods_to_wrap = []


class TestPaddleImageProcessingMixin(unittest.TestCase):
    def test_accuracy_compatible_rescale_normalize_downcasts_once(self):
        pixels = np.asarray([[[65, 66, 67]]], dtype=np.uint8)
        processor = _TestImageProcessor()

        legacy = processor.normalize(
            processor.rescale(pixels, 1 / 255, input_data_format="channels_last"),
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            input_data_format="channels_last",
        )

        processor.accuracy_compatible_rescale_normalize = True
        aligned = processor.normalize(
            processor.rescale(pixels, 1 / 255, input_data_format="channels_last"),
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            input_data_format="channels_last",
        )
        expected = ((pixels.astype(np.float64) / 255 - 0.5) / 0.5).astype(np.float32)

        self.assertEqual(aligned.dtype, np.float32)
        np.testing.assert_array_equal(aligned, expected)
        self.assertFalse(np.array_equal(legacy, expected))


if __name__ == "__main__":
    unittest.main()
