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
from unittest.mock import MagicMock

from PIL import Image

from paddleformers.datasets.template.augment_utils import (
    JpegCompression,
    RandomApply,
    RandomDiscreteRotation,
    RandomScale,
    RandomSingleSidePadding,
)


class TestRandomApply(unittest.TestCase):
    """Tests for RandomApply."""

    def test_apply_with_probability_one(self):
        transform = MagicMock(return_value=MagicMock())
        ra = RandomApply([transform], p=1.0)
        img = Image.new("RGB", (100, 100))
        ra(img)
        transform.assert_called_once_with(img)

    def test_apply_with_probability_zero(self):
        transform = MagicMock(return_value=MagicMock())
        ra = RandomApply([transform], p=0.0)
        img = Image.new("RGB", (100, 100))
        result = ra(img)
        transform.assert_not_called()
        self.assertEqual(result, img)

    def test_multiple_transforms(self):
        t1 = MagicMock(side_effect=lambda x: x)
        t2 = MagicMock(side_effect=lambda x: x)
        ra = RandomApply([t1, t2], p=1.0)
        img = Image.new("RGB", (100, 100))
        ra(img)
        t1.assert_called_once()
        t2.assert_called_once()


class TestRandomDiscreteRotation(unittest.TestCase):
    """Tests for RandomDiscreteRotation."""

    def test_rotation(self):
        rde = RandomDiscreteRotation(degrees=[0, 90, 180, 270])
        img = Image.new("RGB", (100, 100))
        result = rde(img)
        self.assertIsInstance(result, Image.Image)

    def test_zero_degree_rotation(self):
        rde = RandomDiscreteRotation(degrees=[0])
        img = Image.new("RGB", (100, 100))
        result = rde(img)
        self.assertEqual(result.size, img.size)


class TestJpegCompression(unittest.TestCase):
    """Tests for JpegCompression."""

    def test_compression(self):
        jc = JpegCompression(quality_range=(50, 90))
        img = Image.new("RGB", (100, 100))
        result = jc(img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")

    def test_low_quality(self):
        jc = JpegCompression(quality_range=(10, 10))
        img = Image.new("RGB", (100, 100))
        result = jc(img)
        self.assertIsInstance(result, Image.Image)


class TestRandomScale(unittest.TestCase):
    """Tests for RandomScale."""

    def test_scale(self):
        rs = RandomScale(scale_range=(0.5, 1.5))
        img = Image.new("RGB", (100, 100))
        result = rs(img)
        self.assertIsInstance(result, Image.Image)

    def test_scale_one(self):
        rs = RandomScale(scale_range=(1.0, 1.0))
        img = Image.new("RGB", (100, 100))
        result = rs(img)
        self.assertEqual(result.size[0], 100)
        self.assertEqual(result.size[1], 100)


class TestRandomSingleSidePadding(unittest.TestCase):
    """Tests for RandomSingleSidePadding."""

    def test_basic_padding(self):
        rsp = RandomSingleSidePadding(padding_range=(10, 20), fill="white")
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        result = rsp(img)
        self.assertIsInstance(result, Image.Image)

    def test_zero_padding(self):
        rsp = RandomSingleSidePadding(padding_range=(0, 0), fill="white")
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        result = rsp(img)
        self.assertEqual(result.size, img.size)

    def test_invalid_padding_range(self):
        with self.assertRaises(AssertionError):
            RandomSingleSidePadding(padding_range=5, fill="white")

    def test_padding_range_length_mismatch(self):
        with self.assertRaises(AssertionError):
            RandomSingleSidePadding(padding_range=(1, 2, 3), fill="white")


if __name__ == "__main__":
    unittest.main()
