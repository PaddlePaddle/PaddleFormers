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

import base64
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from paddleformers.utils.image_utils import (
    BaseOperator,
    Bbox,
    DecodeImage,
    NormalizeImage,
    PadBatch,
    Permute,
    ResizeImage,
    check,
    img2base64,
    np2base64,
    pil2base64,
    two_dimension_sort_box,
    two_dimension_sort_layout,
)


class TestBaseOperator(unittest.TestCase):
    def test_default_name(self):
        op = BaseOperator()
        self.assertIn("BaseOperator", str(op))

    def test_custom_name(self):
        op = BaseOperator(name="MyOp")
        self.assertIn("MyOp", str(op))

    def test_call_returns_sample(self):
        op = BaseOperator()
        sample = {"image": "dummy"}
        result = op(sample)
        self.assertEqual(result, sample)

    def test_str_representation(self):
        op = BaseOperator(name="TestOp")
        s = str(op)
        self.assertTrue(s.startswith("TestOp_"))


class TestDecodeImage(unittest.TestCase):
    def setUp(self):
        # Create a simple test image and its base64 encoding
        self.img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        buf = __import__("io").BytesIO()
        self.img.save(buf, format="JPEG")
        self.image_bytes = buf.getvalue()
        self.image_b64 = base64.b64encode(self.image_bytes).decode("utf-8")

    def test_decode_from_base64(self):
        sample = {"im_base64": self.image_b64}
        decoder = DecodeImage()
        result = decoder(sample)
        self.assertIn("image", result)
        self.assertIsInstance(result["image"], np.ndarray)
        self.assertEqual(result["image"].shape[2], 3)  # RGB channels

    def test_decode_sets_h_w(self):
        sample = {"im_base64": self.image_b64}
        decoder = DecodeImage()
        result = decoder(sample)
        self.assertIn("h", result)
        self.assertIn("w", result)
        self.assertEqual(result["h"], result["image"].shape[0])
        self.assertEqual(result["w"], result["image"].shape[1])

    def test_decode_sets_im_info(self):
        sample = {"im_base64": self.image_b64}
        decoder = DecodeImage()
        result = decoder(sample)
        self.assertIn("im_info", result)
        self.assertEqual(len(result["im_info"]), 3)

    def test_decode_with_existing_image(self):
        sample = {"image": self.image_bytes}
        decoder = DecodeImage()
        result = decoder(sample)
        self.assertIn("image", result)
        self.assertIsInstance(result["image"], np.ndarray)

    def test_decode_overrides_h_w_if_mismatch(self):
        sample = {"im_base64": self.image_b64, "h": 999, "w": 999}
        decoder = DecodeImage()
        result = decoder(sample)
        self.assertNotEqual(result["h"], 999)


class TestResizeImage(unittest.TestCase):
    def setUp(self):
        self.img = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)

    def test_resize_with_int_target(self):
        sample = {"image": self.img.copy()}
        resizer = ResizeImage(target_size=32)
        result = resizer(sample)
        self.assertEqual(result["image"].shape[0], 32)
        self.assertEqual(result["image"].shape[1], 32)

    def test_resize_with_list_target(self):
        sample = {"image": self.img.copy()}
        resizer = ResizeImage(target_size=[32, 64])
        # Should pick one of the target sizes
        result = resizer(sample)
        self.assertIn(result["image"].shape[0], [32, 64])
        self.assertIn(result["image"].shape[1], [32, 64])

    def test_invalid_target_size_type(self):
        with self.assertRaises(TypeError):
            ResizeImage(target_size=3.5)

    def test_non_numpy_image_raises(self):
        sample = {"image": [[1, 2, 3]]}
        resizer = ResizeImage(target_size=32)
        with self.assertRaises(TypeError):
            resizer(sample)

    def test_zero_min_size_raises(self):
        img = np.zeros((0, 48, 3), dtype=np.uint8)
        sample = {"image": img}
        resizer = ResizeImage(target_size=32)
        with self.assertRaises(ZeroDivisionError):
            resizer(sample)


class TestPermute(unittest.TestCase):
    def test_single_sample_hwc_to_chw(self):
        img = np.random.rand(32, 48, 3).astype(np.float32)
        sample = {"image": img}
        permute = Permute(to_bgr=True)
        result = permute(sample)
        self.assertEqual(result["image"].shape, (3, 32, 48))

    def test_single_sample_no_bgr(self):
        img = np.random.rand(32, 48, 3).astype(np.float32)
        sample = {"image": img}
        permute = Permute(to_bgr=False)
        result = permute(sample)
        self.assertEqual(result["image"].shape, (3, 32, 48))

    def test_batch_input(self):
        img = np.random.rand(32, 48, 3).astype(np.float32)
        samples = [{"image": img}, {"image": img}]
        permute = Permute(to_bgr=False)
        result = permute(samples)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["image"].shape, (3, 32, 48))

    def test_missing_image_key_raises(self):
        sample = {"data": np.zeros(5)}
        permute = Permute(to_bgr=False)
        with self.assertRaises(AssertionError):
            permute(sample)


class TestNormalizeImage(unittest.TestCase):
    def test_normalize_channel_first(self):
        img = np.random.rand(3, 32, 48).astype(np.float32) * 255
        sample = {"image": img}
        norm = NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_channel_first=True,
            is_scale=True,
        )
        result = norm(sample)
        # After normalization, values should be near zero
        self.assertTrue(np.all(np.isfinite(result["image"])))

    def test_normalize_channel_last(self):
        img = np.random.rand(32, 48, 3).astype(np.float32) * 255
        sample = {"image": img}
        norm = NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_channel_first=False,
            is_scale=True,
        )
        result = norm(sample)
        self.assertTrue(np.all(np.isfinite(result["image"])))

    def test_zero_std_raises(self):
        with self.assertRaises(ValueError):
            NormalizeImage(mean=[0.5], std=[0])

    def test_normalize_batch_input(self):
        img = np.random.rand(3, 32, 48).astype(np.float32) * 255
        samples = [{"image": img}, {"image": img}]
        norm = NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_channel_first=True,
            is_scale=False,
        )
        result = norm(samples)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


class TestPadBatch(unittest.TestCase):
    def test_pad_to_stride(self):
        img1 = np.random.rand(3, 30, 40).astype(np.float32)
        img2 = np.random.rand(3, 32, 48).astype(np.float32)
        samples = [
            {"image": img1, "im_info": np.array([30, 40, 1.0])},
            {"image": img2, "im_info": np.array([32, 48, 1.0])},
        ]
        padder = PadBatch(pad_to_stride=32)
        result = padder(samples)
        # Both should be padded to 32 stride
        self.assertEqual(result[0]["image"].shape[1] % 32, 0)
        self.assertEqual(result[0]["image"].shape[2] % 32, 0)

    def test_zero_stride_returns_unchanged(self):
        img = np.random.rand(3, 30, 40).astype(np.float32)
        samples = [{"image": img, "im_info": np.array([30, 40, 1.0])}]
        padder = PadBatch(pad_to_stride=0)
        result = padder(samples)
        self.assertEqual(result[0]["image"].shape, img.shape)

    def test_updates_im_info(self):
        img = np.random.rand(3, 30, 40).astype(np.float32)
        samples = [{"image": img, "im_info": np.array([30, 40, 1.0])}]
        padder = PadBatch(pad_to_stride=32, use_padded_im_info=True)
        result = padder(samples)
        # im_info should reflect padded size
        self.assertEqual(result[0]["im_info"][0], result[0]["image"].shape[1])
        self.assertEqual(result[0]["im_info"][1], result[0]["image"].shape[2])


class TestCheck(unittest.TestCase):
    def test_english_returns_true(self):
        self.assertTrue(check("Hello"))

    def test_number_returns_true(self):
        self.assertTrue(check("123"))

    def test_mixed_returns_true(self):
        self.assertTrue(check("abc123"))

    def test_non_english_returns_false(self):
        self.assertFalse(check("\u4e2d\u6587"))

    def test_empty_returns_false(self):
        self.assertFalse(check(""))

    def test_special_chars_returns_false(self):
        self.assertFalse(check("!@#"))


class TestImg2Base64(unittest.TestCase):
    def test_encode_image(self):
        img = Image.new("RGB", (10, 10), color=(0, 255, 0))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f, format="JPEG")
            f.flush()
            path = f.name
        try:
            result = img2base64(path)
            self.assertIsInstance(result, str)
            # Verify it is valid base64
            decoded = base64.b64decode(result)
            self.assertTrue(len(decoded) > 0)
        finally:
            os.unlink(path)


class TestNp2Base64(unittest.TestCase):
    def test_encode_numpy_array(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = np2base64(arr)
        self.assertIsInstance(result, str)
        decoded = base64.b64decode(result)
        self.assertTrue(len(decoded) > 0)


class TestPil2Base64(unittest.TestCase):
    def test_default_format(self):
        img = Image.new("RGB", (10, 10), color=(0, 0, 255))
        result = pil2base64(img)
        self.assertIsInstance(result, str)

    def test_png_format(self):
        img = Image.new("RGB", (10, 10), color=(0, 0, 255))
        result = pil2base64(img, image_type="PNG")
        self.assertIsInstance(result, str)

    def test_with_size(self):
        img = Image.new("RGB", (10, 20), color=(0, 0, 255))
        result, size = pil2base64(img, size=True)
        self.assertEqual(size, (10, 20))

    def test_without_size(self):
        img = Image.new("RGB", (10, 20), color=(0, 0, 255))
        result = pil2base64(img, size=False)
        self.assertIsInstance(result, str)


class TestBbox(unittest.TestCase):
    def test_creation(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        self.assertEqual(bbox.left, 10)
        self.assertEqual(bbox.top, 20)
        self.assertEqual(bbox.width, 30)
        self.assertEqual(bbox.height, 40)

    def test_negative_width_raises(self):
        with self.assertRaises(AssertionError):
            Bbox(left=0, top=0, width=-1, height=10)

    def test_negative_height_raises(self):
        with self.assertRaises(AssertionError):
            Bbox(left=0, top=0, width=10, height=-1)

    def test_zero_dimensions(self):
        bbox = Bbox(left=0, top=0, width=0, height=0)
        self.assertEqual(bbox.area(), 0)

    def test_right_property(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        self.assertEqual(bbox.right, 40)

    def test_bottom_property(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        self.assertEqual(bbox.bottom, 60)

    def test_set_right(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        bbox.right = 50
        self.assertEqual(bbox.width, 40)

    def test_set_right_less_than_left_raises(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        with self.assertRaises(AssertionError):
            bbox.right = 5

    def test_set_bottom(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        bbox.bottom = 70
        self.assertEqual(bbox.height, 50)

    def test_set_bottom_less_than_top_raises(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        with self.assertRaises(AssertionError):
            bbox.bottom = 10

    def test_set_negative_width_raises(self):
        bbox = Bbox(left=0, top=0, width=10, height=10)
        with self.assertRaises(AssertionError):
            bbox.width = -5

    def test_set_negative_height_raises(self):
        bbox = Bbox(left=0, top=0, width=10, height=10)
        with self.assertRaises(AssertionError):
            bbox.height = -5

    def test_equality(self):
        b1 = Bbox(left=10, top=20, width=30, height=40)
        b2 = Bbox(left=10, top=20, width=30, height=40)
        self.assertTrue(b1 == b2)

    def test_inequality(self):
        b1 = Bbox(left=10, top=20, width=30, height=40)
        b2 = Bbox(left=10, top=20, width=30, height=50)
        self.assertFalse(b1 == b2)

    def test_tuple(self):
        bbox = Bbox(left=10.5, top=20.3, width=30, height=40)
        result = bbox.tuple()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_list_int(self):
        bbox = Bbox(left=10.7, top=20.3, width=30, height=40)
        result = bbox.list_int()
        self.assertIsInstance(result, list)
        self.assertEqual(result, [10, 20, 30, 40])

    def test_points_tuple(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        result = bbox.points_tuple()
        self.assertEqual(result, (10, 20, 40, 60))

    def test_is_vertical(self):
        bbox = Bbox(left=0, top=0, width=10, height=20)
        self.assertTrue(bbox.is_vertical())

    def test_is_horizontal(self):
        bbox = Bbox(left=0, top=0, width=20, height=10)
        self.assertTrue(bbox.is_horizontal())

    def test_is_square(self):
        bbox = Bbox(left=0, top=0, width=10, height=10)
        self.assertTrue(bbox.is_square())

    def test_center(self):
        bbox = Bbox(left=0, top=0, width=10, height=20)
        cx, cy = bbox.center()
        self.assertAlmostEqual(cx, 5.0)
        self.assertAlmostEqual(cy, 10.0)

    def test_points(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        p1, p2 = bbox.points()
        self.assertEqual(p1, (10, 20))
        self.assertEqual(p2, (40, 60))

    def test_contain(self):
        outer = Bbox(left=0, top=0, width=100, height=100)
        inner = Bbox(left=10, top=10, width=20, height=20)
        self.assertTrue(outer.contain(inner))
        self.assertFalse(inner.contain(outer))

    def test_overlap_vertically(self):
        b1 = Bbox(left=0, top=0, width=10, height=20)
        b2 = Bbox(left=0, top=10, width=10, height=20)
        self.assertTrue(b1.overlap_vertically(b2))

    def test_no_overlap_vertically(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=0, top=20, width=10, height=10)
        self.assertFalse(b1.overlap_vertically(b2))

    def test_overlap_horizontally(self):
        b1 = Bbox(left=0, top=0, width=20, height=10)
        b2 = Bbox(left=10, top=0, width=20, height=10)
        self.assertTrue(b1.overlap_horizontally(b2))

    def test_no_overlap_horizontally(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=20, top=0, width=10, height=10)
        self.assertFalse(b1.overlap_horizontally(b2))

    def test_overlap(self):
        b1 = Bbox(left=0, top=0, width=20, height=20)
        b2 = Bbox(left=10, top=10, width=20, height=20)
        self.assertTrue(b1.overlap(b2))

    def test_hoverlap(self):
        b1 = Bbox(left=0, top=0, width=20, height=10)
        b2 = Bbox(left=10, top=0, width=20, height=10)
        self.assertEqual(b1.hoverlap(b2), 10)

    def test_hoverlap_no_overlap(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=20, top=0, width=10, height=10)
        self.assertEqual(b1.hoverlap(b2), 0)

    def test_voverlap(self):
        b1 = Bbox(left=0, top=0, width=10, height=20)
        b2 = Bbox(left=0, top=10, width=10, height=20)
        self.assertEqual(b1.voverlap(b2), 10)

    def test_hdistance(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=20, top=0, width=10, height=10)
        self.assertEqual(b1.hdistance(b2), 10)

    def test_hdistance_overlapping(self):
        b1 = Bbox(left=0, top=0, width=20, height=10)
        b2 = Bbox(left=10, top=0, width=20, height=10)
        self.assertEqual(b1.hdistance(b2), 0)

    def test_vdistance(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=0, top=20, width=10, height=10)
        self.assertEqual(b1.vdistance(b2), 10)

    def test_area(self):
        bbox = Bbox(left=0, top=0, width=10, height=20)
        self.assertEqual(bbox.area(), 200)

    def test_translate(self):
        bbox = Bbox(left=10, top=20, width=30, height=40)
        moved = bbox.translate((5, 10))
        self.assertEqual(moved.left, 15)
        self.assertEqual(moved.top, 30)
        self.assertEqual(moved.width, 30)
        self.assertEqual(moved.height, 40)

    def test_union(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=5, top=5, width=10, height=10)
        union = Bbox.union(b1, b2)
        self.assertEqual(union.left, 0)
        self.assertEqual(union.top, 0)
        self.assertEqual(union.right, 15)
        self.assertEqual(union.bottom, 15)

    def test_intersection(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=5, top=5, width=10, height=10)
        inter = Bbox.intersection(b1, b2)
        self.assertEqual(inter.left, 5)
        self.assertEqual(inter.top, 5)
        self.assertEqual(inter.right, 10)
        self.assertEqual(inter.bottom, 10)

    def test_intersection_no_overlap(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=20, top=20, width=10, height=10)
        inter = Bbox.intersection(b1, b2)
        self.assertEqual(inter.area(), 0)

    def test_iou(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=0, top=0, width=10, height=10)
        iou = Bbox.iou(b1, b2)
        self.assertAlmostEqual(iou, 1.0)

    def test_from_points(self):
        bbox = Bbox.from_points((10, 20), (40, 60))
        self.assertEqual(bbox.left, 10)
        self.assertEqual(bbox.top, 20)
        self.assertEqual(bbox.width, 30)
        self.assertEqual(bbox.height, 40)

    def test_from_points_invalid_width_raises(self):
        with self.assertRaises(AssertionError):
            Bbox.from_points((20, 0), (10, 10))

    def test_from_points_invalid_height_raises(self):
        with self.assertRaises(AssertionError):
            Bbox.from_points((0, 20), (10, 10))

    def test_repr(self):
        bbox = Bbox(left=1, top=2, width=3, height=4)
        self.assertEqual(repr(bbox), "(x=1, y=2, w=3, h=4)")

    def test_str(self):
        bbox = Bbox(left=1, top=2, width=3, height=4)
        self.assertEqual(str(bbox), "(x=1, y=2, w=3, h=4)")

    def test_is_cross_boundary(self):
        bbox = Bbox(left=5, top=5, width=20, height=20)
        self.assertTrue(bbox.is_cross_boundary(30, 30))
        self.assertFalse(bbox.is_cross_boundary(10, 10))

    def test_adjacency(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=5, top=5, width=10, height=10)
        adj = Bbox.adjacency(b1, b2)
        self.assertIsNotNone(adj)


class TestTwoDimensionSortBox(unittest.TestCase):
    def test_sort_horizontal_first(self):
        b1 = Bbox(left=0, top=0, width=10, height=10)
        b2 = Bbox(left=20, top=0, width=10, height=10)
        result = two_dimension_sort_box(b1, b2)
        self.assertLess(result, 0)

    def test_sort_vertical_when_no_horizontal_overlap(self):
        b1 = Bbox(left=0, top=0, width=10, height=5)
        b2 = Bbox(left=0, top=20, width=10, height=5)
        result = two_dimension_sort_box(b1, b2)
        self.assertLess(result, 0)


class TestTwoDimensionSortLayout(unittest.TestCase):
    def test_sort_layout(self):
        layout1 = {"bbox": Bbox(left=0, top=0, width=10, height=10)}
        layout2 = {"bbox": Bbox(left=20, top=0, width=10, height=10)}
        result = two_dimension_sort_layout(layout1, layout2)
        self.assertLess(result, 0)


if __name__ == "__main__":
    unittest.main()
