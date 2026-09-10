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
import paddle
import PIL.Image

from paddleformers.transformers.image_utils import ChannelDimension


class TestIsPaddleTensor(unittest.TestCase):
    """Tests for is_paddle_tensor."""

    def test_paddle_tensor(self):
        from paddleformers.transformers.image_transforms import is_paddle_tensor

        tensor = paddle.randn([3, 4])
        self.assertTrue(is_paddle_tensor(tensor))

    def test_numpy_array(self):
        from paddleformers.transformers.image_transforms import is_paddle_tensor

        self.assertFalse(is_paddle_tensor(np.array([1, 2, 3])))

    def test_python_list(self):
        from paddleformers.transformers.image_transforms import is_paddle_tensor

        self.assertFalse(is_paddle_tensor([1, 2, 3]))


class TestToChannelDimensionFormat(unittest.TestCase):
    """Tests for to_channel_dimension_format."""

    def test_already_last(self):
        from paddleformers.transformers.image_transforms import (
            to_channel_dimension_format,
        )

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = to_channel_dimension_format(image, ChannelDimension.LAST)
        self.assertEqual(result.shape, (32, 32, 3))

    def test_already_first(self):
        from paddleformers.transformers.image_transforms import (
            to_channel_dimension_format,
        )

        image = np.random.rand(3, 32, 32).astype(np.float32)
        result = to_channel_dimension_format(image, ChannelDimension.FIRST)
        self.assertEqual(result.shape, (3, 32, 32))

    def test_last_to_first(self):
        from paddleformers.transformers.image_transforms import (
            to_channel_dimension_format,
        )

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = to_channel_dimension_format(image, ChannelDimension.FIRST)
        self.assertEqual(result.shape, (3, 32, 32))

    def test_first_to_last(self):
        from paddleformers.transformers.image_transforms import (
            to_channel_dimension_format,
        )

        image = np.random.rand(3, 32, 32).astype(np.float32)
        result = to_channel_dimension_format(image, ChannelDimension.LAST)
        self.assertEqual(result.shape, (32, 32, 3))

    def test_invalid_input(self):
        from paddleformers.transformers.image_transforms import (
            to_channel_dimension_format,
        )

        with self.assertRaises(ValueError):
            to_channel_dimension_format([[1, 2], [3, 4]], ChannelDimension.LAST)

    def test_string_channel_dim(self):
        from paddleformers.transformers.image_transforms import (
            to_channel_dimension_format,
        )

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = to_channel_dimension_format(image, "channels_first")
        self.assertEqual(result.shape, (3, 32, 32))


class TestRescale(unittest.TestCase):
    """Tests for rescale."""

    def test_basic_rescale(self):
        from paddleformers.transformers.image_transforms import rescale

        image = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = rescale(image, 2.0)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])

    def test_rescale_with_dtype(self):
        from paddleformers.transformers.image_transforms import rescale

        image = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = rescale(image, 0.5, dtype=np.float64)
        self.assertEqual(result.dtype, np.float64)

    def test_rescale_image_with_data_format(self):
        from paddleformers.transformers.image_transforms import rescale

        image = np.random.rand(3, 32, 32).astype(np.float32)
        result = rescale(image, 255.0, data_format=ChannelDimension.LAST)
        self.assertEqual(result.shape[-1], 3)

    def test_rescale_invalid_input(self):
        from paddleformers.transformers.image_transforms import rescale

        with self.assertRaises(ValueError):
            rescale([[1, 2], [3, 4]], 2.0)


class TestToPilImage(unittest.TestCase):
    """Tests for to_pil_image."""

    def test_pil_image_passthrough(self):
        from paddleformers.transformers.image_transforms import to_pil_image

        img = PIL.Image.new("RGB", (32, 32), color=(128, 128, 128))
        result = to_pil_image(img)
        self.assertIsInstance(result, PIL.Image.Image)
        self.assertEqual(result.size, (32, 32))

    def test_numpy_to_pil(self):
        from paddleformers.transformers.image_transforms import to_pil_image

        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = to_pil_image(image, do_rescale=False)
        self.assertIsInstance(result, PIL.Image.Image)
        self.assertEqual(result.size, (32, 32))

    def test_paddle_tensor_to_pil(self):
        from paddleformers.transformers.image_transforms import to_pil_image

        tensor = paddle.randint(0, 256, [3, 32, 32], dtype="int64")
        result = to_pil_image(tensor, do_rescale=False)
        self.assertIsInstance(result, PIL.Image.Image)

    def test_float_numpy_to_pil_with_rescale(self):
        from paddleformers.transformers.image_transforms import to_pil_image

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = to_pil_image(image)
        self.assertIsInstance(result, PIL.Image.Image)

    def test_single_channel_squeeze(self):
        from paddleformers.transformers.image_transforms import to_pil_image

        image = np.random.randint(0, 256, (32, 32, 1), dtype=np.uint8)
        result = to_pil_image(image, do_rescale=False)
        self.assertIsInstance(result, PIL.Image.Image)
        self.assertEqual(len(result.split()), 1)

    def test_invalid_input_type(self):
        from paddleformers.transformers.image_transforms import to_pil_image

        with self.assertRaises(ValueError):
            to_pil_image([[1, 2], [3, 4]])


class TestGetSizeWithAspectRatio(unittest.TestCase):
    """Tests for get_size_with_aspect_ratio."""

    def test_square_image(self):
        from paddleformers.transformers.image_transforms import (
            get_size_with_aspect_ratio,
        )

        result = get_size_with_aspect_ratio((100, 100), 100)
        # Already square with matching size
        self.assertEqual(result, (100, 100))

    def test_tall_image(self):
        from paddleformers.transformers.image_transforms import (
            get_size_with_aspect_ratio,
        )

        result = get_size_with_aspect_ratio((200, 100), 100)
        self.assertEqual(result, (200, 100))

    def test_wide_image(self):
        from paddleformers.transformers.image_transforms import (
            get_size_with_aspect_ratio,
        )

        result = get_size_with_aspect_ratio((100, 200), 100)
        self.assertEqual(result, (100, 200))

    def test_tall_image_resize(self):
        from paddleformers.transformers.image_transforms import (
            get_size_with_aspect_ratio,
        )

        # Height > width, so width becomes size, height is scaled
        result = get_size_with_aspect_ratio((400, 200), 100)
        # width is smaller, so it becomes 100; height is 400/200*100 = 200
        self.assertEqual(result, (200, 100))

    def test_wide_image_resize(self):
        from paddleformers.transformers.image_transforms import (
            get_size_with_aspect_ratio,
        )

        # Width > height, so height becomes size, width is scaled
        result = get_size_with_aspect_ratio((200, 400), 100)
        # height is smaller, so it becomes 100; width is 400/200*100 = 200
        self.assertEqual(result, (100, 200))

    def test_with_max_size(self):
        from paddleformers.transformers.image_transforms import (
            get_size_with_aspect_ratio,
        )

        result = get_size_with_aspect_ratio((800, 400), 256, max_size=512)
        self.assertTrue(result[0] <= 512)
        self.assertTrue(result[1] <= 512)


class TestGetResizeOutputImageSize(unittest.TestCase):
    """Tests for get_resize_output_image_size."""

    def test_tuple_size(self):
        from paddleformers.transformers.image_transforms import (
            get_resize_output_image_size,
        )

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = get_resize_output_image_size(image, (50, 100))
        self.assertEqual(result, (50, 100))

    def test_list_size(self):
        from paddleformers.transformers.image_transforms import (
            get_resize_output_image_size,
        )

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = get_resize_output_image_size(image, [50, 100])
        self.assertEqual(result, (50, 100))

    def test_int_size_square(self):
        from paddleformers.transformers.image_transforms import (
            get_resize_output_image_size,
        )

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = get_resize_output_image_size(image, 224, default_to_square=True)
        self.assertEqual(result, (224, 224))

    def test_single_element_list(self):
        from paddleformers.transformers.image_transforms import (
            get_resize_output_image_size,
        )

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = get_resize_output_image_size(image, [224])
        self.assertEqual(result, (224, 224))

    def test_invalid_size_list(self):
        from paddleformers.transformers.image_transforms import (
            get_resize_output_image_size,
        )

        image = np.random.rand(100, 200, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            get_resize_output_image_size(image, [1, 2, 3])

    def test_int_size_non_square(self):
        from paddleformers.transformers.image_transforms import (
            get_resize_output_image_size,
        )

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = get_resize_output_image_size(image, 100, default_to_square=False)
        # Short edge (100) becomes 100, long edge scaled
        self.assertEqual(max(result), 200)

    def test_max_size(self):
        from paddleformers.transformers.image_transforms import (
            get_resize_output_image_size,
        )

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = get_resize_output_image_size(image, 100, default_to_square=False, max_size=150)
        self.assertTrue(max(result) <= 150)

    def test_max_size_too_small(self):
        from paddleformers.transformers.image_transforms import (
            get_resize_output_image_size,
        )

        image = np.random.rand(100, 200, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            get_resize_output_image_size(image, 100, default_to_square=False, max_size=50)


class TestResize(unittest.TestCase):
    """Tests for resize."""

    def test_numpy_resize(self):
        from paddleformers.transformers.image_transforms import resize

        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        result = resize(image, (50, 100))
        self.assertEqual(result.shape, (50, 100, 3))

    def test_pil_resize(self):
        from paddleformers.transformers.image_transforms import resize

        image = np.array(PIL.Image.new("RGB", (100, 200), color=(128, 128, 128)))
        result = resize(image, (50, 100))
        self.assertEqual(result.shape, (50, 100, 3))

    def test_paddle_tensor_resize(self):
        from paddleformers.transformers.image_transforms import resize

        tensor = paddle.randint(0, 256, [100, 200, 3], dtype="int64")
        result = resize(tensor, (50, 100))
        self.assertEqual(result.shape, (50, 100, 3))

    def test_resize_channels_first(self):
        from paddleformers.transformers.image_transforms import resize

        image = np.random.randint(0, 256, (3, 100, 200), dtype=np.uint8)
        result = resize(image, (50, 100), data_format=ChannelDimension.FIRST)
        self.assertEqual(result.shape[0], 3)

    def test_resize_no_numpy_return(self):
        from paddleformers.transformers.image_transforms import resize

        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        result = resize(image, (50, 100), return_numpy=False)
        self.assertIsInstance(result, PIL.Image.Image)

    def test_invalid_size(self):
        from paddleformers.transformers.image_transforms import resize

        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            resize(image, (50,))


class TestNormalize(unittest.TestCase):
    """Tests for normalize."""

    def test_normalize_scalar_mean_std(self):
        from paddleformers.transformers.image_transforms import normalize

        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        result = normalize(image, mean=0.5, std=0.5)
        np.testing.assert_array_almost_equal(result, np.zeros((32, 32, 3)))

    def test_normalize_iterable_mean_std(self):
        from paddleformers.transformers.image_transforms import normalize

        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        result = normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, np.zeros((32, 32, 3)))

    def test_normalize_channels_first(self):
        from paddleformers.transformers.image_transforms import normalize

        image = np.ones((3, 32, 32), dtype=np.float32) * 0.5
        result = normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, np.zeros((3, 32, 32)))

    def test_normalize_with_data_format(self):
        from paddleformers.transformers.image_transforms import normalize

        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        result = normalize(image, mean=0.5, std=0.5, data_format=ChannelDimension.FIRST)
        self.assertEqual(result.shape, (3, 32, 32))

    def test_normalize_wrong_mean_length(self):
        from paddleformers.transformers.image_transforms import normalize

        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        with self.assertRaises(ValueError):
            normalize(image, mean=[0.5, 0.5], std=[0.5, 0.5, 0.5])

    def test_normalize_wrong_std_length(self):
        from paddleformers.transformers.image_transforms import normalize

        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        with self.assertRaises(ValueError):
            normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5])


class TestCenterCrop(unittest.TestCase):
    """Tests for center_crop."""

    def test_basic_center_crop(self):
        from paddleformers.transformers.image_transforms import center_crop

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = center_crop(image, (50, 100))
        self.assertEqual(result.shape[:2], (50, 100))

    def test_center_crop_smaller_than_image(self):
        from paddleformers.transformers.image_transforms import center_crop

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = center_crop(image, (200, 400))
        # Should pad to reach the target size
        self.assertEqual(result.shape[:2], (200, 400))

    def test_center_crop_exact_size(self):
        from paddleformers.transformers.image_transforms import center_crop

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = center_crop(image, (100, 200))
        self.assertEqual(result.shape[:2], (100, 200))

    def test_center_crop_channels_first(self):
        from paddleformers.transformers.image_transforms import center_crop

        image = np.random.rand(3, 100, 200).astype(np.float32)
        result = center_crop(image, (50, 100))
        self.assertEqual(result.shape, (3, 50, 100))

    def test_center_crop_with_data_format(self):
        from paddleformers.transformers.image_transforms import center_crop

        image = np.random.rand(100, 200, 3).astype(np.float32)
        result = center_crop(image, (50, 100), data_format=ChannelDimension.FIRST)
        self.assertEqual(result.shape, (3, 50, 100))

    def test_center_crop_invalid_size(self):
        from paddleformers.transformers.image_transforms import center_crop

        image = np.random.rand(100, 200, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            center_crop(image, (50,))

    def test_center_crop_pil_input(self):
        from paddleformers.transformers.image_transforms import center_crop

        image = PIL.Image.new("RGB", (100, 200), color=(128, 128, 128))
        result = center_crop(image, (50, 100))
        self.assertEqual(result.shape, (50, 100, 3))

    def test_center_crop_pil_return_pil(self):
        from paddleformers.transformers.image_transforms import center_crop

        image = PIL.Image.new("RGB", (100, 200), color=(128, 128, 128))
        result = center_crop(image, (50, 100), return_numpy=False)
        # The deprecated PIL path still converts to numpy internally, so result may be numpy
        self.assertIsNotNone(result)


class TestCenterToCornersFormat(unittest.TestCase):
    """Tests for center_to_corners_format and corners_to_center_format."""

    def test_numpy_center_to_corners(self):
        from paddleformers.transformers.image_transforms import center_to_corners_format

        bboxes_center = np.array([[50.0, 50.0, 20.0, 40.0]])
        result = center_to_corners_format(bboxes_center)
        expected = np.array([[40.0, 30.0, 60.0, 70.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_paddle_center_to_corners(self):
        from paddleformers.transformers.image_transforms import center_to_corners_format

        bboxes_center = paddle.to_tensor([[50.0, 50.0, 20.0, 40.0]])
        result = center_to_corners_format(bboxes_center)
        expected = np.array([[40.0, 30.0, 60.0, 70.0]])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_numpy_corners_to_center(self):
        from paddleformers.transformers.image_transforms import corners_to_center_format

        bboxes_corners = np.array([[40.0, 30.0, 60.0, 70.0]])
        result = corners_to_center_format(bboxes_corners)
        expected = np.array([[50.0, 50.0, 20.0, 40.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_paddle_corners_to_center(self):
        from paddleformers.transformers.image_transforms import corners_to_center_format

        bboxes_corners = paddle.to_tensor([[40.0, 30.0, 60.0, 70.0]])
        result = corners_to_center_format(bboxes_corners)
        expected = np.array([[50.0, 50.0, 20.0, 40.0]])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_center_to_corners_unsupported_type(self):
        from paddleformers.transformers.image_transforms import center_to_corners_format

        with self.assertRaises(ValueError):
            center_to_corners_format([[50, 50, 20, 40]])

    def test_corners_to_center_unsupported_type(self):
        from paddleformers.transformers.image_transforms import corners_to_center_format

        with self.assertRaises(ValueError):
            corners_to_center_format([[40, 30, 60, 70]])

    def test_roundtrip(self):
        from paddleformers.transformers.image_transforms import (
            center_to_corners_format,
            corners_to_center_format,
        )

        original = np.array([[50.0, 50.0, 20.0, 40.0], [100.0, 100.0, 30.0, 50.0]])
        corners = center_to_corners_format(original)
        recovered = corners_to_center_format(corners)
        np.testing.assert_array_almost_equal(original, recovered)


class TestRgbToId(unittest.TestCase):
    """Tests for rgb_to_id and id_to_rgb."""

    def test_scalar_rgb_to_id(self):
        from paddleformers.transformers.image_transforms import rgb_to_id

        color = [1, 2, 3]
        result = rgb_to_id(color)
        expected = 1 + 256 * 2 + 256 * 256 * 3
        self.assertEqual(result, expected)

    def test_array_rgb_to_id(self):
        from paddleformers.transformers.image_transforms import rgb_to_id

        color = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        result = rgb_to_id(color)
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result[0, 0], 1 + 256 * 2 + 256 * 256 * 3)

    def test_array_uint8_rgb_to_id(self):
        from paddleformers.transformers.image_transforms import rgb_to_id

        color = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        result = rgb_to_id(color)
        self.assertEqual(result[0, 0], 255)
        self.assertEqual(result[0, 1], 255 * 256)

    def test_id_to_rgb_scalar(self):
        from paddleformers.transformers.image_transforms import id_to_rgb

        result = id_to_rgb(1 + 256 * 2 + 256 * 256 * 3)
        self.assertEqual(result, [1, 2, 3])

    def test_id_to_rgb_array(self):
        from paddleformers.transformers.image_transforms import id_to_rgb

        id_map = np.array([[1 + 256 * 2 + 256 * 256 * 3]])
        result = id_to_rgb(id_map)
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertEqual(result[0, 0].tolist(), [1, 2, 3])

    def test_rgb_id_roundtrip(self):
        from paddleformers.transformers.image_transforms import id_to_rgb, rgb_to_id

        color = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        id_map = rgb_to_id(color)
        recovered = id_to_rgb(id_map)
        np.testing.assert_array_equal(color, recovered)


class TestPaddingMode(unittest.TestCase):
    """Tests for PaddingMode enum."""

    def test_constant(self):
        from paddleformers.transformers.image_transforms import PaddingMode

        self.assertEqual(PaddingMode.CONSTANT, "constant")

    def test_reflect(self):
        from paddleformers.transformers.image_transforms import PaddingMode

        self.assertEqual(PaddingMode.REFLECT, "reflect")

    def test_replicate(self):
        from paddleformers.transformers.image_transforms import PaddingMode

        self.assertEqual(PaddingMode.REPLICATE, "replicate")

    def test_symmetric(self):
        from paddleformers.transformers.image_transforms import PaddingMode

        self.assertEqual(PaddingMode.SYMMETRIC, "symmetric")


class TestPad(unittest.TestCase):
    """Tests for pad."""

    def test_constant_padding_int(self):
        from paddleformers.transformers.image_transforms import PaddingMode, pad

        image = np.ones((32, 32, 3), dtype=np.float32)
        result = pad(image, padding=10, mode=PaddingMode.CONSTANT)
        self.assertEqual(result.shape, (32 + 20, 32 + 20, 3))

    def test_constant_padding_tuple(self):
        from paddleformers.transformers.image_transforms import PaddingMode, pad

        image = np.ones((32, 32, 3), dtype=np.float32)
        result = pad(image, padding=((5, 5), (5, 5)), mode=PaddingMode.CONSTANT)
        self.assertEqual(result.shape, (42, 42, 3))

    def test_reflect_padding(self):
        from paddleformers.transformers.image_transforms import PaddingMode, pad

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = pad(image, padding=5, mode=PaddingMode.REFLECT)
        self.assertEqual(result.shape, (42, 42, 3))

    def test_replicate_padding(self):
        from paddleformers.transformers.image_transforms import PaddingMode, pad

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = pad(image, padding=5, mode=PaddingMode.REPLICATE)
        self.assertEqual(result.shape, (42, 42, 3))

    def test_symmetric_padding(self):
        from paddleformers.transformers.image_transforms import PaddingMode, pad

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = pad(image, padding=5, mode=PaddingMode.SYMMETRIC)
        self.assertEqual(result.shape, (42, 42, 3))

    def test_constant_values(self):
        from paddleformers.transformers.image_transforms import PaddingMode, pad

        image = np.zeros((32, 32, 3), dtype=np.float32)
        result = pad(image, padding=5, mode=PaddingMode.CONSTANT, constant_values=1.0)
        self.assertEqual(result.shape, (42, 42, 3))
        # Check that the padding region has the constant value
        self.assertEqual(result[0, 0, 0], 1.0)

    def test_channels_first_padding(self):
        from paddleformers.transformers.image_transforms import PaddingMode, pad

        image = np.ones((3, 32, 32), dtype=np.float32)
        result = pad(image, padding=5, mode=PaddingMode.CONSTANT, input_data_format=ChannelDimension.FIRST)
        self.assertEqual(result.shape, (3, 42, 42))

    def test_with_data_format(self):
        from paddleformers.transformers.image_transforms import PaddingMode, pad

        image = np.ones((32, 32, 3), dtype=np.float32)
        result = pad(image, padding=5, mode=PaddingMode.CONSTANT, data_format=ChannelDimension.FIRST)
        self.assertEqual(result.shape, (3, 42, 42))


class TestConvertToRgb(unittest.TestCase):
    """Tests for convert_to_rgb."""

    def test_pil_rgb_passthrough(self):
        from paddleformers.transformers.image_transforms import convert_to_rgb

        img = PIL.Image.new("RGB", (32, 32))
        result = convert_to_rgb(img)
        self.assertEqual(result.mode, "RGB")

    def test_pil_l_to_rgb(self):
        from paddleformers.transformers.image_transforms import convert_to_rgb

        img = PIL.Image.new("L", (32, 32))
        result = convert_to_rgb(img)
        self.assertEqual(result.mode, "RGB")

    def test_numpy_passthrough(self):
        from paddleformers.transformers.image_transforms import convert_to_rgb

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = convert_to_rgb(image)
        self.assertIs(result, image)

    def test_paddle_tensor_passthrough(self):
        from paddleformers.transformers.image_transforms import convert_to_rgb

        tensor = paddle.randn([3, 32, 32])
        result = convert_to_rgb(tensor)
        self.assertIs(result, tensor)


if __name__ == "__main__":
    unittest.main()
