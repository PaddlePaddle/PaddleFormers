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

"""Image processor class for PaddleOCR-VL."""

import io
import math
import random
from typing import Dict, List, Optional, Union

import numpy as np
from paddle.vision import transforms
from PIL import Image, ImageOps

from ...utils.log import logger
from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import BaseImageProcessor
from ..image_transforms import convert_to_rgb, to_channel_dimension_format
from ..image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)

__all__ = [
    "PaddleOCRVLImageProcessor",
]


# --- Transformation Classes ---


class RandomApply:
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            for t in self.transforms:
                x = t(x)
        return x


class RandomDiscreteRotation:
    def __init__(self, degrees, interpolation="nearest", expand=True):
        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand

    def __call__(self, img):
        angle = random.choice(self.degrees)
        return img.rotate(angle, self.interpolation, self.expand)


class JpegCompression:
    def __init__(self, quality_range=(20, 80)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        output = io.BytesIO()
        img.convert("RGB").save(output, "JPEG", quality=quality)
        output.seek(0)
        return Image.open(output)


class RandomScale:
    def __init__(self, scale_range=(0.7, 1.3), interpolation="bicubic"):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        original_width, original_height = img.size
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        new_size = (new_height, new_width)  # transforms.Resize需要 (h, w)

        return transforms.functional.resize(img, new_size, self.interpolation)


class RandomSingleSidePadding:
    def __init__(self, padding_range=(0, 20), fill="white"):
        assert (
            isinstance(padding_range, (tuple, list)) and len(padding_range) == 2
        ), "padding_range must be the tuple or list like (min, max)"
        self.min_pad, self.max_pad = padding_range
        self.fill = fill

    def __call__(self, img):

        pad_amount = random.randint(self.min_pad, self.max_pad)
        if pad_amount == 0:
            return img

        chosen_edge = random.choice(["left", "top", "right", "bottom"])

        pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0

        if chosen_edge == "left":
            pad_left = pad_amount
        elif chosen_edge == "top":
            pad_top = pad_amount
        elif chosen_edge == "right":
            pad_right = pad_amount
        else:  # 'bottom'
            pad_bottom = pad_amount

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return ImageOps.expand(img, border=padding, fill=self.fill)


def get_ocr_augmentations(
    scale_range=(0.8, 1.2),
    scale_p=0.5,
    padding_range=(0, 15),
    padding_p=0.5,
    rotation_degrees=[0],
    rotation_p=0.5,
    color_jitter_p=0.5,
    jpeg_quality_range=(40, 90),
    jpeg_p=0.5,
):

    augmentations = []

    if scale_p > 0:
        scale_transform = RandomScale(scale_range=scale_range)
        augmentations.append(RandomApply([scale_transform], p=scale_p))

    if padding_p > 0:
        padding_transform = RandomSingleSidePadding(padding_range=padding_range, fill="white")
        augmentations.append(RandomApply([padding_transform], p=padding_p))

    if rotation_p > 0 and rotation_degrees:
        rotation_transform = RandomDiscreteRotation(degrees=rotation_degrees, interpolation="nearest", expand=True)
        augmentations.append(RandomApply([rotation_transform], p=rotation_p))

    if color_jitter_p > 0:
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        augmentations.append(RandomApply([color_jitter], p=color_jitter_p))

    if jpeg_p > 0:
        jpeg_transform = JpegCompression(quality_range=jpeg_quality_range)
        augmentations.append(RandomApply([jpeg_transform], p=jpeg_p))

    return transforms.Compose(augmentations)


image_augmentation = get_ocr_augmentations(
    rotation_degrees=[90, 270],
    rotation_p=0.1,
    jpeg_quality_range=(60, 100),
    jpeg_p=0.3,
    scale_range=(0.5, 1.5),
    scale_p=0.5,
    padding_range=(0, 15),
    padding_p=0.1,
    color_jitter_p=0.1,
)


def is_scaled_image(image: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    if image.dtype == np.uint8:
        return False

    # It's possible the image has pixel values in [0, 255] but is of floating type
    return np.min(image) >= 0 and np.max(image) <= 1


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


def adjust_size(size, patch_size):
    num_patches = size // patch_size
    if num_patches % 2 != 0:
        num_patches -= 1
    return num_patches * patch_size


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28 * 130,
    max_pixels: int = 28 * 28 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """

    if height < factor:
        logger.debug(f"smart_resize: height={height} < factor={factor}, reset height=factor")
        width = round((width * factor) / height)
        height = factor

    if width < factor:
        logger.debug(f"smart_resize: width={width} < factor={factor}, reset width=factor")
        height = round((height * factor) / width)
        width = factor

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class PaddleOCRVLImageProcessor(BaseImageProcessor):
    model_input_names = [
        "pixel_values",
        "image_grid_thw",
    ]

    def __init__(
        self,
        do_resize: bool = True,
        resample: int = 3,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 28 * 28 * 130,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.temporal_conv_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}  # not used
        self.do_convert_rgb = do_convert_rgb

    def set_pixels(self, min_pixels=None, max_pixels=None, msg=""):
        """set_pixels"""
        if min_pixels is not None:
            assert isinstance(min_pixels, int) and min_pixels >= 0, "min_pixels must be positive int"
            logger.info(f"{msg} PaddleOCRImageProcessor set min_pixels = {min_pixels}")
            self.min_pixels = min_pixels
            self.size["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            assert isinstance(max_pixels, int) and max_pixels > 0, "max_pixels must be positive int"
            logger.info(f"{msg} PaddleOCRImageProcessor set max_pixels = {max_pixels}")
            self.max_pixels = max_pixels
            self.size["max_pixels"] = int(max_pixels)

    def get_smarted_resize(self, height, width, min_pixels=None, max_pixels=None):
        """dummy"""
        actual_min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        actual_max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=actual_min_pixels,
            max_pixels=actual_max_pixels,
        )
        return (resized_height, resized_width), (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

    def _preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        predetermined_grid_thw=None,
    ):
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        if is_scaled_image(np.array(images[0])) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(np.array(images[0]))

        width, height = images[0].size
        resized_height, resized_width = height, width
        processed_images = []

        if predetermined_grid_thw is not None:
            assert len(predetermined_grid_thw) == len(
                images
            ), f"len(predetermined_grid_thw) {len(predetermined_grid_thw)} == len(images) {len(images)}"

        for img_idx, image in enumerate(images):
            if do_resize:
                if predetermined_grid_thw is not None:
                    (resized_height, resized_width) = predetermined_grid_thw[img_idx]
                    resized_height *= self.patch_size
                    resized_width *= self.patch_size
                else:
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=self.patch_size * self.merge_size,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )

                image = image.resize((resized_width, resized_height), resample=resample)

            image = to_numpy_array(image)

            if do_rescale:
                image = (image * rescale_factor).astype(np.float32)

            if do_normalize:
                image = image.astype(np.float32)
                image -= np.array(image_mean, dtype=np.float32)
                image /= np.array(image_std, dtype=np.float32)

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose([0, 3, 1, 2])
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
        assert self.temporal_patch_size == 1
        flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel, self.patch_size, self.patch_size)
        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images,
        videos=None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors=None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        predetermined_grid_thw=None,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = make_batched_images(images)
        if videos is not None:
            raise NotImplementedError("Videos are not yet supported")

        if images is not None and not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, " "paddle.Tensor.")

        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for img_idx, image in enumerate(images):
                if predetermined_grid_thw is not None:
                    predetermined_grid_thw_one = [predetermined_grid_thw[img_idx]]
                else:
                    predetermined_grid_thw_one = None

                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    do_convert_rgb=do_convert_rgb,
                    data_format=data_format,
                    input_data_format=input_data_format,
                    predetermined_grid_thw=predetermined_grid_thw_one,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}

        return BatchFeature(data=data, tensor_type=return_tensors)
