# Copyright 2025 the HuggingFace Team. All rights reserved.
# PaddlePaddle adaptation
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

import math
from typing import Dict, List, Optional, Union
from PIL import Image
import numpy as np
import paddle

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



def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
):
    """
    Dynamically resize image dimensions while respecting pixel constraints.

    Args:
        num_frames: Number of frames (temporal dimension).
        height: Original image height.
        width: Original image width.
        temporal_factor: Temporal alignment factor.
        factor: Spatial alignment factor (patch_size * merge_size).
        min_pixels: Minimum total pixel count.
        max_pixels: Maximum total pixel count.

    Returns:
        Tuple[int, int]: Resized (height, width).
    """
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar

def is_scaled_image(image: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    if image.dtype == np.uint8:
        return False

    # It's possible the image has pixel values in [0, 255] but is of floating type
    return np.min(image) >= 0 and np.max(image) <= 1

def get_image_size(image: np.ndarray, channel_dim=None) -> tuple:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray`):
            The image to get the dimensions of.
        channel_dim (`str`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's (height, width).
    """
    if channel_dim is None:
        channel_dim = infer_channel_format(image)

    if channel_dim == "channels_first":
        return image.shape[-2], image.shape[-1]
    elif channel_dim == "channels_last":
        return image.shape[-3], image.shape[-2]
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")

def resize_image(
    image: np.ndarray,
    size: tuple,
    resample=Image.BICUBIC,
) -> np.ndarray:
    h, w = size
    do_rescale = False
    if image.dtype == np.float32 or image.dtype == np.float64:
        # 与HF版一致：float图先*255转uint8给PIL，resize后再/255还原
        do_rescale = True
        image = (image * 255).clip(0, 255).astype(np.uint8)
    
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((w, h), resample=resample)
    resized = np.array(pil_img)
    
    if do_rescale:
        resized = resized.astype(np.float32) / 255.0
    
    return resized

def rescale_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Rescale image pixel values by scale factor."""
    return image.astype(np.float32) * scale

def pil_to_numpy(image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    if isinstance(image, Image.Image):
        return np.array(image)
    return image

def normalize_image(
    image: np.ndarray,
    mean: List[float],
    std: List[float],
) -> np.ndarray:
    """
    Normalize image with mean and std per channel.

    Args:
        image: Float numpy array in CHW format.
        mean: Per-channel mean.
        std: Per-channel std.
    Returns:
        Normalized numpy array.
    """
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    return (image - mean) / std


def to_channel_first(image: np.ndarray) -> np.ndarray:
    """Convert HWC numpy array to CHW."""
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        return image.transpose(2, 0, 1)
    return image


def infer_channel_format(image: np.ndarray) -> str:
    """Infer whether image is CHW or HWC."""
    if image.ndim == 3:
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            return "channels_first"
    return "channels_last"


class Glm46VImageProcessor:
    """
    PaddlePaddle implementation of GLM-4V image processor.

    Dynamically resizes images and prepares patch-based inputs for
    the GLM-4V vision encoder.

    Args:
        do_resize (bool): Whether to resize images. Default: True.
        size (dict): Size constraints with 'shortest_edge' and 'longest_edge'.
        resample: PIL resampling filter. Default: Image.BICUBIC.
        do_rescale (bool): Whether to rescale pixel values. Default: True.
        rescale_factor (float): Rescale factor. Default: 1/255.
        do_normalize (bool): Whether to normalize. Default: True.
        image_mean (list): Per-channel mean for normalization.
        image_std (list): Per-channel std for normalization.
        do_convert_rgb (bool): Whether to convert images to RGB. Default: True.
        patch_size (int): Spatial patch size of vision encoder. Default: 14.
        temporal_patch_size (int): Temporal patch size. Default: 2.
        merge_size (int): Merge size from vision to LLM encoder. Default: 2.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample=Image.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
    ) -> None:
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        elif size is None:
            size = {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 15000}

        self.size = size
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling | None = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: ChannelDimension | None = ChannelDimension.FIRST,
        input_data_format: str | ChannelDimension | None = None,
    ):
        """
        Core preprocessing pipeline for a single image or video (list of frames).

        Returns:
            flatten_patches (np.ndarray): Shape [grid_t*grid_h*grid_w, C*temporal_patch_size*patch_size*patch_size].
            grid_thw (tuple): (grid_t, grid_h, grid_w).
        """
        images = make_list_of_images(images)

        # Convert to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(img) for img in images]

        images = [to_numpy_array(img) for img in images]

        # Warn if already scaled
        if do_rescale and is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []

        for image in images:
            # Resize
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=temporal_patch_size,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = resize_image(image, size=(resized_height, resized_width), resample=resample)

            # Rescale
            if do_rescale:
                image = rescale_image(image, scale=rescale_factor)

            # Normalize (HWC格式下)
            if do_normalize:
                image = normalize_image(image, mean=image_mean, std=image_std)

            # 最后转 CHW
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        # Stack to [N, C, H, W]
        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        # Pad temporal dimension if needed
        if patches.shape[0] % temporal_patch_size != 0:
            pad_count = temporal_patch_size - (patches.shape[0] % temporal_patch_size)
            repeats = np.repeat(patches[-1][np.newaxis], pad_count, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        # Reshape and transpose to extract patches
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample=None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[str] = None,
    ) -> dict:
        """
        Preprocess a single image or batch of images.

        Args:
            images: PIL Image, numpy array, or list of these.
            do_resize: Override instance do_resize.
            size: Override instance size dict.
            resample: Override resampling filter.
            do_rescale: Override instance do_rescale.
            rescale_factor: Override instance rescale_factor.
            do_normalize: Override instance do_normalize.
            image_mean: Override instance image_mean.
            image_std: Override instance image_std.
            patch_size: Override instance patch_size.
            temporal_patch_size: Override instance temporal_patch_size.
            merge_size: Override instance merge_size.
            do_convert_rgb: Override instance do_convert_rgb.
            return_tensors: If 'pd', return paddle.Tensor; if 'np', return np.ndarray.

        Returns:
            dict with keys:
                'pixel_values': [total_patches, C*temporal_patch_size*patch_size*patch_size]
                'image_grid_thw': [num_images, 3]
        """
        # Resolve parameters with instance defaults
        size = size if size is not None else self.size
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        # Normalize images to a list
        if images is None:
            raise ValueError("images must not be None")
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]

        pixel_values_list = []
        vision_grid_thws = []

        for image in images:
            patches, grid_thw = self._preprocess(
                image,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
                do_convert_rgb=do_convert_rgb,
            )
            pixel_values_list.extend(patches)
            vision_grid_thws.append(grid_thw)

        pixel_values = np.array(pixel_values_list, dtype=np.float32)
        vision_grid_thws = np.array(vision_grid_thws, dtype=np.int64)

        if return_tensors == "pd":
            pixel_values = paddle.to_tensor(pixel_values)
            vision_grid_thws = paddle.to_tensor(vision_grid_thws)

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": vision_grid_thws,
        }

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        images_kwargs: Optional[dict] = None,
    ) -> int:
        """
        Return the number of image patches for a given image size.

        Args:
            height: Image height.
            width: Image width.
            images_kwargs: Optional overrides for patch_size, merge_size, size.

        Returns:
            Number of image patches (grid_h * grid_w).
        """
        if images_kwargs is None:
            images_kwargs = {}

        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        size = images_kwargs.get(
            "size",
            {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 15000},
        )

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            factor=factor,
            min_pixels=size["shortest_edge"],
            max_pixels=size["longest_edge"],
            temporal_factor=self.temporal_patch_size,
        )
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        return grid_h * grid_w


__all__ = ["Glm46VImageProcessor", "smart_resize"]