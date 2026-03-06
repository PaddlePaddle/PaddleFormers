# Copyright 2025 the HuggingFace Team & PaddlePaddle Authors. All Rights Reserved.
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

"""Image processor class for GLM-4.6V (PaddlePaddle version)."""

import math
from typing import Dict, List, Optional, Union

import numpy as np

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
    "Glm46VImageProcessor",
]


def is_scaled_image(image: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    if image.dtype == np.uint8:
        return False
    return np.min(image) >= 0 and np.max(image) <= 1


def make_batched_images(images) -> List[ImageInput]:
    """
    Accepts images in list or nested list format, and makes a flat list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image(s).

    Returns:
        list: A flat list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]
    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images
    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
) -> tuple:
    """
    Rescales the image (and temporal dimension) so that the following conditions are met:

    1. num_frames is divisible by `temporal_factor`.
    2. Both height and width are divisible by `factor`.
    3. The total number of pixels (t * h * w) is within the range [min_pixels, max_pixels].
    4. The aspect ratio of the image is maintained as closely as possible.

    Args:
        num_frames (`int`): Number of frames (temporal dimension).
        height (`int`): Image height.
        width (`int`): Image width.
        temporal_factor (`int`): Temporal patch size; num_frames must be >= this value.
        factor (`int`): Spatial alignment factor (patch_size * merge_size).
        min_pixels (`int`): Minimum total pixels (t * h * w).
        max_pixels (`int`): Maximum total pixels (t * h * w).

    Returns:
        `Tuple[int, int]`: Resized (height, width).
    """
    if num_frames < temporal_factor:
        raise ValueError(f"num_frames={num_frames} must be >= temporal_factor={temporal_factor}")

    # Ensure minimum spatial size
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


class Glm46VImageProcessor(BaseImageProcessor):
    r"""
    Constructs a GLM-4.6V image processor (PaddlePaddle version) that dynamically resizes
    images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`int`, *optional*, defaults to `3` i.e. BICUBIC):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to OPENAI_CLIP_MEAN):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*, defaults to OPENAI_CLIP_STD):
            Standard deviation to use if normalizing the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `112 * 112`):
            Minimum total pixels (used as `shortest_edge` in size).
        max_pixels (`int`, *optional*, defaults to `14 * 14 * 2 * 2 * 2 * 6144`):
            Maximum total pixels (used as `longest_edge` in size).
        patch_size (`int`, *optional*, defaults to `14`):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to `2`):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to `2`):
            The merge size of the vision encoder to LLM encoder.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: int = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 112 * 112,
        max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
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
        self.do_convert_rgb = do_convert_rgb
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}

    def get_smarted_resize(self, height, width, min_pixels=None, max_pixels=None):
        actual_min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        actual_max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        resized_height, resized_width = smart_resize(
            self.temporal_patch_size,
            height,
            width,
            temporal_factor=self.temporal_patch_size,
            factor=self.patch_size * self.merge_size,
            min_pixels=actual_min_pixels,
            max_pixels=actual_max_pixels,
        )
        return (resized_height, resized_width), (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

    def set_pixels(self, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None, msg: str = ""):
        """Dynamically update min/max pixel constraints."""
        if min_pixels is not None:
            assert isinstance(min_pixels, int) and min_pixels >= 0, "min_pixels must be a non-negative int"
            logger.info(f"{msg} Glm46VImageProcessor set min_pixels = {min_pixels}")
            self.min_pixels = min_pixels
            self.size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            assert isinstance(max_pixels, int) and max_pixels > 0, "max_pixels must be a positive int"
            logger.info(f"{msg} Glm46VImageProcessor set max_pixels = {max_pixels}")
            self.max_pixels = max_pixels
            self.size["longest_edge"] = max_pixels

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> int:
        """
        Returns the number of image patches (grid_h * grid_w) for a given image size.

        Args:
            height (`int`): Image height.
            width (`int`): Image width.
            min_pixels (`int`, *optional*): Override self.min_pixels.
            max_pixels (`int`, *optional*): Override self.max_pixels.

        Returns:
            `int`: Number of image patches.
        """
        actual_min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        actual_max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        factor = self.patch_size * self.merge_size

        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
            min_pixels=actual_min_pixels,
            max_pixels=actual_max_pixels,
        )
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        return grid_h * grid_w

    def _preprocess(
        self,
        images: Union[ImageInput, List[ImageInput]],
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Core preprocessing pipeline for a single image or a temporal sequence of frames.

        Steps:
          1. RGB conversion (optional)
          2. Smart resize to satisfy patch/merge constraints (optional)
          3. Rescale pixel values to [0, 1] (optional)
          4. Normalize with mean/std (optional)
          5. Convert to target channel format
          6. Reshape into flattened patches with grid metadata

        Args:
            images: A single image or list of frames (for video / temporal input).
            do_resize, resample, do_rescale, rescale_factor,
            do_normalize, image_mean, image_std, do_convert_rgb: Per-call overrides.
            data_format: Output channel dimension format.
            input_data_format: Input channel dimension format (inferred if None).

        Returns:
            flatten_patches (`np.ndarray`):
                Shape ``(grid_t * grid_h * grid_w,
                          channel * temporal_patch_size * patch_size * patch_size)``.
            grid_thw (`Tuple[int, int, int]`): ``(grid_t, grid_h, grid_w)``.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # Convert to numpy for uniform handling
        images_np = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images_np[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. "
                "If the input images have pixel values between 0 and 1, set "
                "`do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images_np[0])

        # Use the size of the first frame as the reference for smart_resize
        from ..image_utils import get_image_size

        height, width = get_image_size(images_np[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width

        factor = self.patch_size * self.merge_size
        processed_images = []

        for image in images_np:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=self.temporal_patch_size,
                    height=height,
                    width=width,
                    temporal_factor=self.temporal_patch_size,
                    factor=factor,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                # PIL resize expects (width, height)
                from PIL import Image as PILImage

                pil_img = PILImage.fromarray(
                    image.transpose(1, 2, 0).astype(np.uint8)
                    if input_data_format == ChannelDimension.FIRST
                    else image.astype(np.uint8)
                )
                pil_img = pil_img.resize((resized_width, resized_height), resample=resample)
                image = to_numpy_array(pil_img)
                # After PIL, image is HWC; reset input format accordingly
                _cur_format = ChannelDimension.LAST
            else:
                _cur_format = input_data_format

            if do_rescale:
                image = image.astype(np.float32) * rescale_factor

            if do_normalize:
                image = image.astype(np.float32)
                image -= np.array(image_mean, dtype=np.float32)
                image /= np.array(image_std, dtype=np.float32)

            image = to_channel_dimension_format(image, data_format, input_channel_dim=_cur_format)
            processed_images.append(image)

        # Stack frames: shape (num_frames, C, H, W) if FIRST
        patches = np.array(processed_images)  # (T, C, H, W)

        # Ensure channels-first for internal reshape
        if data_format == ChannelDimension.LAST:
            # (T, H, W, C) -> (T, C, H, W)
            patches = patches.transpose(0, 3, 1, 2)

        # Pad temporal dimension to be divisible by temporal_patch_size
        if patches.shape[0] % self.temporal_patch_size != 0:
            pad_len = self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size)
            repeats = np.repeat(patches[-1][np.newaxis], pad_len, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)

        # patches: (T_padded, C, H, W)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        # Reshape with merge_size (same as torch version)
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        # (grid_t, t_p, C, gh/ms, ms, ps, gw/ms, ms, ps)
        # -> (grid_t, gh/ms, gw/ms, ms, ms, C, t_p, ps, ps)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        videos=None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors=None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocess one or more images.

        Args:
            images (`ImageInput`):
                Image(s) to preprocess. Expects pixel values in [0, 255]. If pixel values
                are already in [0, 1], set ``do_rescale=False``.
            videos:
                Not yet supported; raises ``NotImplementedError``.
            do_resize (`bool`, *optional*): Override ``self.do_resize``.
            size (`Dict[str, int]`, *optional*):
                Must contain ``'shortest_edge'`` and ``'longest_edge'``. Overrides the
                min/max pixel bounds stored on the processor.
            resample: Override ``self.resample``.
            do_rescale, rescale_factor: Override rescaling settings.
            do_normalize, image_mean, image_std: Override normalization settings.
            do_convert_rgb: Override ``self.do_convert_rgb``.
            return_tensors (`str`, *optional*):
                Type of tensors to return (``'pd'`` for PaddlePaddle, ``'np'`` for NumPy, etc.).
            data_format: Output channel dimension format.
            input_data_format: Input channel dimension format (inferred if ``None``).

        Returns:
            `BatchFeature` with keys:
                - ``pixel_values``: ``np.ndarray`` of shape
                  ``(total_patches, C * temporal_patch_size * patch_size * patch_size)``.
                - ``image_grid_thw``: ``np.ndarray`` of shape ``(num_images, 3)``
                  containing ``(grid_t, grid_h, grid_w)`` per image.
        """
        # Resolve per-call overrides
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        # Allow caller to override pixel bounds via `size`
        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            self.min_pixels = size["shortest_edge"]
            self.max_pixels = size["longest_edge"]

        if videos is not None:
            raise NotImplementedError("Video input is not yet supported in Glm46VImageProcessor.")

        if images is not None:
            images = make_batched_images(images)

        if images is not None and not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or paddle.Tensor.")

        data = {}
        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
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
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)

            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data.update({"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws})

        return BatchFeature(data=data, tensor_type=return_tensors)
