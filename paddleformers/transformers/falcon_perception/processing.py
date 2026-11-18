# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Adapted from TII Falcon Perception (Apache-2.0).
import io
import math

import einops as E
import numpy as np
import paddle
import requests
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import convert_to_rgb, resize
from transformers.image_utils import (
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)

IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]


def load_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            response = requests.get(image, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        if image.endswith(".npy"):
            img_array = io.BytesIO(np.load(image))
            return Image.open(img_array)
        return Image.open(image)
    if isinstance(image, np.bytes_):
        return Image.open(io.BytesIO(image))
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError(f"Unknown image format {image}")


def load_images(images_input, min_dimension: int, max_dimension: int):
    images = []
    if images_input is not None:
        for inp in images_input:
            img = load_image(inp)
            img = resize_image_if_necessary(img, min_dimension, max_dimension)
            images.append(img)
    return images


def resize_image_if_necessary(image, shortest_dimension=224, longest_dimension=896):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    if (
        shortest_dimension <= original_width <= longest_dimension
        and shortest_dimension <= original_height <= longest_dimension
    ):
        return image
    is_vertical_image = original_width < original_height
    if original_width < shortest_dimension or original_height < shortest_dimension:
        if is_vertical_image:
            new_width = shortest_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = shortest_dimension
            new_width = int(new_height * aspect_ratio)
    elif is_vertical_image:
        new_width = longest_dimension
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = longest_dimension
        new_width = int(new_height * aspect_ratio)
    if new_width > longest_dimension:
        new_width = longest_dimension
        new_height = int(new_width / aspect_ratio)
    if new_height > longest_dimension:
        new_height = longest_dimension
        new_width = int(new_height * aspect_ratio)
    resized_image = image.resize((new_width, new_height))
    return resized_image


def smart_resize(
    image, factor: int, resample, input_data_format, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    height, width = get_image_size(image, channel_dim=input_data_format)
    if height < factor or width < factor:
        raise ValueError(f"height={height!r} or width={width!r} must be larger than factor={factor!r}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt(height * width / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    image = resize(image, size=(h_bar, w_bar), resample=resample, input_data_format=input_data_format)
    return image


class ImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        patch_size,
        merge_size,
        do_resize: bool = True,
        resample: Image.Resampling = Image.Resampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or IMAGE_MEAN
        self.image_std = image_std or IMAGE_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        self.do_convert_rgb = do_convert_rgb
        validate_preprocess_arguments(
            rescale_factor=self.rescale_factor,
            do_normalize=self.do_normalize,
            image_mean=self.image_mean,
            image_std=self.image_std,
            do_resize=self.do_resize,
            size=self.size,
            resample=self.resample,
        )

    def _preprocess(self, image: ImageInput, do_rescale=None, do_normalize=None):
        if self.do_convert_rgb:
            image = convert_to_rgb(image)
        image = to_numpy_array(image)
        input_data_format = infer_channel_dimension_format(image)
        if self.do_resize:
            image = smart_resize(
                image,
                factor=self.patch_size * self.merge_size,
                resample=self.resample,
                input_data_format=input_data_format,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        if do_rescale or self.do_rescale:
            image = self.rescale(image, scale=self.rescale_factor, input_data_format=input_data_format)
        if do_normalize or self.do_normalize:
            image = self.normalize(
                image=image, mean=self.image_mean, std=self.image_std, input_data_format=input_data_format
            )
        return image

    def preprocess(self, images: list[ImageInput] | None, do_rescale=None, do_normalize=None, **kwargs):
        del kwargs
        if images is None:
            return []
        images = [item for item in images if item is not None]
        if not valid_images(images):
            raise ValueError("Invalid image type. Use a PIL image or NumPy array.")
        pixel_values = []
        for image in images:
            processed_image = self._preprocess(image, do_rescale, do_normalize)
            processed_image = processed_image[None, ...]
            pixel_values.append(processed_image)
        return pixel_values

    def batch_images_with_mask(self, pixel_values, max_image_height, max_image_width):
        if pixel_values is None:
            return None
        pixel_values = [item for item in pixel_values if item is not None and len(item) != 0]
        if len(pixel_values) == 0:
            return None
        pixel_values = [paddle.to_tensor(img) for img in pixel_values]
        max_temporal = max((img.shape[0] for img in pixel_values))

        def pad_image_and_mask(img):
            time_steps, height, width, channels = img.shape
            if channels != 3:
                raise ValueError(f"Expected 3-channel RGB images, got {channels} channels.")
            padded_image = paddle.zeros([max_temporal, max_image_height, max_image_width, channels], dtype=img.dtype)
            padded_image[:time_steps, :height, :width] = img
            mask = paddle.zeros((max_temporal, max_image_height, max_image_width), dtype=paddle.int64)
            mask[:time_steps, :height, :width] = 1
            return (padded_image, mask)

        padded_pixel_values, padding_masks = zip(*[pad_image_and_mask(img) for img in pixel_values])
        padded_pixel_values = paddle.stack(list(padded_pixel_values))
        padding_masks = paddle.stack(list(padding_masks))
        return {"pixel_values": padded_pixel_values, "padding_mask": padding_masks}


def _compute_image_spatial_positions(
    pixel_mask_THW: paddle.Tensor, spatial_patch_size: int, temporal_patch_size: int = 1
) -> tuple[paddle.Tensor, paddle.Tensor]:
    mask_thw = E.reduce(
        pixel_mask_THW,
        "(t tp) (h hp) (w wp) -> t h w",
        reduction="any",
        tp=temporal_patch_size,
        hp=spatial_patch_size,
        wp=spatial_patch_size,
    )
    width = E.reduce(mask_thw.sum(axis=-1).int(), "t h -> ", reduction="max")
    height = E.reduce(mask_thw.sum(axis=-2).int(), "t w -> ", reduction="max")
    xlim = paddle.sqrt(width / height)
    ylim = paddle.sqrt(height / width)
    xpos = paddle.linspace(-xlim, xlim, int(width))
    ypos = paddle.linspace(-ylim, ylim, int(height))
    wpos, hpos = paddle.meshgrid(xpos, ypos, indexing="xy")
    return (hpos.flatten(), wpos.flatten())


def _get_image_token_masks(tokens, config):
    spatial_mask = tokens == config.img_id
    no_increase_mask = (
        spatial_mask
        | (tokens == config.image_reg_1_token_id)
        | (tokens == config.image_reg_2_token_id)
        | (tokens == config.image_reg_3_token_id)
        | (tokens == config.image_reg_4_token_id)
        | (tokens == config.img_end_id)
    )
    return (spatial_mask, no_increase_mask)


def get_pos_thw(
    tokens: paddle.Tensor,
    pixel_masks_NTHW: paddle.Tensor,
    config,
    spatial_patch_size: int,
    temporal_patch_size: int = 1,
    pad_token_id: int = None,
):
    assert pad_token_id is not None
    assert tokens.ndim == 2
    assert pixel_masks_NTHW.ndim == 4
    spatial_img_token_mask_BS, no_increase_idx_img_token_mask_BS = _get_image_token_masks(tokens, config)
    hpos_parts, wpos_parts = ([], [])
    for i in range(pixel_masks_NTHW.shape[0]):
        h, w = _compute_image_spatial_positions(pixel_masks_NTHW[i], spatial_patch_size, temporal_patch_size)
        hpos_parts.append(h)
        wpos_parts.append(w)
    hpos_N = paddle.cat(hpos_parts) if hpos_parts else paddle.empty(0)
    wpos_N = paddle.cat(wpos_parts) if wpos_parts else paddle.empty(0)
    expected_tokens = spatial_img_token_mask_BS.sum().item()
    actual_tokens = hpos_N.numel()
    assert (
        actual_tokens == expected_tokens
    ), f"Mismatch between spatial image tokens ({expected_tokens}) and generated positions ({actual_tokens})."
    hpos_BS = paddle.full_like(tokens, fill_value=paddle.nan, dtype=paddle.float32, device=tokens.place)
    wpos_BS = paddle.full_like(tokens, fill_value=paddle.nan, dtype=paddle.float32, device=tokens.place)
    hpos_BS = hpos_BS.masked_scatter_(spatial_img_token_mask_BS, hpos_N)
    wpos_BS = wpos_BS.masked_scatter_(spatial_img_token_mask_BS, wpos_N)
    tpos_BS = paddle.ones_like(tokens, dtype=paddle.float32, device=tokens.place)
    tpos_BS[no_increase_idx_img_token_mask_BS] = 0
    tpos_BS = paddle.cumsum(tpos_BS, axis=1) - 1
    tpos_BS[tokens == pad_token_id] = 0
    hw_pos_BS2 = paddle.stack([hpos_BS, wpos_BS], axis=-1)
    return (tpos_BS.long(), hw_pos_BS2)


def calculate_image_tokens(image, patch_size, merge_size):
    height, width = get_image_size(image)
    return int(height * width / (patch_size * patch_size * merge_size * merge_size))


def tokenize_inputs(prompt, images, tokenizer, config, patch_size, merge_size, max_length):
    img_reg_ids = [
        config.image_reg_1_token_id,
        config.image_reg_2_token_id,
        config.image_reg_3_token_id,
        config.image_reg_4_token_id,
    ]
    if images is not None and len(images) > 0:
        image_token_counts = [calculate_image_tokens(image, patch_size, merge_size) for image in images]
    else:
        image_token_counts = []
    image_token = tokenizer.convert_ids_to_tokens(config.img_id)
    prompt_chunks = [tokenizer.encode(chunk) for chunk in prompt.split(image_token)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, sep) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and (bos_id is not None)
        and (prompt_chunks[0][0] == bos_id)
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    separators = []
    for count in image_token_counts:
        tokens = [config.img_id] * count
        image_block = [config.image_cls_token_id, *img_reg_ids, *tokens, config.img_end_id]
        separators.append(image_block)
    if len(separators) != 0 and len(separators) != len(prompt_chunks):
        separators.append(separators[-1])
    selected_images = []
    if len(separators) == 0:
        input_ids = prompt_chunks[0]
    else:
        for index, x in enumerate(insert_separator(prompt_chunks, separators)):
            if index % 2 != 0:
                if len(input_ids) + len(x) < max_length:
                    input_ids.extend(x)
                    selected_images.append(images[index // 2])
            elif index % 2 == 0:
                input_ids.extend(x[offset:])
    input_ids = paddle.to_tensor(input_ids, dtype="int64")
    return (input_ids, selected_images)


def process_batch(
    tokenizer, config, image_prompt_pairs, max_length, min_dimension, max_dimension, patch_size=16, merge_size=1
):
    """
    Process a batch of images with text prompts.
    Uses LEFT PADDING for proper batch generation with causal models.
    """
    all_input_ids = []
    all_selected_images = []
    processor_local = ImageProcessor(patch_size, merge_size)
    for img_input, prompt in image_prompt_pairs:
        img = load_image(img_input)
        if img is not None:
            img = resize_image_if_necessary(img, min_dimension, max_dimension)
        images = processor_local.preprocess(images=[img] if img else [])
        input_ids, selected_images = tokenize_inputs(
            prompt, images, tokenizer, config, patch_size, merge_size, max_length
        )
        all_input_ids.append(input_ids)
        all_selected_images.extend(selected_images)
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    max_len = max(len(ids) for ids in all_input_ids)
    padded_input_ids = paddle.stack(
        [paddle.concat([paddle.full([max_len - len(ids)], pad_token_id, dtype="int64"), ids]) for ids in all_input_ids]
    )
    processed = processor_local.batch_images_with_mask(all_selected_images, max_dimension, max_dimension)
    assert processed is not None
    pos_t, pos_hw = get_pos_thw(
        padded_input_ids, processed["padding_mask"], config, patch_size, pad_token_id=pad_token_id
    )
    return {
        "tokens": padded_input_ids,
        "pixel_values": processed["pixel_values"],
        "pixel_mask": processed["padding_mask"],
        "pos_t": pos_t,
        "pos_hw": pos_hw,
        "pad_token_id": pad_token_id,
    }
