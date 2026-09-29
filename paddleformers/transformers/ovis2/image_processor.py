# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Image processor for Ovis2."""

from functools import lru_cache

import numpy as np

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import BaseImageProcessor
from ..image_transforms import convert_to_rgb, normalize, rescale, resize, to_channel_dimension_format
from ..image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(min_image_tiles, max_image_tiles):
    ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if min_image_tiles <= width * height <= max_image_tiles:
                ratios.append((width, height))
    return sorted(ratios, key=lambda item: item[0] * item[1])


def compute_patch_covering_area(left, upper, right, lower, side):
    width, height = right - left, lower - upper
    width, height = max(width, height), min(width, height)
    if width > side:
        height = height / width * side
        width = side
    return width * height


def split_image_into_grid(height, width, grid):
    row_height = height // grid[0]
    column_width = width // grid[1]
    return [
        (
            column * column_width,
            row * row_height,
            width if column == grid[1] - 1 else (column + 1) * column_width,
            height if row == grid[0] - 1 else (row + 1) * row_height,
        )
        for row in range(grid[0])
        for column in range(grid[1])
    ]


@lru_cache(maxsize=100)
def get_min_tile_covering_grid(image_size, target_patch_size, max_image_tiles, covering_threshold=0.9):
    image_height, image_width = image_size
    image_area = image_width * image_height
    evaluated, sufficient = [], []
    for grid in get_all_supported_aspect_ratios(1, max_image_tiles):
        regions = split_image_into_grid(image_height, image_width, grid)
        ratio = sum(compute_patch_covering_area(*region, target_patch_size) for region in regions) / image_area
        evaluated.append((grid, ratio))
        if ratio > covering_threshold:
            sufficient.append((grid, ratio))
    if sufficient:
        return min(sufficient, key=lambda item: (item[0][0] * item[0][1], -item[1]))[0]
    return min(evaluated, key=lambda item: (-item[1], item[0][0] * item[0][1]))[0]


@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(original_image_size, target_tile_size, min_image_tiles, max_image_tiles):
    possible_grids = get_all_supported_aspect_ratios(min_image_tiles, max_image_tiles)
    original_height, original_width = original_image_size
    target_height, target_width = target_tile_size
    aspect_ratio = original_width / original_height
    area = original_width * original_height
    best_difference, best_grid = float("inf"), (1, 1)
    for grid in possible_grids:
        difference = abs(aspect_ratio - grid[0] / grid[1])
        if difference < best_difference:
            best_difference, best_grid = difference, grid
        elif difference == best_difference and area > 0.5 * target_height * target_width * grid[0] * grid[1]:
            best_grid = grid
    return best_grid


class Ovis2ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "grids"]

    def __init__(
        self,
        do_resize=True,
        size=None,
        resample=PILImageResampling.BICUBIC,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        crop_to_patches=False,
        min_patches=1,
        max_patches=12,
        use_covering_area_grid=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 384, "width": 384}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.crop_to_patches = crop_to_patches
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.use_covering_area_grid = use_covering_area_grid

    def crop_image_to_patches(
        self,
        image,
        min_patches,
        max_patches,
        use_covering_area_grid=True,
        covering_threshold=0.9,
        patch_size=None,
        resample_filter=None,
    ):
        image = to_channel_dimension_format(
            image, ChannelDimension.FIRST, infer_channel_dimension_format(image)
        )
        patch_height, patch_width = patch_size["height"], patch_size["width"]
        original_height, original_width = image.shape[-2:]
        if use_covering_area_grid:
            num_columns, num_rows = get_min_tile_covering_grid(
                (original_height, original_width), patch_height, max_patches, covering_threshold
            )
        else:
            num_columns, num_rows = get_optimal_tiled_canvas(
                (original_height, original_width),
                (patch_height, patch_width),
                min_patches,
                max_patches,
            )
        target_width = patch_width * num_columns
        target_height = patch_height * num_rows
        resized = resize(
            image,
            (target_height, target_width),
            resample=resample_filter,
            data_format=ChannelDimension.FIRST,
        )
        patches = []
        for index in range(num_columns * num_rows):
            column, row = index % num_columns, index // num_columns
            patches.append(
                resized[
                    :,
                    row * patch_height : (row + 1) * patch_height,
                    column * patch_width : (column + 1) * patch_width,
                ]
            )
        if len(patches) != 1:
            patches.insert(
                0,
                resize(
                    image,
                    (patch_height, patch_width),
                    resample=resample_filter,
                    data_format=ChannelDimension.FIRST,
                ),
            )
        return patches, [num_rows, num_columns]

    def preprocess(
        self,
        images,
        do_resize=None,
        size=None,
        resample=None,
        do_rescale=None,
        rescale_factor=None,
        do_normalize=None,
        image_mean=None,
        image_std=None,
        do_convert_rgb=None,
        crop_to_patches=None,
        min_patches=None,
        max_patches=None,
        use_covering_area_grid=None,
        return_tensors=None,
        data_format=ChannelDimension.FIRST,
        **kwargs,
    ):
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        resample_filter = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb
        crop_to_patches = self.crop_to_patches if crop_to_patches is None else crop_to_patches
        min_patches = self.min_patches if min_patches is None else min_patches
        max_patches = self.max_patches if max_patches is None else max_patches
        use_covering_area_grid = (
            self.use_covering_area_grid if use_covering_area_grid is None else use_covering_area_grid
        )

        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError("Invalid image type. Expected PIL, NumPy, or Paddle images.")
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
        images = [to_numpy_array(image) for image in images]

        processed_images, grids = [], []
        for image in images:
            image = to_channel_dimension_format(
                image, ChannelDimension.FIRST, infer_channel_dimension_format(image)
            )
            if crop_to_patches and max_patches > 1:
                patches, grid = self.crop_image_to_patches(
                    image,
                    min_patches,
                    max_patches,
                    use_covering_area_grid=use_covering_area_grid,
                    patch_size=size,
                    resample_filter=resample_filter,
                )
            else:
                patches, grid = [image], [1, 1]
            grids.append(grid)
            for patch in patches:
                if do_resize:
                    patch = resize(
                        patch,
                        (size["height"], size["width"]),
                        resample=resample_filter,
                        data_format=ChannelDimension.FIRST,
                    )
                if do_rescale:
                    patch = rescale(patch, rescale_factor, data_format=ChannelDimension.FIRST)
                if do_normalize:
                    patch = normalize(patch, image_mean, image_std, data_format=ChannelDimension.FIRST)
                patch = to_channel_dimension_format(patch, data_format, ChannelDimension.FIRST)
                processed_images.append(patch)

        return BatchFeature(
            data={"pixel_values": np.stack(processed_images), "grids": np.asarray(grids, dtype=np.int64)},
            tensor_type=return_tensors,
        )


__all__ = ["Ovis2ImageProcessor"]
