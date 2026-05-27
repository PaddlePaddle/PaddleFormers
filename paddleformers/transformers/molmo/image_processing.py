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
"""Image processor for Molmo."""

from __future__ import annotations

import sys
from typing import Optional

import einops
import numpy as np

if "torch" in sys.modules and sys.modules.get("torch") is None:
    del sys.modules["torch"]
if "torchvision" in sys.modules and sys.modules.get("torchvision") is None:
    del sys.modules["torchvision"]
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from ..image_processing_utils import BaseImageProcessor


def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width, value=0):
    height, width = image.shape[:2]
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    return np.pad(
        image,
        [[offset_height, after_padding_height], [offset_width, after_padding_width], [0, 0]],
        constant_values=value,
    )


def normalize_image(image, offset, scale):
    image -= np.array(offset, dtype=np.float32)[None, None, :]
    image /= np.array(scale, dtype=np.float32)[None, None, :]
    return image


def resize_and_pad(
    image,
    desired_output_size,
    resize_method="torch-bilinear",
    pad_value=0,
    normalize=True,
    image_mean=OPENAI_CLIP_MEAN,
    image_std=OPENAI_CLIP_STD,
):
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)

    if resize_method == "torch-bilinear":
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
        image = convert_image_dtype(image)
        image = torchvision.transforms.Resize(
            [scaled_height, scaled_width], InterpolationMode.BILINEAR, antialias=True
        )(image)
        image = torch.clip(image, 0.0, 1.0)
        image = torch.permute(image, [1, 2, 0]).numpy()
    else:
        raise NotImplementedError(resize_method)

    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0],
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)
    if normalize:
        image = normalize_image(image, offset=image_mean, scale=image_std)
    return image, image_mask


def select_tiling(h, w, patch_size, max_num_patches):
    original_res = h * w
    tilings = []
    for i in range(1, max_num_patches + 1):
        for j in range(1, max_num_patches + 1):
            if i * j <= max_num_patches:
                tilings.append((i, j))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size

    original_size = np.stack([h, w], dtype=np.float32)
    required_scale_d = candidate_resolutions.astype(np.float32) / original_size
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)
    if np.all(required_scale < 1):
        ix = np.argmax(required_scale)
    else:
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    del original_res
    return candidate_tilings[ix]


class MolmoImageProcessor(BaseImageProcessor):
    model_input_names = ["images", "image_masks", "image_input_idx"]

    def __init__(
        self,
        max_crops: int = 12,
        overlap_margins: list[int] = (4, 4),
        base_image_input_size: list[int] = (336, 336),
        image_token_length_w: int = 12,
        image_token_length_h: int = 12,
        image_patch_size: int = 14,
        image_padding_mask: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_crops = max_crops
        self.overlap_margins = overlap_margins
        self.base_image_input_size = base_image_input_size
        self.image_token_length_w = image_token_length_w
        self.image_token_length_h = image_token_length_h
        self.image_patch_size = image_patch_size
        self.image_padding_mask = image_padding_mask
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

    def image_to_patches_and_tokens(
        self,
        image,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        max_crops: Optional[int] = None,
        overlap_margins: Optional[list[int]] = None,
        base_image_input_size: Optional[list[int]] = None,
        image_token_length_w: Optional[int] = None,
        image_token_length_h: Optional[int] = None,
        image_patch_size: Optional[int] = None,
    ):
        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        tokens_per_image = image_token_length_w * image_token_length_h
        image_base_patch_w = base_image_input_size[1] // base_image_input_d
        image_base_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]
        left_margin, right_margin = overlap_margins
        assert left_margin % 2 == 0
        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d
        tiling = select_tiling(
            original_image_h - total_margin_pixels,
            original_image_w - total_margin_pixels,
            crop_window_size,
            max_crops,
        )
        src, img_mask = resize_and_pad(
            image,
            [tiling[0] * crop_window_size + total_margin_pixels, tiling[1] * crop_window_size + total_margin_pixels],
            normalize=self.do_normalize,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )

        patches_arr = []
        mask_arr = []
        patch_ordering_arr = []

        assert (crop_patches + 1) // 2 == image_token_length_h
        assert (crop_patches + 1) // 2 == image_token_length_w
        on = 0
        for i in range(tiling[0]):
            y0 = i * crop_window_size
            crop_y0 = 0 if i == 0 else left_margin // 2

            crop_h = image_base_patch_h - (right_margin + left_margin)
            if i == 0:
                crop_h += left_margin
            if i == (tiling[0] - 1):
                crop_h += right_margin
            for j in range(tiling[1]):
                x0 = j * crop_window_size
                crop_x0 = 0 if j == 0 else left_margin // 2

                crop_w = image_base_patch_w - (right_margin + left_margin)
                if j == 0:
                    crop_w += left_margin
                if j == (tiling[1] - 1):
                    crop_w += right_margin

                pooled_w = (crop_w + 1) // 2
                pooled_h = (crop_h + 1) // 2
                patch_ordering_arr.append(
                    pad_to_bounding_box(
                        np.reshape(np.arange(on, on + pooled_h * pooled_w, dtype=np.int32), (pooled_h, pooled_w, 1)),
                        crop_y0,
                        crop_x0,
                        image_token_length_h,
                        image_token_length_w,
                        value=-1,
                    )[:, :, 0]
                )
                patches_arr.append(src[y0 : y0 + crop_size, x0 : x0 + crop_size])
                mask_arr.append(img_mask[y0 : y0 + crop_size, x0 : x0 + crop_size])
                on += pooled_h * pooled_w

        patches = np.stack(patches_arr)
        patch_ordering = np.stack(patch_ordering_arr)
        img_mask = np.stack(mask_arr)

        patches = einops.rearrange(
            patches,
            "p (h dh) (w dw) c -> p (h w) (dh dw c)",
            dh=base_image_input_d,
            dw=base_image_input_d,
            h=image_base_patch_h,
            w=image_base_patch_w,
        )
        img_mask = einops.rearrange(
            img_mask,
            "p (h dh) (w dw) -> p (h w) (dh dw)",
            dh=base_image_input_d,
            dw=base_image_input_d,
            h=image_base_patch_h,
            w=image_base_patch_w,
        )

        img_mask = img_mask.astype(np.float32).mean(axis=-1)
        patch_ordering = np.reshape(patch_ordering, [-1])
        valid = patch_ordering >= 0

        patch_ordering_rh = np.reshape(
            patch_ordering, [tiling[0], tiling[1], image_token_length_h, image_token_length_w]
        )
        patch_ordering_rh = np.transpose(patch_ordering_rh, [0, 2, 1, 3])
        patch_ordering_rh = np.reshape(patch_ordering_rh, [-1])
        patch_ordering[valid] = patch_ordering_rh[patch_ordering_rh >= 0]

        h = tiling[0] * crop_window_patches + (right_margin + left_margin)
        w = tiling[1] * crop_window_patches + (right_margin + left_margin)
        per_row = np.full(((w + 1) // 2,), image_patch_token_id)
        per_row = np.concatenate([per_row, [image_col_token_id]], 0)
        joint = np.concatenate([[image_start_token_id], np.tile(per_row, [(h + 1) // 2]), [image_end_token_id]], 0)

        resized, _ = resize_and_pad(
            image,
            base_image_input_size,
            normalize=self.do_normalize,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        resized = einops.rearrange(
            resized,
            "(h dh) (w dw) c -> (h w) (dh dw c)",
            dh=base_image_input_d,
            dw=base_image_input_d,
            h=image_base_patch_h,
            w=image_base_patch_w,
        )
        patches = np.concatenate([np.expand_dims(resized, 0), patches], 0)

        patch_ordering = np.where(patch_ordering >= 0, patch_ordering + tokens_per_image, -1)
        patch_ordering = np.concatenate([np.arange(0, tokens_per_image), patch_ordering], 0)
        per_row = np.full((image_token_length_w,), image_patch_token_id)
        per_row = np.concatenate([per_row, [image_col_token_id]], 0)
        extra_tokens = np.tile(per_row, [image_token_length_h])
        joint = np.concatenate([[image_start_token_id], extra_tokens, [image_end_token_id], joint], 0)
        img_mask = np.pad(img_mask, [[0, 1], [0, 0]], constant_values=-1)
        return patches, joint, patch_ordering, img_mask

    def build_image_input_idx(
        self,
        image_tokens: np.ndarray,
        patch_order: np.ndarray,
        image_patch_token_id: int,
        no_image: Optional[bool] = None,
        image_token_length_w: Optional[int] = None,
        image_token_length_h: Optional[int] = None,
    ):
        tokens_per_image = image_token_length_w * image_token_length_h
        if no_image is not None and no_image:
            return np.zeros((0, tokens_per_image), np.int32)

        image_input_idx = image_tokens == image_patch_token_id
        image_input_idx = np.nonzero(image_input_idx)[0].astype(np.int32)

        if patch_order is not None:
            n_tokens = image_input_idx.shape[0]
            patch_order = np.reshape(patch_order, [-1])
            valid = patch_order >= 0
            assert len(image_input_idx) == valid.sum()

            sorted_patch_ixs = np.zeros([n_tokens], np.int32)
            sorted_patch_ixs[patch_order[valid]] = np.arange(valid.sum(), dtype=np.int32)

            sorted_patch_ixs_ex = np.full(np.shape(patch_order), -1)
            sorted_patch_ixs_ex[valid] = sorted_patch_ixs

            valid = (sorted_patch_ixs_ex >= 0).astype(np.int32)
            image_input_idx = image_input_idx[sorted_patch_ixs_ex * valid]
            image_input_idx = image_input_idx * valid - 100 * (1 - valid)
            image_input_idx = np.reshape(image_input_idx, [-1, tokens_per_image])
        return image_input_idx

    def preprocess(
        self,
        image: np.ndarray,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        max_crops: Optional[int] = None,
        overlap_margins: Optional[list[int]] = None,
        base_image_input_size: Optional[list[int]] = None,
        image_token_length_w: Optional[int] = None,
        image_token_length_h: Optional[int] = None,
        image_patch_size: Optional[int] = None,
        **kwargs,
    ):
        max_crops = max_crops or self.max_crops
        overlap_margins = overlap_margins or self.overlap_margins
        base_image_input_size = base_image_input_size or self.base_image_input_size
        image_token_length_w = image_token_length_w or self.image_token_length_w
        image_token_length_h = image_token_length_h or self.image_token_length_h
        image_patch_size = image_patch_size or self.image_patch_size

        crops, image_tokens, patch_ordering, img_mask = self.image_to_patches_and_tokens(
            image,
            image_patch_token_id,
            image_col_token_id,
            image_start_token_id,
            image_end_token_id,
            max_crops,
            overlap_margins,
            base_image_input_size,
            image_token_length_w,
            image_token_length_h,
            image_patch_size,
        )
        patch_idx = self.build_image_input_idx(
            image_tokens,
            patch_ordering,
            image_patch_token_id,
            image_token_length_w=image_token_length_w,
            image_token_length_h=image_token_length_h,
        )
        return crops, image_tokens, patch_idx, img_mask

    def multimodal_preprocess(
        self,
        images,
        tokens,
        image_idx,
        sequence_length,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        **kwargs,
    ):
        max_total_crops = kwargs.get("max_crops") or self.max_crops
        image_token_length_w = kwargs.get("image_token_length_w") or self.image_token_length_w
        image_token_length_h = kwargs.get("image_token_length_h") or self.image_token_length_h
        image_patch_size = kwargs.get("image_patch_size") or self.image_patch_size
        base_image_input_size = kwargs.get("base_image_input_size") or self.base_image_input_size
        image_num_patch = (
            base_image_input_size[0] // image_patch_size,
            base_image_input_size[1] // image_patch_size,
        )
        image_padding_mask = kwargs.get("image_padding_mask") or self.image_padding_mask

        tokens_per_image = image_token_length_w * image_token_length_h
        n_pixels = image_patch_size * image_patch_size * 3
        n_patches = image_num_patch[0] * image_num_patch[1]
        del max_total_crops, tokens_per_image, n_pixels, n_patches, sequence_length

        if images is None:
            return {"input_ids": tokens}

        all_crops = []
        all_image_idx = []
        out_tokens = []
        all_crop_masks = []

        for ix, image in enumerate(images):
            token_ix = image_idx[ix]
            crops, image_tokens, patch_idx, img_mask = self.preprocess(
                image,
                image_patch_token_id,
                image_col_token_id,
                image_start_token_id,
                image_end_token_id,
                **kwargs,
            )

            if token_ix == -1:
                start = 0
                token_ix = 0
                end = 0
            else:
                start = 0 if ix == 0 else image_idx[ix - 1] + 1
                end = token_ix + 1

            all_image_idx.append(patch_idx + token_ix)
            all_crops.append(crops)
            out_tokens.append(tokens[start:token_ix])
            out_tokens.append(image_tokens)
            if ix == (len(images) - 1):
                out_tokens.append(tokens[end:])
            if image_padding_mask:
                all_crop_masks.append(img_mask)

        input_ids = np.concatenate(out_tokens, 0)
        images = np.concatenate(all_crops, 0)
        image_input_idx = np.concatenate(all_image_idx, 0)
        out = {"input_ids": input_ids, "images": images, "image_input_idx": image_input_idx}
        if image_padding_mask:
            out["image_masks"] = np.concatenate(all_crop_masks, 0)
        return out


__all__ = ["MolmoImageProcessor"]
