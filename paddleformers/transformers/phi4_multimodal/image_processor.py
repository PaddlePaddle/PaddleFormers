# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Phi-4 Multimodal."""

import math

import numpy as np
import paddle
from PIL import Image, ImageOps

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import BaseImageProcessor
from ..image_utils import PILImageResampling, make_flat_list_of_images


class Phi4MultimodalImageProcessor(BaseImageProcessor):
    model_input_names = ["image_pixel_values", "image_sizes", "image_attention_mask"]

    def __init__(
        self,
        size=None,
        patch_size=14,
        dynamic_hd=36,
        image_mean=None,
        image_std=None,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        do_convert_rgb=True,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        **kwargs,
    ):
        self.size = size or {"height": 448, "width": 448}
        self.patch_size = patch_size
        self.dynamic_hd = dynamic_hd
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.resample = resample
        self.rescale_factor = rescale_factor
        super().__init__(**kwargs)

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)

    def preprocess(
        self,
        images,
        size=None,
        patch_size=None,
        dynamic_hd=None,
        image_mean=None,
        image_std=None,
        do_rescale=None,
        do_normalize=None,
        rescale_factor=None,
        return_tensors=None,
        **kwargs,
    ):
        size = size or self.size
        if isinstance(size, int):
            size = {"height": size, "width": size}
        patch_size = patch_size if patch_size is not None else self.patch_size
        dynamic_hd = dynamic_hd if dynamic_hd is not None else self.dynamic_hd
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor

        height = size["height"]
        width = size["width"]
        if height != width:
            raise ValueError("Phi4MultimodalImageProcessor only supports square sizes.")

        images = make_flat_list_of_images(images)
        mask_size = height // patch_size
        images_transformed = []
        masks_transformed = []
        image_tokens = []
        image_sizes = []

        for image in images:
            image = self._to_pil_image(image)
            resized_image, attention_mask = self.dynamic_preprocess(
                image, height, patch_size, mask_size, max_num=dynamic_hd
            )
            processed_image = self._to_chw_array(resized_image)
            if do_rescale:
                processed_image = processed_image * rescale_factor
            if do_normalize:
                mean = np.asarray(image_mean, dtype=np.float32)[:, None, None]
                std = np.asarray(image_std, dtype=np.float32)[:, None, None]
                processed_image = (processed_image - mean) / std

            global_image = self._resize_chw(processed_image, height, height)
            image_height, image_width = processed_image.shape[-2:]
            mask_height, mask_width = attention_mask.shape[-2:]
            global_attention_mask = np.ones((1, mask_size, mask_size), dtype=bool)

            hd_image = processed_image.reshape(1, 3, image_height // height, height, image_width // width, width)
            hd_image = hd_image.transpose(0, 2, 4, 1, 3, 5).reshape(-1, 3, height, width)

            attention_mask = attention_mask.reshape(
                mask_height // mask_size, mask_size, mask_width // mask_size, mask_size
            )
            attention_mask = attention_mask.transpose(0, 2, 1, 3).reshape(-1, mask_size, mask_size)

            downsample_attention_mask = attention_mask[:, 0::2, 0::2]
            pooled_mask_size = mask_size // 2 + mask_size % 2
            downsample_attention_mask = downsample_attention_mask.reshape(
                mask_height // mask_size,
                mask_width // mask_size,
                pooled_mask_size,
                pooled_mask_size,
            )
            downsample_attention_mask = downsample_attention_mask.transpose(0, 2, 1, 3)
            downsample_attention_mask = downsample_attention_mask.reshape(
                downsample_attention_mask.shape[0] * downsample_attention_mask.shape[1],
                downsample_attention_mask.shape[2] * downsample_attention_mask.shape[3],
            )

            base_feat_size = mask_size // 2 + mask_size % 2
            num_img_tokens = (
                base_feat_size**2
                + 1
                + int(downsample_attention_mask.sum().item())
                + int(downsample_attention_mask[:, 0].sum().item())
                + base_feat_size
            )

            hd_image = np.concatenate([global_image[None, ...], hd_image], axis=0)
            attention_mask = np.concatenate([global_attention_mask, attention_mask], axis=0)

            images_transformed.append(hd_image.astype(np.float32))
            masks_transformed.append(attention_mask.astype(bool))
            image_tokens.append(num_img_tokens)
            image_sizes.append([image_height, image_width])

        max_crops = max(image.shape[0] for image in images_transformed)
        images_transformed = np.stack(
            [self._pad_images_to_max_crops(image, max_crops) for image in images_transformed], axis=0
        )
        masks_transformed = np.stack(
            [self._pad_masks_to_max_crops(mask, max_crops) for mask in masks_transformed], axis=0
        )

        data = {
            "image_pixel_values": images_transformed,
            "image_sizes": np.asarray(image_sizes, dtype=np.int64),
            "image_attention_mask": masks_transformed,
            "num_img_tokens": image_tokens,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, image_size, patch_size, mask_size, max_num=36, min_num=1):
        orig_width, orig_height = image.size
        w_crop_num = math.ceil(orig_width / float(image_size))
        h_crop_num = math.ceil(orig_height / float(image_size))
        if w_crop_num * h_crop_num > max_num:
            aspect_ratio = orig_width / orig_height
            target_ratios = {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if min_num <= i * j <= max_num
            }
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
        else:
            target_width = image_size * w_crop_num
            target_height = image_size * h_crop_num
            target_aspect_ratio = (w_crop_num, h_crop_num)

        ratio_width = target_width / orig_width
        ratio_height = target_height / orig_height
        if ratio_width < ratio_height:
            new_size = (target_width, int(orig_height * ratio_width))
            padding_width = 0
            padding_height = target_height - int(orig_height * ratio_width)
        else:
            new_size = (int(orig_width * ratio_height), target_height)
            padding_width = target_width - int(orig_width * ratio_height)
            padding_height = 0

        attention_mask = np.ones(
            (int(mask_size * target_aspect_ratio[1]), int(mask_size * target_aspect_ratio[0])),
            dtype=bool,
        )
        if padding_width >= patch_size:
            attention_mask[:, -math.floor(padding_width / patch_size) :] = False
        if padding_height >= patch_size:
            attention_mask[-math.floor(padding_height / patch_size) :, :] = False

        if min(new_size[1], target_height) < 10 or min(new_size[0], target_width) < 10:
            raise ValueError(f"the aspect ratio is very extreme {new_size}")

        image = image.resize((int(new_size[0]), int(new_size[1])), resample=self.resample)
        resized_img = ImageOps.expand(
            image, border=(0, 0, int(padding_width), int(padding_height)), fill=(255, 255, 255)
        )
        return resized_img, attention_mask

    def _to_pil_image(self, image):
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            if isinstance(image, paddle.Tensor):
                image = image.detach().cpu().numpy()
            image = np.asarray(image)
            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}.")
            if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
                image = image.transpose(1, 2, 0)
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = image * 255
                image = np.clip(image, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        return pil_image.convert("RGB") if self.do_convert_rgb else pil_image

    @staticmethod
    def _to_chw_array(image):
        return np.asarray(image).astype(np.float32).transpose(2, 0, 1)

    def _resize_chw(self, image, height, width):
        resized = []
        for channel in image:
            pil_channel = Image.fromarray(channel.astype(np.float32), mode="F")
            pil_channel = pil_channel.resize((width, height), resample=self.resample)
            resized.append(np.asarray(pil_channel, dtype=np.float32))
        return np.stack(resized, axis=0)

    @staticmethod
    def _pad_images_to_max_crops(images, max_crops):
        if max_crops <= images.shape[0]:
            return images
        pad = np.zeros((max_crops - images.shape[0], *images.shape[1:]), dtype=images.dtype)
        return np.concatenate([images, pad], axis=0)

    @staticmethod
    def _pad_masks_to_max_crops(masks, max_crops):
        if max_crops <= masks.shape[0]:
            return masks
        pad = np.ones((max_crops - masks.shape[0], *masks.shape[1:]), dtype=masks.dtype)
        return np.concatenate([masks, pad], axis=0)


__all__ = ["Phi4MultimodalImageProcessor"]
