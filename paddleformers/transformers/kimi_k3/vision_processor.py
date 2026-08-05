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

"""Image processor class for Kimi-K3."""

import json
from typing import Any, Dict, Optional, Union

import numpy as np
import paddle
from PIL import Image

from ..image_processing_utils import BaseImageProcessor, BatchFeature
from ..tokenizer_utils_base import TensorType
from .media_utils import (
    MediaInput,
    TransparentBgConfig,
    _to_tensor,
    ensure_media_type,
    image_to_np,
    navit_patchify,
    navit_resize_image,
    normalize,
)

__all__ = ["KimiK3VisionProcessor"]


class KimiK3VisionProcessor(BaseImageProcessor):
    model_type = "kimi_k3"

    def __init__(
        self,
        media_proc_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.media_proc_cfg = media_proc_cfg

    @property
    def _transparent_bg_config(self) -> Optional[TransparentBgConfig]:
        cfg = self.media_proc_cfg.get("transparent_bg_config")
        if cfg is None:
            return None
        if isinstance(cfg, TransparentBgConfig):
            return cfg
        return TransparentBgConfig(**cfg)

    @property
    def _transparent_bg_fill_stage(self) -> str:
        return self.media_proc_cfg.get("transparent_bg_fill_stage", "before_resize")

    def media_tokens_calculator(self, media: MediaInput):
        media = ensure_media_type(
            media,
            transparent_bg_config=self._transparent_bg_config,
            transparent_bg_fill_stage=self._transparent_bg_fill_stage,
        )
        ret = self.get_resize_config(media)
        return ret["num_tokens"]

    @classmethod
    def make_image_prompt(cls, width: int, height: int) -> str:
        """Build the K3 image placeholder with resolution info."""
        return f"<|media_begin|>image {width}x{height}" f"<|media_content|><|media_pad|><|media_end|>"

    def get_resize_config(self, media_input: MediaInput) -> dict:
        if media_input["type"] == "image":
            w, h = media_input["image"].size
            ret = navit_resize_image(
                w,
                h,
                self.media_proc_cfg["patch_size"],
                self.media_proc_cfg["merge_kernel_size"],
                self.media_proc_cfg["in_patch_limit"],
                self.media_proc_cfg["patch_limit_on_one_side"],
                self.media_proc_cfg["fixed_output_tokens"],
            )
            return ret
        else:
            raise ValueError("Unsupported type: {}".format(media_input["type"]))

    def resize_image(
        self, image: Image.Image, new_width: int, new_height: int, pad_width: int, pad_height: int
    ) -> np.ndarray:
        image_np = image_to_np(
            image,
            (new_width, new_height),
            "resize",
            transparent_bg_config=self._transparent_bg_config,
            transparent_bg_fill_stage=self._transparent_bg_fill_stage,
        )
        image_np = np.pad(
            image_np,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return image_np

    def preprocess(
        self,
        medias: list[MediaInput],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Preprocess a atom vision input (images) into model-ready tensors.

        Args:
            medias: List of MediaInput.
            return_tensors: Desired output format ('pt', 'np', 'tf', or None).

        Returns:
            BatchFeature containing 'pixel_values' and 'grid_thws' tensors.
        """
        if not isinstance(medias, list):
            medias = [medias]
        if medias:
            pixel_values = []
            for item in medias:
                item = ensure_media_type(
                    item,
                    transparent_bg_config=self._transparent_bg_config,
                    transparent_bg_fill_stage=self._transparent_bg_fill_stage,
                )
                resize_config = self.get_resize_config(item)
                new_width, new_height, pad_width, pad_height = (
                    resize_config["new_width"],
                    resize_config["new_height"],
                    resize_config["pad_width"],
                    resize_config["pad_height"],
                )
                if item["type"] == "image":
                    image = item["image"]
                    image_np = self.resize_image(image, new_width, new_height, pad_width, pad_height)
                    pixel_values.append(np.expand_dims(image_np, axis=0))
                else:
                    raise ValueError("Unsupported type: {}".format(item["type"]))
            normalized_pixel_values = []
            image_std_inv = 1.0 / np.array(self.media_proc_cfg["image_std"])
            image_mean = np.array(self.media_proc_cfg["image_mean"])
            for pixels in pixel_values:
                pixels = normalize(pixels, image_mean, image_std_inv)
                pixels_and_thw = navit_patchify(
                    pixels,
                    self.media_proc_cfg["patch_size"],
                )
                normalized_pixel_values.append(pixels_and_thw)

            pixel_values = paddle.concat(
                [_to_tensor(pixel_value["pixel_values"]) for pixel_value in normalized_pixel_values]
            )
            grid_thws = paddle.concat(
                [
                    _to_tensor(pixel_value["grid_thw"], dtype="int64").unsqueeze(0)
                    for pixel_value in normalized_pixel_values
                ]
            )

            data = {
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
            }

        else:
            data = {}

        return BatchFeature(data=data, tensor_type="pd")

    def __repr__(self):
        return f"KimiK3VisionProcessor(media_proc_cfg={self.media_proc_cfg})"

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["media_proc_cfg"] = self.media_proc_cfg
        if "media_processor" in output:
            del output["media_processor"]
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        media_proc_cfg = config.pop("media_proc_cfg", {})

        from_pretrained_only_keys = ["subfolder", "revision", "cache_dir", "local_files_only", "trust_remote_code"]
        for key in from_pretrained_only_keys:
            kwargs.pop(key, None)
        merged_kwargs = {**config, **kwargs}

        return cls(media_proc_cfg=media_proc_cfg, **merged_kwargs)

    def to_json_string(self):
        dictionary = self.to_dict()
        for key, value in dictionary.items():
            if hasattr(value, "tolist"):
                dictionary[key] = value.tolist()
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"
