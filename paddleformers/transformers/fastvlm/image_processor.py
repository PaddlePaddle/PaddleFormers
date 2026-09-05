# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Image preprocessing for FastVLM."""

from transformers import CLIPImageProcessor

from ..image_processing_utils import PaddleImageProcessingMixin


class FastVLMImageProcessor(PaddleImageProcessingMixin, CLIPImageProcessor):
    def __init__(self, image_size=1024, **kwargs):
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"shortest_edge": image_size})
        kwargs.setdefault("do_center_crop", True)
        kwargs.setdefault("crop_size", {"height": image_size, "width": image_size})
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.0, 0.0, 0.0])
        kwargs.setdefault("image_std", [1.0, 1.0, 1.0])
        super().__init__(**kwargs)


__all__ = ["FastVLMImageProcessor"]
