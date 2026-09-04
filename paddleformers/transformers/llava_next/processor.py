# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Processor for Llava-NeXT."""

from __future__ import annotations

from typing import TypedDict

from ..feature_extraction_utils import BatchFeature
from ..processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from .image_processor import LlavaNextImageProcessor, select_best_resolution


class LlavaNextImageProcessorKwargs(TypedDict, total=False):
    do_convert_rgb: bool
    do_resize: bool
    size: int | list[int] | tuple[int, ...] | dict[str, int]
    default_to_square: bool
    resample: int
    do_rescale: bool
    rescale_factor: float
    do_normalize: bool
    image_mean: float | list[float] | tuple[float, ...]
    image_std: float | list[float] | tuple[float, ...]
    do_center_crop: bool
    do_pad: bool
    crop_size: int | list[int] | tuple[int, ...] | dict[str, int]
    data_format: str
    input_data_format: str
    return_tensors: str
    image_grid_pinpoints: list[list[int]]


class LlavaNextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "do_pad": True,
        },
    }
    images_kwargs: LlavaNextImageProcessorKwargs


LlavaNextProcessorKwargs.__annotations__["images_kwargs"] = LlavaNextImageProcessorKwargs


class LlavaNextProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LlavaNextImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    model_input_names = ["input_ids", "attention_mask", "pixel_values", "image_sizes", "image_grid_thw"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        num_additional_image_tokens=0,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.num_additional_image_tokens = num_additional_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = getattr(tokenizer, "image_token", image_token) if tokenizer is not None else image_token
        self.image_token_id = (
            getattr(tokenizer, "image_token_id", None) if tokenizer is not None else kwargs.get("image_token_id", None)
        )
        if self.image_token_id is None and tokenizer is not None:
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor or LlavaNextImageProcessor(), tokenizer, chat_template=chat_template)

    def __call__(self, images=None, text=None, **kwargs: Unpack[LlavaNextProcessorKwargs]):
        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")
        output_kwargs = self._merge_kwargs(
            LlavaNextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if self.tokenizer is not None else {},
            **kwargs,
        )
        if isinstance(text, str):
            text = [text]
        if text is None:
            text = [self.image_token]

        image_inputs = (
            self.image_processor(images, return_tensors=None, **output_kwargs["images_kwargs"])
            if images is not None
            else {}
        )
        prompt_strings = text
        if image_inputs and self.patch_size is not None:
            height = image_inputs["pixel_values"][0][0].shape[-2]
            width = image_inputs["pixel_values"][0][0].shape[-1]
            image_sizes = iter(image_inputs["image_sizes"])
            prompt_strings = []
            for sample in text:
                while self.image_token in sample:
                    orig_height, orig_width = next(image_sizes)
                    num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
                    if self.vision_feature_select_strategy == "default":
                        num_image_tokens -= 1
                    sample = sample.replace(self.image_token, "<placeholder>" * num_image_tokens, 1)
                prompt_strings.append(sample.replace("<placeholder>", self.image_token))

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = (
            self.tokenizer(prompt_strings, return_tensors=None, **output_kwargs["text_kwargs"])
            if self.tokenizer is not None
            else {}
        )
        if text_inputs:
            self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])
        if return_mm_token_type_ids and text_inputs:
            import numpy as np

            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        data = {**text_inputs, **image_inputs}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], self.image_processor.image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width
        patches_height = height // self.patch_size
        patches_width = width // self.patch_size
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )
        base_features = patches_height * patches_width + self.num_additional_image_tokens
        return unpadded_features + newline_features + base_features

    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width
        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2
        return current_height * current_width, current_height

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = dict(LlavaNextProcessorKwargs._defaults.get("images_kwargs", {}))
            images_kwargs.update(kwargs)
            size = images_kwargs.get("size", None) or self.image_processor.size
            if "shortest_edge" in size:
                processed_height = processed_width = size["shortest_edge"]
            else:
                processed_height = min(size["height"], size["width"])
                processed_width = processed_height

            num_image_tokens = []
            num_image_patches = [1] * len(image_sizes)
            for image_size in image_sizes:
                orig_height, orig_width = image_size
                tokens = self._get_number_of_features(orig_height, orig_width, processed_height, processed_width)
                if self.vision_feature_select_strategy == "default":
                    tokens -= 1
                num_image_tokens.append(tokens)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["LlavaNextProcessor", "LlavaNextImageProcessor"]
