# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Combined text and image processor for LFM2-VL."""

import math

from ..feature_extraction_utils import BatchFeature
from ..processing_utils import ProcessorMixin


class Lfm2VlProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Lfm2VlImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, chat_template=None, **kwargs):
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_start_token = getattr(tokenizer, "image_start_token", "<|image_start|>")
        self.image_end_token = getattr(tokenizer, "image_end_token", "<|image_end|>")
        self.image_thumbnail_token = getattr(tokenizer, "image_thumbnail_token", "<|img_thumbnail|>")
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _tokens_per_image(self, rows, columns, image_size, use_special_tokens=True):
        patch = self.image_processor.encoder_patch_size
        factor = self.image_processor.downsample_factor
        tile_patches = self.image_processor.tile_size // patch
        tokens_per_tile = math.ceil(tile_patches / factor) ** 2
        image_height, image_width = image_size
        image_tokens = math.ceil((image_height // patch) / factor) * math.ceil((image_width // patch) / factor)
        parts = [self.image_start_token] if use_special_tokens else []
        if rows > 1 or columns > 1:
            for row in range(rows):
                for column in range(columns):
                    if use_special_tokens:
                        parts.append(f"<|img_row_{row + 1}_col_{column + 1}|>")
                    parts.append(self.image_token * tokens_per_tile)
            if self.image_processor.use_thumbnail:
                if use_special_tokens:
                    parts.append(self.image_thumbnail_token)
                parts.append(self.image_token * image_tokens)
        else:
            parts.append(self.image_token * image_tokens)
        if use_special_tokens:
            parts.append(self.image_end_token)
        return "".join(parts)

    def __call__(self, images=None, text=None, return_tensors=None, use_image_special_tokens=True, **kwargs):
        if text is None:
            raise ValueError("Lfm2VlProcessor requires text.")
        texts = [text] if isinstance(text, str) else list(text)
        image_list = [] if images is None else (list(images) if isinstance(images, (list, tuple)) else [images])
        if image_list and len(image_list) != len(texts):
            raise ValueError("LFM2-VL currently expects one image per prompt.")
        image_inputs = {}
        if image_list:
            image_inputs = self.image_processor(image_list, return_tensors=return_tensors, return_row_col_info=True)
            for index, prompt in enumerate(texts):
                if prompt.count(self.image_token) != 1:
                    raise ValueError("Each LFM2-VL image prompt must contain exactly one <image> token.")
                replacement = self._tokens_per_image(
                    image_inputs["image_rows"][index],
                    image_inputs["image_cols"][index],
                    image_inputs["image_sizes"][index],
                    use_image_special_tokens,
                )
                texts[index] = prompt.replace(self.image_token, replacement)
        tokenized = self.tokenizer(texts, return_tensors=return_tensors, **kwargs)
        data = dict(tokenized)
        for key in ["pixel_values", "pixel_attention_mask", "spatial_shapes"]:
            if key in image_inputs:
                data[key] = image_inputs[key]
        return BatchFeature(data=data)

    @property
    def model_input_names(self):
        return list(self.tokenizer.model_input_names + self.image_processor.model_input_names)


__all__ = ["Lfm2VlProcessor"]
