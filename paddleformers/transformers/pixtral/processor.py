# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import os

from ..auto.tokenizer import AutoTokenizer
from ..processing_utils import ProcessorMixin
from .image_processor import PixtralImageProcessor


class PixtralProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        chat_template=None,
        image_token: str = "[IMG]",
        image_break_token: str = "[IMG_BREAK]",
        image_end_token: str = "[IMG_END]",
        image_max_pixels: int = 384 * 384,
        image_min_pixels: int = 32 * 32,
        video_max_pixels: int = 256 * 256,
        video_min_pixels: int = 16 * 16,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token = image_token
        self.image_break_token = image_break_token
        self.image_end_token = image_end_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token) if tokenizer is not None else None
        self.image_break_token_id = (
            tokenizer.convert_tokens_to_ids(image_break_token) if tokenizer is not None else None
        )
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(image_end_token) if tokenizer is not None else None
        self.image_max_pixels = image_max_pixels
        self.image_min_pixels = image_min_pixels
        self.video_max_pixels = video_max_pixels
        self.video_min_pixels = video_min_pixels

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None) or AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        preprocessor_config = {}
        preprocessor_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, encoding="utf-8") as f:
                preprocessor_config = json.load(f)

        processor_config = {}
        processor_path = os.path.join(pretrained_model_name_or_path, "processor_config.json")
        if os.path.exists(processor_path):
            with open(processor_path, encoding="utf-8") as f:
                processor_config = json.load(f)

        image_processor = kwargs.pop("image_processor", None) or PixtralImageProcessor(**preprocessor_config)
        chat_template = kwargs.pop("chat_template", None)
        chat_template_path = os.path.join(pretrained_model_name_or_path, "chat_template.json")
        if chat_template is None and os.path.exists(chat_template_path):
            with open(chat_template_path, encoding="utf-8") as f:
                chat_template = json.load(f).get("chat_template")

        init_kwargs = {**processor_config, **kwargs}
        init_kwargs.pop("processor_class", None)
        return cls(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template, **init_kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names if self.tokenizer is not None else []
        image_input_names = self.image_processor.model_input_names if self.image_processor is not None else []
        return tokenizer_input_names + image_input_names


__all__ = ["PixtralProcessor"]
