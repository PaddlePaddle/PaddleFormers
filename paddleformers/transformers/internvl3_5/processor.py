# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 OpenGVLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import List, Union

from ..feature_extraction_utils import BatchFeature
from ..image_utils import ImageInput
from ..processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils_base import PreTokenizedInput, TextInput

__all__ = ["InternVLProcessor"]


class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
        "images_kwargs": {},
    }


class InternVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "InternVLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_seq_length=256,
        image_token="<image>",
        img_start_token="<img>",
        img_end_token="</img>",
        img_context_token="<IMG_CONTEXT>",
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token = image_token
        self.img_start_token = img_start_token
        self.img_end_token = img_end_token
        self.img_context_token = img_context_token
        self.img_context_token_id = (
            tokenizer.convert_tokens_to_ids(img_context_token) if tokenizer is not None else 151671
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _expand_image_tokens(self, text, num_patches_list):
        if num_patches_list is None:
            return text
        patch_index = 0
        expanded = []
        for sample in text:
            if self.image_token not in sample and len(num_patches_list) > 0 and len(text) == 1:
                sample = self.image_token + "\n" + sample
            while self.image_token in sample:
                if patch_index >= len(num_patches_list):
                    raise ValueError("More <image> placeholders than processed images.")
                image_tokens = (
                    self.img_start_token
                    + self.img_context_token * self.image_seq_length * num_patches_list[patch_index]
                    + self.img_end_token
                )
                sample = sample.replace(self.image_token, image_tokens, 1)
                patch_index += 1
            expanded.append(sample)
        if patch_index != len(num_patches_list):
            raise ValueError("The number of images does not match <image> placeholders in text.")
        return expanded

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=getattr(self.tokenizer, "init_kwargs", {}),
            **kwargs,
        )
        return_tensors = kwargs.get("return_tensors", None)
        output_kwargs["images_kwargs"].pop("return_tensors", None)
        output_kwargs["text_kwargs"].pop("return_tensors", None)

        image_inputs = {}
        num_patches_list = None
        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                return_tensors=return_tensors,
                **output_kwargs["images_kwargs"],
            )
            num_patches_list = image_inputs["num_patches_list"]
            image_inputs.pop("num_patches_list", None)

        if text is None:
            data = dict(image_inputs)
            return BatchFeature(data=data)
        if not isinstance(text, list):
            text = [text]
        text = self._expand_image_tokens(text.copy(), num_patches_list)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=return_tensors)
        data = dict(text_inputs)
        data.update(dict(image_inputs))
        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
