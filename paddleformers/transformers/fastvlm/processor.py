# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Processor for FastVLM prompts and images."""

import numpy as np

from ..feature_extraction_utils import BatchFeature
from ..processing_utils import ProcessorMixin

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200


class FastVLMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FastVLMImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, chat_template=None, **kwargs):
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_INDEX
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _encode_prompt(self, prompt):
        chunks = prompt.split(self.image_token)
        input_ids = []
        for index, chunk in enumerate(chunks):
            chunk_ids = self.tokenizer(chunk, add_special_tokens=False)["input_ids"]
            input_ids.extend(chunk_ids)
            if index != len(chunks) - 1:
                input_ids.append(self.image_token_id)
        return input_ids

    def __call__(self, images=None, text=None, return_tensors=None, padding=True, **kwargs):
        if text is None:
            raise ValueError("FastVLMProcessor requires text.")
        texts = [text] if isinstance(text, str) else list(text)
        if images is not None:
            image_list = list(images) if isinstance(images, (list, tuple)) else [images]
            if len(image_list) != len(texts):
                raise ValueError("FastVLM expects one image per prompt.")
            for index, prompt in enumerate(texts):
                if self.image_token not in prompt:
                    texts[index] = self.image_token + "\n" + prompt
                elif prompt.count(self.image_token) != 1:
                    raise ValueError("FastVLM expects exactly one <image> token per prompt.")
        elif any(self.image_token in prompt for prompt in texts):
            raise ValueError("An image must be supplied for each <image> token.")

        encoded = [self._encode_prompt(prompt) for prompt in texts]
        max_length = max(len(item) for item in encoded)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        input_ids = np.full([len(encoded), max_length], pad_token_id, dtype="int64")
        attention_mask = np.zeros_like(input_ids)
        for batch_index, item in enumerate(encoded):
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                input_ids[batch_index, -len(item) :] = item
                attention_mask[batch_index, -len(item) :] = 1
            else:
                input_ids[batch_index, : len(item)] = item
                attention_mask[batch_index, : len(item)] = 1
        data = {"input_ids": input_ids, "attention_mask": attention_mask}
        if images is not None:
            data.update(self.image_processor(image_list, return_tensors=return_tensors))
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        return list(self.tokenizer.model_input_names + self.image_processor.model_input_names)


__all__ = ["FastVLMProcessor"]
