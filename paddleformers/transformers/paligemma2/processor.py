# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional, Union

import numpy as np

from ..feature_extraction_utils import BatchFeature
from ..image_utils import ImageInput, is_valid_image
from ..processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils import AddedToken
from ..tokenizer_utils_base import PreTokenizedInput, TextInput

IMAGE_TOKEN = "<image>"
EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [f"<seg{i:0>3}>" for i in range(128)]


class PaliGemmaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
        "images_kwargs": {"data_format": "channels_first"},
    }


class PaliGemmaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_seq_length = getattr(image_processor, "image_seq_length", 1024)
        image_processor.image_seq_length = self.image_seq_length
        if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken(IMAGE_TOKEN, normalized=False)]})
        tokenizer.add_tokens(EXTRA_TOKENS)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.image_token = IMAGE_TOKEN
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs: Unpack[PaliGemmaProcessorKwargs],
    ) -> BatchFeature:
        if images is None:
            raise ValueError("`images` are required by PaliGemmaProcessor.")
        if text is None:
            text = ""
        if not isinstance(text, list):
            text = [text]
        output_kwargs = self._merge_kwargs(
            PaliGemmaProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)
        if suffix is not None and not isinstance(suffix, list):
            suffix = [suffix]
        if suffix is not None:
            if len(suffix) != len(text):
                raise ValueError("`suffix` must have the same batch size as `text`.")
            suffix = [value + self.tokenizer.eos_token for value in suffix]

        image_batches = self._normalize_images(images, len(text))
        images = [image for image_batch in image_batches for image in image_batch]
        input_strings = []
        for prompt, sample_images in zip(text, image_batches):
            if IMAGE_TOKEN in prompt:
                if prompt.count(IMAGE_TOKEN) != len(sample_images):
                    raise ValueError("The number of `<image>` tokens must match the number of images for each prompt.")
                expanded = prompt.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                last_image = expanded.rfind(IMAGE_TOKEN)
                prompt = (
                    expanded[: last_image + len(IMAGE_TOKEN)]
                    + self.tokenizer.bos_token
                    + expanded[last_image + len(IMAGE_TOKEN) :]
                )
            else:
                prompt = IMAGE_TOKEN * (self.image_seq_length * len(sample_images)) + self.tokenizer.bos_token + prompt
            input_strings.append(prompt + "\n")

        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=True,
            **output_kwargs["text_kwargs"],
        )
        data = {**inputs, "pixel_values": pixel_values}
        if suffix is not None:
            labels = np.asarray(inputs["input_ids"]).copy()
            labels[np.asarray(inputs["token_type_ids"]) == 0] = -100
            labels = np.concatenate([labels[:, 1:], np.full((labels.shape[0], 1), -100, dtype=labels.dtype)], axis=1)
            data["labels"] = labels
        return BatchFeature(data=data, tensor_type=return_tensors)

    @staticmethod
    def _normalize_images(images, batch_size):
        if is_valid_image(images):
            if batch_size != 1:
                raise ValueError("The number of images must match the number of prompts.")
            return [[images]]
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError("`images` must be an image or a list containing one image per prompt.")
        if not is_valid_image(images[0]):
            raise ValueError(
                "PaliGemma2 supports exactly one image per prompt; nested image batches are not supported."
            )
        if len(images) != batch_size:
            raise ValueError("The number of images must match the number of prompts.")
        return [[image] for image in images]

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        if image_sizes is None:
            return MultiModalData()
        return MultiModalData(
            num_image_tokens=[self.image_seq_length] * len(image_sizes), num_image_patches=[1] * len(image_sizes)
        )

    @property
    def model_input_names(self):
        return list(
            self.tokenizer.model_input_names + ["token_type_ids", "labels"] + self.image_processor.model_input_names
        )


__all__ = ["PaliGemmaProcessor"]
