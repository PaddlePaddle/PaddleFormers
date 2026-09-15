# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Paddle Idefics3 processor."""

from itertools import accumulate
from typing import Optional, TypedDict, Union

import numpy as np

from ..feature_extraction_utils import BatchFeature
from ..image_utils import ImageInput, is_valid_image
from ..processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils import AddedToken
from ..tokenizer_utils_base import PreTokenizedInput, TextInput


class Idefics3ImageProcessorKwargs(TypedDict, total=False):
    """Image processor kwargs for IDEfics3, including return_row_col_info."""

    do_convert_rgb: bool
    do_resize: bool
    do_rescale: bool
    do_normalize: bool
    do_image_splitting: bool
    do_pad: bool
    return_row_col_info: bool
    size: dict
    max_image_size: dict
    image_mean: list[float]
    image_std: list[float]
    resample: int
    return_tensors: str


class Idefics3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "return_row_col_info": True,
        },
    }
    images_kwargs: Idefics3ImageProcessorKwargs


class Idefics3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, image_seq_len: int = 169, chat_template=None, **kwargs):
        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True).content
        self.image_token = AddedToken("<image>", normalized=False, special=True).content
        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True).content
        self.global_image_tag = "<global-img>"
        self.image_seq_len = image_seq_len
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token) if tokenizer is not None else None
        self.fake_image_token_id = (
            tokenizer.convert_tokens_to_ids(self.fake_image_token) if tokenizer is not None else None
        )
        self.global_image_token_id = (
            tokenizer.convert_tokens_to_ids(self.global_image_tag) if tokenizer is not None else None
        )
        self.row_col_ids = (
            [tokenizer.convert_tokens_to_ids(f"<row_{i + 1}_col_{j + 1}>") for i in range(6) for j in range(6)]
            if tokenizer is not None
            else []
        )

        if tokenizer is not None:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        self.fake_image_token,
                        self.image_token,
                        self.end_of_utterance_token,
                    ]
                }
            )
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
            self.fake_image_token_id = tokenizer.convert_tokens_to_ids(self.fake_image_token)

        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        image_seq_len: Optional[int] = None,
        **kwargs: Unpack[Idefics3ProcessorKwargs],
    ) -> BatchFeature:
        images, text = self.prepare_inputs_layout(images=images, text=text)
        self.validate_inputs(images=images, text=text)

        output_kwargs = self._merge_kwargs(
            Idefics3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)

        image_inputs = {}
        text_inputs = {}
        if images is not None:
            image_kwargs = output_kwargs["images_kwargs"].copy()
            image_kwargs.pop("return_tensors", None)
            image_inputs = self.image_processor(images, return_tensors=None, **image_kwargs)
            images_replacements = [
                self.replace_image_token(image_inputs, idx, image_seq_len) for idx in range(sum(map(len, images)))
            ]
            image_inputs.pop("rows", None)
            image_inputs.pop("cols", None)

            if text is not None:
                text = self.get_text_with_replacements(text, images_replacements)
                text_inputs = self.tokenizer(text, return_tensors=None, **output_kwargs["text_kwargs"])
                self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
            else:
                text = images_replacements
                text_inputs = self.tokenizer(text, return_tensors=None, **output_kwargs["text_kwargs"])
        elif text is not None:
            text_inputs = self.tokenizer(text=text, return_tensors=None, **output_kwargs["text_kwargs"])

        if return_mm_token_type_ids and text_inputs:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def prepare_inputs_layout(self, images=None, text=None):
        if text is not None:
            if isinstance(text, str):
                text = [text]
            else:
                text = list(text)

        if images is not None:
            images = self.image_processor.fetch_images(images)
            if is_valid_image(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and images and is_valid_image(images[0]):
                if text is not None:
                    n_images_in_text = [sample.count(self.image_token) for sample in text]
                    cumsum_images = [0] + list(accumulate(n_images_in_text))
                    images = [images[cumsum_images[i] : cumsum_images[i + 1]] for i in range(len(n_images_in_text))]
                else:
                    images = [list(images)]
            else:
                images = [list(sample) for sample in images]
        return images, text

    def validate_inputs(self, images=None, text=None):
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")
        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]
            if images is not None:
                n_images_in_images = [len(sample) for sample in images]
                if n_images_in_text != n_images_in_images:
                    raise ValueError(
                        f"The total number of {self.image_token} tokens in the prompts should be the same as the "
                        f"number of images passed. Found {n_images_in_text} tokens and {n_images_in_images} images."
                    )
            elif any(n_images_in_text):
                raise ValueError(f"Found {sum(n_images_in_text)} {self.image_token} tokens but no images were passed.")

    def replace_image_token(self, image_inputs: dict, image_idx: int, image_seq_len: int) -> str:
        image_rows = [row for row_list in image_inputs["rows"] for row in row_list][image_idx]
        image_cols = [col for col_list in image_inputs["cols"] for col in col_list][image_idx]
        if image_rows == 0 and image_cols == 0:
            return (
                f"{self.fake_image_token}"
                f"{self.global_image_tag}"
                f"{self.image_token * image_seq_len}"
                f"{self.fake_image_token}"
            )

        text_split_images = ""
        for row in range(image_rows):
            for col in range(image_cols):
                text_split_images += (
                    f"{self.fake_image_token}" f"<row_{row + 1}_col_{col + 1}>" f"{self.image_token * image_seq_len}"
                )
            text_split_images += "\n"
        text_split_images += (
            f"\n{self.fake_image_token}"
            f"{self.global_image_tag}"
            f"{self.image_token * image_seq_len}"
            f"{self.fake_image_token}"
        )
        return text_split_images

    def get_text_with_replacements(self, text, images_replacements):
        image_idx = 0
        replaced_text = []
        for sample in text:
            parts = sample.split(self.image_token)
            num_image_tokens = len(parts) - 1
            if image_idx + num_image_tokens > len(images_replacements):
                raise ValueError("The number of image tokens exceeds the available image replacements.")

            sample_with_replacements = parts[0]
            for part in parts[1:]:
                if image_idx >= len(images_replacements):
                    raise ValueError("The number of image tokens exceeds the available image replacements.")
                sample_with_replacements += images_replacements[image_idx] + part
                image_idx += 1
            replaced_text.append(sample_with_replacements)
        if image_idx != len(images_replacements):
            raise ValueError("The number of image replacements does not match the number of image tokens.")
        return replaced_text

    def create_mm_token_type_ids(self, input_ids):
        mm_token_type_ids = []
        for ids in input_ids:
            array_ids = np.array(ids)
            token_types = np.zeros_like(array_ids)
            token_types[array_ids == self.image_token_id] = 1
            mm_token_type_ids.append(token_types.tolist())
        return mm_token_type_ids

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = Idefics3ProcessorKwargs._defaults.get("images_kwargs", {}).copy()
            images_kwargs.update(kwargs)
            base_image_length = self.image_seq_len + 3
            col_length = self.image_seq_len + 2
            extra_split_newline = len(self.tokenizer("\n\n", add_special_tokens=False)["input_ids"]) - 1
            num_image_tokens = []
            num_image_patches = []
            for image_size in image_sizes:
                num_patches, num_rows, num_cols = self.image_processor.get_number_of_image_patches(
                    *image_size, images_kwargs
                )
                row_length = col_length * num_cols + 1
                split_extra = extra_split_newline if num_rows > 0 else 0
                num_image_tokens.append(base_image_length + split_extra + (row_length * num_rows))
                num_image_patches.append(num_patches)
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

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["Idefics3Processor"]
