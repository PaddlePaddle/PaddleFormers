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
"""Processor class for Phi-4 Multimodal."""

import inspect
import json
import re

from ...utils.download import resolve_file_path
from ..feature_extraction_utils import FEATURE_EXTRACTOR_NAME, BatchFeature
from ..processing_utils import ProcessingKwargs, ProcessorMixin, Unpack

_RESOLVE_FILE_PATH_KWARGS = (
    "subfolder",
    "repo_type",
    "revision",
    "library_version",
    "cache_dir",
    "local_dir",
    "local_dir_use_symlinks",
    "user_agent",
    "force_download",
    "proxies",
    "etag_timeout",
    "resume_download",
    "token",
    "local_files_only",
    "endpoint",
    "download_hub",
)


class Phi4MultimodalProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {},
    }


class Phi4MultimodalProcessor(ProcessorMixin):
    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        from ..auto.tokenizer import AutoTokenizer
        from .feature_extraction import Phi4MultimodalFeatureExtractor
        from .image_processor import Phi4MultimodalImageProcessor

        preprocessor_config = {}
        resolve_file_path_kwargs = {key: kwargs[key] for key in _RESOLVE_FILE_PATH_KWARGS if key in kwargs}
        resolve_file_path_kwargs["force_return"] = True
        preprocessor_config_path = resolve_file_path(
            pretrained_model_name_or_path,
            FEATURE_EXTRACTOR_NAME,
            **resolve_file_path_kwargs,
        )
        if preprocessor_config_path is not None:
            with open(preprocessor_config_path, encoding="utf-8") as f:
                preprocessor_config = json.load(f)

        def _filter_config(processor_cls):
            valid_keys = {
                key
                for key, value in inspect.signature(processor_cls.__init__).parameters.items()
                if key != "self"
                and value.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            return {key: value for key, value in preprocessor_config.items() if key in valid_keys}

        image_processor = Phi4MultimodalImageProcessor(**_filter_config(Phi4MultimodalImageProcessor))
        feature_extractor = Phi4MultimodalFeatureExtractor(**_filter_config(Phi4MultimodalFeatureExtractor))
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return [image_processor, feature_extractor, tokenizer]

    def __init__(
        self,
        image_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_processor=None,
        **kwargs,
    ):
        if feature_extractor is None:
            feature_extractor = audio_processor
        self.image_token = getattr(tokenizer, "image_token", "<|endoftext10|>")
        self.audio_token = getattr(tokenizer, "audio_token", "<|endoftext11|>")
        self.image_token_id = getattr(tokenizer, "image_token_id", None)
        if self.image_token_id is None and tokenizer is not None:
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.audio_token_id = getattr(tokenizer, "audio_token_id", None)
        if self.audio_token_id is None and tokenizer is not None:
            self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        super().__init__(image_processor, feature_extractor, tokenizer, chat_template=chat_template, **kwargs)
        self.audio_processor = self.feature_extractor

    def __call__(self, text, images=None, audio=None, **kwargs: Unpack[Phi4MultimodalProcessorKwargs]) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Phi4MultimodalProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"]) if images is not None else {}
        audio_inputs = self.audio_processor(audio, **output_kwargs["audio_kwargs"]) if audio is not None else {}

        num_img_tokens = image_inputs.pop("num_img_tokens", [])
        audio_embed_sizes = audio_inputs.get("audio_embed_sizes", [])
        if hasattr(audio_embed_sizes, "numpy"):
            audio_embed_sizes = audio_embed_sizes.numpy().tolist()
        elif not isinstance(audio_embed_sizes, list):
            audio_embed_sizes = list(audio_embed_sizes)

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) or (len(text) > 0 and not isinstance(text[0], str)):
            raise TypeError("Invalid input text. Please provide a string or a list of strings.")

        concatenated_prompt = "".join(text)
        if concatenated_prompt.count(self.image_token) != len(num_img_tokens):
            raise ValueError(
                f"You should add as many image tokens `{self.image_token}` in your prompt as images passed to the processor. "
                f"Input contains {concatenated_prompt.count(self.image_token)} tokens != {len(num_img_tokens)} images."
            )
        if concatenated_prompt.count(self.audio_token) != len(audio_embed_sizes):
            raise ValueError(
                f"You should add as many audio tokens `{self.audio_token}` in your prompt as audios passed to the processor. "
                f"Input contains {concatenated_prompt.count(self.audio_token)} tokens != {len(audio_embed_sizes)} audios."
            )

        image_count_iter = iter(num_img_tokens)
        audio_count_iter = iter(audio_embed_sizes)
        processed_text = [
            re.sub(re.escape(self.image_token), lambda _: self.image_token * int(next(image_count_iter)), sample)
            for sample in text
        ]
        processed_text = [
            re.sub(re.escape(self.audio_token), lambda _: self.audio_token * int(next(audio_count_iter)), sample)
            for sample in processed_text
        ]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(processed_text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(processed_text, text_inputs, modalities=["image", "audio"])

        if images is not None and audio is not None:
            input_mode = 3
        elif images is not None:
            input_mode = 1
        elif audio is not None:
            input_mode = 2
        else:
            input_mode = 0

        return BatchFeature(
            data={**text_inputs, **image_inputs, **audio_inputs, "input_mode": [input_mode]},
            tensor_type=return_tensors,
        )


__all__ = ["Phi4MultimodalProcessor"]
