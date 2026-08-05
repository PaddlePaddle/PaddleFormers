# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The Moonshot AI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""Processor class for Kimi-K3: wraps the vision processor and the tokenizer."""

from ..image_processing_utils import BatchFeature
from ..processing_utils import ProcessorMixin
from .media_utils import ensure_media_type

__all__ = ["KimiK3Processor"]


class KimiK3Processor(ProcessorMixin):
    r"""
    Constructs a Kimi-K3 processor which wraps a [`KimiK3VisionProcessor`] and a
    [`KimiK3TikTokenTokenizer`] into a single processor.

    Kimi-K3 emits exactly one `<|media_pad|>` per image; the placeholder is expanded
    inside the model, so this processor keeps the text length independent of the
    number of vision tokens.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_processor = self.media_processor = image_processor
        self.image_placeholder = "<|kimi_image_placeholder|>"

    def update_raw_text(self, text: str, image_prompts: list) -> str:
        image_count = text.count(self.image_placeholder)
        if image_count == 0:
            return text
        assert image_count == len(
            image_prompts
        ), f"image placeholder count {image_count} != image_prompts count {len(image_prompts)}"
        text_parts = text.split(self.image_placeholder)
        text = "".join([text_parts[i] + image_prompts[i] for i in range(len(image_prompts))])
        return text + text_parts[-1]

    def preprocess_medias(self, medias: list, **kwargs):
        updated_medias = []
        image_prompts = []
        for media in medias:
            if media["type"] != "image":
                raise ValueError(f"unsupported media type: {media['type']}")
            updated_medias.append(media)
            img = ensure_media_type(
                media,
                transparent_bg_config=self.media_processor._transparent_bg_config,
                transparent_bg_fill_stage=self.media_processor._transparent_bg_fill_stage,
            )["image"]
            width, height = img.size
            image_prompts.append(self.media_processor.make_image_prompt(width, height))
        return updated_medias, image_prompts

    def __call__(
        self,
        messages: list = None,
        medias: list = None,
        text: str = None,
        return_tensors: str = "pd",
        **kwargs,
    ) -> BatchFeature:
        """Process multimodal inputs into `input_ids`/`attention_mask`/`pixel_values`/`grid_thws`."""
        if messages is None and (medias is None or text is None):
            raise ValueError("Provide either 'messages' or both 'medias' and 'text'")

        if medias is None:
            medias = self._extract_medias_from_messages(messages)
        updated_medias, image_prompts = self.preprocess_medias(medias, **kwargs)
        preprocessed = self.media_processor.preprocess(updated_medias, return_tensors=return_tensors)

        if text is None:
            text = self.tokenizer.apply_chat_template(messages, image_prompts=image_prompts, **kwargs)
        else:
            text = self.update_raw_text(text, image_prompts)

        text_inputs = self.tokenizer(text, add_special_tokens=False, return_tensors=return_tensors)
        return BatchFeature(data={**text_inputs, **preprocessed.data})

    @staticmethod
    def _extract_medias_from_messages(messages: list) -> list:
        medias = []
        for msg in messages:
            if msg["role"] != "user" or not msg.get("content"):
                continue
            for content_part in msg["content"]:
                if not isinstance(content_part, dict):
                    continue
                content_type = content_part.get("type")
                if content_type in ("image_url", "image"):
                    image_data = content_part.get(content_type)
                    assert image_data is not None, f"image data is missing for content part: {content_part}"
                    medias.append({"type": "image", "image": image_data})
        return medias

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values", "grid_thws"]
