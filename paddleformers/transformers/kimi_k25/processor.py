# coding=utf-8
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
"""processor class for Kimi-K2.5."""

from ..image_processing_utils import BatchFeature
from ..processing_utils import ProcessorMixin


class KimiK25Processor(ProcessorMixin):
    r"""
    Constructs a KimiK25 processor which wraps a KimiK25 image processor and a tokenizer into a single processor.
    [`KimiK25Processor`] offers all the functionalities of [`KimiK25ImageProcessor`] and [`TikTokenTokenizer`]. See the
    [`~KimiK25Processor.__call__`] and [`~KimiK25Processor.decode`] for more information.
    Args:
        image_processor ([`KimiK25ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`TikTokenTokenizer`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_processor = self.media_processor = image_processor
        # A special temporal placeholder to be replaced by actual video placeholders
        self.video_placeholder = "<|kimi_k25_video_placeholder|>"

    def update_raw_text(self, text: str, video_prompts: list[str]) -> str:
        # replace video prompt in text with video chunk prompts
        video_count = text.count(self.video_placeholder)
        if video_count == 0:
            return text
        assert video_count == len(video_prompts)
        text_parts = text.split(self.video_placeholder)
        assert len(text_parts) == len(video_prompts) + 1
        text = "".join([text_parts[i] + video_prompts[i] for i in range(len(video_prompts))])
        text += text_parts[-1]
        return text

    def preprocess_medias(self, medias: list[dict], **kwargs) -> list[dict]:
        updated_medias = []
        video_prompts = []
        for media in medias:
            if media["type"] == "image":
                updated_medias.append(media)
            elif media["type"] == "video":
                video_chunks = self.media_processor.split_video_chunks(media["video"], **kwargs)
                updated_medias.extend(video_chunks)
                video_prompts.append("".join([vc["prompt"] for vc in video_chunks]))
            else:
                raise ValueError(f"unsupported media type: {media['type']}")
        return updated_medias, video_prompts

    def __call__(
        self,
        messages: list[dict] = None,
        medias: list[dict] = None,
        text: str = None,
        return_tensors: str = "pd",
        **kwargs
    ) -> BatchFeature:
        """
        Process multimodal inputs for Kimi-K2.5 model.
        This processor accepts ordered messages and extracts both media and text in a single pass.
        text will be automatically updated if video input detected in messages
        Args:
            messages: List of message dicts with 'role' and 'content' fields.
                     If provided, medias and text will be extracted automatically.
            medias: Pre-extracted list of media dicts. If None, extracted from messages.
            text: Pre-formatted text string. If None, generated via apply_chat_template.
            return_tensors: Format of returned tensors ('pt', 'np', 'tf'). Default: 'pt'.
            **kwargs: Additional arguments passed to tokenizer.apply_chat_template.
        Returns:
            BatchFeature with fields: input_ids, attention_mask, pixel_values, grid_thws.
        """
        if messages is None and (medias is None or text is None):
            raise ValueError("Provide either 'messages' or both 'medias' and 'text'")

        if medias is not None and text is not None:
            updated_medias, video_prompts = self.preprocess_medias(medias, **kwargs)
            preprocessed = self.media_processor.preprocess(updated_medias, return_tensors=return_tensors)
            text = self.update_raw_text(text, video_prompts)
            text_inputs = self.tokenizer(text, add_special_tokens=False, return_tensors=return_tensors, **kwargs)
            return BatchFeature(data={**text_inputs, **preprocessed.data})

        if medias is None:
            medias = self._extract_medias_from_messages(messages)
        updated_medias, video_prompts = self.preprocess_medias(medias, **kwargs)
        preprocessed = self.media_processor.preprocess(updated_medias, return_tensors=return_tensors)

        # Generate text if not provided
        if text is None:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)

        text = self.update_raw_text(text, video_prompts)

        text_inputs = self.tokenizer(text, add_special_tokens=False, return_tensors=return_tensors, **kwargs)
        return BatchFeature(data={**text_inputs, **preprocessed.data})

    @staticmethod
    def _extract_medias_from_messages(messages: list[dict]) -> list[dict]:
        """
        Extract media items from messages in a single pass.

        This is an optimized version that processes messages only once.
        Kept as internal method since external callers should use __call__.
        """
        medias = []
        for msg in messages:
            if msg["role"] != "user" or not msg.get("content"):
                continue

            for content_part in msg["content"]:
                if not isinstance(content_part, dict):
                    continue

                content_type = content_part.get("type")
                if content_type in ["video_url", "video"]:
                    medias.append(
                        {"type": "video", "video": content_part["video_url"]["url"], "first_frame_timestamp": 0.0}
                    )
                elif content_type in ["image_url", "image"]:
                    medias.append(
                        {
                            "type": "image",
                            "image": content_part["image_url"],
                        }
                    )
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
