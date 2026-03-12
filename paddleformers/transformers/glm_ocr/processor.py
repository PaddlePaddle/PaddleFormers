# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional, Union

import numpy as np
import paddle

from ..image_processing_utils import BatchFeature
from ..image_utils import ImageInput
from ..processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils_base import PreTokenizedInput, TextInput


class Glm46VProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


class Glm46VProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<|image|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def apply_chat_template(
        self,
        conversation,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        """
        PaddleFormers Glm46VProcessor fallback apply_chat_template.

        This method intentionally bypasses ProcessorMixin.apply_chat_template to avoid
        processing_utils kwargs-grouping issues (mm_load_kwargs/template_kwargs).
        It formats messages using tokenizer.apply_chat_template (text-only),
        then calls __call__(images=..., text=...) to build multimodal tensors.
        """
        tokenize = kwargs.get("tokenize", False)
        add_generation_prompt = kwargs.get("add_generation_prompt", False)
        return_dict = kwargs.get("return_dict", False)
        return_tensors = kwargs.get("return_tensors", None)

        # If user only wants the rendered string, we can return it directly.
        # If tokenize/return_dict requested, we will build full BatchFeature via __call__.

        def _load_image_from_url_or_path(u: str):
            # Best-effort local path support (HF examples often pass local path via "url")
            try:
                from PIL import Image

                return Image.open(u).convert("RGB")
            except Exception:
                # If PIL not available or file not found, just raise with context
                raise ValueError(f"Failed to load image from path/url: {u}")

        def _flatten_conv_and_collect_images(conv_one):
            """
            conv_one: list[dict] with {"role":..., "content":...}
            Returns: (flat_conv_for_tokenizer, images_list)
            flat_conv_for_tokenizer uses content as a STRING with image tokens inserted.
            """
            images = []
            flat_conv = []

            for msg in conv_one:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # content can be str or list of {"type": "...", ...}
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        t = item.get("type", None)
                        if t == "image":
                            # Paddle tests sometimes pass {"type":"image","image": PIL.Image}
                            if "image" in item and item["image"] is not None:
                                images.append(item["image"])
                            elif "url" in item and item["url"] is not None:
                                images.append(_load_image_from_url_or_path(item["url"]))
                            else:
                                raise ValueError("Image item must contain either 'image' or 'url'.")
                            parts.append(self.image_token)
                        elif t == "text":
                            parts.append(item.get("text", ""))
                        else:
                            # Unknown part: try to treat as text if possible
                            if "text" in item:
                                parts.append(item["text"])
                    content_str = "".join(parts)
                else:
                    content_str = str(content)

                flat_conv.append({"role": role, "content": content_str})

            return flat_conv, images

        # Normalize to batch or single
        is_batched = isinstance(conversation, list) and len(conversation) > 0 and isinstance(conversation[0], list)

        if not is_batched:
            conv_list = [conversation]
        else:
            conv_list = conversation

        rendered_texts = []
        all_images = []

        for conv_one in conv_list:
            flat_conv, images = _flatten_conv_and_collect_images(conv_one)
            all_images.extend(images)

            # Render text via tokenizer (NOT ProcessorMixin) to avoid mm_load_kwargs bug
            if hasattr(self.tokenizer, "apply_chat_template"):
                rendered = self.tokenizer.apply_chat_template(
                    flat_conv,
                    chat_template=chat_template,
                    tokenize=False,  # render only; tokenization handled in __call__
                    add_generation_prompt=add_generation_prompt,
                )
            else:
                # Minimal fallback if tokenizer lacks apply_chat_template
                # (keeps basic semantics; may be less faithful than real template)
                rendered = ""
                for m in flat_conv:
                    rendered += f"{m['role']}: {m['content']}\n"
                if add_generation_prompt:
                    rendered += "assistant: "
            rendered_texts.append(rendered)

        # If user only wants the rendered string (HF-style)
        if not tokenize and not return_dict:
            return rendered_texts[0] if not is_batched else rendered_texts

        # Build multimodal tensors via existing __call__
        # We pass return_tensors through text_kwargs (your __call__ pops it from text_kwargs)
        # Also preserve padding option if provided.
        padding = kwargs.get("padding", False)

        # __call__ expects either a single text or list; we already have list
        features = self(
            images=all_images if len(all_images) > 0 else None,
            text=rendered_texts if is_batched or len(rendered_texts) > 1 else rendered_texts[0],
            padding=padding,
            return_tensors=return_tensors,
        )

        # ProcessorMixin.apply_chat_template in HF returns dict-like when return_dict=True.
        # BatchFeature is dict-like enough for tests; keep it.
        if return_dict:
            return features

        # If return_dict=False but tokenize=True, try to mimic returning tokenized ids.
        # We'll return input_ids (and pixel_values exist in features if images were provided).
        return features["input_ids"]

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs: Unpack[Glm46VProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Glm46VProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens.item(), 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = Glm46VProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        if isinstance(generated_outputs, paddle.Tensor):
            generated_outputs = generated_outputs.numpy()
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["Glm46VProcessor"]
