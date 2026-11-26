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

"""LLaVA-style preprocessing for the native FastVLM checkpoint format."""

from ..fastvlm.image_processor import FastVLMImageProcessor
from ..feature_extraction_utils import BatchFeature
from ..processing_utils import ProcessorMixin


class FastVlmImageProcessor(FastVLMImageProcessor):
    def __init__(self, **kwargs):
        # Native HF FastVLM uses CLIP resize/center-crop, not remote-code square padding.
        kwargs.setdefault("image_aspect_ratio", None)
        super().__init__(**kwargs)


class FastVlmProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FastVlmImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        patch_size=64,
        image_token="<image>",
        vision_feature_select_strategy="full",
        num_additional_image_tokens=0,
        chat_template=None,
        **kwargs
    ):
        if vision_feature_select_strategy != "full" or num_additional_image_tokens != 0:
            raise ValueError("FastVLM uses all spatial tokens and no additional CLS token")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.num_additional_image_tokens = num_additional_image_tokens
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        if text is None:
            raise ValueError("FastVLM requires text containing the image placeholders")
        texts = [text] if isinstance(text, str) else list(text)
        image_inputs = (
            self.image_processor(images, return_tensors="np", **kwargs.pop("images_kwargs", {}))
            if images is not None
            else {}
        )
        count = sum(sample.count(self.image_token) for sample in texts)
        if images is None and count:
            raise ValueError("Image placeholders require images")
        expected_tokens = 0
        if images is not None:
            pixels = image_inputs["pixel_values"]
            if count != len(pixels):
                raise ValueError("Supply one image for each image placeholder")
            height, width = pixels.shape[-2:]
            tokens_per_image = (height // self.patch_size) * (width // self.patch_size)
            if tokens_per_image == 0:
                raise ValueError("Image dimensions are smaller than the effective vision patch size")
            expected_tokens = count * tokens_per_image
            texts = [sample.replace(self.image_token, self.image_token * tokens_per_image) for sample in texts]
        text_kwargs = kwargs.pop("text_kwargs", {})
        text_kwargs.update(kwargs)
        text_inputs = self.tokenizer(texts, return_tensors=None, **text_kwargs)
        actual_tokens = sum(row.count(self.image_token_id) for row in text_inputs["input_ids"])
        if actual_tokens != expected_tokens:
            raise ValueError("Tokenization or truncation changed the number of image tokens")
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))


__all__ = ["FastVlmProcessor", "FastVlmImageProcessor"]
