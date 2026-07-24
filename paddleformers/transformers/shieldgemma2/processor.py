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

from collections.abc import Mapping, Sequence

from ..feature_extraction_utils import BatchFeature
from ..gemma3.processor import Gemma3Processor, Gemma3ProcessorKwargs
from ..image_utils import ImageInput
from ..processing_utils import Unpack


DEFAULT_SHIELDGEMMA2_POLICIES: Mapping[str, str] = {
    "dangerous": (
        "No Dangerous Content: The image shall not contain content that"
        " facilitates or encourages activities that could cause real-world harm"
        " (e.g., building firearms and explosive devices, promotion of"
        " terrorism, instructions for suicide)."
    ),
    "sexual": (
        "No Sexually Explicit content: The image shall not contain content that"
        " depicts explicit or graphic sexual acts (e.g., pornography, erotic"
        " nudity, depictions of rape or sexual assault)."
    ),
    "violence": (
        "No Violence/Gore content: The image shall not contain content that"
        " depicts shocking, sensational, or gratuitous violence (e.g.,"
        " excessive blood and gore, gratuitous violence against animals,"
        " extreme injury or moment of death)."
    ),
}


class ShieldGemma2ProcessorKwargs(Gemma3ProcessorKwargs, total=False):
    policies: Sequence[str] | None
    custom_policies: Mapping[str, str] | None
    _defaults = {
        "text_kwargs": {"padding": True},
        "images_kwargs": {"do_pan_and_scan": False},
    }


class ShieldGemma2Processor(Gemma3Processor):
    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        image_seq_length=256,
        policy_definitions=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template, image_seq_length, **kwargs)
        self.policy_definitions = (
            DEFAULT_SHIELDGEMMA2_POLICIES if policy_definitions is None else policy_definitions
        )

    def __call__(
        self,
        images: ImageInput | None = None,
        text=None,
        **kwargs: Unpack[ShieldGemma2ProcessorKwargs],
    ) -> BatchFeature:
        if not images:
            raise ValueError("ShieldGemma 2 needs images to classify")
        if not isinstance(images, Sequence):
            images = [images]

        if not self.chat_template:
            raise ValueError("ShieldGemma 2 requires the use of a specific chat template")

        common_kwargs = kwargs.setdefault("common_kwargs", {})
        if "return_tensors" in kwargs:
            common_kwargs["return_tensors"] = kwargs.pop("return_tensors")

        images_kwargs = kwargs.setdefault("images_kwargs", {})
        if images_kwargs.get("do_pan_and_scan") is True:
            images_kwargs["do_pan_and_scan"] = False

        text_kwargs = kwargs.setdefault("text_kwargs", {})
        if "padding" not in text_kwargs:
            text_kwargs["padding"] = kwargs.pop("padding", True)
            text_kwargs["padding_side"] = kwargs.pop("padding_side", "left")

        policy_definitions: Mapping[str, str] = {
            **self.policy_definitions,
            **(kwargs.get("custom_policies") or {}),
        }
        policies = kwargs.get("policies")
        if policies is None:
            policies = list(policy_definitions.keys())

        messages = []
        expanded_images = []
        for image in images:
            if not isinstance(image, list):
                image = [image]
            elif len(image) > 1:
                raise ValueError(
                    f"ShieldGemma2 can process at most one image per sample, but got {len(image)} images"
                )

            for policy in policies:
                if policy not in policy_definitions:
                    raise ValueError(f"Unknown ShieldGemma2 policy: {policy}")
                content = []
                if image:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": policy_definitions[policy]})
                messages.append([{"role": "user", "content": content}])
                expanded_images.append(image)

        text = self.apply_chat_template(messages, tokenize=False)
        return super().__call__(images=expanded_images, text=text, **kwargs)


__all__ = [
    "DEFAULT_SHIELDGEMMA2_POLICIES",
    "ShieldGemma2Processor",
    "ShieldGemma2ProcessorKwargs",
]
