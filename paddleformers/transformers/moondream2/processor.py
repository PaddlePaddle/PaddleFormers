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

"""Processor helpers for Moondream2."""

from typing import List, Optional, Union

from ..feature_extraction_utils import BatchFeature
from ..image_utils import ImageInput
from ..processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils_base import PreTokenizedInput, TextInput

__all__ = ["Moondream2Processor"]


class Moondream2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}, "images_kwargs": {}}


class Moondream2Processor(ProcessorMixin):
    """Combine Moondream2's image crop processor with its CodeGen tokenizer.

    The task helpers return the exact numeric prompt templates published by the
    reference checkpoint.  They are useful because these control tokens are not
    intended to be reconstructed from text strings.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = [
        "crop_size",
        "patch_size",
        "max_crops",
        "overlap_margin",
        "image_mean",
        "image_std",
        "image_processor_type",
    ]

    TASK_TOKENS = {
        "caption": {"short": [1, 32708, 2, 12492, 3], "normal": [1, 32708, 2, 6382, 3], "long": [1, 32708, 2, 4059, 3]},
        "query": {"prefix": [1, 15381, 2], "suffix": [3]},
        "detect": {"prefix": [1, 7235, 476, 2], "suffix": [3]},
        "point": {"prefix": [1, 2581, 2], "suffix": [3]},
    }

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @classmethod
    def build_task_prompt(cls, task, text=None, length="normal"):
        if task == "caption":
            if length not in cls.TASK_TOKENS[task]:
                raise ValueError(f"caption length must be one of {tuple(cls.TASK_TOKENS[task])}")
            return list(cls.TASK_TOKENS[task][length])
        if task not in cls.TASK_TOKENS:
            raise ValueError(f"unsupported Moondream2 task: {task}")
        if text is None:
            raise ValueError(f"text is required for the {task} task")
        return list(cls.TASK_TOKENS[task]["prefix"]), str(text), list(cls.TASK_TOKENS[task]["suffix"])

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        task: Optional[str] = None,
        task_length="normal",
        **kwargs: Unpack[Moondream2ProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Moondream2ProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"]) if images is not None else {}
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        if task is not None:
            if task == "caption":
                prompt_count = len(image_inputs["pixel_values"]) if images is not None else 1
                prompts = [None] * prompt_count
            elif isinstance(text, str):
                prompts = [text]
            elif isinstance(text, list) and text and all(isinstance(item, str) for item in text):
                prompts = text
            else:
                raise ValueError("task text must be a string or a non-empty list of strings")
            if images is not None and len(prompts) != len(image_inputs["pixel_values"]):
                raise ValueError("task prompts and images must have the same batch size")
            input_ids = []
            for prompt_text in prompts:
                prompt = self.build_task_prompt(task, text=prompt_text, length=task_length)
                if isinstance(prompt, tuple):
                    prefix, question, suffix = prompt
                    tokenized_question = self.tokenizer(question, add_special_tokens=False)["input_ids"]
                    if tokenized_question and isinstance(tokenized_question[0], list):
                        tokenized_question = tokenized_question[0]
                    input_ids.append(prefix + tokenized_question + suffix)
                else:
                    input_ids.append(prompt)
            text_inputs = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=output_kwargs["text_kwargs"].get("padding", False),
                max_length=output_kwargs["text_kwargs"].get("max_length"),
                pad_to_multiple_of=output_kwargs["text_kwargs"].get("pad_to_multiple_of"),
                return_attention_mask=True,
                return_tensors=None,
            )
        elif text is not None:
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        else:
            text_inputs = {}
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def post_process_image_text_to_text(self, generated_outputs, **kwargs):
        return self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True, **kwargs)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))
