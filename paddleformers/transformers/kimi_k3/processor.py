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

from ..processing_utils import ProcessorMixin

__all__ = ["KimiK3Processor"]


class KimiK3Processor(ProcessorMixin):
    r"""
    Constructs a Kimi-K3 processor which wraps a [`KimiK3VisionProcessor`] and a
    [`KimiK3TikTokenTokenizer`] into a single processor.

    Training reaches the vision processor through ``self.image_processor`` (see
    ``KimiK3Plugin`` in ``paddleformers/datasets/template/mm_plugin.py``) and
    renders text with the registered ``kimi_k3`` template, so chat rendering and
    generation are not implemented here.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
