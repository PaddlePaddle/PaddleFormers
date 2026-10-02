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

"""FastVLM configuration."""

from ..qwen2.configuration import Qwen2Config


class FastVLMConfig(Qwen2Config):
    """Configuration for Apple FastVLM checkpoints based on LLaVA-Qwen2."""

    model_type = "llava_qwen2"

    def __init__(
        self,
        mm_vision_tower="mobileclip_l_1024",
        mm_hidden_size=3072,
        mm_projector_type="mlp2x_gelu",
        mm_patch_merge_type="flat",
        image_aspect_ratio="pad",
        image_token_index=-200,
        unfreeze_mm_vision_tower=True,
        tokenizer_model_max_length=8192,
        tokenizer_padding_side="right",
        vision_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mm_vision_tower = mm_vision_tower
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_type = mm_projector_type
        self.mm_patch_merge_type = mm_patch_merge_type
        self.image_aspect_ratio = image_aspect_ratio
        self.image_token_index = image_token_index
        self.unfreeze_mm_vision_tower = unfreeze_mm_vision_tower
        self.tokenizer_model_max_length = tokenizer_model_max_length
        self.tokenizer_padding_side = tokenizer_padding_side
        self.vision_config = vision_config


__all__ = ["FastVLMConfig"]
