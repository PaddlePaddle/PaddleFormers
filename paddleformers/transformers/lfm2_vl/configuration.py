# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""LFM2-VL configuration."""

from typing import Optional, Union

from ..configuration_utils import PretrainedConfig
from ..lfm2.configuration import Lfm2Config


class Siglip2VisionConfig(PretrainedConfig):
    model_type = "siglip2_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_patches=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_patches = num_patches


class Lfm2VlConfig(PretrainedConfig):
    model_type = "lfm2_vl"
    sub_configs = {"text_config": Lfm2Config, "vision_config": Siglip2VisionConfig}

    def __init__(
        self,
        vision_config: Optional[Union[dict, Siglip2VisionConfig]] = None,
        text_config: Optional[Union[dict, Lfm2Config]] = None,
        image_token_id=396,
        projector_hidden_act="gelu",
        projector_hidden_size=2560,
        projector_bias=True,
        projector_use_layernorm=True,
        downsample_factor=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = Siglip2VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Siglip2VisionConfig(**vision_config)
        if text_config is None:
            text_config = Lfm2Config()
        elif isinstance(text_config, dict):
            text_config = Lfm2Config(**text_config)
        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_id = image_token_id
        self.image_token_index = image_token_id
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        self.projector_bias = projector_bias
        self.projector_use_layernorm = projector_use_layernorm
        self.downsample_factor = downsample_factor
        super().__init__(tie_word_embeddings=kwargs.pop("tie_embedding", tie_word_embeddings), **kwargs)


__all__ = ["Lfm2Config", "Lfm2VlConfig", "Siglip2VisionConfig"]
