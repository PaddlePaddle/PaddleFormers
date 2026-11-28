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

import os
from typing import Union

from ..configuration_utils import PretrainedConfig
from ..utils import logger


class InternVisionConfig(PretrainedConfig):
    model_type = "intern_vit_6b"

    def __init__(
        self,
        num_channels=3,
        patch_size=14,
        image_size=224,
        qkv_bias=False,
        hidden_size=3200,
        num_attention_heads=25,
        intermediate_size=12800,
        qk_normalization=True,
        num_hidden_layers=48,
        use_flash_attn=True,
        hidden_act="gelu",
        norm_type="rms_norm",
        layer_norm_eps=1e-6,
        dropout=0.0,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=0.1,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.use_flash_attn = use_flash_attn
        self.recompute_granularity = recompute_granularity
        self.recompute_method = recompute_method
        self.recompute_num_layers = recompute_num_layers

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


__all__ = ["InternVisionConfig"]
