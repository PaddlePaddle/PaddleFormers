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

from ..cohere2.configuration import Cohere2Config
from ..configuration_utils import PretrainedConfig
from ..siglip_vision_model.configuration import SiglipVisionConfig


class AyaVisionConfig(PretrainedConfig):
    model_type = "aya_vision"
    is_composition = True
    sub_configs = {"vision_config": SiglipVisionConfig, "text_config": Cohere2Config}
    attribute_map = {"image_token_id": "image_token_index"}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        downsample_factor=2,
        adapter_layer_norm_eps=1e-6,
        image_token_index=255036,
        alignment_intermediate_size=None,
        alignment_activation_fn="swiglu",
        projector_hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = SiglipVisionConfig(
                hidden_size=1152,
                intermediate_size=4304,
                patch_size=14,
                image_size=364,
                num_hidden_layers=26,
                num_attention_heads=14,
                vision_use_head=False,
            )

        if isinstance(text_config, dict):
            text_config = Cohere2Config(**text_config)
        elif text_config is None:
            text_config = Cohere2Config()

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'. "
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_config = vision_config
        self.text_config = text_config
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.downsample_factor = downsample_factor
        self.adapter_layer_norm_eps = adapter_layer_norm_eps
        self.image_token_index = image_token_index
        self.alignment_intermediate_size = alignment_intermediate_size or text_config.hidden_size
        self.alignment_activation_fn = alignment_activation_fn
        self.projector_hidden_act = projector_hidden_act

        # Sync _attn_implementation to sub-configs (align with Qwen3-VL pattern)
        if "_attn_implementation" in kwargs:
            self.vision_config._attn_implementation = kwargs["_attn_implementation"]
            self.text_config._attn_implementation = kwargs["_attn_implementation"]


__all__ = ["AyaVisionConfig"]
