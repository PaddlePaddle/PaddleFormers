# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Llava-NeXT model configuration."""

from ..auto.configuration import CONFIG_MAPPING
from ..configuration_utils import PretrainedConfig
from ..siglip_vision_model.configuration import SiglipVisionConfig


class LlavaNextConfig(PretrainedConfig):
    model_type = "llava_next"
    attribute_map = {"image_token_id": "image_token_index", "num_classes": "num_labels"}
    sub_configs = {"vision_config": PretrainedConfig, "text_config": PretrainedConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        multimodal_projector_bias=True,
        tie_word_embeddings=False,
        image_grid_pinpoints=None,
        image_seq_length=576,
        **kwargs,
    ):
        dtype = kwargs.get("dtype", None)
        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            if dtype is not None:
                vision_config["dtype"] = dtype
                vision_config["torch_dtype"] = dtype
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            model_type = vision_config["model_type"]
            if model_type == "siglip_vision_model":
                vision_config = SiglipVisionConfig(**vision_config)
            else:
                vision_config = CONFIG_MAPPING[model_type](**vision_config)
        elif vision_config is None:
            vision_config = SiglipVisionConfig(image_size=336, patch_size=14, vision_use_head=False)
            if dtype is not None:
                vision_config.dtype = dtype

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            if dtype is not None:
                text_config["dtype"] = dtype
                text_config["torch_dtype"] = dtype
            text_config["model_type"] = text_config.get("model_type", "llama")
            model_type = text_config["model_type"]
            text_config = CONFIG_MAPPING[model_type](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()
            if dtype is not None:
                text_config.dtype = dtype

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'. "
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.multimodal_projector_bias = multimodal_projector_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.image_grid_pinpoints = image_grid_pinpoints or [
            [336, 672],
            [672, 336],
            [672, 672],
            [1008, 336],
            [336, 1008],
        ]
        self.image_seq_length = image_seq_length
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def image_token_id(self):
        return self.image_token_index

    def __setattr__(self, key, value):
        config_dict = super().__getattribute__("__dict__")
        if key == "dtype":
            for sub_config_key in self.sub_configs:
                sub_config = config_dict.get(sub_config_key)
                if sub_config is not None:
                    setattr(sub_config, "dtype", value)
            super().__setattr__(key, value)
        elif (
            (text_config := config_dict.get("text_config")) is not None
            and key
            not in [
                "_name_or_path",
                "model_type",
                "dtype",
                "_attn_implementation_internal",
                "id2label",
                "label2id",
                "num_labels",
            ]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key not in [
            "_name_or_path",
            "model_type",
            "dtype",
            "_attn_implementation_internal",
            "id2label",
            "label2id",
            "num_labels",
        ]:
            config_dict = super().__getattribute__("__dict__")
            if "text_config" in config_dict and key in config_dict["text_config"].__dict__:
                return getattr(config_dict["text_config"], key)
        return super().__getattribute__(key)


__all__ = ["LlavaNextConfig"]
