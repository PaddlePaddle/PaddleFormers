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

"""Mistral3 model configuration."""

from ..configuration_utils import PretrainedConfig
from ..llama.configuration import LlamaConfig
from ..pixtral.configuration import PixtralVisionConfig


class Mistral3Config(PretrainedConfig):
    r"""Configuration for Mistral-Small-3.1 style multimodal models."""

    model_type = "mistral3"
    attribute_map = {"image_token_id": "image_token_index"}
    sub_configs = {"vision_config": PixtralVisionConfig, "text_config": LlamaConfig}
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=10,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
        multimodal_projector_bias=False,
        spatial_merge_size=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        self.multimodal_projector_bias = multimodal_projector_bias
        self.spatial_merge_size = spatial_merge_size

        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            vision_config.pop("model_type", None)
            self.vision_config = PixtralVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = PixtralVisionConfig(
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=1540,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                head_dim=64,
                hidden_act="gelu",
            )
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.pop("model_type", None)
            self.text_config = LlamaConfig(**text_config)
        elif text_config is None:
            self.text_config = LlamaConfig(
                attention_dropout=0.0,
                head_dim=128,
                hidden_act="silu",
                hidden_size=5120,
                initializer_range=0.02,
                intermediate_size=32768,
                max_position_embeddings=131072,
                num_attention_heads=32,
                num_hidden_layers=40,
                num_key_value_heads=8,
                rms_norm_eps=1e-5,
                rope_theta=1000000000.0,
                sliding_window=None,
                use_cache=True,
                vocab_size=131072,
            )
        else:
            self.text_config = text_config
