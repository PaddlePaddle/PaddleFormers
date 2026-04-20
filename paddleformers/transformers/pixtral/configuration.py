# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class PixtralVisionConfig(PretrainedConfig):
    model_type = "pixtral"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        attention_dropout=0.0,
        initializer_range=0.02,
        rope_theta=10000.0,
        rope_scaling=None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_scaling
        self.head_dim = self.hidden_size // self.num_attention_heads

        standardize_rope_params(self, rope_theta=self.rope_theta)
        rope_config_validation(self)

        super().__init__(**kwargs)


__all__ = ["PixtralVisionConfig"]
