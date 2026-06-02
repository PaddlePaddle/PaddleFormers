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

"""Pixtral Vision model configuration"""

from ..configuration_utils import PretrainedConfig


class PixtralVisionConfig(PretrainedConfig):
    """Configuration class for PixtralVisionModel.

    Args:
        hidden_size (`int`, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, defaults to 16):
            Number of attention heads for each attention layer.
        num_channels (`int`, defaults to 3):
            Number of input image channels.
        image_size (`int`, defaults to 1024):
            The size (resolution) of each image.
        patch_size (`int`, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str`, defaults to "gelu"):
            The non-linear activation function in the encoder.
        attention_dropout (`float`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_theta (`float`, defaults to 10000.0):
            The base period of the RoPE embeddings.
        initializer_range (`float`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

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
        rope_theta=10000.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Build rope_parameters dict for compatibility
        if not hasattr(self, "rope_parameters") or self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
            }


__all__ = ["PixtralVisionConfig"]
