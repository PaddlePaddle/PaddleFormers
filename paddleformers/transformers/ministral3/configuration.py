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

from ..configuration_utils import PretrainedConfig

__all__ = ["Mistral3Config", "Ministral3TextConfig"]


class Ministral3TextConfig:
    """
    Simple wrapper around the text_config dict for Ministral3 language model.
    Provides attribute-style access to text config keys, compatible with RoPE utilities.

    original source: transformers.models.ministral3.configuration_ministral3.Ministral3Config
    """

    def __init__(self, cfg_dict: dict):
        self.attention_dropout = cfg_dict.get("attention_dropout", 0.0)
        self.head_dim = cfg_dict.get("head_dim", 128)
        self.hidden_act = cfg_dict.get("hidden_act", "silu")
        self.hidden_size = cfg_dict.get("hidden_size", 4096)
        self.initializer_range = cfg_dict.get("initializer_range", 0.02)
        self.intermediate_size = cfg_dict.get("intermediate_size", 14336)
        self.max_position_embeddings = cfg_dict.get("max_position_embeddings", 262144)
        self.model_type = cfg_dict.get("model_type", "ministral3")
        self.num_attention_heads = cfg_dict.get("num_attention_heads", 32)
        self.num_hidden_layers = cfg_dict.get("num_hidden_layers", 34)
        self.num_key_value_heads = cfg_dict.get("num_key_value_heads", 8)
        self.rms_norm_eps = cfg_dict.get("rms_norm_eps", 1e-5)
        self.rope_parameters = cfg_dict.get(
            "rope_parameters",
            {
                "rope_type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "llama_4_scaling_beta": 0.1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            },
        )
        self.sliding_window = cfg_dict.get("sliding_window", None)
        self.use_cache = cfg_dict.get("use_cache", True)
        self.vocab_size = cfg_dict.get("vocab_size", 131072)
        self.pad_token_id = cfg_dict.get("pad_token_id", None)

    @classmethod
    def from_dict(cls, cfg_dict):
        if isinstance(cfg_dict, cls):
            return cfg_dict
        return cls(cfg_dict)


class Mistral3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Mistral3ForConditionalGeneration`].
    It is used to instantiate a Mistral3 model according to the specified arguments, defining the model architecture.

    Args:
        vision_config (`dict`, *optional*):
            The config dictionary of the vision backbone.
        text_config (`dict`, *optional*):
            The config dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 10):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_layer (`Union[int, list[int]]`, *optional*, defaults to -1):
            The index of the layer to select the vision feature.
        multimodal_projector_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the multimodal projector.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The downsampling factor for the spatial merge operation.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the input and output embeddings.
    """

    model_type = "mistral3"
    attribute_map = {"image_token_id": "image_token_index"}
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
        tie_word_embeddings: bool = False,
        initializer_range=0.02,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        self.initializer_range = initializer_range

        # Default vision config (Pixtral-like)
        if vision_config is None:
            vision_config = {
                "intermediate_size": 4096,
                "hidden_size": 1024,
                "patch_size": 14,
                "image_size": 1540,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "vocab_size": 32000,
                "head_dim": 64,
                "hidden_act": "gelu",
            }
        if isinstance(vision_config, dict):
            vision_config = Mistral3VisionConfig(**vision_config)
        self.vision_config = vision_config

        # Default text config (Mistral-like)
        if text_config is None:
            text_config = {
                "attention_dropout": 0.0,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 5120,
                "initializer_range": 0.02,
                "intermediate_size": 32768,
                "max_position_embeddings": 131072,
                "model_type": "mistral",
                "num_attention_heads": 32,
                "num_hidden_layers": 40,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-05,
                "rope_theta": 1000000000.0,
                "sliding_window": None,
                "use_cache": True,
                "vocab_size": 131072,
            }
        if isinstance(text_config, dict):
            text_config = Mistral3TextConfig(**text_config)
        self.text_config = text_config

        self.multimodal_projector_bias = multimodal_projector_bias
        self.spatial_merge_size = spatial_merge_size
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)


class Mistral3VisionConfig(PretrainedConfig):
    """Vision configuration for Mistral3 model."""

    model_type = "mistral3_vision"

    def __init__(
        self,
        intermediate_size=4096,
        hidden_size=1024,
        patch_size=14,
        image_size=1540,
        num_hidden_layers=24,
        num_attention_heads=16,
        vocab_size=32000,
        head_dim=64,
        hidden_act="gelu",
        **kwargs,
    ):
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.hidden_act = hidden_act

        super().__init__(**kwargs)


class Mistral3TextConfig(PretrainedConfig):
    """Text configuration for Mistral3 model."""

    model_type = "mistral3_text"

    def __init__(
        self,
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
        rms_norm_eps=1e-05,
        rope_theta=1000000000.0,
        sliding_window=None,
        use_cache=True,
        vocab_size=131072,
        **kwargs,
    ):
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.use_cache = use_cache
        self.vocab_size = vocab_size

        super().__init__(**kwargs)
