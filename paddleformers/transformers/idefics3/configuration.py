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

"""Idefics3 model configuration."""

from ..auto.configuration import CONFIG_MAPPING
from ..configuration_utils import PretrainedConfig


class Idefics3VisionConfig(PretrainedConfig):
    model_type = "idefics3_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
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
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range


class Idefics3Config(PretrainedConfig):
    model_type = "idefics3"
    is_composition = True
    sub_configs = {"vision_config": Idefics3VisionConfig, "text_config": CONFIG_MAPPING}
    keys_to_ignore_at_inference = ["past_key_values"]
    _text_config_passthrough_exclusions = {
        "_name_or_path",
        "model_type",
        "dtype",
        "_attn_implementation_internal",
        "architectures",
        "sub_configs",
        "vision_config",
        "text_config",
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        use_cache=True,
        image_token_id=128257,
        tie_word_embeddings=False,
        scale_factor=2,
        pad_token_id=128002,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = Idefics3VisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Idefics3VisionConfig()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config = text_config.copy()
            text_config["model_type"] = text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["llama"](vocab_size=128258, rms_norm_eps=1e-5, pad_token_id=pad_token_id)
        else:
            self.text_config = text_config

        self.text_config.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.image_token_id = image_token_id
        self.scale_factor = scale_factor
        self.architectures = kwargs.get("architectures", ["Idefics3ForConditionalGeneration"])
        if "_attn_implementation" in kwargs:
            self.vision_config._attn_implementation = kwargs["_attn_implementation"]
            self.text_config._attn_implementation = kwargs["_attn_implementation"]

    def __setattr__(self, key, value):
        text_config = super().__getattribute__("__dict__").get("text_config")
        if (
            text_config is not None
            and key not in self._text_config_passthrough_exclusions
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key not in super().__getattribute__("_text_config_passthrough_exclusions"):
            config_dict = super().__getattribute__("__dict__")
            text_config = config_dict.get("text_config")
            if text_config is not None and key in text_config.__dict__:
                return getattr(text_config, key)
        return super().__getattribute__(key)


__all__ = ["Idefics3Config", "Idefics3VisionConfig"]
