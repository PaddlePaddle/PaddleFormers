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
"""ShieldGemma2 model configuration."""
from ..configuration_utils import PretrainedConfig
from ..gemma3.configuration import Gemma3TextConfig


class ShieldGemma2VisionConfig(PretrainedConfig):
    model_type = "shieldgemma2_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        vision_use_head=False,
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
        self.vision_use_head = vision_use_head


class ShieldGemma2Config(PretrainedConfig):
    r"""
    Configuration class for [`ShieldGemma2ForImageClassification`].
    """

    model_type = "shieldgemma2"
    is_composition = True
    sub_configs = {"vision_config": ShieldGemma2VisionConfig, "text_config": Gemma3TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }
    _text_config_passthrough_exclusions = {
        "_name_or_path",
        "model_type",
        "dtype",
        "_attn_implementation_internal",
        "id2label",
        "label2id",
        "num_labels",
        "num_classes",
        "architectures",
        "sub_configs",
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        mm_tokens_per_image=256,
        boi_token_index=255999,
        eoi_token_index=256000,
        image_token_index=262144,
        initializer_range=0.02,
        yes_token_index=10784,
        no_token_index=3771,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        self.mm_tokens_per_image = mm_tokens_per_image
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index
        self.initializer_range = initializer_range
        self.yes_token_index = yes_token_index
        self.no_token_index = no_token_index
        self.architectures = kwargs.get("architectures", ["ShieldGemma2ForImageClassification"])

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in self._text_config_passthrough_exclusions
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in super().__getattribute__(
            "_text_config_passthrough_exclusions"
        ):
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)
        return super().__getattribute__(key)

    @property
    def num_classes(self):
        return self.num_labels

    @num_classes.setter
    def num_classes(self, num_classes):
        self.num_labels = num_classes


__all__ = ["ShieldGemma2Config", "ShieldGemma2VisionConfig"]
