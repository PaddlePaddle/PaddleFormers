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
from ..configuration_utils import PretrainedConfig, layer_type_validation
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class Gemma3TextConfig(PretrainedConfig):
    model_type = "gemma3_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = {"global": 1_000_000.0, "local": 10_000.0}

    def __init__(
        self,
        vocab_size=262208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        layer_types=None,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_scaling=None,
        rope_parameters=None,
        rope_local_base_freq=10000.0,
        use_bidirectional_attention=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.layer_types = layer_types
        self.use_bidirectional_attention = use_bidirectional_attention
        if use_bidirectional_attention:
            self.sliding_window = self.sliding_window // 2 + 1
        self.rope_local_base_freq = rope_local_base_freq
        self.rope_scaling = rope_scaling

        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)
        if self.layer_types is None:
            self.layer_types = [
                ("sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention")
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        default_rope_parameters = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        if rope_parameters is not None and all(
            layer_type in self.layer_types for layer_type in rope_parameters.keys()
        ):
            self.rope_parameters = {
                layer_type: dict(rope_parameters[layer_type]) if rope_parameters.get(layer_type) is not None else None
                for layer_type in set(self.layer_types)
            }
        else:
            self.rope_parameters = {
                layer_type: dict(default_rope_parameters[layer_type]) for layer_type in default_rope_parameters
            }
            if rope_parameters is not None:
                self.rope_parameters["full_attention"].update(rope_parameters)
            elif rope_scaling is not None:
                self.rope_parameters["full_attention"].update(rope_scaling)

        if self.rope_parameters.get("full_attention") is None:
            self.rope_parameters["full_attention"] = {"rope_type": "default"}
        self.rope_parameters["full_attention"].setdefault("rope_theta", rope_theta)

        if self.rope_parameters.get("sliding_attention") is None:
            self.rope_parameters["sliding_attention"] = {"rope_type": "default"}
        self.rope_parameters["sliding_attention"].setdefault("rope_theta", rope_local_base_freq)

        self.rope_theta = {
            "full_attention": self.rope_parameters["full_attention"]["rope_theta"],
            "sliding_attention": self.rope_parameters["sliding_attention"]["rope_theta"],
        }

        standardize_rope_params(self, rope_theta=self.rope_theta)
        rope_config_validation(self)


class SiglipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
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


class Gemma3Config(PretrainedConfig):
    model_type = "gemma3"
    is_composition = True
    sub_configs = {"text_config": Gemma3TextConfig, "vision_config": SiglipVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_classes": "num_labels",
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
        "vision_config",
        "text_config",
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
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Composition configs receive classification metadata at the outer level as
        # well as in the text sub-config.  Keep the outer label maps in sync with
        # ``num_classes`` (the legacy alias used by ConfigTester and checkpoints).
        if "num_classes" in kwargs:
            self.num_labels = kwargs["num_classes"]

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
        self.tie_word_embeddings = tie_word_embeddings
        self.architectures = kwargs.get("architectures", ["Gemma3ForConditionalGeneration"])

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
        if "text_config" in super().__getattribute__("__dict__") and key not in super().__getattribute__(
            "_text_config_passthrough_exclusions"
        ):
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)
        return super().__getattribute__(key)


__all__ = ["Gemma3Config", "Gemma3TextConfig", "SiglipVisionConfig"]
