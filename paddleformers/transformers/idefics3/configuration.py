# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
    sub_configs = {"vision_config": Idefics3VisionConfig, "text_config": CONFIG_MAPPING}
    keys_to_ignore_at_inference = ["past_key_values"]

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

        self.use_cache = use_cache
        self.image_token_id = image_token_id
        self.scale_factor = scale_factor
        if "_attn_implementation" in kwargs:
            self.vision_config._attn_implementation = kwargs["_attn_implementation"]
            self.text_config._attn_implementation = kwargs["_attn_implementation"]

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in ["_name_or_path", "model_type", "dtype", "_attn_implementation_internal"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "_name_or_path",
            "model_type",
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                # Check self first: attributes explicitly set on Idefics3Config
                # (e.g. tie_word_embeddings) take priority over text_config defaults.
                self_dict = super().__getattribute__("__dict__")
                if key in self_dict:
                    return self_dict[key]
                return getattr(text_config, key)
        return super().__getattribute__(key)


__all__ = ["Idefics3Config", "Idefics3VisionConfig"]
