# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ovis2 model configuration."""

from ..configuration_utils import PretrainedConfig
from ..qwen2.configuration import Qwen2Config


class Ovis2VisionConfig(PretrainedConfig):
    model_type = "ovis2_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_channels=3,
        image_size=224,
        patch_size=14,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        qkv_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        vocab_size=16384,
        hidden_stride=1,
        num_visual_indicator_tokens=5,
        initializer_range=0.02,
        tokenize_function="softmax",
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
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.qkv_bias = qkv_bias
        self.mlp_bias = mlp_bias
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.hidden_stride = hidden_stride
        self.num_visual_indicator_tokens = num_visual_indicator_tokens
        self.initializer_range = initializer_range
        self.tokenize_function = tokenize_function


class Ovis2Config(PretrainedConfig):
    model_type = "ovis2"
    is_composition = True
    sub_configs = {"text_config": Qwen2Config, "vision_config": Ovis2VisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151665,
        visual_indicator_token_ids=(151666, 151667, 151668, 151669, 151670),
        vocab_size=151643,
        hidden_size=1536,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        if isinstance(vision_config, dict):
            vision_config = Ovis2VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Ovis2VisionConfig(num_visual_indicator_tokens=len(visual_indicator_token_ids))

        if isinstance(text_config, dict):
            model_type = text_config.pop("model_type", "qwen2")
            if model_type != "qwen2":
                raise ValueError(f"Ovis2 currently supports a Qwen2 text backbone, got {model_type!r}.")
            text_config = Qwen2Config(**text_config)
        elif text_config is None:
            text_config = Qwen2Config()

        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_id = image_token_id
        self.visual_indicator_token_ids = list(visual_indicator_token_ids)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings
        self.architectures = kwargs.get("architectures", ["Ovis2ForConditionalGeneration"])


__all__ = ["Ovis2Config", "Ovis2VisionConfig"]
