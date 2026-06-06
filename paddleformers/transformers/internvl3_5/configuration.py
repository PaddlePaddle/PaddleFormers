# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 OpenGVLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import copy

from ..configuration_utils import PretrainedConfig
from ..qwen3.configuration import Qwen3Config

__all__ = ["InternVisionConfig", "InternVLChatConfig"]


class InternVisionConfig(PretrainedConfig):
    model_type = "intern_vit_6b"
    base_config_key = "vision_config"

    def __init__(
        self,
        num_channels=3,
        patch_size=14,
        image_size=224,
        qkv_bias=False,
        hidden_size=3200,
        num_attention_heads=25,
        intermediate_size=12800,
        qk_normalization=True,
        num_hidden_layers=48,
        use_flash_attn=True,
        hidden_act="gelu",
        norm_type="rms_norm",
        layer_norm_eps=1e-6,
        dropout=0.0,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.use_flash_attn = use_flash_attn


class InternVLChatConfig(PretrainedConfig):
    model_type = "internvl_chat"
    is_composition = True
    sub_configs = {"vision_config": InternVisionConfig, "llm_config": Qwen3Config}

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version="v1",
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        img_context_token_id=151671,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = {"architectures": ["InternVisionModel"]}
        if llm_config is None:
            llm_config = {"architectures": ["Qwen3ForCausalLM"]}

        self.vision_config = (
            InternVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
        )
        self.llm_config = Qwen3Config(**llm_config) if isinstance(llm_config, dict) else llm_config

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.img_context_token_id = img_context_token_id
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings
        self.vocab_size = self.llm_config.vocab_size
        self.hidden_size = self.llm_config.hidden_size
        self.pad_token_id = getattr(self.llm_config, "pad_token_id", getattr(self, "pad_token_id", None))
        self.eos_token_id = getattr(self.llm_config, "eos_token_id", getattr(self, "eos_token_id", None))
        self.bos_token_id = getattr(self.llm_config, "bos_token_id", getattr(self, "bos_token_id", None))

    def to_dict(self, *args, **kwargs):
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
