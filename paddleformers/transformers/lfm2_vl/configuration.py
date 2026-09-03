# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""LFM2-VL configuration."""

from typing import Optional, Union

from ..configuration_utils import PretrainedConfig


class Lfm2Config(PretrainedConfig):
    model_type = "lfm2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=2560,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=128000,
        initializer_range=0.02,
        norm_eps=1e-5,
        use_cache=True,
        rope_theta=1000000.0,
        rope_parameters=None,
        conv_bias=False,
        conv_L_cache=3,
        layer_types=None,
        full_attn_idxs=None,
        block_auto_adjust_ff_dim=False,
        block_ffn_dim_multiplier=None,
        block_multiple_of=256,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        fuse_rms_norm=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = kwargs.pop("block_ff_dim", intermediate_size)
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.rms_norm_eps = norm_eps
        # Preserve the reference implementation's unfused FP32 RMSNorm computation.
        self.fuse_rms_norm = fuse_rms_norm
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters or {"rope_type": "default", "rope_theta": rope_theta}
        self.conv_bias = conv_bias
        self.conv_L_cache = conv_L_cache
        self.block_auto_adjust_ff_dim = block_auto_adjust_ff_dim
        self.block_ffn_dim_multiplier = block_ffn_dim_multiplier
        self.block_multiple_of = block_multiple_of
        self.full_attn_idxs = full_attn_idxs
        if layer_types is None:
            full_attn_idxs = full_attn_idxs if full_attn_idxs is not None else list(range(num_hidden_layers))
            layer_types = ["full_attention" if i in full_attn_idxs else "conv" for i in range(num_hidden_layers)]
        self.layer_types = layer_types
        super().__init__(
            tie_word_embeddings=kwargs.pop("tie_embedding", tie_word_embeddings),
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class Siglip2VisionConfig(PretrainedConfig):
    model_type = "siglip2_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_patches=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_patches = num_patches


class Lfm2VlConfig(PretrainedConfig):
    model_type = "lfm2_vl"
    sub_configs = {"text_config": Lfm2Config, "vision_config": Siglip2VisionConfig}

    def __init__(
        self,
        vision_config: Optional[Union[dict, Siglip2VisionConfig]] = None,
        text_config: Optional[Union[dict, Lfm2Config]] = None,
        image_token_id=396,
        projector_hidden_act="gelu",
        projector_hidden_size=2560,
        projector_bias=True,
        projector_use_layernorm=True,
        downsample_factor=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = Siglip2VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Siglip2VisionConfig(**vision_config)
        if text_config is None:
            text_config = Lfm2Config()
        elif isinstance(text_config, dict):
            text_config = Lfm2Config(**text_config)
        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_id = image_token_id
        self.image_token_index = image_token_id
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        self.projector_bias = projector_bias
        self.projector_use_layernorm = projector_use_layernorm
        self.downsample_factor = downsample_factor
        super().__init__(tie_word_embeddings=kwargs.pop("tie_embedding", tie_word_embeddings), **kwargs)


__all__ = ["Lfm2Config", "Lfm2VlConfig", "Siglip2VisionConfig"]
