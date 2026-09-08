# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class Cohere2Config(PretrainedConfig):
    model_type = "cohere2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=8192,
        intermediate_size=22528,
        logit_scale=0.0625,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        model_max_length=16384,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=5,
        eos_token_id=255001,
        tie_word_embeddings=True,
        rope_parameters=None,
        rope_theta=50000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        use_qk_norm=False,
        attention_bias=False,
        sliding_window=4096,
        layer_types=None,
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
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.logit_scale = logit_scale
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.model_max_length = model_max_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rms_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        if rope_parameters is None:
            rope_parameters = self.rope_scaling
        elif "rope_theta" not in rope_parameters:
            rope_parameters = dict(rope_parameters)
            rope_parameters["rope_theta"] = rope_theta
        self.rope_parameters = rope_parameters
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 4)
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)


__all__ = ["Cohere2Config"]
