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

import math

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class StableLmConfig(PretrainedConfig):
    model_type = "stablelm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        intermediate_size=6912,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        layer_norm_eps=1.0e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        rope_parameters=None,
        use_qkv_bias=False,
        qk_layernorm=False,
        use_parallel_residual=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=None,
        **kwargs,
    ):
        for name, value in {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "max_position_embeddings": max_position_embeddings,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        for name, value in {"hidden_dropout": hidden_dropout, "attention_dropout": attention_dropout}.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1), got {value!r}")
        if not isinstance(partial_rotary_factor, (int, float)) or not 0 < partial_rotary_factor <= 1:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        rotary_dim = int(hidden_size // num_attention_heads * partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("partial RoPE dimension must be positive and even")
        for name, value in {
            "layer_norm_eps": layer_norm_eps,
            "initializer_range": initializer_range,
            "rope_theta": rope_theta,
        }.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_qkv_bias = use_qkv_bias
        self.qk_layernorm = qk_layernorm
        self.use_parallel_residual = use_parallel_residual
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout

        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        rope_scaling = kwargs.pop("rope_scaling", None)
        if rope_parameters is None:
            rope_parameters = rope_scaling
        if rope_parameters is not None and not isinstance(rope_parameters, dict):
            raise ValueError("rope_parameters and rope_scaling must be dicts or None")
        self.rope_parameters = dict(rope_parameters) if rope_parameters is not None else None
        standardize_rope_params(self, rope_theta=self.rope_theta)
        effective_rope_theta = self.rope_parameters["rope_theta"]
        if (
            not isinstance(effective_rope_theta, (int, float))
            or not math.isfinite(effective_rope_theta)
            or effective_rope_theta <= 0
        ):
            raise ValueError("rope_parameters.rope_theta must be positive and finite")
        rope_config_validation(self, ignore_keys={"partial_rotary_factor"})

        self.norm_eps = self.layer_norm_eps
        self.mlp_bias = False

        if kwargs.get("architectures") is None:
            kwargs["architectures"] = ["StableLmForCausalLM"]
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if getattr(self, "head_dim", None) not in (None, hidden_size // num_attention_heads):
            raise ValueError("StableLM head_dim must equal hidden_size // num_attention_heads")
        self.head_dim = hidden_size // num_attention_heads
