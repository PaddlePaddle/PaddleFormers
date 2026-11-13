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
#
# This file incorporates code from the Qwen team, Alibaba Group and the
# HuggingFace Inc. team. Copyright 2024 The Qwen team, Alibaba Group and the
# HuggingFace Inc. team. All rights reserved.
"""Hunyuan model configuration"""

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class HunyuanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HunyuanModel`]. It is used to instantiate a
    Hunyuan Dense V1 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    Hunyuan-0.5B [tencent/Hunyuan-0.5B-Pretrain](https://huggingface.co/tencent/Hunyuan-0.5B-Pretrain).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 120818):
            Vocabulary size of the Hunyuan model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HunyuanModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 3584):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. If it is not
            specified, will default to `8`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Scaling configuration for the RoPE embeddings. Hunyuan checkpoints encode DynamicNTKAlpha here using the
            legacy `type="dynamic"` field together with `alpha`; these values are folded into the RoPE base once at
            initialization. Defaults to `None`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply per-head RMS normalization to the query and key states after RoPE.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from paddleformers.transformers import HunyuanModel, HunyuanConfig

    >>> # Initializing a Hunyuan style configuration
    >>> configuration = HunyuanConfig()

    >>> # Initializing a model from the Hunyuan-0.5B style configuration
    >>> model = HunyuanModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hunyuan_v1_dense"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=120818,
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_qk_norm=True,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = dict(rope_scaling) if rope_scaling is not None else None
        self.attention_bias = attention_bias
        self.use_qk_norm = use_qk_norm
        self.attention_dropout = attention_dropout
        # Hunyuan checkpoints use the legacy `type` field and add DynamicNTKAlpha
        # parameters which are intentionally retained in `rope_parameters`.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.rope_parameters = dict(self.rope_scaling) if self.rope_scaling is not None else None

        # Keep Hunyuan's DynamicNTKAlpha parameters while validating the common
        # RoPE fields understood by PaddleFormers.
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(
            self,
            ignore_keys={"alpha", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"},
        )

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HunyuanConfig"]
