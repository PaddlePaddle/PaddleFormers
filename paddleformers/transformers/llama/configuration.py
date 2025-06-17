# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
""" Llama model configuration"""

from ..configuration_utils import PretrainedConfig

__all__ = [
    "LLAMA_PRETRAINED_INIT_CONFIGURATION",
    "LlamaConfig",
    "LLAMA_PRETRAINED_RESOURCE_FILES_MAP",
]

LLAMA_PRETRAINED_INIT_CONFIGURATION = {
    # Hypothetical model weights (tiny-random-llama & micro-random-llama) for test only
    "__internal_testing__/micro-random-llama": {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 1000,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 8,
        "num_hidden_layers": 1,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    },
    "__internal_testing__/tiny-random-llama": {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 8,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    },
}

# Hypothetical model weights (tiny-random-llama) for test only
LLAMA_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "__internal_testing__/micro-random-llama": "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/micro-random-llama/model_state.pdparams",
        "__internal_testing__/tiny-random-llama": "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/tiny-random-llama/model_state.pdparams",
    },
}


class LlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~LlamaModel`]. It is used to instantiate an Llama
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Llama-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Llama model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LlamaModel`] or [`~TFLlamaModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
            Enable rope fusion or not.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        Example:
    ```python
    >>> from paddleformers.transformer import LlamaModel, LlamaConfig

    >>> # Initializing a Llama llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"
    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }
    pretrained_init_configuration = LLAMA_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        max_position_embeddings=2048,
        seq_length=2048,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_cache=True,
        fuse_attention_qkv=False,
        fuse_attention_ffn=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        alibi=False,
        rope_scaling_factor=1.0,
        rope_scaling_type=None,
        long_sequence_strategy_type=None,
        long_sequence_strategy_name=None,
        long_sequence_init_args=None,
        use_long_sequence_strategies=False,
        use_flash_attention_for_generation=False,
        use_last_token_for_generation=False,
        immediate_clear_past_key_value=False,
        dpo_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = seq_length
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        self.use_cache = use_cache
        self.fuse_attention_qkv = fuse_attention_qkv
        self.fuse_attention_ffn = fuse_attention_ffn

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.alibi = alibi

        self.rope_scaling_factor = rope_scaling_factor
        self.rope_scaling_type = rope_scaling_type

        self.long_sequence_strategy_type = long_sequence_strategy_type
        self.long_sequence_strategy_name = long_sequence_strategy_name
        self.long_sequence_init_args = {} if long_sequence_init_args is None else long_sequence_init_args
        self.use_long_sequence_strategies = use_long_sequence_strategies
        self.use_flash_attention_for_generation = use_flash_attention_for_generation
        self.use_last_token_for_generation = use_last_token_for_generation
        self.immediate_clear_past_key_value = immediate_clear_past_key_value
        self.dpo_config = dpo_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rope(self):
        return not self.alibi
