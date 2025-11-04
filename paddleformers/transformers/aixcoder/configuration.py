# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
""" Aixcoder model configuration"""

from ..configuration_utils import PretrainedConfig

__all__ = [
    "AIXCODER_PRETRAINED_INIT_CONFIGURATION",
    "AixcoderConfig",
    "AIXCODER_PRETRAINED_RESOURCE_FILES_MAP",
]

AIXCODER_PRETRAINED_INIT_CONFIGURATION = {}

AIXCODER_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {},
}


class AixcoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~AixcoderModel`]. It is used to instantiate an Aixcoder
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Aixcoder-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Aixcoder model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~AixcoderModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        scale_factor (`int`, *optional*, defaults to 8):
            The scale factor for the rotary position embeddings.
        factor (`int`, *optional*, defaults to 8):
            The factor for the rotary position embeddings.
        tensor_parallel_output (`bool`, *optional*, defaults to `False`):
            whether to return the output in multiple tensor parallel ranks, or return the concatenated output.
            Defaults to `False`.
        sequence_parallel (`bool`, *optional*, defaults to `False`):
            whether to use Megatron-style's sequence parallel. Defaults to `False`.
    """
    model_type = "aixcoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=49152,
        hidden_size=4096,
        intermediate_size=11008,
        max_position_embeddings=32768,
        seq_length=32768,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        use_cache=True,
        fuse_attention_qkv=False,
        fuse_attention_ffn=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        alibi=False,
        rope_scaling_factor=1.0,
        rope_scaling_type=None,
        rope_scaling=None,
        attention_dropout=0.0,
        scale_factor=8,
        factor=8,
        tensor_parallel_output=False,
        sequence_parallel=False,
        use_flash_attention=False,
        use_fused_rms_norm=False,
        use_fused_rope=False,
        long_sequence_strategy_type=None,
        long_sequence_strategy_name=None,
        long_sequence_init_args=None,
        use_long_sequence_strategies=False,
        use_flash_attention_for_generation=False,
        use_last_token_for_generation=False,
        immediate_clear_past_key_value=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = seq_length
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # Aixcoder uses GQA with num_key_value_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
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
        self.rope_scaling = rope_scaling

        self.attention_dropout = attention_dropout
        self.scale_factor = scale_factor
        self.factor = factor
        self.tensor_parallel_output = tensor_parallel_output
        self.sequence_parallel = sequence_parallel

        self.use_flash_attention = use_flash_attention
        self.use_fused_rms_norm = use_fused_rms_norm
        self.use_fused_rope = use_fused_rope

        self.long_sequence_strategy_type = long_sequence_strategy_type
        self.long_sequence_strategy_name = long_sequence_strategy_name
        self.long_sequence_init_args = {} if long_sequence_init_args is None else long_sequence_init_args
        self.use_long_sequence_strategies = use_long_sequence_strategies
        self.use_flash_attention_for_generation = use_flash_attention_for_generation
        self.use_last_token_for_generation = use_last_token_for_generation
        self.immediate_clear_past_key_value = immediate_clear_past_key_value

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )

        # Re-set fuse attributes after super().__init__ in case they were overwritten
        self.fuse_attention_qkv = fuse_attention_qkv
        self.fuse_attention_ffn = fuse_attention_ffn

    @property
    def rope(self):
        return not self.alibi
