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
""" MiniMax (Text-01) model configuration"""

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class MiniMaxConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxModel`]. It is used to instantiate a
    MiniMax (Text-01) model according to the specified arguments, defining the model architecture.

    MiniMax (Text-01) uses a hybrid attention mechanism:
    - Some layers are full attention (standard causal self-attention with RoPE)
    - Some layers are linear attention ("lightning attention") with intra-/inter-block attention

    The layer type is controlled by `layer_types`. By default, the pattern is alternating
    `full_attention` and `linear_attention` (full on odd-indexed layers, linear on even).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 200064):
            Vocabulary size of the MiniMax model.
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the routed expert MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key_value heads for implementing Grouped Query Attention.
        head_dim (`int`, *optional*):
            Dimension of each attention head. If None, defaults to hidden_size // num_attention_heads.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 10240000):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        fuse_rms_norm (`bool`, *optional*, defaults to `False`):
            Whether to use the fused RMSNorm kernel. The default preserves the FP32 accumulation order used by the
            Transformers MiniMax implementation.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            Number of selected experts per token.
        num_local_experts (`int`, *optional*, defaults to 32):
            Number of routed experts.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Reserved for checkpoint compatibility. MiniMax does not currently return router logits;
            enabling this option raises `NotImplementedError` during the forward pass.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Reserved for checkpoint compatibility and not applied while router-logit output is unsupported.
        router_jitter_noise (`float`, *optional*, defaults to 0.0):
            Jitter noise for the router.
        layer_types (`list[str]`, *optional*):
            A list that maps each layer index to its attention type. Can be `"full_attention"` or `"linear_attention"`.
        block_size (`int`, *optional*, defaults to 256):
            The length of each attention block for the lightning attention.
            Lightning attention does not currently support packed sequences. Disable packing and
            `use_attn_mask_startend_row_indices` when `layer_types` contains `"linear_attention"`.
        full_attn_alpha_factor (`float`, *optional*, defaults to 1):
            Weight for residual value in residual connection after full attention.
        full_attn_beta_factor (`float`, *optional*, defaults to 1):
            Weight for hidden state value in residual connection after full attention.
        linear_attn_alpha_factor (`float`, *optional*, defaults to 1):
            Weight for residual value in residual connection after lightning attention.
        linear_attn_beta_factor (`float`, *optional*, defaults to 1):
            Weight for hidden state value in residual connection after lightning attention.
        mlp_alpha_factor (`float`, *optional*, defaults to 1):
            Weight for residual value in residual connection after MLP.
        mlp_beta_factor (`float`, *optional*, defaults to 1):
            Weight for hidden state value in residual connection after MLP.

    ```python
    >>> from paddleformers.transformers import MiniMaxModel, MiniMaxConfig

    >>> # Initializing a MiniMax (Text-01) style configuration
    >>> configuration = MiniMaxConfig()

    >>> # Initializing a model from the MiniMax (Text-01) style configuration
    >>> model = MiniMaxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "minimax"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=6144,
        intermediate_size=9216,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=10240000,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        fuse_rms_norm=False,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=32,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        attn_type_list=None,
        rope_theta=10000000.0,
        rope_scaling=None,
        layer_types=None,
        block_size=256,
        full_attn_alpha_factor=1.0,
        full_attn_beta_factor=1.0,
        linear_attn_alpha_factor=1.0,
        linear_attn_beta_factor=1.0,
        mlp_alpha_factor=1.0,
        mlp_beta_factor=1.0,
        **kwargs,
    ):
        full_attn_alpha_factor = kwargs.pop("layernorm_full_attention_alpha", full_attn_alpha_factor)
        full_attn_beta_factor = kwargs.pop("layernorm_full_attention_beta", full_attn_beta_factor)
        linear_attn_alpha_factor = kwargs.pop("layernorm_linear_attention_alpha", linear_attn_alpha_factor)
        linear_attn_beta_factor = kwargs.pop("layernorm_linear_attention_beta", linear_attn_beta_factor)
        mlp_alpha_factor = kwargs.pop("layernorm_mlp_alpha", mlp_alpha_factor)
        mlp_beta_factor = kwargs.pop("layernorm_mlp_beta", mlp_beta_factor)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.fuse_rms_norm = fuse_rms_norm
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.rope_parameters = self.rope_scaling
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        if layer_types is None:
            if attn_type_list is not None:
                if len(attn_type_list) != num_hidden_layers:
                    raise ValueError(
                        f"attn_type_list length ({len(attn_type_list)}) must equal "
                        f"num_hidden_layers ({num_hidden_layers})."
                    )
                self.layer_types = [
                    "linear_attention" if int(attn_type) == 0 else "full_attention" for attn_type in attn_type_list
                ]
            else:
                self.layer_types = [
                    "full_attention" if bool((i + 1) % 2) else "linear_attention"
                    for i in range(self.num_hidden_layers)
                ]
        else:
            if len(layer_types) != num_hidden_layers:
                raise ValueError(
                    f"layer_types length ({len(layer_types)}) must equal num_hidden_layers ({num_hidden_layers})."
                )
            self.layer_types = list(layer_types)
        self.attn_type_list = [0 if layer_type == "linear_attention" else 1 for layer_type in self.layer_types]
        self.block_size = block_size
        self.full_attn_alpha_factor = full_attn_alpha_factor
        self.full_attn_beta_factor = full_attn_beta_factor
        self.linear_attn_alpha_factor = linear_attn_alpha_factor
        self.linear_attn_beta_factor = linear_attn_beta_factor
        self.mlp_alpha_factor = mlp_alpha_factor
        self.mlp_beta_factor = mlp_beta_factor

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            sliding_window=sliding_window,
            **kwargs,
        )


__all__ = ["MiniMaxConfig"]
