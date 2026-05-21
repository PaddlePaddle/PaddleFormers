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

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class MiniMaxM2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxM2Model`]. It is used to instantiate a
    MiniMaxM2 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 200064):
            Vocabulary size of the MiniMaxM2 model.
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Intermediate size of the routed expert.
        num_hidden_layers (`int`, *optional*, defaults to 62):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 48):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 196608):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        rope_theta (`float`, *optional*, defaults to 5000000):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        rotary_dim (`int`, *optional*, defaults to 64):
            Dimension of the rotary position embedding.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts per token.
        n_shared_experts (`int`, *optional*, defaults to 0):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers.
        use_qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use query-key normalization in the attention.
        qk_norm_type (`str`, *optional*, defaults to `"per_layer"`):
            Type of QK normalization.
        use_mtp (`bool`, *optional*, defaults to `True`):
            Whether to use multi-token prediction.
        num_mtp_modules (`int`, *optional*, defaults to 3):
            Number of MTP modules.
        mtp_transformer_layers (`int`, *optional*, defaults to 1):
            Number of transformer layers in each MTP module.
        use_routing_bias (`bool`, *optional*, defaults to `True`):
            Whether to use routing bias in MoE.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            Frequency of MoE layers.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
            Scoring function for MoE routing.
        disable_ffn_model_parallel (`bool`, *optional*, defaults to `False`):
            Whether to use tp in the moe.
        fd_fallback (`bool`, *optional*, defaults to `False`):
            Whether fastdeploy fallback.
    """

    model_type = "minimax_m2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=3072,
        head_dim=128,
        moe_intermediate_size=1536,
        num_hidden_layers=62,
        num_attention_heads=48,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=196608,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=5000000,
        rope_scaling=None,
        rotary_dim=64,
        attention_bias=False,
        attention_dropout=0.0,
        num_experts_per_tok=8,
        n_shared_experts=0,
        n_routed_experts=256,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        use_qk_norm=True,
        qk_norm_type="per_layer",
        pp_seg_method="layer:Glm4MoeDecoderLayer",
        disable_ffn_model_parallel=False,
        scoring_func="sigmoid",
        seq_aux=True,
        topk_method="noaux_tc",
        using_flex_token=True,
        moe_subbatch_token_num_before_dispatch=0,
        sliding_window=None,
        fd_fallback=False,
        use_mtp=True,
        num_mtp_modules=3,
        mtp_transformer_layers=1,
        use_routing_bias=True,
        moe_layer_freq=1,
        attn_type_list=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rotary_dim = rotary_dim
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.fd_fallback = fd_fallback
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.rope_parameters = self.rope_scaling
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        # MoE arguments
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.scoring_func = scoring_func
        self.seq_aux = seq_aux
        self.topk_method = topk_method
        self.using_flex_token = using_flex_token
        self.use_fp8 = False
        self.moe_subbatch_token_num_before_dispatch = moe_subbatch_token_num_before_dispatch
        self.use_mtp = use_mtp
        self.num_mtp_modules = num_mtp_modules
        self.mtp_transformer_layers = mtp_transformer_layers
        self.use_routing_bias = use_routing_bias
        self.moe_layer_freq = moe_layer_freq
        self.attn_type_list = attn_type_list

        self.pp_seg_method = pp_seg_method
        self.disable_ffn_model_parallel = disable_ffn_model_parallel

        super().__init__(
            **kwargs,
        )


__all__ = ["MiniMaxM2Config"]
