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


class DeepseekV32Config(PretrainedConfig):
    r"""
    Configuration for DeepSeek V3.2 model.

    Architecture: MLA (Multi-Latent Attention) + DSA Indexer (DeepSeek Sparse Attention)
                 + MoE (Mixture of Experts) + MTP (Multi-Token Prediction)

    Field names are kept consistent with the HuggingFace config.json so that
    ``TransformerConfig.from_config()`` can map them to the PaddleFleet provider
    dataclass fields without any manual renaming.
    """

    model_type = "deepseek_v32"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        num_attention_heads=128,
        num_key_value_heads=128,
        max_position_embeddings=163840,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        initializer_range=0.02,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        # MLA parameters
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        head_dim=None,
        multi_latent_attention=True,
        use_qk_norm=True,
        # DSA Indexer parameters (field names match HF config.json)
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
        indexer_loss_coeff=0.0,
        indexer_use_sparse_loss=False,
        # RoPE format control for DSA Indexer
        # False = non-interleaved (default, compatible with MLA's interleaved YaRN)
        # True = interleaved (paired frequency format)
        indexer_rotary_interleaved=False,
        # MoE parameters
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        topk_method="noaux_tc",
        first_k_dense_replace=3,
        moe_layer_freq=1,
        # MTP parameters
        num_nextn_predict_layers=1,
        # Pipeline parallel segmentation
        pp_seg_method="layer:TransformerLayer|EmptyLayer",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # head_dim must equal v_head_dim for MLA: o_proj input size = num_heads * head_dim,
        # and the attention output per head = v_head_dim.
        self.head_dim = head_dim if head_dim is not None else v_head_dim
        self.multi_latent_attention = multi_latent_attention
        self.use_qk_norm = use_qk_norm

        # DSA Indexer
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.indexer_loss_coeff = indexer_loss_coeff
        self.indexer_use_sparse_loss = indexer_use_sparse_loss
        self.indexer_rotary_interleaved = indexer_rotary_interleaved

        # MoE
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.topk_method = topk_method
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq

        # MTP
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # PP
        self.pp_seg_method = pp_seg_method

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        # Re-set after super().__init__ because LlmMetaConfig defaults override these
        self.multi_latent_attention = multi_latent_attention
        self.use_qk_norm = use_qk_norm
        self.num_nextn_predict_layers = num_nextn_predict_layers


__all__ = ["DeepseekV32Config"]
