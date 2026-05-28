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


class DeepseekV4Config(PretrainedConfig):
    r"""
    Configuration class for DeepSeek-V4 model.

    DeepSeek-V4 introduces several innovations over V3:
    - DSv4 Hybrid Attention: Single-head MQA with CSA (Compressed Sparse Attention)
    - mHC (Multi-stream Hyper-Connection): 4-stream residual connections
    - Grouped LoRA output projection (8 groups)
    - Hash-based routing for early MoE layers
    - CSAIndexer + DSAIndexer for learned sparse attention

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 43):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Intermediate size of MoE expert FFN.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function in FFN.
        max_position_embeddings (`int`, *optional*, defaults to 65536):
            Maximum sequence length.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMSNorm.
        multi_latent_attention (`bool`, *optional*, defaults to `True`):
            Enable Multi-Latent Attention (required for DSv4 Hybrid).
        q_lora_rank (`int`, *optional*, defaults to 1024):
            Low-rank dimension for Q projection.
        v_head_dim (`int`, *optional*, defaults to 512):
            Head dimension for value (key == value in DSv4).
        qk_nope_head_dim (`int`, *optional*, defaults to 448):
            Non-positional dimension in QK head.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Positional (RoPE) dimension in QK head.
        experimental_attention_variant (`str`, *optional*, defaults to `"dsv4_hybrid"`):
            Selects DSv4 Hybrid Attention in PaddleFleet.
        o_groups (`int`, *optional*, defaults to 8):
            Number of groups for grouped LoRA output projection.
        o_lora_rank (`int`, *optional*, defaults to 1024):
            Low-rank dimension per group for output projection.
        qk_pos_emb_head_dim (`int`, *optional*, defaults to 64):
            Positional embedding dimension in DSv4 Hybrid.
        csa_window_size (`int`, *optional*, defaults to 128):
            Sliding window size for CSA.
        csa_compress_ratios (`list`, *optional*):
            Per-layer compression ratios for CSA. Must be a Python list of length
            num_hidden_layers. Values in {0, 4, 128}.
        csa_compress_rotary_base (`float`, *optional*, defaults to 160000.0):
            RoPE base for compressed KV positions.
        enable_hyper_connections (`bool`, *optional*, defaults to `True`):
            Enable mHC multi-stream residual connections.
        num_residual_streams (`int`, *optional*, defaults to 4):
            Number of residual streams in mHC.
        mhc_sinkhorn_iterations (`int`, *optional*, defaults to 20):
            Sinkhorn-Knopp iterations for doubly stochastic matrix.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 6):
            Top-k experts per token.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Expert FFN intermediate size.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        scoring_func (`str`, *optional*, defaults to `"sqrtsoftplus"`):
            MoE router scoring function.
        moe_n_hash_layers (`int`, *optional*, defaults to 3):
            Number of leading layers using hash-based routing.
        actual_vocab_size (`int`, *optional*, defaults to 102400):
            Padded vocab size for hash routing tid2eid table.
        mtp_num_layers (`int`, *optional*, defaults to 1):
            Number of Multi-Token Prediction layers.
        mtp_loss_scaling_factor (`float`, *optional*, defaults to 0.1):
            Loss weight for MTP auxiliary loss.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    # HuggingFace config.json field name -> PaddleFleet internal field name
    # Used to accept both naming conventions when loading from HF checkpoints.
    _HF_TO_FLEET_FIELD_MAP = {
        "compress_ratios": "csa_compress_ratios",
        "num_hash_layers": "moe_n_hash_layers",
        "compress_rope_theta": "csa_compress_rotary_base",
        "sliding_window": "csa_window_size",
        "hc_mult": "num_residual_streams",
        "hc_sinkhorn_iters": "mhc_sinkhorn_iterations",
        "hc_eps": "_hc_eps",  # not directly used, keep for reference
        "head_dim": "v_head_dim",
        "index_n_heads": "dsa_index_n_heads",
        "index_head_dim": "dsa_index_head_dim",
        "index_topk": "dsa_index_topk",
    }

    def __init__(
        self,
        # === Basic architecture ===
        vocab_size=129280,
        hidden_size=4096,
        num_hidden_layers=43,
        num_attention_heads=64,
        intermediate_size=2048,
        hidden_act="silu",
        max_position_embeddings=65536,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        attention_bias=False,
        attention_dropout=0.0,
        # === MLA (Multi-Latent Attention) ===
        multi_latent_attention=True,
        q_lora_rank=1024,
        v_head_dim=512,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        kv_lora_rank=None,
        use_qk_norm=True,
        # === DSv4 Hybrid Attention ===
        experimental_attention_variant="dsv4_hybrid",
        o_groups=8,
        o_lora_rank=1024,
        qk_pos_emb_head_dim=64,
        # === CSA (Compressed Sparse Attention) ===
        csa_window_size=128,
        csa_compress_ratios=None,
        csa_compress_rotary_base=160000.0,
        csa_dense_mode=False,
        # === DSA Indexer ===
        dsa_index_n_heads=64,
        dsa_index_head_dim=128,
        dsa_index_topk=512,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        # === mHC (Hyper-Connection) ===
        enable_hyper_connections=True,
        num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        # === MoE ===
        n_routed_experts=256,
        num_experts_per_tok=6,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        moe_n_hash_layers=3,
        actual_vocab_size=129280,
        moe_layer_freq=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        moe_expert_fusion=False,
        # === MTP (Multi-Token Prediction) ===
        use_mtp=True,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        use_dense_mtp=False,
        # === RoPE ===
        rope_theta=10000,
        rope_type="yarn",
        rotary_scaling_factor=16,
        original_max_position_embeddings=65536,
        rotary_base=10000,
        # === Parallelism (overridden at runtime) ===
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        # === Pipeline ===
        pp_seg_method="layer:DeepseekV4DecoderLayer",
        # === Other ===
        tie_word_embeddings=False,
        activation_func_clamp_value=10.0,
        normalization="RMSNorm",
        **kwargs,
    ):
        # Remap HF-style field names passed via kwargs to Fleet-internal names
        for hf_name, fleet_name in self._HF_TO_FLEET_FIELD_MAP.items():
            if hf_name in kwargs:
                val = kwargs.pop(hf_name)
                # Only apply if the Fleet-equivalent parameter was not explicitly set
                # (i.e. still at default). For positional params, we check if they differ
                # from their default value.
                setattr(self, f"_hf_{hf_name}", val)  # keep original for save_pretrained

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # MLA
        self.multi_latent_attention = multi_latent_attention
        self.q_lora_rank = q_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.use_qk_norm = use_qk_norm

        # DSv4 Hybrid Attention
        self.experimental_attention_variant = experimental_attention_variant
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim

        # CSA
        self.csa_window_size = csa_window_size
        if csa_compress_ratios is None:
            # Default for 43-layer DSv4 (44 elements, one more than num_hidden_layers
            # to accommodate num_head_empty_layers offset in Fleet):
            # [0]: reserved for offset
            # [1-2]: ratio=0 (pure window attention)
            # [3-42]: alternating ratio=4 and ratio=128 (20 pairs)
            # [43]: ratio=0 (last layer, no compression)
            self.csa_compress_ratios = [0, 0] + [4, 128] * 20 + [4, 0]
        else:
            self.csa_compress_ratios = csa_compress_ratios
        self.csa_compress_rotary_base = csa_compress_rotary_base
        self.csa_dense_mode = csa_dense_mode

        # DSA Indexer
        self.dsa_index_n_heads = dsa_index_n_heads
        self.dsa_index_head_dim = dsa_index_head_dim
        self.dsa_index_topk = dsa_index_topk
        self.dsa_indexer_loss_coeff = dsa_indexer_loss_coeff
        self.dsa_indexer_use_sparse_loss = dsa_indexer_use_sparse_loss

        # mHC (Hyper-Connection)
        self.enable_hyper_connections = enable_hyper_connections
        self.num_residual_streams = num_residual_streams
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations

        # MoE
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.moe_n_hash_layers = moe_n_hash_layers
        self.actual_vocab_size = actual_vocab_size
        self.moe_layer_freq = moe_layer_freq
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_expert_fusion = moe_expert_fusion

        # MTP
        self.use_mtp = use_mtp
        self.mtp_num_layers = mtp_num_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.use_dense_mtp = use_dense_mtp

        # RoPE
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.rotary_scaling_factor = rotary_scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rotary_base = rotary_base

        # Parallelism
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.expert_model_parallel_size = expert_model_parallel_size
        self.context_parallel_size = context_parallel_size
        self.sequence_parallel = sequence_parallel

        # Other
        self.tie_word_embeddings = tie_word_embeddings
        self.activation_func_clamp_value = activation_func_clamp_value
        self.normalization = normalization
        self.pp_seg_method = pp_seg_method

        # Apply HF->Fleet field mappings (override defaults with HF values)
        # This happens after setting defaults so HF values take precedence
        for hf_name, fleet_name in self._HF_TO_FLEET_FIELD_MAP.items():
            hf_attr = f"_hf_{hf_name}"
            if hasattr(self, hf_attr):
                val = getattr(self, hf_attr)
                if not fleet_name.startswith("_"):
                    setattr(self, fleet_name, val)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            # Pass fields managed by PretrainedConfig's LlmMetaConfig to avoid
            # being overridden by set_expected_keys() defaults.
            multi_latent_attention=multi_latent_attention,
            mtp_num_layers=mtp_num_layers,
            mtp_loss_scaling_factor=mtp_loss_scaling_factor,
            sequence_parallel=sequence_parallel,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            **kwargs,
        )


__all__ = ["DeepseekV4Config"]
