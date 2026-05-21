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

"""
DeepSeek V3.2 Model Providers for PaddleFleet-based pretraining.

Architecture: MLA (Multi-Latent Attention) + DSA Indexer (DeepSeek Sparse Attention)
             + MoE (Mixture of Experts) + MTP (Multi-Token Prediction)

Reference: DeepSeek-V3.2-Exp/inference/model.py
Config:    DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json

Usage:
    provider = DeepSeekV3_2_671BProvider()
    model = provider.provide(loss_fn=loss_fn)

Pattern follows glm45_provider.py exactly.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import paddle
import paddle.nn.functional as F

from paddleformers.transformers.gpt_provider import GPTModelProvider

logger = logging.getLogger(__name__)


@dataclass
class DeepSeekV3_2BaseProvider(GPTModelProvider):
    """
    Base provider for DeepSeek V3.2 architecture.

    Key components:
    - MLA: Multi-Latent Attention with low-rank KV compression
    - DSA: DeepSeek Sparse Attention (Indexer selects top-2048 tokens per query)
    - MoE: Mixture of Experts with group-limited routing
    - MTP: Multi-Token Prediction auxiliary loss
    """

    # ---- Normalization and activation ----
    normalization: str = "RMSNorm"
    hidden_act: Callable = F.silu
    gated_linear_unit: bool = True
    use_bias: bool = False
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6

    # ---- Precision ----
    autocast_dtype: paddle.dtype = paddle.bfloat16
    params_dtype: paddle.dtype = paddle.bfloat16
    bf16: bool = True

    # ---- Embedding ----
    tie_word_embeddings: bool = False

    # ---- Sequence ----
    seq_length: int = 4096
    max_sequence_length: int = 4096
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    init_method_std: float = 0.006  # ~1/sqrt(7168)

    # ---- MLA: Multi-Latent Attention ----
    # MLA de-interleave in rope_utils is NOT needed when rotary_interleaved=True,
    # because _rotate_half(interleaved=True) already pairs adjacent dims correctly
    # (matching DeepSeek-V3.2 reference apply_rotary_emb(interleaved=True)).
    multi_latent_attention: bool = False
    num_attention_heads: int = 128
    # head_dim matches v_head_dim=128 so o_proj sizing in Attention base is correct
    head_dim: int = 128
    # num_key_value_heads must be set for Attention base class;
    # in MLA, KV is latent-compressed but we set this equal to num_attention_heads
    # so TP sharding logic in Attention.__init__ works correctly
    num_key_value_heads: int = 128

    # MLA low-rank projection dimensions (matches DeepSeek V3.2 671B config)
    q_lora_rank: int = 1536  # wq_a: hidden -> q_lora_rank
    kv_lora_rank: int = 512  # wkv_a: hidden -> kv_lora_rank + qk_rope_head_dim
    qk_nope_head_dim: int = 128  # per-head non-RoPE Q/K dim
    qk_rope_head_dim: int = 64  # per-head RoPE Q/K dim
    v_head_dim: int = 128  # per-head V dim (= head_dim, so o_proj ok)

    # ---- DSA: DeepSeek Sparse Attention Indexer ----
    # Non-None activates the DeepSeek V3.2 path in gpt_builders.py
    # Field names mirror HuggingFace config.json keys for zero-copy from_config().
    index_n_heads: int = 64  # Indexer scoring heads
    index_head_dim: int = 128  # Indexer Q/K head dim
    index_topk: int = 2048  # Tokens selected per query
    # KL loss trains wq_b/wk/weights_proj via KL(true_attn_dist || indexer_dist)
    # Coefficient ~0.01 matches Megatron-Core default; set to None to disable
    indexer_loss_coeff: float = 0.01
    indexer_use_sparse_loss: bool = False  # use full-sequence KL (denser gradients)

    # ---- RoPE ----
    position_embedding_type: str = "rope"
    # DeepSeek V3.2 uses YaRN-style RoPE with base 10000
    rotary_base: float = 10000.0
    # MLA uses interleaved RoPE; Indexer uses non-interleaved (handled internally)
    # Setting rotary_interleaved=True here enables the interleaved path for MLA Q/K
    rotary_interleaved: bool = True
    # Disable fused RoPE kernel: MLA applies RoPE only to qk_rope_head_dim subspace,
    # which is incompatible with the fused kernel that expects full head_dim
    apply_rope_fusion: bool = False
    # Use fp32 RoPE for numerical stability (matches reference implementation)
    high_precision_rope: bool = True

    # ---- MoE routing ----
    scoring_func: str = "sigmoid"  # Score experts with sigmoid
    num_experts_per_tok: int = 8  # n_activated_experts
    n_group: int = 8  # n_expert_groups: 256 experts / 8 groups = 32 per group
    topk_group: int = 4  # n_limited_groups: select top-4 groups
    routed_scaling_factor: float = 2.5  # route_scale: scale selected expert weights
    topk_method: str = "group_limited_greedy"  # group-limited top-k routing
    norm_topk_prob: bool = True  # normalize expert weights to sum to 1
    moe_token_dispatcher_type: str = "deepep"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_router_pre_softmax: bool = False
    moe_expert_fusion: bool = False
    moe_shared_expert_overlap: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0.0

    # ---- MTP: Multi-Token Prediction ----
    # 1 MTP layer for auxiliary next-token prediction loss
    num_nextn_predict_layers: Optional[int] = 1
    mtp_loss_scaling_factor: float = 0.1  # MTP loss weight

    # ---- Optimization ----
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True


@dataclass
class DeepSeekV3_2_671BProvider(DeepSeekV3_2BaseProvider):
    """
    Provider for DeepSeek V3.2 671B model (full production config).

    Architecture:
    - 61 transformer layers: first 3 dense MLP + 58 MoE
    - All layers use MLA + DSA Indexer attention
    - 256 routed experts + 1 shared expert per MoE layer

    Config reference: DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json
    """

    # ---- Model dimensions ----
    hidden_size: int = 7168  # dim
    num_hidden_layers: int = 61  # n_layers
    vocab_size: int = 129280

    # ---- FFN dimensions ----
    intermediate_size: int = 18432  # inter_dim: dense MLP hidden size
    moe_intermediate_size: int = 2048  # moe_inter_dim: per-expert MLP hidden size

    # ---- MoE architecture ----
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    # Layer pattern: first 3 layers dense (0), then 58 MoE (1)
    moe_layer_freq: Union[int, List[int]] = field(default_factory=lambda: [0] * 3 + [1] * 58)


@dataclass
class DeepSeekV3_2_671BDebugProvider(DeepSeekV3_2_671BProvider):
    """
    Small debug variant of DeepSeek V3.2 for single-card validation.

    Reduces all dimensions to fit on a single GPU for smoke testing.
    Pattern: 1 dense layer + 3 MoE layers.
    """

    # ---- Reduced model dimensions ----
    num_hidden_layers: int = 4
    hidden_size: int = 1024
    vocab_size: int = 129280

    # ---- Reduced attention dimensions ----
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64

    # ---- Reduced Indexer dimensions ----
    index_n_heads: int = 8
    index_head_dim: int = 64
    index_topk: int = 128
    indexer_loss_coeff: float = 0.01
    indexer_use_sparse_loss: bool = False

    # ---- Reduced FFN dimensions ----
    intermediate_size: int = 2048
    moe_intermediate_size: int = 512

    # ---- Reduced MoE ----
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_layer_freq: Union[int, List[int]] = field(default_factory=lambda: [0] * 1 + [1] * 3)

    # ---- Disable MTP for simplicity ----
    num_nextn_predict_layers: Optional[int] = 0

    # ---- Short sequence for debug ----
    seq_length: int = 512
    max_sequence_length: int = 512

    # ---- Single card: no model parallel ----
    sequence_parallel: bool = False
    expert_model_parallel_size: int = 1
    tensor_model_parallel_size: int = 1
    moe_router_force_load_balancing: bool = True


@dataclass
class DeepSeekV3_2_8GPUDebugProvider(DeepSeekV3_2BaseProvider):
    """
    Debug provider for DeepSeek V3.2 on a single node with 8 GPUs.

    Scales up from the single-card DebugProvider to exercise multi-card
    communication paths (all-reduce, all-gather, DeepEP routing) without
    the memory footprint of the full 671B model.

    Key dimension constraints for parallelism:
        num_attention_heads (32) and index_n_heads (16) must be
        divisible by whatever tensor_model_parallel_size is used.
        n_routed_experts (16) must be divisible by expert_model_parallel_size.

    Pattern: 2 dense layers + 6 MoE layers (8 total).
    """

    # ---- Reduced model dimensions ----
    num_hidden_layers: int = 8
    hidden_size: int = 2048
    vocab_size: int = 129280

    # ---- Reduced attention dimensions ----
    num_attention_heads: int = 32  # divisible by TP=1/2/4/8
    num_key_value_heads: int = 32
    head_dim: int = 64
    q_lora_rank: int = 512
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64

    # ---- Reduced Indexer dimensions ----
    index_n_heads: int = 16  # divisible by TP=1/2/4/8
    index_head_dim: int = 64
    index_topk: int = 256
    indexer_loss_coeff: float = 0.01
    indexer_use_sparse_loss: bool = False

    # ---- Reduced FFN dimensions ----
    intermediate_size: int = 4096
    moe_intermediate_size: int = 1024

    # ---- Reduced MoE ----
    n_routed_experts: int = 16  # divisible by EP=1/2/4/8
    n_shared_experts: int = 1
    moe_layer_freq: Union[int, List[int]] = field(default_factory=lambda: [0] * 2 + [1] * 6)

    # ---- Disable MTP for simplicity ----
    num_nextn_predict_layers: Optional[int] = 0

    # ---- Moderate sequence length ----
    seq_length: int = 1024
    max_sequence_length: int = 1024
