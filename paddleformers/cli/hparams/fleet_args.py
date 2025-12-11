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

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional


@dataclass
class FleetArguments:
    """
    Core configuration arguments for Fleet Model.
    """

    # ------------------------------ Position Embedding Configuration ------------------------------
    position_embedding_type: str = "rope"
    """Type of position embedding. Defaults to RoPE (Rotary Position Embedding)."""

    # ------------------------------ MoE Router & Expert Configuration ------------------------------
    moe_router_enable_expert_bias: bool = False
    """Whether to enable expert-specific bias terms in the MoE router. Fine-tunes router preference for individual experts.
    Defaults to False (simplifies computation and avoids overfitting)."""

    moe_router_force_load_balancing: bool = True
    """Whether to enforce load balancing across MoE experts. Prevents overutilization of a small subset of experts.
    Defaults to True (critical optimization for MoE stability and efficiency)."""

    moe_router_load_balancing_type: str = "seq_aux_loss"
    """Strategy for MoE expert load balancing."""

    moe_router_bias_update_rate: float = 0.01
    """Update rate for MoE router biases (only effective if `moe_router_enable_expert_bias=True`).
    Controls the magnitude of bias adjustments to prevent unstable updates. Defaults to 0.01."""

    moe_shared_expert_overlap: bool = False
    """Whether to allow shared experts to be reused across layers/modules. Reduces memory footprint but may limit model expressivity.
    Defaults to False (prioritizes model capacity)."""

    moe_dequant_input: bool = False
    """Whether to dequantize inputs to MoE experts (only applicable if inputs are quantized).
    Defaults to False (enable only for quantized inference/training pipelines)."""

    moe_expert_fusion: bool = True
    """Whether to enable operator fusion for MoE expert layers (e.g., Linear + Activation fusion).
    Improves training/inference throughput by reducing kernel launch overhead. Defaults to True."""

    moe_router_fusion: bool = True
    """Whether to enable operator fusion for the MoE router (e.g., Gating + Softmax fusion).
    Reduces computation latency for expert selection. Defaults to True."""

    moe_subbatch_token_num_after_dispatch: int = 4096
    """Number of tokens per sub-batch after MoE expert dispatch. Controls memory usage for expert computations.
    Defaults to 4096 (balances memory efficiency and parallelism for most GPUs)."""

    moe_grouped_gemm: bool = True
    """Whether to enable grouped GEMM (General Matrix Multiplication) for MoE experts.
    Batches computations across multiple experts to improve hardware utilization. Defaults to True."""

    # ------------------------------ Network Architecture Configuration ------------------------------
    gated_linear_unit: bool = False
    """Whether to use Gated Linear Units (GLU) instead of standard Linear layers. Enhances model expressivity (common in SwiGLU).
    Defaults to False (compatible with basic transformer architectures)."""

    normalization: str = "RMSNorm"
    """Type of normalization layer. Defaults to RMSNorm."""

    fp8: bool = False
    """Whether to enable FP8 mixed-precision training/inference. Reduces memory usage and accelerates computation (requires hardware support).
    Defaults to False (enable only for Ampere+/Hopper GPUs with FP8 support)."""

    fp8_wgrad: bool = False
    """Whether to use FP8 for gradient storage during training (only effective if `fp8=True`).
    Further reduces memory footprint but may introduce minor numerical error. Defaults to False."""

    fp32_residual_connection: bool = True
    """Whether to use FP32 precision for residual connections. Mitigates numerical underflow/overflow in deep transformers.
    Defaults to True (standard practice for stable LLM training)."""

    softmax_scale: Optional[float] = None
    """Scaling factor for Softmax inputs. If None, uses automatic scaling (e.g., sqrt(d_model) for attention).
    Defaults to None (adapts to model dimension automatically)."""

    softmax_type: Literal["vanilla", "off-by-one", "learnable"] = "vanilla"
    """Applies modified softmax from https://www.evanmiller.org/attention-is-off-by-one.html.
    Supports both TE FusedAttention and local unfused attention. Supports both a fixed offset andand learnable offset."""

    # ------------------------------ Initialization Configuration ------------------------------
    init_method: Callable | None = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    paddlefleet.utils.init_method_normal(init_method_std) which is paddle nn init normal with
    mean=0.0 and std=init_method_std."""

    output_layer_init_method: Callable | None = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to paddlefleet.utils.scaled_init_method_normal(init_method_std) which is paddle nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_hidden_layers)."""

    embedding_init_method: Callable | None = None
    """
    Method to initialize weights of the embedding layer. If None, will be set as described
    in init_method above.
    """

    embedding_init_method_std: float = 0.02
    """Standard deviation for embedding layer initialization (only effective if `embedding_init_method="normal"`).
    Defaults to 0.02 (common choice for transformer embeddings to avoid saturation)."""

    # ------------------------------ Recomputation (Gradient Checkpointing) Configuration ------------------------------
    recompute_method: str = None
    """Determines which transformer layers will be recomputed. uniform will uniformly divide the
    total number of transformer layers in a transformer block and recompute the input activation of
    each divided chunk at the specified granularity.  block will recompute the input activations for
    only a set number of transformer layers per pipeline stage.  The rest of the layers in the
    pipeline stage will not have any activations recomputed.  If None, and recompute is enabled, all
    layers will do recomputation. If set, must be 'uniform' or 'block'."""

    recompute_num_layers: int = None
    """When recompute_method is uniform, recompute_num_layers is the number of transformer layers in
    each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is
    the number of transformer layers to recompute within each pipeline stage.  Must be None for
    'selective' activation checkpointing."""

    recompute_modules: Optional[List[str]] = None
    """List of module names to apply recomputation."""

    # recompute_granularity: str = None
    """Determines which type of activation recompute to use.  Fleet-core supports 'selective'
    activation checkpointing where the sublayers set in --recompute-modules is checkpointed.
    The default is "core_attn" which is the memory intensive part of attention.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

    recompute_mtp_granularity: str = "none"
    """Recomputation granularity for MTP (Mixture of Token-Parallel) layers.
    """

    recompute_mtp_method: str = "none"
    """Recomputation method for MTP layers.
    """

    recompute_mtp_modules: Optional[List[str]] = None
    """List of MTP module names to apply recomputation."""

    # ------------------------------ Distributed Communication Configuration ------------------------------
    cp_comm_type: str = None
    """Communication type for checkpoint parallelism (CP).
    """

    dp_comm_overlap: bool = True
    """Whether to overlap data parallelism (DP) communication with computation."""

    sharding_comm_overlap: bool = True
    """Whether to overlap sharding parallelism (SP) communication with computation. Reduces latency for sharded models.
    Defaults to True."""

    tp_async_allreduce: bool = False
    """Whether to use asynchronous allreduce for tensor parallelism (TP)."""

    sp_async_reduce_scatter: bool = False
    """Whether to use asynchronous reduce-scatter for sharding parallelism (SP)."""

    overlap_p2p_comm: bool = True
    """Whether to overlap point-to-point (P2P) communication with computation.
    Defaults to True."""

    batch_p2p_comm: bool = True
    """Whether to batch point-to-point (P2P) communication requests."""

    # ------------------------------ General Training/Inference Configuration ------------------------------
    deterministic_mode: bool = False
    """Whether to enable deterministic (reproducible) training/inference. Disables non-deterministic optimizations.
    Defaults to False (prioritizes speed over strict reproducibility)."""

    dynamic_shape: bool = True
    """Whether to support dynamic input shapes (variable sequence lengths). Critical for LLM inference with varying prompt lengths.
    Defaults to True (standard for LLM pipelines)."""

    mtp_loss_scaling_factor: float = 1.0
    """Loss scaling factor for MTP (Mixture of Token-Parallel) training. Adjusts for imbalanced token distributions.
    Defaults to 1.0 (no scaling; tune for MTP-specific stability issues)."""
