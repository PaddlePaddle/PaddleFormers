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

# Referred to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import paddle.nn.functional as F

from ..model_parallel_config import ModelParallelConfig
from ..utils import (
    get_magic_init_method,
    init_method_normal,
    scaled_init_method_normal,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for transformers."""

    ####################
    # model architecture
    ####################

    num_hidden_layers: int = 1
    """Number of transformer layers in a transformer block."""

    pad_token_id: int = 0
    """Token ID used for padding."""

    num_nextn_predict_layers: int = 0
    """Number of Multi-Token Prediction (MTP) Layers."""

    train_mtp_only: bool = False
    """Whether to train MTP only."""

    mtp_distillation_loss: bool = False
    """Whether to use distillation MTP loss."""

    mtp_num_layers: int = 0
    """MTP Layer number."""

    mtp_loss_scaling_factor: float = 0.3
    """Weighting factor of Multi-Token Prediction (MTP) loss."""

    add_mtp_loss: bool = True
    """Add mtp loss to final loss to enable mtp backward and weight update."""

    mtp_load_weight_only: bool = False
    """When True, use WeightOnlyMTPLayer (holds weights but skips MTP computation and embedding processing)."""

    use_dense_mtp: bool = False
    """When True, MTP layers use dense MLP instead of MoE in their internal transformer block."""

    separate_mtp_headloss: bool = False
    """Separate MTP LMHead & Loss calculate for pipeline balance."""

    enable_mtp_magic_send: bool = False
    """When True, use magic send mechanism for MTP: broadcast input_ids to last PP stage
    and re-embed there, instead of pre-computing shifted embeddings at first stage
    and concatenating them through the pipeline."""

    experimental_dataflow: bool = False
    """When True, use new experimental dataflow where mtp_startend_row_indices_all is passed as a
    separate input instead of being appended to attn_mask_startend_row_indices.
    The new dataflow requires: input_ids, labels, startend_row_indices (last dim=1, main seq only),
    mtp_startend_row_indices_all ([B, num_nextn, S, 1]), position_ids."""

    num_empty_layers_add_in_head: int = 0
    """Number of EmptyLayer before the Decoder Layer.
    num_empty_layers_add_in_head=2 Example:
        EmptyLayer, EmptyLayer, Decoder, Dcoder, ...
    0 implies equal layer division across PP ranks."""

    num_empty_layers_add_in_tail: int = 0
    """Number of EmptyLayer after the Decoder Layer.
    num_empty_layers_add_in_tail=2 Example:
        ..., Decoder, Dcoder, EmptyLayer, EmptyLayer
    0 implies equal layer division across PP ranks."""

    # Note: need to implement PipelineParallelLayerLayout and import
    # pipeline_model_parallel_layout: str | list | PipelineParallelLayerLayout = None
    pipeline_model_parallel_layout: str | list = None
    """Custom definition of the pipeline parallel partitioning.
    Support type:
    - str: e.g., 'Et*3|(tt|)*29,m|L'. Stages are split by '|', replicated stages or layers
    can be described with multiplication. Commas can be used cosmetically.
    - list: e.g., [['embedding', 'decoder'], ['decoder', 'decoder', 'decoder', 'loss']].
    - PipelineParallelLayerLayout: a PipelineParallelLayerLayout object.
    If given either a string or a list, it will be transferred into a PipelineParallelLayerLayout
    in post init. Let i = a * pp_size + b, then layout[i] gives a list of the layers
    in the a-th vpp stage and the b-th pp stage, i.e., vpp(0)pp(0), vpp(0)pp(1), ...,
    vpp(i)pp(j), vpp(i)pp(j+1), ..., vpp(-1)pp(-2), vpp(-1)pp(-1).
    In the inner lists of layers, 'embedding' or 'E' denotes the embedding layer, 'loss' or 'L'
    denotes the loss function, and 'decoder' or 't' denotes the transformer decoder layer.
    Examples:
        [['embedding', 'decoder'], ['decoder', 'decoder', 'decoder', 'loss']]:
        pp = 2, vpp = None
        pp rank 0 holds: embedding, decoder
        pp rank 1 holds: decoder*3, loss
        'E|(tt|)*2,(t|)*4,mL':
        pp = 2, vpp = 4
        vpp rank 0 pp rank 0 holds: embedding
        vpp rank 0 pp rank 1~2 holds: decoder*2
        vpp rank 0 pp rank 3 holds: decoder
        vpp rank 1 pp rank 0~2 holds: decoder
        vpp rank 1 pp rank 3 holds: mtp, loss"""

    account_for_embedding_in_pipeline_split: bool = False
    """If set, the embedding layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    account_for_loss_in_pipeline_split: bool = False
    """If set, the loss layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 1
    """Number of transformer attention heads."""

    softmax_scale: float = None
    """Softmax scale for attention scaling."""

    softmax_type: Literal["vanilla", "off-by-one", "learnable"] = "vanilla"
    """Applies modified softmax from https://www.evanmiller.org/attention-is-off-by-one.html.
       Supports both TE FusedAttention and local unfused attention. Supports both a fixed offset and
       and learnable offset."""

    num_key_value_heads: int = None
    """Number of key-value heads for group query attention. If None, normal attention is used."""

    init_method: Callable | None = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    paddlefleet.utils.init_method_normal(init_method_std) which is paddle nn init normal with
    mean=0.0 and std=init_method_std."""

    head_dim: int = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    hidden_dropout_prob: float = 0.0
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.0
    """Post attention dropout probability."""

    _attn_implementation: str = "default"
    """Attention implementation to use."""

    flashmask_use_varlen: bool = False
    """If True, convert flashmask to varlen in attention."""

    intermediate_size: int | None = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    hidden_act: Callable = F.gelu
    """Activation function to use for the non-linearity in the MLP."""

    use_bias: bool = False
    """Include a bias term in all linear layers (QKV projections and Output projections, after core attention, and two in
    MLP layer)."""

    moe_routed_expert_use_bias: bool | None = None
    """Override whether routed MoE expert MLP layers include bias terms. If None, use use_bias."""

    attention_bias: bool = False
    """Include a bias term in QKV projections."""

    output_layer_init_method: Callable | None = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to paddlefleet.utils.scaled_init_method_normal(init_method_std) which is paddle nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_hidden_layers)."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    use_vha_attention: bool = False
    """If True, enables VHA premix/postmix extensions in standard self-attention."""

    vha_shared_kv: bool = False
    """If True, enables Shared KV to reduce KVCache"""

    vha_postmix_rank: int | None = None
    """Rank of the VHA postmix low-rank head mixing matrices."""

    vha_q_lora_rank: int | None = None
    """Rank of the VHA Q low-rank projection. When set, Q projects to this rank per head before premix expansion."""

    swa_vha_q_lora_rank: int | None = None
    """VHA Q low-rank projection rank for SWA layers. Defaults to swa_head_dim in __post_init__."""

    swa_vha_postmix_rank: int | None = None
    """VHA postmix rank for SWA layers. Defaults to swa_num_attention_heads // 4."""

    attention_value_scale: float | None = None
    """Scale factor applied to the value tensor before attention computation. If None, no scaling
    is applied. Used in architectures like MiMo that scale V for training stability."""

    add_full_attention_sink_bias: bool = False
    """Whether to add a learnable attention sink bias for full (non-SWA) attention layers.
    When True, softmax_type is promoted to 'learnable' for full attention layers."""

    add_swa_attention_sink_bias: bool = True
    """Whether to add a learnable attention sink bias for sliding window attention (SWA) layers.
    When True, softmax_type is promoted to 'learnable' for SWA layers."""

    swa_head_dim: int | None = None
    """Dimension of query/key heads for sliding window attention layers. Defaults to head_dim."""

    swa_v_head_dim: int | None = None
    """Dimension of value heads for sliding window attention layers. Defaults to v_head_dim."""

    swa_num_attention_heads: int | None = None
    """Number of attention heads for sliding window attention layers. Defaults to num_attention_heads."""

    swa_num_key_value_heads: int | None = None
    """Number of key/value heads (GQA groups) for sliding window attention layers. Defaults to num_key_value_heads."""

    swa_rope_theta: float | None = None
    """The base period of the RoPE embeddings for sliding window attention layers. Defaults to rope_theta."""

    head_wise_swa_ratio: float = 0.0
    """Ratio of KV heads that use sliding window attention within an SWA layer.
    0.0 means all heads use SWA; values between 0 and 1 create a mix where
    the first (1 - ratio) * num_heads are full attention and the rest are SWA."""

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    heterogeneous_block_specs: bool = False
    """Whether to use heterogeneous block specs (nemotron-nas architecture)."""

    sliding_window: tuple[int, int] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    window_attn_skip_freq: int | list[int] = None
    """Frequency of full attention layers among sliding window attention layers. Accepts either:
    - An integer N: Represents a (N-1):1 ratio, one full attention layer after (N-1) SWA layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,1,0,0,0,0], where 1 represents SWA. """

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

    fp32_residual_connection: bool = False
    """If true, move residual connections to fp32."""

    rope_scaling: dict = None
    """Related parameters for rope_scaling, default is None."""

    rope_theta: float = 10000.0
    """The base period of the RoPE embeddings, default is 10000.0."""

    apply_residual_connection_post_layernorm: bool = False
    """If True, uses the original BERT residue connection ordering."""

    activation_func_clamp_value: float = None
    """Clamp the output of the linear_fc1 in the activation function. Only used when activation_func
    is quick_gelu."""

    glu_linear_offset: float = 0.0
    """Offset term in the GLU activation function: activation_func(x[0]) * (x[1] + offset). Only
    used when gated_linear_unit is True"""

    multimodal_embedding: bool = False
    """Whether to use multimodal embedding."""

    multimax_modules: list[str] | None = None
    """Submodules to apply learnable SegLU-style modulation to before softmax.

    Mirrors the Megatron ``recompute_modules`` style: a list of submodule
    names. ``None`` (default) disables the feature globally. Currently
    supported list entries:

    - ``"lm_head"``: apply SegLU(x, ranges, ts) on the LM-head logits before
      the language-modeling softmax/cross-entropy. Adds two [4]-shape
      learnable parameters (multimax_ranges, multimax_ts) to the LM head.
      These are excluded from weight decay via the "multimax" substring
      filter in the trainer's no-decay rule.
    - ``"attention"``: apply on attention scores before softmax. Reserved;
      not implemented yet (emits a warning if listed).

    YAML/JSON behaviour:
    - unset key, ``multimax_modules: null``, or empty list ``multimax_modules: []``
      all map to Python ``None`` (feature disabled).
    - ``multimax_modules: [lm_head]`` enables the LM-head branch.
    """

    gated_attention: bool = False
    """If True, enables gated attention where a learnable sigmoid gate is applied to the
    attention output before the output projection. The gate is produced alongside the query
    from the fused QKV projection (doubling the query projection size). This allows the model
    to dynamically control the information flow from attention. See Qwen3.5 for reference."""

    ####################
    # block attention residuals
    ####################
    block_attention_residuals: bool = False
    """Whether to use block attention residuals. When True,
    replaces standard fixed-weight residual connections with
    learned softmax attention over block-level representations."""

    attn_res_block_size: int = 1
    """Number of consecutive layers per block for
    block attention residuals. Controls how many layers
    accumulate standard residuals before applying the learned
    attention-weighted combination across blocks."""

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    high_precision_rope: bool = False
    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    normalization: str = "RMSNorm"
    """Norm type"""

    use_qk_norm: bool = False
    """Whether to apply `normalization` type of normalization to the query and key embeddings."""

    qk_norm_fusion: bool = False
    """If True, use Triton fused RMSNorm kernel for QK norm."""

    qk_norm_type: str = "per_head"
    """Type of qk normalization:
    - "per_head": normalize each attention head independently (default for most models)
    - "per_layer": normalize across all heads jointly (full-dimension, used by MiniMax)
    """

    rms_norm_eps: float = 1e-5
    """Epsilon value for norm."""

    layernorm_zero_centered_gamma: bool = False
    """If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves
    numerical stability."""

    bias_dropout_fusion: bool = False
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    sigmoid_gate_fusion: bool = False
    """If True, use Triton fused sigmoid gate kernel."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: str = None
    """Determines which type of activation recompute to use.  Fleet-core supports 'selective'
    activation checkpointing where the sublayers set in --recompute-modules is checkpointed.
    The default is "core_attn" which is the memory intensive part of attention.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

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

    recompute_modules: list[str] | dict = None
    """The submodules to recompute.
    list: contains all submodule need recompute
    dict: keys contains all submodule need recompute, value means submodule in which layers need recompute
    """

    ####################
    # MoE related
    ####################
    n_routed_experts: int | None = None
    """Number of routed experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    n_shared_experts: int | None = None
    """Number of shared experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    num_experts_per_tok: int = 2
    """Number of experts to route to for each token."""

    scoring_func: str = "softmax"
    """Score function for MoE routing. Options: "softmax", "sigmoid", "tanh",
    "relu", "gelu", "leaky_relu", "sftplus" (softplus, non-negative unbounded),
    "sqrtsoftplus" (sqrt(softplus), non-negative unbounded)."""

    moe_intermediate_size: int | None = None
    """MoE Feed-Forward Network hidden size"""

    topk_method: str = "greedy"
    """Options are greedy, group_limited_greedy, no_auxtc"""

    moe_token_dispatcher_type: str = "deepep"
    """The type of token dispatcher to use. The default is 'deepep'.
    Options are 'allgather', 'alltoall', 'deepep', and 'hybridep'."""

    moe_use_fusion_node: bool = True
    """Whether to use fusion node for MoE layer. Default is True"""

    moe_router_load_balancing_type: str = "aux_loss"
    """"Options are aux_loss, seq_aux_loss, global_aux_loss, sinkhorn"""

    moe_layer_freq: int | list[int] | None = None
    """Frequency between MoE layers and Dense layers. Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,0,1,1,1,0,1,1,1,0]"""

    first_k_dense_replace: int | None = None
    """the number of Dense layers.
    - An integer N: Represents the first N layers are dense layers, the remaining ones are moe layers."""

    moe_expert_capacity_factor: float | None = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match
    the expert capacity length, effective only after the moe_expert_capacity_factor is set. The
    default setting is False."""

    moe_token_drop_policy: str = "probs"
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with
    the lowest probabilities will be dropped. If "position", tokens at the end of each batch will
    be dropped.
    """

    router_aux_loss_coef: float = 1e-2
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    norm_topk_prob: bool = True
    """Whether to normalize the topk probabilities."""

    n_group: int = 1
    """Number of groups for routed experts."""

    topk_group: int = 1
    """Number of selected groups per token for expert selection."""

    routed_scaling_factor: float = 1.0
    """Scalar multiplier applied to the selected top-k routing weights after expert selection.
    The final scaled weights are used in ``top_gate`` (``[S, K]``), which is passed to the
    dispatch/combine flow for expert output weighting.

    Default is ``1.0`` (no scaling effect). For example, set to ``2.5`` for DeepSeek-V3 to
    compensate for sigmoid scores not summing to 1 after top-k selection.

    When ``routed_scaling_factor_learnable=True``, this value is used as the initialization
    value for the per-expert learnable parameter."""

    routed_scaling_factor_learnable: bool = False
    """Whether to use a learnable per-expert scaling parameter instead of a fixed scalar.

    - ``False`` (default): apply ``routed_scaling_factor`` as a fixed scalar uniformly.
    - ``True``: create a trainable parameter of shape ``[num_experts]``, initialized to
      ``routed_scaling_factor``, and apply it via per-expert lookup after top-k selection."""

    moe_dequant_input: bool = False
    """Whether to dequantize input."""

    moe_expert_fusion: bool = False
    """Whether to fuse experts."""

    moe_subbatch_token_num_before_dispatch: int | None = None
    """Whether to enable subbatch before dispatch, the value means the number of tokens in one subbatch."""

    moe_subbatch_token_num_after_dispatch: int | None = None
    """Whether to enable subbatch after dispatch, the value means the number of tokens in one subbatch."""

    use_auto_subbatch: bool = False
    """When True, dynamically determine subbatch sizes based on VMM free block analysis
    instead of using a fixed moe_subbatch_token_num_after_dispatch value."""

    moe_subbatch_diag: bool = False
    """When True, print auto_subbatch diagnostic info (path, subbatch_rows, zip_unzip_fusion)
    after each forward/backward pass. Useful for debugging memory behavior."""

    router_z_loss_coef: float = None
    """Scaling coefficient for z-loss. Default is None."""

    moe_router_force_load_balancing: bool = False
    """Force load balancing with random logits for MoE router."""

    moe_n_hash_layers: int = 0
    """Number of leading transformer layers that use hash-based MoE routing.
    Layers with layer_number < moe_n_hash_layers (0-indexed) use a pre-computed
    tid2eid lookup table for expert selection instead of learned top-k routing.
    Score weights are still computed from the gate logits. 0 disables hash routing."""

    actual_vocab_size: int | None = None
    """Padded actual vocabulary size. Required when moe_n_hash_layers > 0 for the
    tid2eid lookup buffer in hash-based MoE routing."""

    moe_router_fusion: bool = False
    """Whether to fuse MoE router."""

    moe_shared_expert_gate: bool = False
    """Enable gate for shared expert."""

    moe_shared_expert_overlap: bool = False
    """Enable overlapping between shared expert computations and a2a combinet"""

    moe_deep_gemm: bool = True
    """Whether to use DeepGEMM for the bf16 grouped-gemm MoE path. This option only takes effect when
    ``moe_expert_fusion=True`` and fp8 is disabled, it is ignored when fp8 is enabled."""

    moe_ep_barrier: bool = True
    """Whether to use barrier for expert parallelism."""

    moe_latent_size: int | None = None
    """The latent dimension size for latent MoE. Positive values enable latent MoE."""

    ##################
    # Context Parallel
    ##################
    cp_comm_type: str | list[str] | None = None
    """Inter-gpu communication type for context parallelism. Not support now.
    str: all layers share same communication type.
    List[str]: each layer has its separate communication type.
    """

    cp_balance_mode: str = "dualchunk_allgather"
    """Context parallel scatter/gather layout mode.
    "dualchunk_allgather": balanced front+rear chunk splitting (default).
    "contiguous_allgather": simple rank-order contiguous slicing.
    """

    ####################
    # fp8
    ####################
    fp8: str | None = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_recipe: str = "blockwise"
    """If set, enables the use of FP8 precision. There are 2 predefined
    choices 1) 'mxfp8' for Blackwell architecture only, 2) 'blockwise' for blockwise scaling recipe"""

    fp8_wgrad: bool = True
    """Whether to use fp8 wgrad."""

    dw_p2p_overlap: bool = False
    """Whether to overlap p2p communication and matmul kernel in pp parallel on Blackwell."""

    use_ue8m0: bool = False
    """Whether to use UE8M0 packed scaling factors for FP8 on Blackwell GPUs."""

    use_fp8_qat: bool = False
    """Whether to enable FP8 Quantization-Aware Training (QAT)."""

    ####################
    # initialization
    ####################
    init_method: callable = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    paddlefleet.utils.init_method_normal(init_method_std) which is paddle nn init normal with
    mean=0.0 and std=init_method_std."""

    embedding_init_method: Callable | None = None
    """
    Method to initialize weights of the embedding layer. If None, will be set as described
    in init_method above.
    """

    embedding_init_method_std: float | None = None
    """
    Standard deviation of the zero mean normal for the default initialization method for the
    embedding layer. If None, will be set to init_method_std.
    """

    output_layer_init_method: callable = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to paddlefleet.utils.scaled_init_method_normal(init_method_std) which is paddle nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_hidden_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    embedding_init_method: callable = None
    """
    Method to initialize weights of the embedding layer. If None, will be set as described
    in init_method above.
    """

    embedding_init_method_std: float = None
    """
    Standard deviation of the zero mean normal for the default initialization method for the
    embedding layer. If None, will be set to init_method_std.
    """

    init_model_with_meta_device: bool = False
    """
    If True, initializes the model with the meta device. This is helpful for
    training of very large models. This feature is only works when custom fsdp is turned on.
    """

    use_cpu_initialization: bool = False

    is_hybrid_model: bool = False
    """ Indicates whether this is a hybrid model. """

    ####################
    # Hyper-Connection (mHC) Configuration
    ####################
    enable_hyper_connections: bool = False
    """Enable mHC (Manifold-Constrained Hyper-Connections) residual connections."""

    num_residual_streams: int = 4
    """Number of residual streams (n in mHC paper)."""

    mhc_sinkhorn_iterations: int = 20
    """Number of Sinkhorn-Knopp iterations for doubly stochastic projection."""

    mhc_init_gating_factor: float = 0.01
    """Initial value of Gating Factor (alpha in paper)."""

    use_fused_mhc: bool = False
    """Use fused triton kernels for mHC operations (sinkhorn, h_aggregate, h_post_bda, proj_rms).
    Requires cuTile to be available."""

    high_precision_mhc: bool = True
    """Use high precision (float32) for mHC forward and backward computation."""

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    using_sonic_moe: bool = False
    """When using_sonic_moe is enabled, the computation part of the moelayer will use the implementation provided by SonicMoE."""

    ####################
    # MLA
    ####################
    """Configuration object for paddlefleet Multi-Latent Attention (MLA) transformers.

    The initialization function has an argument for each parameter, including those in
    ModelParallelConfig. Included YaRN RoPE parameters that is fused in MLA.
    """

    q_lora_rank: int = 512
    """Rank of Query tensor's low rank representation."""

    kv_lora_rank: int = 512
    """Rank of Key and Value tensors' low rank representation."""

    qk_nope_head_dim: int = 64
    """Dimension of the head in the QK projection. q_head_dim = qk_nope_head_dim + qk_rope_head_dim. Original qk_head_dim"""

    qk_rope_head_dim: int = 64
    """Dimension of the position embedding in the QK projection. Original qk_pos_emb_head_dim."""

    v_head_dim: int | None = None
    """Dimension of the head in the V projection."""

    rope_type: str = "yarn"
    """Type of RoPE to use. Default to yarn, options are rope and yarn."""

    rotary_base: float = 10000
    """Rotary base for the rotary embeddings, used by rope and yarn."""

    rotary_percent: float = 1.0
    """Rotary percent for the rotary embeddings, used by rope."""

    rotary_scaling_factor: float = 40
    """Rotary scaling factor for the rotary embeddings, used by yarn."""

    original_max_position_embeddings: int = 4096
    """Original maximum position embeddings for the original model, used by yarn."""

    beta_fast: float = 32
    """Beta fast for YaRN RoPE, used by yarn."""

    beta_slow: float = 1
    """Beta slow for YaRN RoPE, used by yarn."""

    mscale: float = 1.0
    """Mscale for YaRN RoPE in Multi-Latent Attention, used by yarn."""

    mscale_all_dim: float = 0.0
    """Mscale all dimensions for YaRN RoPE in Multi-Latent Attention, used by yarn."""

    loss_subbatch_sequence_length: int = -1
    """Sequence length of subbatch for loss computation."""

    fused_linear_ce_loss_chunk: int = 0
    """Enable fused linear + cross-entropy loss when > 0.

    When set to a positive integer N, LM head skips materializing the full
    [B, S, V] logits tensor and instead passes (hidden_states, weight, bias)
    to LanguageLoss, which dispatches to LigerFusedLinearCrossEntropyFunction
    with num_chunks=N. Only compatible with tensor_model_parallel_size == 1
    (or parallel_output disabled)."""

    # cache_mla_latents: bool = False

    ####################
    # DSA (DeepSeek Sparse Attention)
    ####################

    dsa_index_n_heads: int | None = None
    """Number of DSA Indexer heads. None disables DSA; non-None activates
    DeepSeek V3.2 sparse attention path.

    Note: This field corresponds to the HuggingFace config.json field "index_n_heads".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_index_head_dim: int = 128
    """Per-head dimension for Indexer Q/K vectors.

    Note: This field corresponds to the HuggingFace config.json field "index_head_dim".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_index_topk: int = 2048
    """Number of token positions selected by Indexer per query token.

    Note: This field corresponds to the HuggingFace config.json field "index_topk".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_indexer_loss_coeff: float | None = None
    """KL loss coefficient for DSA Indexer training. None disables the KL loss.

    Note: This field corresponds to the HuggingFace config.json field "indexer_loss_coeff".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_indexer_use_sparse_loss: bool = False
    """Whether to restrict DSA KL loss to top-k positions only.

    Note: This field corresponds to the HuggingFace config.json field "indexer_use_sparse_loss".
    The mapping from HuggingFace field name to PaddleFleet internal field name is handled
    by TransformerConfig.transform_rules.
    """

    dsa_indexer_rotary_interleaved: bool = False
    """
    Whether Indexer uses interleaved Rotary Position Embeddings.

    When False (default), Indexer uses non-interleaved RoPE with
    half-head frequencies [θ₁,θ₂,...,θ₁,θ₂,...].

    When True, Indexer uses interleaved RoPE with paired frequencies
    [θ₁,θ₁,θ₂,θ₂,...].

    This allows compatibility with MLA's YaRN RoPE which always generates
    interleaved frequencies.
    """

    dsa_indexer_loss_coeff: float = 0.01
    """KL loss coefficient for DSA Indexer training. None disables the KL loss."""

    ####################
    # CSA / DSv4 Hybrid Attention
    ####################

    experimental_attention_variant: str | None = None
    """Which experimental attention variant to use.
    Supported values: None (disabled), 'dsa', 'dsv4_hybrid'.
    When 'dsv4_hybrid', enables DeepSeekV4 Hybrid Attention with Compressed Sparse Attention.
    """

    csa_window_size: int = 128
    """Sliding window size for Compressed Sparse Attention (CSA).
    Each query attends to the last csa_window_size tokens via a sliding window.
    """

    csa_compress_ratios: list | None = None
    """Per-layer compression ratios for CSA. Length must equal num_hidden_layers.
    Each value must be one of {0, 4, 128}:
      - 0: window-only attention (no compression)
      - 4: overlapping compression with learned CSAIndexer
      - 128: non-overlapping compression, attend to all compressed positions
    """

    csa_compress_rotary_base: float = 40000.0
    """Rotary base for compressed KV positions in CSA.
    Used instead of the standard rotary_base when compress_ratio > 1 for a layer.
    """

    csa_dense_mode: bool = False
    """If True, skip CSAIndexer for ratio==4 layers and attend to all compressed positions.
    Useful for debugging or ablation studies.
    """

    csa_tilelang_backend: str | None = None
    """Optional CSA TileLang backend.

    None keeps the default Paddle implementation. 'attention_paddle_compat'
    enables the TileLang indexer and sparse attention paths by default while
    preserving Paddle-compatible tensor layouts and algorithm semantics.
    """

    csa_tilelang_enable_indexer: bool | None = None
    """Optional override for the CSA TileLang indexer path.

    None follows csa_tilelang_backend. True requires
    csa_tilelang_backend='attention_paddle_compat'. False disables only the
    CSA indexer TileLang path.
    """

    csa_indexer_backend: str = "tilelang"
    """CSA indexer backward backend.

    One of {"tilelang", "cudnn"}. Default "tilelang" preserves the legacy
    path.
    """

    csa_sparse_attn_backend: str = "tilelang"
    """CSA sparse attention backend. Single switch selecting one of three
    implementations of the final sparse MQA attention.

    One of {"unfused", "tilelang", "cudnn"}:
      * "unfused": pure-Paddle einsum forward + Paddle autograd backward
        (non-fused reference path).
      * "tilelang" (default): TileLang sparse MQA kernel forward + backward.
      * "cudnn": FlashMLA sparse forward kernel + cuDNN DSA backward
        kernel.
    """

    o_groups: int = 8
    """Number of groups for grouped low-rank output projection (wo_a) in DSv4 Hybrid.
    Set to 0 to use a single linear output projection instead.
    """

    o_lora_rank: int = 1024
    """Low-rank dimension per group for the grouped output projection in DSv4 Hybrid."""

    qk_pos_emb_head_dim: int | None = None
    """Dimension of positional embedding portion in each QK head for DSv4 Hybrid.
    When set, the total head dim is split as: v_head_dim = qk_nope_dim + qk_pos_emb_head_dim.
    The positional embedding (RoPE) is applied only to the last qk_pos_emb_head_dim dims.
    """

    gpt_model_use_experimental_version: bool = False
    """Enable experimental version code paths for precision alignment."""

    moe_topk_fusion: bool = False
    """If True, use Triton fused MoE TopK kernel for expert selection."""

    routing_map_fusion: bool = False
    """If True, use Triton fused routing map kernel for MoE routing."""

    magic_init: bool = False
    """Use the magic initialization method."""

    # Field name mapping rules: HuggingFace config.json name -> TransformerConfig name
    transform_rules = {
        # DSA field mapping
        "index_n_heads": "dsa_index_n_heads",
        "index_head_dim": "dsa_index_head_dim",
        "index_topk": "dsa_index_topk",
        "indexer_loss_coeff": "dsa_indexer_loss_coeff",
        "indexer_use_sparse_loss": "dsa_indexer_use_sparse_loss",
        "indexer_rotary_interleaved": "dsa_indexer_rotary_interleaved",
        "indexer_rope_interleave": "dsa_indexer_rotary_interleaved",
        # CSA / DSv4 Hybrid field mapping
        "csa_window_size": "csa_window_size",
        "csa_compress_ratios": "csa_compress_ratios",
        "csa_compress_rotary_base": "csa_compress_rotary_base",
        "csa_dense_mode": "csa_dense_mode",
        "csa_tilelang_backend": "csa_tilelang_backend",
        "csa_tilelang_enable_indexer": "csa_tilelang_enable_indexer",
        "csa_indexer_backend": "csa_indexer_backend",
        "csa_sparse_attn_backend": "csa_sparse_attn_backend",
        "o_groups": "o_groups",
        "o_lora_rank": "o_lora_rank",
        "qk_pos_emb_head_dim": "qk_pos_emb_head_dim",
    }

    @classmethod
    def from_config(cls, config_dict):
        # note(zhangweilong): if cls(),will call __post_init__ directly,but __new__ will skip some attr init .please check provider attr
        instance = object.__new__(cls)
        instance.register_attributes(config_dict)
        instance.__post_init__()
        return instance

    def register_attributes(self, config):
        transform_rules = None
        if hasattr(self, "transform_rules"):
            transform_rules = self.transform_rules

        for key, value in config.__dict__.items():
            if transform_rules and key in transform_rules:
                self._process_attribute(transform_rules[key], value)
            else:
                self._process_attribute(key, value)

    def _process_attribute(self, key, value):
        if not isinstance(key, str) or not key.isidentifier():
            print(f"invalid key name: {key}")
            return

        if key == "hidden_act":
            if isinstance(value, str):
                if value == "gelu_pytorch_tanh":
                    func = functools.partial(F.gelu, approximate=True)
                else:
                    func = getattr(F, value)
                setattr(self, key, func)
            elif callable(value):
                setattr(self, key, value)
            else:
                raise TypeError(
                    f"hidden_act must be str or callable, but get {type(value)}"
                )
        elif key == "dtype":
            self.params_dtype = value
        else:
            setattr(self, key, value)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()
        if self.enable_mtp_magic_send:
            assert self.num_nextn_predict_layers == 1, (
                "enable_mtp_magic_send only supports num_nextn_predict_layers=1"
            )
            assert self.pipeline_model_parallel_size > 1, (
                "enable_mtp_magic_send requires pipeline_model_parallel_size > 1"
            )
            if (
                self.virtual_pipeline_model_parallel_size is not None
                and self.virtual_pipeline_model_parallel_size > 1
            ):
                assert self.overlap_p2p_comm, (
                    "enable_mtp_magic_send with vpp requires overlap_p2p_comm=True"
                )
                assert self.variable_seq_lengths, (
                    "enable_mtp_magic_send with vpp requires variable_seq_lengths=True"
                )

        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.swa_head_dim is None:
            self.swa_head_dim = self.head_dim
        if self.swa_v_head_dim is None:
            self.swa_v_head_dim = self.v_head_dim
        if self.swa_num_attention_heads is None:
            self.swa_num_attention_heads = self.num_attention_heads
        if self.swa_num_key_value_heads is None:
            self.swa_num_key_value_heads = self.num_key_value_heads
        if self.swa_rope_theta is None:
            self.swa_rope_theta = self.rope_theta

        if self.vha_q_lora_rank is None:
            self.vha_q_lora_rank = self.head_dim

        if self.swa_vha_q_lora_rank is None:
            self.swa_vha_q_lora_rank = self.swa_head_dim

        if self.num_key_value_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        # Set the embedding init method
        if self.embedding_init_method_std is None:
            # By default, use the same init std as you use for every other non-output layer.
            self.embedding_init_method_std = self.init_method_std

        if self.embedding_init_method is None:
            if self.init_method is None or (
                self.embedding_init_method_std != self.init_method_std
            ):
                # In this case, we set both the init method and the embedding init method to
                #  whatever std value requested (or defaulted) for the embedding_init_layer
                self.embedding_init_method = init_method_normal(
                    self.embedding_init_method_std
                )
            else:
                # Replicate the current behavior where if you are not changing the std of the
                #  embedding init differently and the init method is set, we fallback to the
                #  init method for this layer. Since we are here after an OR we know that
                #  init_method is not None
                self.embedding_init_method = self.init_method

        if self.magic_init:
            if self.hidden_size == 0:
                raise ValueError(
                    "hidden_size must be non-zero when magic_init is True."
                )
            sigma = math.sqrt(0.3333 / self.hidden_size)
            self.init_method = get_magic_init_method(sigma)
            self.init_method_std = sigma
        elif self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if (
            self.first_k_dense_replace
            and self.moe_layer_freq is not None
            and not isinstance(self.moe_layer_freq, int)
        ):
            raise ValueError(
                "Cannot specify both first_k_dense_replace and moe_layer_freq."
            )
        if self.first_k_dense_replace is None and self.moe_layer_freq is None:
            self.moe_layer_freq = 1
        if self.first_k_dense_replace:
            if self.moe_layer_freq:
                moe_layer_pattern = [
                    1 if ((i + 1) % self.moe_layer_freq == 0) else 0
                    for i in range(
                        self.num_hidden_layers - self.first_k_dense_replace
                    )
                ]
            else:
                moe_layer_pattern = [1] * (
                    self.num_hidden_layers - self.first_k_dense_replace
                )
            self.moe_layer_freq = [
                0
            ] * self.first_k_dense_replace + moe_layer_pattern
        if self.recompute_granularity == "":
            self.recompute_granularity = None

        # recompute config check
        if self.recompute_granularity is not None:
            assert self.recompute_granularity in ["full", "selective"], (
                "recompute_granularity must be one of full and selective"
            )
            if self.recompute_granularity == "full":
                assert self.recompute_method in [
                    "block",
                    "first_n",
                    "uniform",
                ], (
                    "when recompute_granularity=full, recompute_method must be one of block, first_n and uniform"
                )
                assert self.recompute_num_layers is not None, (
                    "when recompute_granularity=full, recompute_num_layers mustn't be None"
                )
            elif self.recompute_granularity == "selective":
                assert self.recompute_method in ["block", "first_n", None], (
                    "when recompute_granularity=selective, recompute_method must be one of block and first_n"
                )
                assert self.recompute_modules is not None
            else:
                raise ValueError(
                    "recompute_granularity must be one of full and selective"
                )

        if self.magic_init:
            self.output_layer_init_method = self.init_method
        elif self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std,
                self.num_hidden_layers,
                multiplier=2.0 if not self.is_hybrid_model else 1.0,
            )

        # Set the embedding init method
        if self.embedding_init_method_std is None:
            # By default, use the same init std as you use for every other non-output layer.
            self.embedding_init_method_std = self.init_method_std

        if self.magic_init:
            self.embedding_init_method = self.init_method
            self.embedding_init_method_std = self.init_method_std
        elif self.embedding_init_method is None:
            if self.init_method is None or (
                self.embedding_init_method_std != self.init_method_std
            ):
                # In this case, we set both the init method and the embedding init method to
                #  whatever std value requested (or defaulted) for the embedding_init_layer
                self.embedding_init_method = init_method_normal(
                    self.embedding_init_method_std
                )
            else:
                # Replicate the current behavior where if you are not changing the std of the
                #  embedding init differently and the init method is set, we fallback to the
                #  init method for this layer. Since we are here after an OR we know that
                #  init_method is not None
                self.embedding_init_method = self.init_method

        # DSv4 Hybrid Attention validation
        if self.experimental_attention_variant == "dsv4_hybrid":
            if self.csa_compress_ratios is None:
                raise ValueError(
                    "experimental_attention_variant='dsv4_hybrid' requires "
                    "csa_compress_ratios to be set."
                )
            if (
                len(self.csa_compress_ratios)
                != self.num_hidden_layers + self.num_nextn_predict_layers
            ):
                raise ValueError(
                    f"csa_compress_ratios length ({len(self.csa_compress_ratios)}) "
                    f"must equal num_hidden_layers ({self.num_hidden_layers + self.num_nextn_predict_layers})."
                )
            valid_ratios = {0, 4, 128}
            for i, r in enumerate(self.csa_compress_ratios):
                if r not in valid_ratios:
                    raise ValueError(
                        f"csa_compress_ratios[{i}]={r} is invalid. "
                        f"Must be one of {valid_ratios}."
                    )

            valid_tilelang_backends = {None, "attention_paddle_compat"}
            if self.csa_tilelang_backend not in valid_tilelang_backends:
                raise ValueError(
                    f"csa_tilelang_backend={self.csa_tilelang_backend!r} is invalid. "
                    f"Must be one of {valid_tilelang_backends}."
                )
            if (
                self.csa_tilelang_backend is None
                and self.csa_tilelang_enable_indexer
            ):
                raise ValueError(
                    "csa_tilelang_enable_indexer=True requires csa_tilelang_backend='attention_paddle_compat'."
                )
            if (
                getattr(self, "csa_tilelang_enable_sparse_attn", None)
                is not None
            ):
                raise ValueError(
                    "csa_tilelang_enable_sparse_attn has been removed. Use "
                    "csa_sparse_attn_backend in {'unfused', 'tilelang', 'cudnn'} "
                    "instead (unfused=non-fused Paddle, tilelang=TileLang "
                    "fwd/bwd, cudnn=FlashMLA fwd + cuDNN bwd)."
                )
            if self.csa_indexer_backend not in {"tilelang", "cudnn"}:
                raise ValueError(
                    f"csa_indexer_backend={self.csa_indexer_backend!r} is invalid. "
                    "Must be one of {'tilelang', 'cudnn'}."
                )
            if self.csa_sparse_attn_backend not in {
                "unfused",
                "tilelang",
                "cudnn",
            }:
                raise ValueError(
                    f"csa_sparse_attn_backend={self.csa_sparse_attn_backend!r} is invalid. "
                    "Must be one of {'unfused', 'tilelang', 'cudnn'}."
                )

        # Hash-based MoE routing consistency checks.
        if self.moe_n_hash_layers > 0:
            if self.actual_vocab_size is None:
                raise ValueError(
                    "actual_vocab_size must be set when moe_n_hash_layers > 0; "
                    "it is required to allocate the tid2eid lookup buffer."
                )
            if self.actual_vocab_size <= 0:
                raise ValueError(
                    f"actual_vocab_size must be positive, got "
                    f"{self.actual_vocab_size}."
                )
            if self.moe_n_hash_layers > self.num_hidden_layers:
                raise ValueError(
                    f"moe_n_hash_layers ({self.moe_n_hash_layers}) cannot exceed "
                    f"num_hidden_layers ({self.num_hidden_layers})."
                )
            if self.scoring_func not in ("softmax", "sigmoid", "sqrtsoftplus"):
                raise ValueError(
                    f"Hash routing requires scoring_func in "
                    f"{{'softmax', 'sigmoid', 'sqrtsoftplus'}}, got "
                    f"{self.scoring_func!r}."
                )
            if (
                self.num_experts_per_tok is None
                or self.num_experts_per_tok <= 0
            ):
                raise ValueError(
                    "num_experts_per_tok (top-k) must be a positive integer "
                    "when moe_n_hash_layers > 0."
                )
            if (
                self.n_routed_experts is None
                or self.n_routed_experts < self.num_experts_per_tok
            ):
                raise ValueError(
                    f"n_routed_experts ({self.n_routed_experts}) must be >= "
                    f"num_experts_per_tok ({self.num_experts_per_tok}) "
                    f"when moe_n_hash_layers > 0."
                )

        if self.window_attn_skip_freq is not None:
            if (
                isinstance(self.window_attn_skip_freq, int)
                and self.window_attn_skip_freq <= 0
            ):
                raise ValueError(
                    f"window_attn_skip_freq must be a positive integer when "
                    f"specified as int, but got {self.window_attn_skip_freq}."
                )

        if (
            self.num_nextn_predict_layers > 0
            and self.window_attn_skip_freq is not None
        ):
            if not isinstance(self.window_attn_skip_freq, list):
                raise TypeError(
                    f"window_attn_skip_freq must be a list of length "
                    f"num_hidden_layers + num_nextn_predict_layers "
                    f"({self.num_hidden_layers} + {self.num_nextn_predict_layers} = "
                    f"{self.num_hidden_layers + self.num_nextn_predict_layers}) "
                    f"when num_nextn_predict_layers > 0, "
                    f"but got {type(self.window_attn_skip_freq).__name__} instead."
                )
            if (
                len(self.window_attn_skip_freq)
                != self.num_hidden_layers + self.num_nextn_predict_layers
            ):
                raise ValueError(
                    f"self.window_attn_skip_freq ({len(self.window_attn_skip_freq)}) "
                    f"must equal num_hidden_layers + num_nextn_predict_layers ({self.num_hidden_layers + self.num_nextn_predict_layers})."
                )

        if (
            self.num_nextn_predict_layers == 0
            and self.window_attn_skip_freq is not None
        ):
            if (
                isinstance(self.window_attn_skip_freq, list)
                and len(self.window_attn_skip_freq) != self.num_hidden_layers
            ):
                raise ValueError(
                    f"self.window_attn_skip_freq ({len(self.window_attn_skip_freq)}) "
                    f"must equal num_hidden_layers ({self.num_hidden_layers})."
                )

        if not (0.0 <= self.head_wise_swa_ratio <= 1.0):
            raise ValueError(
                f"head_wise_swa_ratio must be between 0.0 and 1.0, "
                f"but got {self.head_wise_swa_ratio}."
            )

        # Multimax validation + grep-friendly confirmation banner.
        # Operators can verify the setting reached the model with:
        #   grep MULTIMAX <train.log>
        import warnings as _warnings

        _multimax = getattr(self, "multimax_modules", None)
        # YAML entry path returns OmegaConf containers (ListConfig), not
        # builtin list. Normalize to a plain Python list before any
        # isinstance(_multimax, list) check; otherwise the recommended
        # `multimax_modules: [lm_head]` form is rejected.
        try:
            from omegaconf import (
                ListConfig as _ListConfig,
                OmegaConf as _OmegaConf,
            )

            if isinstance(_multimax, _ListConfig):
                _multimax = _OmegaConf.to_container(_multimax, resolve=True)
                self.multimax_modules = _multimax
        except ImportError:
            pass
        # Allow yaml/json to leave the field unset, set to ``null``, pass an
        # empty string, or pass an empty list -- all map to the canonical
        # disabled sentinel ``None``.
        if _multimax in ("", []):
            _multimax = None
            self.multimax_modules = None
        # Back-compat: a plain string is treated as a single-element list
        # so older configs (multimax_modules: lm_head) keep working.
        if isinstance(_multimax, str):
            _multimax = [_multimax]
            self.multimax_modules = _multimax
        if _multimax is not None:
            if not isinstance(_multimax, list) or not all(
                isinstance(x, str) for x in _multimax
            ):
                raise ValueError(
                    f"multimax_modules must be None or a list[str], "
                    f"got {_multimax!r}."
                )
            _valid = {"lm_head", "attention"}
            _bad = [x for x in _multimax if x not in _valid]
            if _bad:
                raise ValueError(
                    f"multimax_modules entries must each be one of "
                    f"{sorted(_valid)}, got invalid entries {_bad!r} "
                    f"in {_multimax!r}."
                )
            if "attention" in _multimax:
                _warnings.warn(
                    f"[MULTIMAX-CONFIG] multimax_modules={_multimax}: "
                    "'attention' branch is not implemented yet; only the "
                    "lm_head modulation will take effect."
                )
            _warnings.warn(f"[MULTIMAX-CONFIG] multimax_modules={_multimax}")
