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
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import paddle

logger = logging.getLogger(__name__)


@dataclass
class GPTModelEstimator:
    """
    Estimator for GPT-like model parameters and computational metrics.

    This class estimates:
    - Model parameters number
    - Activation parameters number per forward pass
    - FLOPs per token and per training step

    Features:
    - Supports both dense and MoE (Mixture of Experts) architectures
    - Supports multiple attention configurations (MHA, GQA, MLA)
    - Supports Multi-Token Prediction (MTP) layers
    """

    # Model
    seq_length: int = 0
    vocab_size: int = 0
    untie_embeddings_and_output_weights: bool = False
    num_hidden_layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    gated_linear_unit: bool = False

    # Attention
    num_attention_heads: int = 0
    head_dim: int = 0
    num_kv_heads: int = 0
    causal_mask: bool = False

    # MLA
    multi_latent_attention: bool = False
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_head_dim: int | None = None
    qk_pos_emb_head_dim: int | None = None
    v_head_dim: int | None = None

    # MoE parameters
    moe_layer_freq: list[int] = field(default_factory=list)
    moe_num_experts: int | None = None
    moe_intermediate_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_topk: int | None = None

    # MTP
    num_nextn_predict_layers: int | None = None

    # Training dtypes
    bf16: bool = False
    fp16: bool = False
    fp8: bool = False

    def estimate_num_parameters(self) -> tuple[int, int]:
        """Estimate total number of model parameters."""

        # 1. Embedding and lm_head parameters
        embedding_params = self.vocab_size * self.hidden_size
        if self.untie_embeddings_and_output_weights:
            embedding_params *= 2

        # 2. Transformer layers
        def dense_mlp_params() -> int:
            """Estimate MLP layer parameters."""
            scale_factor = 3 if self.gated_linear_unit else 2
            return scale_factor * self.hidden_size * self.intermediate_size

        def moe_params(only_activated: bool) -> int:
            """Estimate MoE layer parameters."""
            params = 0
            # Router
            params += self.hidden_size * self.moe_num_experts
            # Routed Experts
            scale_factor = 3 if self.gated_linear_unit else 2
            if only_activated:
                params += (
                    scale_factor
                    * self.hidden_size
                    * self.moe_intermediate_size
                    * self.moe_topk
                )
            else:
                params += (
                    scale_factor
                    * self.hidden_size
                    * self.moe_intermediate_size
                    * self.moe_num_experts
                )
            # Shared Experts
            if self.moe_shared_expert_intermediate_size is not None:
                params += (
                    scale_factor
                    * self.hidden_size
                    * self.moe_shared_expert_intermediate_size
                )
            return params

        def attention_params() -> int:
            """Estimate attention layer parameters."""
            params = 0
            if self.multi_latent_attention:
                ### MLA
                # Q projection
                if self.q_lora_rank is None:
                    params_q_proj = (
                        self.hidden_size
                        * self.num_attention_heads
                        * (self.qk_head_dim + self.qk_pos_emb_head_dim)
                    )
                else:
                    params_q_down_proj = self.hidden_size * self.q_lora_rank
                    params_q_up_proj = (
                        self.q_lora_rank
                        * self.num_attention_heads
                        * self.qk_head_dim
                    )
                    params_q_rope = (
                        self.q_lora_rank
                        * self.num_attention_heads
                        * self.qk_pos_emb_head_dim
                    )
                    params_q_proj = (
                        params_q_down_proj + params_q_up_proj + params_q_rope
                    )
                params += params_q_proj
                # KV projection
                params_kv_down_proj = self.hidden_size * self.kv_lora_rank
                params_kv_up_proj = (
                    self.kv_lora_rank
                    * self.num_attention_heads
                    * (self.qk_head_dim + self.v_head_dim)
                )
                params_k_rope = self.hidden_size * self.qk_pos_emb_head_dim
                params += (
                    params_kv_down_proj + params_kv_up_proj + params_k_rope
                )
                # Output projection
                params += (
                    self.num_attention_heads
                    * self.v_head_dim
                    * self.hidden_size
                )
            else:
                ### MHA / GQA
                # Q projections
                params += (
                    self.hidden_size * self.num_attention_heads * self.head_dim
                )
                # K, V projections
                params += (
                    2 * self.num_kv_heads * self.hidden_size * self.head_dim
                )
                # Output projection
                params += (
                    self.num_attention_heads * self.head_dim * self.hidden_size
                )

            return params

        num_moe_layers = sum(self.moe_layer_freq)
        total_params = embedding_params + (
            (self.num_hidden_layers - num_moe_layers) * dense_mlp_params()
            + num_moe_layers * moe_params(only_activated=False)
            + self.num_hidden_layers * attention_params()
        )
        activated_params = embedding_params + (
            (self.num_hidden_layers - num_moe_layers) * dense_mlp_params()
            + num_moe_layers * moe_params(only_activated=True)
            + self.num_hidden_layers * attention_params()
        )

        return total_params, activated_params

    def estimate_flops_per_token(self) -> float:
        """Estimate FLOPs per token (forward + backward)."""
        num_moe_layers = sum(self.moe_layer_freq)
        num_dense_layers = self.num_hidden_layers - num_moe_layers

        if self.num_nextn_predict_layers is not None:
            last_layer_is_moe = self.moe_layer_freq[-1]
            num_moe_layers += last_layer_is_moe * self.num_nextn_predict_layers
            num_dense_layers += (
                1 - last_layer_is_moe
            ) * self.num_nextn_predict_layers
            num_hidden_layers = (
                self.num_hidden_layers + self.num_nextn_predict_layers
            )
        else:
            num_hidden_layers = self.num_hidden_layers

        gated_linear_multiplier = 3 / 2 if self.gated_linear_unit else 1.0

        # 1. Embedding layer (typically negligible, just lookup)
        # We skip this as it's not a matrix multiplication

        # 2. Transformer layers
        def mlp_flops() -> float:
            """Calculate FLOPs for dense MLP and MoE layer."""
            moe_shared_expert_intermediate_size = (
                self.moe_shared_expert_intermediate_size
                if self.moe_shared_expert_intermediate_size is not None
                else 0
            )
            return (
                3  # forward + backward wgrad + backward dgrad
                * 2  # mxn @ nxk matmul needs 2mnk flops
                * self.hidden_size
                * (
                    2  # ffn1 + ffn2
                    * gated_linear_multiplier
                    * (
                        # dense layers
                        (num_dense_layers * self.intermediate_size)
                        # moe routed experts
                        + (
                            num_moe_layers
                            * self.moe_intermediate_size
                            * self.moe_topk
                        )
                        # moe shared experts
                        + (num_moe_layers * moe_shared_expert_intermediate_size)
                    )
                    # moe router
                    + num_moe_layers * self.moe_num_experts
                )
            )

        def attention_flops() -> float:
            """Calculate FLOPs for attention layer."""
            if self.multi_latent_attention:
                ### MLA
                # Q projection
                if self.q_lora_rank is None:
                    q_term = (
                        self.hidden_size
                        * self.num_attention_heads
                        * (self.qk_head_dim + self.qk_pos_emb_head_dim)
                    )
                else:
                    q_term = (
                        # q_down_proj
                        self.hidden_size * self.q_lora_rank
                        # q_up_proj + q_rope
                        + self.q_lora_rank
                        * self.num_attention_heads
                        * (self.qk_head_dim + self.qk_pos_emb_head_dim)
                    )
                # KV projection
                kv_term = (
                    # kv_down_proj + k_rope
                    self.hidden_size
                    * (self.kv_lora_rank + self.qk_pos_emb_head_dim)
                    # kv_up_proj
                    + self.kv_lora_rank
                    * self.num_attention_heads
                    * (self.qk_head_dim + self.v_head_dim)
                )
                # Output projection
                out_term = (
                    self.num_attention_heads
                    * self.v_head_dim
                    * self.hidden_size
                )
                # Core Attention computation
                attn_term = (
                    self.seq_length
                    * self.num_attention_heads
                    * (
                        # QK^T
                        self.qk_head_dim
                        + self.qk_pos_emb_head_dim
                        # Attn@V
                        + self.v_head_dim
                    )
                )
                attn_term //= 2 if self.causal_mask else 1

                total_term = q_term + kv_term + out_term + attn_term
            else:
                ### MHA / GQA
                proj_term = (
                    self.hidden_size
                    * self.head_dim
                    * (
                        2 * self.num_attention_heads + 2 * self.num_kv_heads
                    )  # q_proj + out_proj  # k_proj + v_proj
                )
                attn_term = (
                    self.seq_length
                    * self.num_attention_heads
                    * self.head_dim
                    * 2
                )  # QK^T + Attn@V
                attn_term //= 2 if self.causal_mask else 1

                total_term = proj_term + attn_term

            return 3 * 2 * num_hidden_layers * total_term

        # 3. Output logits computation
        def output_logits_flops() -> float:
            """Calculate FLOPs of one token for output logits computation."""
            return (
                3
                * 2
                * self.hidden_size
                * self.vocab_size
                * (
                    1 + self.num_nextn_predict_layers
                    if self.num_nextn_predict_layers is not None
                    else 0
                )
            )

        # 4. MTP (Multi-Token Prediction) Layers
        def mtp_flops() -> float:
            """
            Calculate FLOPs of one token for MTP layers.
            Note: attention and mlp block in mtp layer have been already accounted above
            """
            if self.num_nextn_predict_layers is None:
                return 0.0
            return (
                3
                * 2
                * self.num_nextn_predict_layers
                * 2  # input projection: 2 * hidden_size -> hidden_size
                * self.hidden_size
                * self.hidden_size
            )

        return (
            mlp_flops()
            + attention_flops()
            + output_logits_flops()
            + mtp_flops()
        )

    def estimate_flops_per_step(self, batch_size: int) -> float:
        """Estimate FLOPs per training step (batch_size tokens)."""
        return batch_size * self.seq_length * self.estimate_flops_per_token()

    def estimate_mfu(self, tokens_per_second_per_gpu: float) -> float:
        """Estimate MFU (Model FLOPs Utilization)"""
        device_peak_tflops = self._get_device_peak_tflops()
        if device_peak_tflops is None:
            return 0
        return (
            tokens_per_second_per_gpu
            * self.estimate_flops_per_token()
            / 1e12  # convert to TFLOPS
            / device_peak_tflops
        )

    def _get_device_peak_tflops(self):
        """Get the peak FLOPS on the current device"""
        if not paddle.device.is_compiled_with_cuda():
            return None

        device_name = paddle.device.cuda.get_device_name().upper()
        dtype_key = "FP32_TFLOPS"
        if self.bf16:
            dtype_key = "BF16_TFLOPS"
        elif self.fp16:
            dtype_key = "FP16_TFLOPS"
        elif self.fp8:
            dtype_key = "FP8_TFLOPS"

        for spec in GPU_SPECIFICATIONS_REGISTRATION:
            if any(n in device_name for n in spec.names):
                return getattr(spec, dtype_key, None)
        logger.warning(
            f"{device_name} is not supported yet. "
            "Please register it in GPU_SPECIFICATIONS_REGISTRATION."
        )
        return None


@dataclass
class GPUSpecifications:
    """GPU specifications used for estimating mfu"""

    names: list[str]
    FP32_TFLOPS: float
    BF16_TFLOPS: float
    FP16_TFLOPS: float
    FP8_TFLOPS: float | None


GPU_SPECIFICATIONS_REGISTRATION = [
    GPUSpecifications(
        names=["A100", "A800"],
        FP32_TFLOPS=19.5,
        BF16_TFLOPS=312,
        FP16_TFLOPS=312,
        FP8_TFLOPS=None,
    ),
    GPUSpecifications(
        names=["H100", "H200", "H800"],
        FP32_TFLOPS=67,
        BF16_TFLOPS=989,
        FP16_TFLOPS=989,
        FP8_TFLOPS=1979,
    ),
    GPUSpecifications(
        names=["B200", "B300"],
        FP32_TFLOPS=75,
        BF16_TFLOPS=2200,
        FP16_TFLOPS=2200,
        FP8_TFLOPS=4500,
    ),
    GPUSpecifications(
        names=["GB200", "GB300"],
        FP32_TFLOPS=80,
        BF16_TFLOPS=2500,
        FP16_TFLOPS=2500,
        FP8_TFLOPS=5000,
    ),
]


def fill_feature(input_embeds, target_index, value):
    """
    Fill positions in `input_embeds` specified by `target_index` with the given `value`.

    Padding-Token embedding will be set to `value` to avoid gradient propagation updates.

    Args:
        input_embeds (Tensor): Input feature tensor of shape [..., D].
        target_index (Tensor): Bool index tensor specifying positions to fill.
        value (float): Scalar value to fill with.

    Returns:
        Tensor: Feature tensor with specified positions filled by `value`,
                same shape as `input_embeds`.
    """
    input_embeds_shape = input_embeds.shape
    input_embeds = input_embeds.reshape([-1, input_embeds.shape[-1]])
    indices = paddle.nonzero(target_index.flatten()).flatten()
    assert not isinstance(value, paddle.Tensor), type(value)
    if input_embeds.size > 0 and indices.size > 0:
        input_embeds[indices] = value
    input_embeds = input_embeds.reshape(input_embeds_shape)
    return input_embeds
