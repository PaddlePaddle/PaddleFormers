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
DeepSeek Sparse Attention (DSA) extension.

This module provides:
  - DSAIndexer: Token scoring module that selects top-k relevant positions
  - DSAttention: Core attention component with DSA support (pluggable)
  - FusedDSAIndexerLoss: Fused KL-divergence loss with full manual backward
  - DSAIndexerLossAutoScaler: Loss scaling helper
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddleformers.fleet import parallel_state
from paddleformers.fleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddleformers.fleet.packed_seq_params import PackedSeqParams
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )


logger = logging.getLogger(__name__)


def hadamard_transform(x: Tensor, scale: float = 1.0) -> Tensor:
    """Fast Walsh-Hadamard Transform using the butterfly algorithm.

    Pure Paddle implementation, equivalent to:
        F.linear(x, hadamard_matrix(dim)) * scale

    Uses O(N log N) butterfly operations instead of O(N^2) matrix multiply.
    The Hadamard matrix is symmetric and orthogonal, so backward is the same
    transform applied to grad_output (handled automatically by Paddle autograd).

    Reference:
        - fast-hadamard-transform (Tri Dao): csrc/fast_hadamard_transform_cuda.cu
        - PaddleFormers/paddleformers/quantization/hadamard_utils.py (matmul_hadU)

    Args:
        x: Input tensor of shape (..., dim). dim must be a power of 2.
        scale: Scaling factor applied to the output.

    Returns:
        Hadamard-transformed tensor of the same shape.
    """
    original_shape = x.shape
    output_dtype = x.dtype
    dim = original_shape[-1]
    assert dim > 0 and (dim & (dim - 1)) == 0, (
        f"hadamard_transform requires dim to be a power of 2, got {dim}"
    )

    # Megatron uses fast_hadamard_transform, whose bf16 path accumulates in fp32
    # and casts back to bf16. Keep the same numeric contract here.
    x = x.cast("float32")

    # Flatten batch dims: (..., dim) -> (batch, dim)
    x = x.reshape([-1, dim])

    h = 1
    while h < dim:
        x = x.reshape([-1, dim // (2 * h), 2, h])
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = paddle.stack([a + b, a - b], axis=2)
        x = x.reshape([-1, dim])
        h *= 2

    return (x.reshape(original_shape) * scale).cast(output_dtype)


def rotate_activation(x: Tensor) -> Tensor:
    """Apply Hadamard rotation activation.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert x.dtype == paddle.bfloat16, (
        f"rotate_activation only support bf16 input, but got {x.dtype}"
    )
    hidden_size = x.shape[-1]
    return hadamard_transform(x, scale=hidden_size**-0.5)


# ---------------------------------------------------------------------------
# Unfused DSA attention (explicit bmm, supports asymmetric Q/K vs V dims)
# ---------------------------------------------------------------------------
def _unfused_dsa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    combined_mask: Tensor | None,
    softmax_scale: float,
) -> Tensor:
    """Unfused DSA sparse attention

    Uses explicit bmm instead of flash attention to support:
    - Different Q/K head_dim vs V head_dim (MLA architecture)
    - Arbitrary per-token sparse masks from DSA Indexer

    Args:
        query: [b, s, nhpp, qk_head_dim]
        key:   [b, s, nhpp, qk_head_dim]
        value: [b, s, nhpp, v_head_dim]   (v_head_dim may differ from qk_head_dim)
        combined_mask: [b, 1, s, s]  (causal + sparse index mask, -inf for masked)
        softmax_scale: 1/sqrt(qk_head_dim)

    Returns:
        output: [b, s, nhpp * v_head_dim]
    """
    b, s, nhpp, qk_hd = query.shape
    v_hd = value.shape[-1]

    # Reshape for bmm: [b*nhpp, s, hd]
    q = query.transpose([0, 2, 1, 3]).reshape([b * nhpp, s, qk_hd])
    k = key.transpose([0, 2, 1, 3]).reshape([b * nhpp, s, qk_hd])
    v = value.transpose([0, 2, 1, 3]).reshape([b * nhpp, s, v_hd])

    # Q * K^T with scale: [b*nhpp, s, s]
    attn_scores = (
        paddle.bmm(q.cast("float32"), k.cast("float32").transpose([0, 2, 1]))
        * softmax_scale
    )

    # Apply combined mask (causal + sparse index mask)
    if combined_mask is not None:
        mask = (
            combined_mask.expand([b, nhpp, s, s])
            .contiguous()
            .reshape([b * nhpp, s, s])
        )
        attn_scores = attn_scores + mask.cast("float32")

    attn_weights = F.softmax(attn_scores, axis=-1)

    # Attention_weights * V: [b*nhpp, s, v_hd]
    output = paddle.bmm(attn_weights.cast(v.dtype), v)

    # [b*nhpp, s, v_hd] -> [b, s, nhpp*v_hd]
    output = (
        output.reshape([b, nhpp, s, v_hd])
        .transpose([0, 2, 1, 3])
        .reshape([b, s, nhpp * v_hd])
    )

    return output


def _normalize_dsa_mask(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 4:
        assert mask.shape[1] == 1, "DSA mask must have singleton head dimension"
        mask = mask.squeeze(1)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    return mask


# ---------------------------------------------------------------------------
# DSA Indexer Sublayers Spec
# ---------------------------------------------------------------------------
@dataclass
class DSAIndexerSublayersSpec:
    """Sublayers spec for DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: LayerSpec | type = None
    linear_wk: LayerSpec | type = None
    k_norm: LayerSpec | type = None
    linear_weights_proj: LayerSpec | type = None


@dataclass
class DSAttentionSublayersSpec:
    """Sublayers spec for DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: LayerSpec | type = None


# ---------------------------------------------------------------------------
# DSA Indexer
# ---------------------------------------------------------------------------
class DSAIndexer(paddle.nn.Layer):
    """DSA Indexer: DeepSeek Sparse Attention token selection module.

    For each query token, scores all cached key positions using a lightweight
    n_heads-head attention mechanism, then selects the top-k most relevant
    positions for the full MLA attention computation.

    Key design notes:
    - Uses non-interleaved RoPE (unlike MLA which uses interleaved)
    - Uses LayerNorm (not RMSNorm) on K
    - nope/pe split order: [nope | pe]
    - Uses ReLU-aggregated scoring across heads
    - Per-head learned importance weights via weights_proj
    - weights absorbs softmax_scale

    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: DSAIndexerSublayersSpec,
        layer_number: int,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.n_heads = config.dsa_index_n_heads
        self.head_dim = config.dsa_index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.index_topk = config.dsa_index_topk
        self.softmax_scale = self.head_dim**-0.5

        # wq_b: q_lora_rank -> n_heads * head_dim (duplicated)
        self.wq_b = build_spec_layer(
            sublayers_spec.linear_wq_b,
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=pg_collection.tp,
            tp_comm_buffer_name="dsa_indexer_wq_b",
        )

        # wk: hidden_size -> head_dim (single shared K, duplicated)
        self.wk = build_spec_layer(
            sublayers_spec.linear_wk,
            config.hidden_size,
            self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=pg_collection.tp,
            tp_comm_buffer_name="dsa_indexer_wk",
        )

        # k_norm: LayerNorm (NOT RMSNorm) per reference
        self.k_norm = build_spec_layer(
            sublayers_spec.k_norm,
            normalized_shape=self.head_dim,
            epsilon=getattr(self.config, "rms_norm_eps", 1e-5),
        )

        # weights_proj: learned per-head importance [hidden -> n_heads]
        self.weights_proj = build_spec_layer(
            sublayers_spec.linear_weights_proj,
            config.hidden_size,
            self.n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=pg_collection.tp,
            tp_comm_buffer_name="dsa_indexer_weights_proj",
        )

        # Initialize Position Embedding.
        # The indexer has its own RoPE to encode positions for the scoring mechanism.
        if config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.rope_head_dim,
                rotary_percent=1.0,
                rotary_interleaved=getattr(
                    config, "dsa_indexer_rotary_interleaved", False
                ),
                rotary_base=config.rope_theta,
                cp_group=pg_collection.cp,
            )
        elif config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.rope_head_dim,
                rotary_interleaved=getattr(
                    config, "dsa_indexer_rotary_interleaved", False
                ),
                rotary_base=config.rope_theta,
                scaling_factor=config.rotary_scaling_factor,
                original_max_position_embeddings=config.original_max_position_embeddings,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
                mscale=config.mscale,
                mscale_all_dim=config.mscale_all_dim,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {config.rope_type}, "
                "supported types are 'rope' and 'yarn'"
            )

    def _apply_rope(
        self, x: Tensor, freqs: Tensor, mscale: float = 1.0
    ) -> Tensor:
        """Apply RoPE to the pe portion of x.

        Split order: [pe | nope], matching DeepSeek-V3.2 Indexer (model.py:462).

        RoPE format is controlled by config.dsa_indexer_rotary_interleaved:
        - False (default): non-interleaved RoPE with half-head frequencies [θ₁,θ₂,...,θ₁,θ₂,...]
        - True: interleaved RoPE with paired frequencies [θ₁,θ₁,θ₂,θ₂,...]

        Args:
            x: [..., head_dim] (rope_dim + nope_dim)
            freqs: RoPE frequencies
            mscale: YaRN concentration factor (1.0 for plain RoPE, ~1.37 for YaRN)
        """
        x_pe = x[..., : self.rope_head_dim]
        x_nope = x[..., self.rope_head_dim :]
        x_pe = _apply_rotary_pos_emb_bshd(
            x_pe,
            freqs,
            rotary_interleaved=self.config.dsa_indexer_rotary_interleaved,
            multi_latent_attention=False,
            mscale=mscale,
        )
        return paddle.concat([x_pe, x_nope], axis=-1)

    def forward_before_topk(
        self,
        hidden_states: Tensor,  # [b, s, hidden_size] or [s/TP, b, hidden_size] (SP mode)
        q_latent: Tensor,  # [b, s, q_lora_rank] or [s/TP, b, q_lora_rank] (SP mode)
    ):
        """Compute q, k, weights before top-k selection.

        RoPE frequencies are computed internally from self.rotary_pos_emb.

        When sequence_parallel is enabled, inputs are seq-first sharded
        [s/TP, b, h]. This method gathers them internally (like Megatron DSA)
        and transposes to batch-first [b, s, h] before processing.
        """
        # Gather from sequence parallel region if needed
        if self.config.sequence_parallel and self.pg_collection.tp.nranks > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )
            q_latent = gather_from_sequence_parallel_region(
                q_latent, group=self.pg_collection.tp
            )
            # Transpose from seq-first [s, b, h] to batch-first [b, s, h]
            hidden_states = hidden_states.transpose([1, 0, 2])
            q_latent = q_latent.transpose([1, 0, 2])

        bsz, seqlen, _ = hidden_states.shape

        # Compute RoPE internally
        rotary_seq_len = seqlen
        if self.config.rope_type == "rope":
            freqs = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            freqs, mscale = self.rotary_pos_emb(
                rotary_seq_len, packed_seq=False
            )

        q, _ = self.wq_b(q_latent)  # [b, s, n_heads * head_dim]
        q = q.reshape([bsz, seqlen, self.n_heads, self.head_dim])
        q = self._apply_rope(q, freqs, mscale)

        k, _ = self.wk(hidden_states)  # [b, s, head_dim]
        k = self.k_norm(k)
        k = self._apply_rope(k.unsqueeze(2), freqs, mscale).squeeze(2)

        # Rotate activation (Hadamard transform)
        q = rotate_activation(q)
        k = rotate_activation(k)

        weights, _ = self.weights_proj(hidden_states)
        weights = weights * (self.n_heads**-0.5) * self.softmax_scale

        return q, k, weights

    def compute_index_scores(
        self,
        q: Tensor,  # [b, s, n_heads, head_dim]
        k: Tensor,  # [b, t, head_dim]
        weights: Tensor,  # [b, s, n_heads]
        mask: Tensor | None = None,
    ):
        """Compute index scores and select top-k."""
        q_fp32 = q.cast("float32")
        k_fp32 = k.cast("float32")

        scores = paddle.einsum("bshd,btd->bsht", q_fp32, k_fp32)
        index_scores = (weights.unsqueeze(-1) * F.relu(scores)).sum(axis=2)

        if mask is not None:
            index_scores = index_scores + _normalize_dsa_mask(mask)

        topk_k = min(self.index_topk, index_scores.shape[-1])
        topk_indices = paddle.topk(index_scores, k=topk_k, axis=-1)[1]
        # Clamp indices to valid range: paddle.topk may return garbage indices
        # for -inf input values
        topk_indices = paddle.clip(
            topk_indices, min=0, max=index_scores.shape[-1] - 1
        )

        return index_scores, topk_indices

    def forward(
        self,
        hidden_states: Tensor,
        q_latent: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute DSA token importance scores and return scores + top-k indices."""
        q, k, weights = self.forward_before_topk(hidden_states, q_latent)
        index_scores, topk_indices = self.compute_index_scores(
            q, k, weights, attention_mask
        )
        return index_scores, topk_indices


def _compute_index_scores_fused(
    q: Tensor, weights: Tensor, k: Tensor
) -> Tensor:
    """Compute index scores from Indexer outputs.

    Args:
        q:       [sq, b, h, d]  (Indexer query, after RoPE + Hadamard)
        weights: [sq, b, h]     (per-head importance weights)
        k:       [sk, b, d]     (Indexer key, after RoPE + Hadamard)

    Returns:
        index_scores: [b, sq, sk]
    """
    # q @ k^T -> [sq, b, h, sk]
    index_scores = paddle.einsum(
        "sbhd,tbd->sbht", q.cast("float32"), k.cast("float32")
    )
    # ReLU activation
    index_scores = F.relu(index_scores)
    # Weight each head: [sq, b, h, sk] * [sq, b, h, 1] -> [sq, b, h, sk]
    index_scores = index_scores * weights.unsqueeze(-1)
    # Sum across heads: [sq, b, h, sk] -> [sq, b, sk]
    index_scores = index_scores.sum(axis=2)
    # Transpose to [b, sq, sk]
    index_scores = index_scores.transpose([1, 0, 2])
    return index_scores


def fused_qk_topk_naive(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk: int,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute index scores and select top-k indices (batch-first interface).

    This is the standalone equivalent of Megatron's fused_qk_topk_naive,
    operating on batch-first tensors for CSA compatibility.

    Args:
        q: [b, sq, n_heads, head_dim] — Indexer query (after RoPE + Hadamard)
        k: [b, sk, head_dim] — Indexer key (after RoPE + Hadamard)
        weights: [b, sq, n_heads] — Per-head importance weights (pre-scaled)
        index_topk: Number of top-k positions to select
        mask: Optional [b, sq, sk] mask with -inf for masked positions

    Returns:
        index_scores: [b, sq, sk]
        topk_indices: [b, sq, topk]
    """
    q_fp32 = q.cast("float32")
    k_fp32 = k.cast("float32")

    # q @ k^T: [b, sq, n_heads, head_dim] x [b, sk, head_dim] -> [b, sq, n_heads, sk]
    scores = paddle.einsum("bshd,btd->bsht", q_fp32, k_fp32)
    # ReLU activation
    scores = F.relu(scores)
    # Weight each head and sum: [b, sq, n_heads, sk] * [b, sq, n_heads, 1] -> sum -> [b, sq, sk]
    index_scores = (scores * weights.unsqueeze(-1)).sum(axis=2)

    if mask is not None:
        index_scores = index_scores + mask

    topk_k = min(index_topk, index_scores.shape[-1])
    topk_values, topk_indices = paddle.topk(index_scores, k=topk_k, axis=-1)
    topk_indices = paddle.clip(
        topk_indices, min=0, max=index_scores.shape[-1] - 1
    )
    # Mark indices whose scores are -inf as invalid (-1). This happens when
    # a document-aware mask blocks cross-document compressed positions.
    # The tilelang kernel handles this internally, but the naive path needs
    # explicit invalidation so downstream sparse attention ignores them.
    invalid_topk = paddle.isinf(topk_values) & (topk_values < 0)
    topk_indices = paddle.where(
        invalid_topk,
        paddle.full_like(topk_indices, -1),
        topk_indices,
    )

    return index_scores, topk_indices


def _compute_dsa_indexer_loss(
    index_scores: Tensor,
    topk_indices: Tensor,
    query: Tensor,
    key: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    tp_group,
    causal_mask_override: Tensor | None = None,
) -> Tensor:
    """Compute KL divergence loss between index_scores and true attention_scores.

    Args:
        index_scores: [b, sq, sk]
        topk_indices: [b, sq, topk]
        query: [sq, b, np, hn]  (MLA query, DETACHED)
        key:   [sk, b, np, hn]  (MLA key, DETACHED)
        softmax_scale: Scale coefficient after q @ k^T
        loss_coeff: Coefficient for the indexer KL divergence loss
        sparse_loss: Whether to apply sparse index mask
        tp_group: TP process group (or None)
        causal_mask_override: Optional [b, sq, sk] or [sq, sk] mask with -inf for
            masked positions. When provided, replaces the standard triangular causal mask.
            Used by CSA where the mask shape differs from standard causal.

    Returns:
        indexer_loss: scalar
    """
    sq, b, np, hn = query.shape
    sk = key.shape[0]

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.transpose([1, 2, 0, 3]).reshape([b * np, sq, hn])
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.transpose([1, 2, 3, 0]).reshape([b * np, hn, sk])
    # Compute attention scores [b * np, sq, sk]
    attention_scores = (
        paddle.bmm(query_reshaped.cast("float32"), key_reshaped.cast("float32"))
        * softmax_scale
    )
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape([b, np, sq, sk])

    # causal_mask [sq, sk] or [b, sq, sk]
    if causal_mask_override is not None:
        causal_mask = causal_mask_override.cast("float32")
    else:
        causal_mask = paddle.triu(
            paddle.full([sq, sk], float("-inf"), dtype="float32"),
            diagonal=1,
        )
    # index_mask [b, sq, sk]
    index_mask = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    index_mask = paddle.put_along_axis(
        index_mask,
        topk_indices,
        paddle.zeros_like(topk_indices, dtype="float32"),
        axis=-1,
    )

    # Apply causal mask
    if causal_mask.ndim == 3:
        attention_scores = attention_scores + causal_mask.unsqueeze(1)
    else:
        attention_scores = attention_scores + causal_mask.reshape(
            [1, 1, sq, sk]
        )
    if sparse_loss:
        attention_scores = attention_scores + index_mask.reshape([b, 1, sq, sk])
        index_scores = index_scores + index_mask

    # Handle fully-masked rows (all -inf) to prevent NaN in softmax
    if causal_mask_override is not None:
        if causal_mask.ndim == 2:
            row_valid = (causal_mask > float("-inf")).any(axis=-1)  # [sq]
            attn_row_mask = row_valid.reshape([1, 1, sq, 1])
            idx_row_mask = row_valid.reshape([1, sq, 1])
        else:
            row_valid = (causal_mask > float("-inf")).any(axis=-1)  # [b, sq]
            attn_row_mask = row_valid.reshape([b, 1, sq, 1])
            idx_row_mask = row_valid.reshape([b, sq, 1])

        attention_scores = paddle.where(
            attn_row_mask, attention_scores, paddle.zeros_like(attention_scores)
        )
        index_scores = paddle.where(
            idx_row_mask, index_scores, paddle.zeros_like(index_scores)
        )

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = F.softmax(attention_scores, axis=-1, dtype="float32")
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = F.softmax(index_scores, axis=-1, dtype="float32")

    # Zero out invalid rows after softmax so they contribute nothing to loss
    if causal_mask_override is not None:
        attention_scores = attention_scores * attn_row_mask.cast("float32")
        index_scores = index_scores * idx_row_mask.cast("float32")

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores = attention_scores.sum(axis=1)
    if tp_group is not None and tp_group.nranks > 1:
        paddle.distributed.all_reduce(
            attention_scores.contiguous(), group=tp_group
        )
    # L1 normalize target on the last dimension
    attention_scores = attention_scores / attention_scores.sum(
        axis=-1, keepdim=True
    ).clip(min=1e-10)

    # KL divergence: KL(target || index) = target * log(target / index)
    kl_per_element = attention_scores * (
        paddle.log(attention_scores + 1e-10) - paddle.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    kl_div = kl_per_element.sum(axis=-1).mean()
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


def _bwd_fused_indexer_loss(
    q: Tensor,
    weights: Tensor,
    k: Tensor,
    query: Tensor,
    key: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    grad_loss: Tensor,
    tp_group,
    causal_mask_override: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Manual backward for fused indexer loss.

    All tensor layouts (sequence-first):
        q:       [sq, b, h, d]
        weights: [sq, b, h]
        k:       [sk, b, d]
        query:   [sq, b, np, hn]  (MLA query)
        key:     [sk, b, np, hn]  (MLA key)

    Returns:
        grad_q:       [sq, b, h, d]
        grad_weights: [sq, b, h]
        grad_k:       [sk, b, d]
    """
    # Recompute index_scores from (q, weights, k)
    index_scores = _compute_index_scores_fused(q, weights, k)  # [b, sq, sk]

    sq, b, np, hn = query.shape
    sk = key.shape[0]

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.transpose([1, 2, 0, 3]).reshape([b * np, sq, hn])
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.transpose([1, 2, 3, 0]).reshape([b * np, hn, sk])
    # Compute attention scores [b * np, sq, sk]
    attention_scores = (
        paddle.bmm(query_reshaped.cast("float32"), key_reshaped.cast("float32"))
        * softmax_scale
    )
    del query_reshaped, key_reshaped

    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape([b, np, sq, sk])

    # causal_mask [sq, sk] or [b, sq, sk]
    if causal_mask_override is not None:
        causal_mask = causal_mask_override.cast("float32")
    else:
        causal_mask = paddle.triu(
            paddle.full([sq, sk], float("-inf"), dtype="float32"),
            diagonal=1,
        )
    # index_mask [b, sq, sk]
    index_mask = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    index_mask = paddle.put_along_axis(
        index_mask,
        topk_indices,
        paddle.zeros_like(topk_indices, dtype="float32"),
        axis=-1,
    )

    # Apply causal mask to both attention and index scores
    if causal_mask.ndim == 3:
        attention_scores = attention_scores + causal_mask.unsqueeze(1)
        index_scores = index_scores + causal_mask
    else:
        attention_scores = attention_scores + causal_mask.reshape(
            [1, 1, sq, sk]
        )
        index_scores = index_scores + causal_mask.unsqueeze(0)

    if sparse_loss:
        attention_scores = attention_scores + index_mask.reshape([b, 1, sq, sk])
        index_scores = index_scores + index_mask

    # Handle fully-masked rows (all -inf) to prevent NaN in softmax
    if causal_mask_override is not None:
        if causal_mask.ndim == 2:
            row_valid = (causal_mask > float("-inf")).any(axis=-1)  # [sq]
            attn_row_mask = row_valid.reshape([1, 1, sq, 1])
            idx_row_mask = row_valid.reshape([1, sq, 1])
        else:
            row_valid = (causal_mask > float("-inf")).any(axis=-1)  # [b, sq]
            attn_row_mask = row_valid.reshape([b, 1, sq, 1])
            idx_row_mask = row_valid.reshape([b, sq, 1])

        attention_scores = paddle.where(
            attn_row_mask, attention_scores, paddle.zeros_like(attention_scores)
        )
        index_scores = paddle.where(
            idx_row_mask, index_scores, paddle.zeros_like(index_scores)
        )

    # Compute softmax for both
    attention_scores_softmax = F.softmax(
        attention_scores, axis=-1, dtype="float32"
    )
    del attention_scores

    index_scores_softmax = F.softmax(index_scores, axis=-1, dtype="float32")
    del index_scores

    # Zero out invalid rows after softmax so they contribute nothing to gradients
    if causal_mask_override is not None:
        attention_scores_softmax = (
            attention_scores_softmax * attn_row_mask.cast("float32")
        )
        index_scores_softmax = index_scores_softmax * idx_row_mask.cast(
            "float32"
        )

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(axis=1)
    del attention_scores_softmax

    if tp_group is not None and tp_group.nranks > 1:
        paddle.distributed.all_reduce(
            attention_scores_sum.contiguous(), group=tp_group
        )

    # L1 normalize
    attention_scores_normalized = (
        attention_scores_sum
        / attention_scores_sum.sum(axis=-1, keepdim=True).clip(min=1e-10)
    )
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss.cast("float32") * loss_coeff  # scalar

    # Backward through mean: distribute gradient equally
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    grad_kl_per_element = grad_kl_per_row.reshape([1, 1, 1]).expand([b, sq, sk])

    # Backward through kl: dkl/d_index_softmax = -target / index_softmax
    grad_index_scores_softmax = (
        -attention_scores_normalized
        / (index_scores_softmax + 1e-10)
        * grad_kl_per_element
    )
    del attention_scores_normalized

    # Backward through softmax:
    # dL/dx = softmax * (dL/d_softmax - sum(dL/d_softmax * softmax))
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(
        axis=-1, keepdim=True
    )
    grad_index_scores_logits = index_scores_softmax * (
        grad_index_scores_softmax - sum_grad
    )
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    # Zero out gradients for masked positions
    if causal_mask_override is not None:
        causal_valid_mask = causal_mask == 0
        if causal_valid_mask.ndim == 2:
            causal_valid_mask = causal_valid_mask.unsqueeze(0)
        elif causal_valid_mask.shape[0] == 1:
            causal_valid_mask = causal_valid_mask.squeeze(0).unsqueeze(0)
        causal_valid_mask = causal_valid_mask.expand([b, sq, sk])
    else:
        causal_valid_mask = (
            paddle.tril(paddle.ones([sq, sk], dtype="bool"))
            .unsqueeze(0)
            .expand([b, sq, sk])
        )
    del causal_mask

    if sparse_loss:
        index_valid_mask = index_mask == 0  # [b, sq, sk]
        del index_mask
        valid_mask = causal_valid_mask & index_valid_mask  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask
        valid_mask = causal_valid_mask  # [b, sq, sk]
    del causal_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.cast(
        "float32"
    )
    del valid_mask

    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(
        [1, 0, 2]
    )  # [sq, b, sk]
    del grad_index_scores_logits

    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    # Compute forward values needed for backward (recomputation)
    scores = paddle.einsum(
        "sbhd,tbd->sbht", q.cast("float32"), k.cast("float32")
    )  # [sq, b, h, sk]
    relu_mask = scores > 0
    scores_after_relu = F.relu(scores)
    del scores

    # Backward through multiplication by weights:
    # dL/d_weights = grad * relu_scores (sum over sk)
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(
        axis=-1
    )  # [sq, b, h]

    # dL/d_relu_scores = grad * weights
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(
        -1
    )  # [sq, b, h, sk]
    del grad_weighted_scores, scores_after_relu

    # Backward through ReLU
    grad_scores = grad_scores_after_relu * relu_mask.cast(
        "float32"
    )  # [sq, b, h, sk]
    del grad_scores_after_relu, relu_mask

    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = paddle.einsum(
        "sbht,tbd->sbhd", grad_scores, k.cast("float32")
    )  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = paddle.einsum(
        "sbht,sbhd->tbd", grad_scores, q.cast("float32")
    )  # [sk, b, d]
    del grad_scores

    return (
        grad_q.cast(q.dtype),
        grad_weights.cast(weights.dtype),
        grad_k.cast(k.dtype),
    )


class FusedDSAIndexerLoss(paddle.autograd.PyLayer):
    """Fused DSA Indexer Loss: index_scores + topk + KL loss + full manual backward."""

    _last_topk_indices: Tensor | None = None

    @staticmethod
    def forward(
        ctx,
        q: Tensor,  # [sq, b, h, d]  — Indexer query output
        weights: Tensor,  # [sq, b, h]     — Indexer per-head weights
        k: Tensor,  # [sk, b, d]     — Indexer key output
        query: Tensor,  # [sq, b, np, hn] — MLA query (DETACHED)
        key: Tensor,  # [sk, b, np, hn] — MLA key (DETACHED)
        # Non-tensor params follow (stored on ctx, not in backward returns)
        softmax_scale: float = 1.0,
        topk: int = 64,
        loss_coeff: float = 1.0,
        mask: Tensor | None = None,
        sparse_loss: bool = True,
        tp_group=None,
    ) -> Tensor:
        """Fused forward: compute index_scores, topk, and KL loss.

        Args:
            q:       Indexer query after RoPE+Hadamard [sq, b, h, d]
            weights: Per-head importance weights [sq, b, h]
            k:       Indexer key after RoPE+Hadamard [sk, b, d]
            query:   MLA query (detached) [sq, b, np, hn]
            key:     MLA key (detached) [sk, b, np, hn]
            softmax_scale: MLA attention softmax scale
            topk:    Number of top-k indices to select
            loss_coeff: Coefficient for KL loss
            mask:    Optional mask for index_scores [b, 1, sq, sk] or [1, 1, sq, sk]
            sparse_loss: Whether to use sparse index mask in loss
            tp_group: TP process group (or None)

        Returns:
            indexer_loss: scalar KL divergence loss
        """
        with paddle.amp.auto_cast(False):
            # Step 1: Compute index_scores from (q, weights, k)
            index_scores = _compute_index_scores_fused(
                q, weights, k
            )  # [b, sq, sk]

            # Step 2: Apply mask and select topk
            mask = _normalize_dsa_mask(mask)
            if mask is not None:
                masked_scores = index_scores + mask
            else:
                masked_scores = index_scores
            topk_k = min(topk, masked_scores.shape[-1])
            topk_values, topk_indices = paddle.topk(
                masked_scores, k=topk_k, axis=-1
            )
            # Clamp indices to valid range: paddle.topk may return garbage indices
            # for -inf input values
            topk_indices = paddle.clip(
                topk_indices, min=0, max=masked_scores.shape[-1] - 1
            )
            # Mark indices whose scores are -inf as invalid (-1). This happens
            # when a document-aware mask blocks cross-document compressed
            # positions.
            invalid_topk = paddle.isinf(topk_values) & (topk_values < 0)
            topk_indices = paddle.where(
                invalid_topk,
                paddle.full_like(topk_indices, -1),
                topk_indices,
            )

            FusedDSAIndexerLoss._last_topk_indices = topk_indices.detach()

            # Step 3: Compute KL loss (use masked_scores)
            indexer_loss = _compute_dsa_indexer_loss(
                masked_scores,
                topk_indices,
                query,
                key,
                softmax_scale,
                loss_coeff,
                sparse_loss,
                tp_group,
                causal_mask_override=mask,
            )

        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.tp_group = tp_group
        ctx.causal_mask_override = mask

        return indexer_loss

    @staticmethod
    def backward(ctx, grad_loss: Tensor):
        """Backward: recompute and manually backprop to (q, weights, k).

        Returns 6 gradients for the 6 Tensor inputs to forward:
            q, weights, k, query, key, mask
        (Paddle PyLayer only counts Tensor params, not float/int/bool/None.)
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensor()

        with paddle.amp.auto_cast(False):
            grad_q, grad_weights, grad_k = _bwd_fused_indexer_loss(
                q,
                weights,
                k,
                query,
                key,
                topk_indices,
                ctx.softmax_scale,
                ctx.loss_coeff,
                ctx.sparse_loss,
                grad_loss,
                ctx.tp_group,
                causal_mask_override=ctx.causal_mask_override,
            )

        return grad_q, grad_weights, grad_k, None, None, None


class DSAIndexerLossAutoScaler(paddle.autograd.PyLayer):
    """Attaches indexer_loss to the backward graph without changing output value."""

    _main_loss_backward_scale: Tensor | None = None

    @staticmethod
    def forward(ctx, output: Tensor, indexer_loss: Tensor) -> Tensor:
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (indexer_loss,) = ctx.saved_tensor()
        scale = DSAIndexerLossAutoScaler._main_loss_backward_scale
        if scale is None:
            scale = paddle.ones([1], dtype=indexer_loss.dtype)
        scaled_grad = paddle.ones_like(indexer_loss) * scale
        return grad_output, scaled_grad

    @staticmethod
    def set_loss_scale(scale: Tensor):
        DSAIndexerLossAutoScaler._main_loss_backward_scale = scale


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses across layers and ranks."""

    tracker = {}
    num_layers = None

    @staticmethod
    def get_total_num_layers(config):
        mtp_num_layers = getattr(config, "mtp_num_layers", 0) or 0
        nextn_num_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
        return config.num_hidden_layers + (mtp_num_layers or nextn_num_layers)

    @staticmethod
    def register_total_num_layers(config):
        DSAIndexerLossLoggingHelper.num_layers = (
            DSAIndexerLossLoggingHelper.get_total_num_layers(config)
        )

    @staticmethod
    def save_loss_to_tracker(
        loss: Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group=None,
        avg_group=None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor (scalar).
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = paddle.zeros([num_layers])
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def _infer_num_layers(num_layers: int | None = None):
        if num_layers is not None:
            return num_layers
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            return tracker["values"].shape[0]
        return DSAIndexerLossLoggingHelper.num_layers

    @staticmethod
    def reduce_loss_in_tracker(num_layers: int | None = None):
        """Collect and reduce the indexer losses across ranks.

        PP all-reduce must be called on every rank in the pipeline group.
        Ranks without local indexer layers lazily create a zero tracker so they
        still participate in the collective and do not hang other ranks.
        """
        tracker = DSAIndexerLossLoggingHelper.tracker
        num_layers = DSAIndexerLossLoggingHelper._infer_num_layers(num_layers)
        if "values" not in tracker:
            if num_layers is None:
                return
            tracker["values"] = paddle.zeros([num_layers])
            tracker["reduce_group"] = None
            tracker["avg_group"] = None
        values = tracker["values"]

        # PP all-reduce
        pp_group = parallel_state.get_pipeline_model_parallel_group(
            check_initialized=False
        )
        if pp_group is not None and pp_group.nranks > 1:
            paddle.distributed.all_reduce(values, group=pp_group)

        # TP reduce
        if tracker.get("reduce_group") is not None:
            paddle.distributed.all_reduce(values, group=tracker["reduce_group"])

        # CP avg
        if tracker.get("avg_group") is not None:
            paddle.distributed.all_reduce(values, group=tracker["avg_group"])
            values /= tracker["avg_group"].nranks

        # DP avg
        dp_group = parallel_state.get_data_parallel_group(
            check_initialized=False
        )
        if dp_group is not None and dp_group.nranks > 1:
            paddle.distributed.all_reduce(values, group=dp_group)
            values /= dp_group.nranks

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer=None,
        total_loss_dict: dict | None = None,
        num_layers: int | None = None,
        csa_compress_ratios: list[int] | None = None,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss (e.g. 1/num_microbatches).
            iteration: Current training iteration.
            writer: TensorBoard writer (optional).
            total_loss_dict: Dictionary to accumulate total losses (optional).
            num_layers: Total number of layers with indexer metrics.
            csa_compress_ratios: Per-layer CSA compress ratios.
        """
        num_layers = DSAIndexerLossLoggingHelper._infer_num_layers(num_layers)
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
            num_layers=num_layers
        )
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        if csa_compress_ratios is not None:
            num_indexer_layers = sum(
                1 for ratio in csa_compress_ratios if ratio == 4
            )
        else:
            num_indexer_layers = indexer_loss_values.shape[0]
        if num_indexer_layers == 0:
            DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
            return
        avg_indexer_loss = indexer_loss_values.sum() / num_indexer_layers

        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar(
                "indexer loss", avg_indexer_loss.item(), iteration
            )

        logger.info(
            "Iteration %d | indexer loss: %.6f",
            iteration,
            avg_indexer_loss.item(),
        )

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


# ---------------------------------------------------------------------------
# DSAttention - Core Attention Component with DSA
# ---------------------------------------------------------------------------
class DSAttention(FleetLayer):
    """Sparse Attention with DSA Indexer as a core_attention component.

    This module implements sparse attention mechanism using a DSA Indexer to compute top-k
    attention indices for reducing computational complexity. It serves as a pluggable
    core_attention component for MLA, compatible with the DotProductAttention interface.

    To use DSAttention, set it as the core_attention in the spec configuration:
        MLASelfAttentionSublayersSpec(
            ...
            core_attention=DSAttention,
            ...
        )
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: DSAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        softmax_scale: float,
        k_channels: int | None = None,
        v_channels: int | None = None,
        is_mtp_layer: bool = False,
        is_swa: bool = False,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)

        DSAIndexerLossLoggingHelper.register_total_num_layers(config)
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.softmax_scale = softmax_scale

        # DSA Indexer - build from spec
        # sublayers_spec.indexer should be a LayerSpec for DSAIndexer
        self.indexer = build_spec_layer(
            sublayers_spec.indexer,
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
        )

        # DSA loss config
        self.dsa_indexer_loss_coeff = getattr(
            config, "dsa_indexer_loss_coeff", None
        )
        self.dsa_indexer_use_sparse_loss = getattr(
            config, "dsa_indexer_use_sparse_loss", False
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_startend_row_indices: Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        use_rr_flash_attention: bool = False,
        # KV cache parameters (ignored by DSAttention, for interface compatibility)
        past_key_values=None,
        layer_idx=None,
        use_cache: bool = False,
        # DSA-specific parameters
        x: Tensor | None = None,
        qr: Tensor | None = None,
        # ignore fastdeploy specific parameters
        kv_compressed: paddle.Tensor = None,
        k_pos_emb: paddle.Tensor = None,
        q_absorbed: paddle.Tensor = None,
        v_b_proj_weight: paddle.Tensor = None,
    ) -> Tensor:
        """Forward pass for Sparse Attention.

        Note: query/key/value are always batch-first [b, s, ...] when entering
        this method. The upstream MLASelfAttention transposes from seq-first to
        batch-first before calling core_attention.

        Args:
            query: Query tensor [b, s, nhpp, qk_head_dim].
            key: Key tensor [b, s, nhpp, qk_head_dim].
            value: Value tensor [b, s, nhpp, hnv].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            x: Original hidden states for indexer. [b, s, hidden_size] or
                [s/TP, b, hidden_size] in sequence_parallel mode.
            qr: Low-rank query representation for indexer. [b, s, q_lora_rank] or
                [s/TP, b, q_lora_rank] in sequence_parallel mode.
            attn_mask_startend_row_indices: Optional row indices for packed seq.
            attn_mask_type: Attention mask type.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.
            use_rr_flash_attention: Whether to use refined recompute flash attention.

        Returns:
            output: Output tensor [b, sq, hidden_size] or [sq, b, hidden_size]
        """
        # DSA requires x and qr (hidden_states and q_latent)
        if x is None or qr is None:
            raise ValueError(
                "DSAttention requires x and qr parameters. "
                "These are passed by MultiLatentAttention when using DSA."
            )

        # Detach indexer inputs to prevent gradients from flowing back to main model
        # Use detach() + stop_gradient=False so that:
        # 1. Gradients don't flow back to the main model (detach breaks the graph)
        # 2. Linear layers can still compute grad_input in backward without PyLayer errors
        x = x.detach()
        x.stop_gradient = False
        qr = qr.detach()
        qr.stop_gradient = False

        # rotate_activation requires bf16 input
        assert x.dtype == paddle.bfloat16, (
            f"DSAttention: x must be bfloat16, got {x.dtype}"
        )
        assert qr.dtype == paddle.bfloat16, (
            f"DSAttention: qr must be bfloat16, got {qr.dtype}"
        )

        # Layout: batch-first [b, sq, np, hn]
        b, sq, np, hn = query.shape
        sk = key.shape[1]

        # Build causal mask
        causal_mask = paddle.triu(
            paddle.full([sq, sk], float("-inf"), dtype="float32"),
            diagonal=1,
        )  # [sq, sk]

        if attn_mask_type is not None and attn_mask_type == AttnMaskType.causal:
            # Use causal mask only
            indexer_float_mask = causal_mask.unsqueeze(0).unsqueeze(
                0
            )  # [1, 1, sq, sk]
        elif attention_mask is not None:
            mask = attention_mask.squeeze(1)
            indexer_float_mask = paddle.zeros_like(
                mask, dtype="float32"
            ).masked_fill(mask.cast("bool"), float("-inf"))

        else:
            indexer_float_mask = causal_mask.unsqueeze(0).unsqueeze(
                0
            )  # [1, 1, sq, sk]

        # Training with indexer loss
        if self.training and self.dsa_indexer_loss_coeff is not None:
            # Indexer forward_before_topk runs WITH gradient tracking
            # RoPE is computed internally by the indexer
            q_idx, k_idx, weights_idx = self.indexer.forward_before_topk(x, qr)

            # Convert to seq-first for FusedDSAIndexerLoss
            q_idx_sf = q_idx.transpose([1, 0, 2, 3])  # [b,s,h,d] -> [s,b,h,d]
            k_idx_sf = k_idx.transpose([1, 0, 2])  # [b,s,d] -> [s,b,d]
            weights_idx_sf = weights_idx.transpose(
                [1, 0, 2]
            )  # [b,s,h] -> [s,b,h]

            # Convert query/key to seq-first for loss computation
            query_sf = query.transpose([1, 0, 2, 3])
            key_sf = key.transpose([1, 0, 2, 3])

            indexer_loss = FusedDSAIndexerLoss.apply(
                q_idx_sf,
                weights_idx_sf,
                k_idx_sf,
                query_sf.detach(),
                key_sf.detach(),
                self.softmax_scale,
                self.indexer.index_topk,
                float(self.dsa_indexer_loss_coeff),
                indexer_float_mask,
                bool(self.dsa_indexer_use_sparse_loss),
                self.pg_collection.tp
                if self.pg_collection.tp is not None
                and self.pg_collection.tp.nranks > 1
                else None,
            )
            topk_indices = FusedDSAIndexerLoss._last_topk_indices
        else:
            # Inference or no loss
            _, topk_indices = self.indexer.forward(x, qr, indexer_float_mask)
            indexer_loss = None

        # Build sparse mask
        index_mask = paddle.full(
            [b, sq, sk],
            fill_value=float("-inf"),
            dtype="float32",
        )
        zeros = paddle.zeros(
            [
                topk_indices.shape[0],
                topk_indices.shape[1],
                topk_indices.shape[2],
            ],
            dtype="float32",
        )
        index_mask = paddle.put_along_axis(
            index_mask, topk_indices, zeros, axis=-1
        )
        # Merge causal + index
        index_mask = index_mask + causal_mask.unsqueeze(0)
        combined_mask = index_mask.unsqueeze(1)  # [b, 1, sq, sk]

        if attention_mask is not None:
            combined_mask = attention_mask.cast("float32") + combined_mask

        # Run sparse attention (batch-first layout)
        core_attn_out = _unfused_dsa_attention(
            query, key, value, combined_mask, self.softmax_scale
        )

        # Attach indexer loss if training
        if self.training and indexer_loss is not None:
            if (
                self.dsa_indexer_loss_coeff is not None
                and self.dsa_indexer_loss_coeff > 0
            ):
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                        self.config
                    ),
                )
            core_attn_out = DSAIndexerLossAutoScaler.apply(
                core_attn_out, indexer_loss
            )

        return core_attn_out


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------
Indexer = DSAIndexer
