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
DeepSeekV4 Hybrid Attention with Compressed Sparse Attention.

Ported from Megatron-LM experimental_attention_variant/deepseek_v4_hybrid_attention.py
(commit bf4e1db).

Components:
  - DSv4HybridAttention: Base class with inverse RoPE, grouped output projection
  - DSv4HybridSelfAttention: Self-attention with Q low-rank, single-head KV
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddleformers.fleet.fp8.qat import fp8_simulate_qat
from paddleformers.fleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from paddleformers.fleet.transformer.attention import Attention

if TYPE_CHECKING:
    from paddleformers.fleet.process_groups_config import ProcessGroupCollection
    from paddleformers.fleet.transformer.enums import AttnMaskType
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )


def _q_rms_norm(q: Tensor, eps: float) -> Tensor:
    """RMS normalization for query (no learnable weight)."""
    return q * paddle.rsqrt(q.square().mean(-1, keepdim=True) + eps)


from paddleformers.fleet.transformer.utils import (
    get_doc_lens,
)


def build_document_rope_freqs(
    rotary_pos_emb: nn.Layer,
    sq: int,
    startend_row_indices: Tensor,
    position_offset: int = 0,
):
    """Build RoPE frequencies that restart from zero for each document."""
    assert (
        startend_row_indices.shape[0] == 1
        and startend_row_indices.shape[1] == 1
    ), "Document RoPE currently expects batch_size == 1 and head == 1."

    doc_lens = get_doc_lens(startend_row_indices)
    max_doc_len = int(doc_lens.max().item())
    _rope_result = rotary_pos_emb(max_doc_len, packed_seq=False)
    if isinstance(_rope_result, tuple):
        freqs, mscale = _rope_result
    else:
        freqs, mscale = _rope_result, 1.0
    freqs = freqs.squeeze(0).squeeze(1)
    doc_freqs = [freqs[:doc_len] for doc_len in doc_lens.tolist()]
    freqs = paddle.concat(doc_freqs, axis=0)
    needed_len = position_offset + sq
    if freqs.shape[0] < needed_len:
        freqs = paddle.concat(
            [
                freqs,
                paddle.zeros(
                    [needed_len - freqs.shape[0], freqs.shape[-1]],
                    dtype=freqs.dtype,
                ),
            ],
            axis=0,
        )

    return freqs.reshape([1, -1, 1, freqs.shape[-1]]), mscale


# ---------------------------------------------------------------------------
# Sublayers spec dataclass
# ---------------------------------------------------------------------------


@dataclass
class DSv4HybridSelfAttentionSublayersSpec:
    """Sublayer specifications for DSv4 Hybrid Self-Attention."""

    linear_q_down_proj: type | LayerSpec = None
    linear_q_up_proj: type | LayerSpec = None
    linear_kv_proj: type | LayerSpec = None
    core_attention: type | LayerSpec | None = None
    o_proj: type | LayerSpec = None
    q_layernorm: type | LayerSpec = None
    kv_layernorm: type | LayerSpec = None


# ---------------------------------------------------------------------------
# DSv4HybridAttention
# ---------------------------------------------------------------------------


class DSv4HybridAttention(Attention):
    """DSv4 Hybrid Attention with CSA core attention, inverse RoPE, and grouped output.

    This class:
    1. Builds per-layer RotaryEmbedding (with configurable base for compressed layers)
    2. Builds CompressedSparseAttention as core attention
    3. Applies inverse RoPE on attention output
    4. Performs grouped low-rank output projection
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: DSv4HybridSelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.num_attention_heads = config.num_attention_heads
        self.v_head_dim = config.v_head_dim
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim or 0
        self.query_projection_size = self.num_attention_heads * self.v_head_dim
        self.q_head_dim = self.v_head_dim
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.v_head_dim

        # Per-layer compress ratio
        if is_mtp_layer:
            layer_idx = self.config.num_hidden_layers + layer_number
            compress_ratio = self.config.csa_compress_ratios[layer_idx]
        else:
            layer_idx = layer_number - self.config.num_empty_layers_add_in_head
            compress_ratio = self.config.csa_compress_ratios[layer_idx]
        # Per-layer RoPE (potentially different base for compressed layers)
        rope_base = getattr(config, "rotary_base", 10000)
        if compress_ratio > 1:
            rope_base = config.csa_compress_rotary_base

        use_compressed_yarn = compress_ratio > 1
        if not use_compressed_yarn:
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=getattr(config, "rotary_percent", 1.0),
                rotary_base=rope_base,
            )
        else:
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=rope_base,
                scaling_factor=getattr(config, "rotary_scaling_factor", 40),
                original_max_position_embeddings=getattr(
                    config, "original_max_position_embeddings", 4096
                ),
                beta_fast=getattr(config, "beta_fast", 32),
                beta_slow=getattr(config, "beta_slow", 1),
                mscale=getattr(config, "mscale", 1.0),
                mscale_all_dim=getattr(config, "mscale_all_dim", 0.0),
            )

        self.core_attention = build_spec_layer(
            sublayers_spec.core_attention,
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=None,
            softmax_scale=getattr(config, "softmax_scale", None),
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=1,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
            is_mtp_layer=is_mtp_layer,
            compress_ratio=compress_ratio,
            rotary_pos_emb=self.rotary_pos_emb,
        )

        # Grouped output projection
        self.o_local_groups = config.o_groups
        assert self.query_projection_size % config.o_groups == 0, (
            "num_attention_heads * v_head_dim must be divisible by o_groups"
        )
        group_proj_in_size = self.query_projection_size // config.o_groups
        group_proj_out_size = config.o_groups * config.o_lora_rank

        self.linear_o_group_proj = self.create_parameter(
            shape=[group_proj_out_size, group_proj_in_size],
            dtype=config.dtype if hasattr(config, "dtype") else "bfloat16",
            default_initializer=nn.initializer.Normal(
                std=getattr(config, "init_method_std", 0.02)
            ),
        )

        linear_proj_in_size = config.o_groups * config.o_lora_rank
        self.o_proj = build_spec_layer(
            sublayers_spec.o_proj,
            linear_proj_in_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )
        self.use_fp8_qat = getattr(config, "use_fp8_qat", False)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, None]:
        """Forward pass.

        Args:
            hidden_states: [b, sq, hidden_size]
            attention_mask: optional mask

        Returns:
            (output [b, sq, hidden_size], bias=None)
        """
        startend_row_indices = kwargs.get(
            "attn_mask_startend_row_indices", None
        )

        # Get Q, K, V tensors
        # In CP mode, pass position_offset so RoPE uses correct global positions.
        cp_pg = getattr(self, "pg_collection", None)
        cp_pg = cp_pg.cp if cp_pg is not None else None
        cp_size = getattr(cp_pg, "nranks", 1) if cp_pg is not None else 1
        if cp_size > 1:
            assert self.config.cp_balance_mode == "contiguous_allgather", (
                f"DSv4HybridAttention requires cp_balance_mode='contiguous_allgather', "
                f"got '{self.config.cp_balance_mode}'"
            )
        cp_rank = (
            getattr(cp_pg, "rank", 0)
            if cp_pg is not None and cp_size > 1
            else 0
        )
        _, sq, _ = hidden_states.shape
        position_offset = cp_rank * sq if cp_size > 1 else 0

        query, key, value, q_compressed, kv_compressed = (
            self.get_query_key_value_tensors(
                hidden_states=hidden_states,
                startend_row_indices=startend_row_indices,
                position_offset=position_offset,
            )
        )

        # Core attention (CompressedSparseAttention)
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            startend_row_indices,
            attention_mask,
            x=hidden_states,
            qr=q_compressed,
        )
        # core_attn_out: [b, sq, np * v_head_dim]

        # Inverse RoPE on last qk_pos_emb_head_dim of each head
        b, sq, _ = core_attn_out.shape
        pos_dim = self.qk_pos_emb_head_dim
        nope_dim = self.v_head_dim - pos_dim

        if pos_dim > 0:
            core_attn_out = core_attn_out.reshape(
                [b, sq, self.num_attention_heads, self.v_head_dim]
            )
            # Get RoPE frequencies for inverse
            if startend_row_indices is not None:
                freqs, mscale = build_document_rope_freqs(
                    self.rotary_pos_emb,
                    sq,
                    startend_row_indices,
                    position_offset=position_offset,
                )
            else:
                # Get RoPE frequencies for inverse; use global positions in CP mode
                rope_len = sq + position_offset
                _rope_result = self.rotary_pos_emb(rope_len, packed_seq=False)
                if isinstance(_rope_result, tuple):
                    freqs, mscale = _rope_result
                else:
                    freqs, mscale = _rope_result, 1.0
            # DSv4 reference uses pure norm-preserving RoPE; YaRN's mscale is not applied.
            mscale = 1.0
            freqs = freqs[:, position_offset : position_offset + sq, :]

            if self.config.apply_rope_fusion:
                from paddleformers.fleet.triton_ops import (
                    fused_apply_mla_rope_inplace,
                )

                # The clone is necessary because sparse attention depends on core_attn_out
                # for backward. However, it is still 10x faster than the unfused path.
                core_attn_out = fused_apply_mla_rope_inplace(
                    core_attn_out,
                    freqs,
                    nope_dim,
                    mscale,
                    inverse=True,
                    clone_input=True,
                )
            else:
                content_part = core_attn_out[..., :nope_dim]
                rot_part = core_attn_out[..., nope_dim:]

                rot_part = _apply_rotary_pos_emb_bshd(
                    rot_part,
                    freqs,
                    mscale=mscale,
                    rotary_interleaved=False,
                    multi_latent_attention=True,
                    inverse=True,
                    mla_output_remove_interleaving=True,
                )
                core_attn_out = paddle.concat([content_part, rot_part], axis=-1)
                core_attn_out = core_attn_out.reshape([b, sq, -1])

        # Grouped output projection
        core_attn_out = core_attn_out.reshape([b, sq, self.o_local_groups, -1])
        wo_a_weight = self.linear_o_group_proj.reshape(
            [self.o_local_groups, self.config.o_lora_rank, -1]
        )
        core_attn_out = paddle.einsum(
            "...gd,grd->...gr", core_attn_out, wo_a_weight
        )
        core_attn_out = core_attn_out.reshape([b, sq, -1])

        # Output projection
        output, bias = self.o_proj(core_attn_out)

        return output, bias

    def get_query_key_value_tensors(
        self, hidden_states: Tensor, startend_row_indices: Tensor | None = None
    ):
        """Override in subclass."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DSv4HybridSelfAttention
# ---------------------------------------------------------------------------


class DSv4HybridSelfAttention(DSv4HybridAttention):
    """DSv4 Hybrid Self-Attention with Q low-rank decomposition and single-head KV.

    Q path: hidden -> q_down_proj -> q_layernorm -> q_up_proj -> rms_norm -> RoPE
    KV path: hidden -> kv_proj -> kv_layernorm -> RoPE (single head, key == value)
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: DSv4HybridSelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.q_lora_rank = config.q_lora_rank
        q_head_dim = self.v_head_dim  # In DSv4 Hybrid, q_head_dim == v_head_dim

        # Q down projection: hidden_size -> q_lora_rank (duplicated)
        self.linear_q_down_proj = build_spec_layer(
            sublayers_spec.linear_q_down_proj,
            config.hidden_size,
            config.q_lora_rank,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )

        # Q layernorm
        self.q_layernorm = build_spec_layer(
            sublayers_spec.q_layernorm,
            config=config,
            hidden_size=config.q_lora_rank,
            eps=getattr(config, "rms_norm_eps", 1e-5),
        )

        # Q up projection: q_lora_rank -> num_heads * q_head_dim (column parallel)
        self.linear_q_up_proj = build_spec_layer(
            sublayers_spec.linear_q_up_proj,
            config.q_lora_rank,
            self.num_attention_heads * q_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        # KV projection: hidden_size -> v_head_dim (single head)
        self.linear_kv_proj = build_spec_layer(
            sublayers_spec.linear_kv_proj,
            config.hidden_size,
            config.v_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        # KV layernorm
        self.kv_layernorm = build_spec_layer(
            sublayers_spec.kv_layernorm,
            config=config,
            hidden_size=config.v_head_dim,
            eps=getattr(config, "rms_norm_eps", 1e-5),
        )

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        startend_row_indices: Tensor | None = None,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Derive query, key, value from hidden_states.

        Args:
            hidden_states: [b, sq, hidden_size]
            position_offset: global position offset for CP (cp_rank * sq_local).
                When non-zero, RoPE frequencies are sliced from the correct
                global starting position.

        Returns:
            query: [b, sq, num_heads, v_head_dim]
            key:   [b, sq, 1, v_head_dim]
            value: [b, sq, 1, v_head_dim]
            q_compressed: [b, sq, q_lora_rank]
            kv_compressed: [b, sq, hidden_size] (== hidden_states)
        """
        b, sq, _ = hidden_states.shape

        # Q path
        q_compressed, _ = self.linear_q_down_proj(
            hidden_states
        )  # [b, sq, q_lora_rank]
        q_compressed = self.q_layernorm(q_compressed)

        q, _ = self.linear_q_up_proj(q_compressed)  # [b, sq, n * v_head_dim]
        q = q.reshape([b, sq, self.num_attention_heads, self.v_head_dim])
        q = _q_rms_norm(q, getattr(self.config, "rms_norm_eps", 1e-5))

        # KV path
        kv, _ = self.linear_kv_proj(hidden_states)  # [b, sq, v_head_dim]
        kv = self.kv_layernorm(kv)

        # Apply RoPE to both Q and KV
        pos_dim = self.qk_pos_emb_head_dim
        nope_dim = self.v_head_dim - pos_dim

        if pos_dim > 0:
            # Get RoPE frequencies
            if startend_row_indices is not None:
                freqs, mscale = build_document_rope_freqs(
                    self.rotary_pos_emb,
                    sq,
                    startend_row_indices,
                    position_offset=position_offset,
                )
            else:
                # Get RoPE frequencies for global positions
                rope_len = sq + position_offset
                _rope_result = self.rotary_pos_emb(rope_len, packed_seq=False)
                if isinstance(_rope_result, tuple):
                    freqs, mscale = _rope_result
                else:
                    freqs, mscale = _rope_result, 1.0
            # DSv4 reference uses pure norm-preserving RoPE; YaRN's mscale is not applied.
            mscale = 1.0
            # Slice to local positions [position_offset : position_offset + sq]
            freqs = freqs[:, position_offset : position_offset + sq, :]

            # Q RoPE: split nope/pe, apply RoPE to pe part
            if self.config.apply_rope_fusion:
                from paddleformers.fleet.triton_ops import (
                    fused_apply_mla_rope_inplace,
                )

                query = fused_apply_mla_rope_inplace(q, freqs, nope_dim, mscale)
            else:
                q_nope = q[..., :nope_dim]
                q_pe = q[..., nope_dim:]
                q_pe = _apply_rotary_pos_emb_bshd(
                    q_pe,
                    freqs,
                    mscale=mscale,
                    rotary_interleaved=False,
                    multi_latent_attention=True,
                    mla_output_remove_interleaving=True,
                )
                query = paddle.concat([q_nope, q_pe], axis=-1)

            # KV RoPE: split nope/pe, apply RoPE to pe part
            kv_nope = kv[..., :nope_dim]
            kv_pe = kv[..., nope_dim:]
            # Add head dim for RoPE: [b, sq, pos_dim] -> [b, sq, 1, pos_dim]
            kv_pe = kv_pe.unsqueeze(2)
            kv_pe = _apply_rotary_pos_emb_bshd(
                kv_pe,
                freqs,
                mscale=mscale,
                rotary_interleaved=False,
                multi_latent_attention=True,
                mla_output_remove_interleaving=True,
            )
            kv_pe = kv_pe.squeeze(2)

            # KV QAT:
            #   kv_nope: bf16 -> fp32 -> fp8e4m3 ->fp32 -> bf16
            if self.use_fp8_qat:
                kv_nope = fp8_simulate_qat(kv_nope, 64)

            kv = paddle.concat([kv_nope, kv_pe], axis=-1)
        else:
            query = q

        # Single head: key = value = [b, sq, 1, v_head_dim]
        key = kv.unsqueeze(2)
        value = key

        return query, key, value, q_compressed, hidden_states
