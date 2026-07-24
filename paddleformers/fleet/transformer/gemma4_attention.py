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
"""Gemma4 SelfAttention with heterogeneous head_dim, K=V tying, V-Norm, and dual RoPE selection."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor

from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)


def startend_row_indices_to_dense_mask(
    startend_row_indices: Tensor, seq_len_q: int
) -> Tensor:
    """Convert flashmask startend_row_indices to dense boolean attention mask for eager path.

    Flashmask semantics (Paddle kernel convention):
    - LTS (column 0, "downstart"): first MASKED row from below. Query q >= LTS[k] is masked.
    - LTE (column 1, when bound_num>=2): exclusive upper bound. Band LTS <= q < LTE is masked.
    - When causal=True, the flashmask kernel applies the standard causal mask (q < k)
      internally and the indices provide ADDITIONAL constraints on top.

    Args:
        startend_row_indices: [b, 1, sk, 1] or [b, 1, sk, 2]
        seq_len_q: query sequence length

    Returns:
        Boolean mask [b, 1, sq, sk] where True = masked (do not attend).
    """
    bsz, num_heads, sk, bound_num = startend_row_indices.shape

    idx_dtype = startend_row_indices.dtype
    rows = paddle.arange(seq_len_q, dtype=idx_dtype).reshape([1, 1, -1, 1])
    cols = paddle.arange(sk, dtype=idx_dtype).reshape([1, 1, 1, -1])

    # downstart per column: [b, nh, 1, sk]
    downstart = startend_row_indices[:, :, :, 0:1].transpose([0, 1, 3, 2])

    # Causal mask: upper triangle (q < k → masked)
    causal_mask = rows < cols

    # Flashmask LTS constraint: q >= LTS[k] → additionally masked
    if bound_num >= 2:
        downend = startend_row_indices[:, :, :, 1:2].transpose([0, 1, 3, 2])
        flashmask = (rows >= downstart) & (rows < downend)
    else:
        flashmask = rows >= downstart

    return causal_mask | flashmask


if TYPE_CHECKING:
    from paddleformers.fleet.process_groups_config import ProcessGroupCollection
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class Gemma4SelfAttention(SelfAttention):
    """Gemma4 self-attention with heterogeneous KV heads and head dims.

    Key differences from standard SelfAttention:
    1. Selects sliding vs global config based on layer_types[layer_number-1]
    2. Picks corresponding RoPE from (rope_sliding, rope_global) tuple
    3. V-Norm: scaleless RMSNorm on value tensor
    4. K=V tying: global layers set value = key
    5. Selects appropriate attention_mask for sliding vs full attention
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: SelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        # Determine layer type: layer_number = i + num_empty_layers_add_in_head
        # from layer_specs factory; logical index into layer_types is layer_number
        # minus the empty-layer offset.
        layer_types = getattr(config, "layer_types", None)
        num_empty = getattr(config, "num_empty_layers_add_in_head", 0) or 0
        logical_idx = layer_number - num_empty
        self.is_sliding = (
            layer_types[logical_idx] == "sliding_attention"
            if layer_types and 0 <= logical_idx < len(layer_types)
            else True
        )

        # Deep copy config and adjust for global layers
        layer_config = copy.deepcopy(config)
        if not self.is_sliding:
            layer_config.head_dim = getattr(
                config, "global_head_dim", config.head_dim
            )
            layer_config.v_head_dim = layer_config.head_dim
            layer_config.num_key_value_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
            layer_config.sliding_window = None
        else:
            # Flashmask LTS semantics: window_size = total attended tokens (including self)
            sw = getattr(config, "sliding_window", None)
            if isinstance(sw, int):
                layer_config.sliding_window = (sw, 0)
            elif isinstance(sw, (tuple, list)):
                layer_config.sliding_window = tuple(sw)

        # Gemma4 uses QK-Norm → fixed softmax_scale=1.0
        layer_config.softmax_scale = 1.0

        # Global layers (head_dim=512) must use eager attention because
        # flashmask and Paddle SDPA kernels only support head_dim<=256.
        if not self.is_sliding:
            layer_config._attn_implementation = "eager"

        # K=V tying flag
        self._tied_kv = not self.is_sliding and getattr(
            config, "attention_k_eq_v", False
        )

        # V-Norm epsilon
        self._v_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        from paddleformers.fleet.transformer.enums import AttnMaskType

        if attn_mask_type is None:
            attn_mask_type = AttnMaskType.causal

        super().__init__(
            config=layer_config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, split_qkv=True
    ):
        if self._tied_kv:
            # For K=V tying (global layers): value = raw key before K-Norm,
            # then apply V-Norm to value and K-Norm to key separately.
            saved_k_norm = self.k_norm
            self.k_norm = None
            try:
                query, key, value = super().get_query_key_value_tensors(
                    hidden_states, key_value_states, split_qkv
                )
            finally:
                self.k_norm = saved_k_norm

            # K=V tying: value = raw key (before K-Norm)
            value = key

            # V-Norm: scaleless RMSNorm
            v_float = value.cast("float32")
            rms = (
                v_float.pow(2).mean(-1, keepdim=True) + self._v_norm_eps
            ).sqrt()
            value = (v_float / rms).cast(value.dtype)

            # Apply K-Norm separately
            key = self.k_norm(key)

            return query, key, value
        else:
            # Standard path (sliding layers)
            query, key, value = super().get_query_key_value_tensors(
                hidden_states, key_value_states, split_qkv
            )

            # V-Norm: scaleless RMSNorm on value
            v_float = value.cast("float32")
            rms = (
                v_float.pow(2).mean(-1, keepdim=True) + self._v_norm_eps
            ).sqrt()
            value = (v_float / rms).cast(value.dtype)

            return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states: Tensor | None = None,
        rotary_pos_emb=None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rope_freqs_cis: Tensor | None = None,
        swa_rotary_pos_emb=None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
        in_recompute: bool = False,
        past_key_values=None,
        layer_idx: int | None = None,
        use_cache: bool = False,
    ):
        # Select RoPE based on layer type (DualRoPEOutput or tuple)
        if (
            hasattr(rotary_pos_emb, "__getitem__")
            and len(rotary_pos_emb) == 2
            and not isinstance(rotary_pos_emb, paddle.Tensor)
        ):
            rotary_pos_emb = (
                rotary_pos_emb[0] if self.is_sliding else rotary_pos_emb[1]
            )

        # Select attention mask
        if isinstance(attention_mask, dict):
            mask_key = (
                "sliding_attention" if self.is_sliding else "full_attention"
            )
            attention_mask = attention_mask.get(mask_key, attention_mask)

        # Global layers (head_dim=512) exceed flashmask kernel limit.
        # Convert startend_row_indices to dense mask for eager path.
        # NOTE: eager path does not support CP or SP; assumes sq == sk.
        if not self.is_sliding and attn_mask_startend_row_indices is not None:
            attention_mask = startend_row_indices_to_dense_mask(
                attn_mask_startend_row_indices,
                attn_mask_startend_row_indices.shape[2],
            )
            attn_mask_startend_row_indices = None

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            key_value_states=key_value_states,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rope_freqs_cis=rope_freqs_cis,
            swa_rotary_pos_emb=swa_rotary_pos_emb,
            swa_rotary_pos_cos=swa_rotary_pos_cos,
            swa_rotary_pos_sin=swa_rotary_pos_sin,
            position_ids=position_ids,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            in_recompute=in_recompute,
            past_key_values=past_key_values,
            layer_idx=layer_idx,
            use_cache=use_cache,
        )
