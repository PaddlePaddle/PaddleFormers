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

import hashlib
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

_LOG_LAYER_MD5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"

if TYPE_CHECKING:
    from .transformer_config import TransformerConfig

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer
from paddle.distributed.fleet.utils import recompute

from paddleformers.fleet import tensor_parallel
from paddleformers.fleet.context_parallel_utils import ContextParallelScatterOp
from paddleformers.fleet.models.common.embeddings import apply_rotary_pos_emb
from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from paddleformers.fleet.parallel_state import get_context_parallel_world_size
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.recompute_utils import (
    need_recompute_in_block,
    need_recompute_in_first_n,
)
from paddleformers.fleet.tensor_parallel import RecomputeWithoutOutput
from paddleformers.fleet.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from paddleformers.fleet.transformer.layer import FleetLayer
from paddleformers.fleet.transformer.utils import is_layer_window_attention
from paddleformers.fleet.utils import divide, get_pg_rank, get_pg_size

from .enums import AttnMaskType


def _md5(t):
    """Compute MD5 of tensor bytes (cast to fp32 via .numpy())."""
    return hashlib.md5(t.detach().numpy().tobytes()).hexdigest()


def _apply_ec_complex_3d_mrope(
    query,
    key,
    position_ids,
    head_dim,
    rope_theta=1000000.0,
    mrope_section=None,
    layer_number=0,
    cp_balance_mode="dualchunk_allgather",
):
    """Apply EC-style complex multiplication 3D MRoPE to query and key tensors."""
    import logging

    if mrope_section is None:
        mrope_section = [16, 1, 1]

    seq_len_q = query.shape[1]
    seq_len_p = position_ids.shape[1]
    # MTP processing trims position_ids by 1; pad back to match query length
    if seq_len_p < seq_len_q:
        last_pos = position_ids[:, -1:, :]  # [B, 1, 3]
        pad_count = seq_len_q - seq_len_p
        # Increment each axis by 1 for each padded position
        pads = []
        for i in range(pad_count):
            pads.append(last_pos + (i + 1))
        position_ids = paddle.concat([position_ids, *pads], axis=1)

    # EC's using_position_axis construction (from ernie_core/models/ernie5/modeling.py:432-442):
    # For mrope_section=[16,1,1], point_num=64:
    #   1) Build [1]*1 + [2]*1 = [1, 2] from mrope_section[1:]
    #   2) Repeat 24 times: [1,2]*24 = [1,2,1,2,...] (48 entries)
    #   3) Append [0]*16 at end
    #   Total: [1,2,1,2,...(48), 0,0,...,0(16)] = 64 entries
    point_num = head_dim // 2
    using_position_axis = []
    for i, n in enumerate(mrope_section[1:]):
        using_position_axis.extend([i + 1] * n)
    repeat_count = (point_num - mrope_section[0]) // sum(mrope_section[1:])
    using_position_axis = using_position_axis * repeat_count
    using_position_axis.extend([0] * mrope_section[0])
    using_position_axis = paddle.to_tensor(using_position_axis, dtype="int64")

    expand_position_ids = paddle.index_select(
        position_ids, using_position_axis, axis=-1
    )

    freqs = 1.0 / (
        rope_theta
        ** (paddle.arange(0, head_dim, 2, dtype="float32") / float(head_dim))
    )
    freqs = freqs.reshape([1, 1, head_dim // 2])
    freqs = freqs.expand(
        [
            expand_position_ids.shape[0],
            expand_position_ids.shape[1],
            head_dim // 2,
        ]
    )

    freqs = expand_position_ids.astype("float32") * freqs

    freqs_cis = paddle.polar(paddle.ones_like(freqs), freqs)
    freqs_cis = freqs_cis.unsqueeze(2)
    if get_context_parallel_world_size() > 1:
        freqs_cis = ContextParallelScatterOp.apply(
            freqs_cis, axis=1, mode=cp_balance_mode
        )
    if _LOG_LAYER_MD5:
        logger = logging.getLogger(__name__)
        rank = paddle.distributed.get_rank()
        q_md5 = _md5(query)
        k_md5 = _md5(key)
        logger.info(
            f"[MD5 Probe PF] Rank={rank} query_before_rope MD5={q_md5} shape={list(query.shape)}"
        )
        logger.info(
            f"[MD5 Probe PF] Rank={rank} key_before_rope MD5={k_md5} shape={list(key.shape)}"
        )
        fc_md5 = _md5(paddle.as_real(freqs_cis))
        logger.info(
            f"[MD5 Probe PF] Rank={rank} freqs_cis MD5={fc_md5} shape={list(freqs_cis.shape)} q_dtype={query.dtype}"
        )

    orig_dtype = query.dtype
    xq = query.reshape([*query.shape[:-1], -1, 2]).cast("float32")
    xk = key.reshape([*key.shape[:-1], -1, 2]).cast("float32")

    xq_ = paddle.as_complex(xq)
    xk_ = paddle.as_complex(xk)

    if _LOG_LAYER_MD5:
        xq_md5 = _md5(paddle.as_real(xq_ * freqs_cis))
        logger.info(
            f"[MD5 Probe PF] Rank={rank} xq_complex MD5={xq_md5} shape={list(xq_.shape)}"
        )

    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis

    query = paddle.as_real(xq_out)
    query = paddle.flatten(query, start_axis=3).cast(orig_dtype)
    key = paddle.as_real(xk_out)
    key = paddle.flatten(key, start_axis=3).cast(orig_dtype)

    return query, key


@dataclass
class SelfAttentionSublayersSpec:
    """
    Configuration class for specifying the sublayers_spec of a self-attention.
    """

    qkv_proj: LayerSpec | type = None
    core_attention: LayerSpec | type = None
    o_proj: LayerSpec | type = None
    q_norm: LayerSpec | type = None
    k_norm: LayerSpec | type = None
    gate_proj: LayerSpec | type = None


@dataclass
class SelfAttentionVHASublayersSpec:
    """
    Configuration class for specifying the sublayers_spec of a VHA self-attention.
    """

    q_proj: LayerSpec | type = None
    k_proj: LayerSpec | type = None
    v_proj: LayerSpec | type = None
    gate_proj: LayerSpec | type = None
    qkv_proj: LayerSpec | type = None  # used for SWA fallback (fused QKV)
    core_attention: LayerSpec | type = None
    o_proj: LayerSpec | type = None
    q_norm: LayerSpec | type = None
    k_norm: LayerSpec | type = None


@dataclass
class CrossAttentionSublayersSpec:
    """
    Configuration class for specifying the sublayers_spec of a cross-attention.
    """

    linear_q: LayerSpec | type = None
    linear_kv: LayerSpec | type = None
    core_attention: LayerSpec | type = None
    o_proj: LayerSpec | type = None


class Attention(FleetLayer, ABC):
    """Attention layer abstract class.

    This layer only contains common layers required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: SelfAttentionSublayersSpec
        | CrossAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.is_mtp_layer = is_mtp_layer

        self.is_swa = False

        if self.config.sliding_window is not None:
            if self.is_mtp_layer:
                for_swa_layer_number = (
                    self.layer_number + self.config.num_hidden_layers
                )
            else:
                # for non-mtp layer, layer_number add num_empty_layers_add_in_head in
                # src/paddleformers.fleet/models/gpt/gpt_layer_specs.py#L533
                # real_layer_number = layer_number + config.num_empty_layers_add_in_head
                for_swa_layer_number = (
                    self.layer_number - self.config.num_empty_layers_add_in_head
                )
                assert for_swa_layer_number >= 0, (
                    f"for_swa_layer_number must be non-negative, but got {for_swa_layer_number} "
                    f"(layer_number={self.layer_number}, "
                    f"num_empty_layers_add_in_head={self.config.num_empty_layers_add_in_head})"
                )

            if is_layer_window_attention(
                self.config.sliding_window,
                self.config.window_attn_skip_freq,
                for_swa_layer_number,
            ):
                self.is_swa = True

        if self.is_swa:
            self.head_dim = self.config.swa_head_dim
            self.v_head_dim = self.config.swa_v_head_dim
            self.num_attention_heads = self.config.swa_num_attention_heads
            self.num_key_value_heads = self.config.swa_num_key_value_heads
            self.rope_theta = self.config.swa_rope_theta
        else:
            self.head_dim = self.config.head_dim
            self.v_head_dim = (
                self.config.v_head_dim
                if isinstance(self.config.v_head_dim, int)
                else self.config.head_dim
            )
            self.num_attention_heads = self.config.num_attention_heads
            self.num_key_value_heads = self.config.num_key_value_heads
            self.rope_theta = self.config.rope_theta

        self.query_projection_size = self.head_dim * self.num_attention_heads
        self.key_projection_size = self.head_dim * self.num_key_value_heads
        self.value_projection_size = self.v_head_dim * self.num_key_value_heads
        self.out_projection_size = self.v_head_dim * self.num_attention_heads
        self.qk_rope_head_dim = self.head_dim
        if (
            isinstance(self.config.rotary_percent, (int, float))
            and self.config.rotary_percent < 1.0
        ):
            self.qk_rope_head_dim = int(
                self.head_dim * self.config.rotary_percent
            )

        self.v_scale = self.config.attention_value_scale

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "cp"]
            )
        else:
            assert hasattr(pg_collection, "tp"), (
                "Attention pg_collection must have tp process group"
            )
            assert hasattr(pg_collection, "cp"), (
                "Attention pg_collection must have cp process group"
            )
        self.pg_collection = pg_collection

        # Per attention head and per partition values
        world_size = get_pg_size(self.pg_collection.tp)
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size,
            self.num_attention_heads,
        )

        self.value_hidden_size_per_attention_head = divide(
            self.value_projection_size, self.num_key_value_heads
        )
        self.num_attention_heads_per_partition = divide(
            self.num_attention_heads,
            world_size,
        )
        self.num_query_groups_per_partition = divide(
            self.num_key_value_heads, world_size
        )

        self.core_attention = build_spec_layer(
            sublayers_spec.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            is_mtp_layer=self.is_mtp_layer,
            is_swa=self.is_swa,
            k_channels=self.head_dim,
            v_channels=self.v_head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            cp_comm_type=cp_comm_type,
            softmax_scale=self.config.softmax_scale,
            pg_collection=self.pg_collection,
        )
        self.use_rr_flash_attention = False
        self.recompute_core_attention = False
        if self.config.recompute_granularity == "selective":
            if isinstance(self.config.recompute_modules, list):
                if self.config.recompute_num_layers is None:
                    # selective all submodels to recompute
                    if "core_attn" in self.config.recompute_modules:
                        self.recompute_core_attention = True
                else:
                    # selective submodels in special layers to recompute
                    assert self.config.recompute_method in ["first_n", "block"]
                    if "core_attn" in self.config.recompute_modules:
                        self.recompute_core_attention = (
                            need_recompute_in_block(
                                self.layer_number,
                                self.config,
                                self.config.recompute_num_layers,
                            )
                            if self.config.recompute_method == "block"
                            else need_recompute_in_first_n(
                                self.layer_number,
                                self.config,
                                self.config.recompute_num_layers,
                            )
                        )
            elif isinstance(self.config.recompute_modules, dict):
                assert self.config.recompute_method in ["first_n", "block"]
                if "core_attn" in self.config.recompute_modules:
                    self.recompute_core_attention = (
                        need_recompute_in_block(
                            self.layer_number,
                            self.config,
                            self.config.recompute_modules["core_attn"],
                        )
                        if self.config.recompute_method == "block"
                        else need_recompute_in_first_n(
                            self.layer_number,
                            self.config,
                            self.config.recompute_modules["core_attn"],
                        )
                    )
        if (
            self.config.recompute_modules is not None
            and "flash_attn" in self.config.recompute_modules
        ):
            assert self.config.recompute_granularity is not None, (
                "rr must be used when recompute is enabled"
            )
            if isinstance(self.config.recompute_modules, list):
                self.use_rr_flash_attention = True
            elif isinstance(self.config.recompute_modules, dict):
                self.use_rr_flash_attention = not need_recompute_in_first_n(
                    self.layer_number,
                    self.config,
                    self.config.recompute_modules["flash_attn"],
                )
        # Output.
        self.o_proj = build_spec_layer(
            sublayers_spec.o_proj,
            self.out_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.use_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        self.recompute_gated_attn = (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "gated_attn" in self.config.recompute_modules
        )

    def _post_core_attention_hook(self, core_attn_out: Tensor) -> Tensor:
        """Hook called after core attention. Override in subclasses (e.g. VHA postmix)."""
        return core_attn_out

    @abstractmethod
    def get_query_key_value_tensors(
        self, hidden_states, key_value_states, split_qkv=True
    ):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states: Tensor | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rope_freqs_cis: Tensor | None = None,
        swa_rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: Tensor | None = None,
        in_recompute: bool = False,
        past_key_values=None,
        layer_idx=None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention layer.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            rotary_pos_emb (Optional[Union[Tensor, tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.

        Return:
            (tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        # no_rope = (
        #    self.config.no_rope_freq[self.layer_number - 1]
        #    if self.config.no_rope_freq
        #    else False
        # )
        no_rope = False

        if self.is_swa:
            if rope_freqs_cis is not None:
                raise ValueError("Sliding Window Not Support rope_freqs_cis")
            rotary_pos_emb = swa_rotary_pos_emb
            rotary_pos_cos = swa_rotary_pos_cos
            rotary_pos_sin = swa_rotary_pos_sin

        if no_rope:
            rotary_pos_emb = None

        # hidden_states: [b, sq, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Check if fused_single_qkv_rope is requested but either unavailable or not
        # supported for the current use case.
        # if self.attention_type != "cross":
        #   assert not (self.config.fused_single_qkv_rope), (
        #        "fused_single_qkv_rope requested but not available/supported for the config."
        #    )

        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        qkv_output = self.get_query_key_value_tensors(
            hidden_states, key_value_states, split_qkv=True
        )
        attn_mask_type = self.attn_mask_type
        block_table = None
        if len(qkv_output) == 4:
            query, key, value, gate = qkv_output
        else:
            query, key, value = qkv_output
            gate = None

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if self.qk_rope_head_dim > 0 and self.qk_rope_head_dim < self.head_dim:
            query, query_nope = query.split(
                [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
                axis=-1,
            )
            key, key_nope = key.split(
                [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
                axis=-1,
            )

        if (
            self.config.gpt_model_use_experimental_version
            and position_ids is not None
            and not self.config.multi_latent_attention
        ):
            # EC-compatible complex multiplication 3D MRoPE
            query, key = _apply_ec_complex_3d_mrope(
                query,
                key,
                position_ids,
                head_dim=self.head_dim,
                rope_theta=self.config.rope_theta,
                mrope_section=getattr(self.config, "mrope_section", [16, 1, 1]),
                layer_number=self.layer_number,
                cp_balance_mode=self.config.cp_balance_mode,
            )
        elif rope_freqs_cis is not None:
            rope_freqs_cis = rope_freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
            # ..., num_heads, head_dim/2
            query_ = paddle.view_as_complex(
                query.float().view(*query.shape[:-1], -1, 2)
            )
            key_ = paddle.view_as_complex(
                key.float().view(*key.shape[:-1], -1, 2)
            )
            query = (
                paddle.view_as_real(query_ * rope_freqs_cis)
                .flatten(-2)
                .to(hidden_states.dtype)
            )  # ..., num_heads, head_dim
            key = (
                paddle.view_as_real(key_ * rope_freqs_cis)
                .flatten(-2)
                .to(hidden_states.dtype)
            )  # ..., num_heads, head_dim

        elif rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                total_seqlen_q = packed_seq_params.total_seqlen_q
                total_seqlen_kv = packed_seq_params.total_seqlen_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
                total_seqlen_q = total_seqlen_kv = None

            if (
                self.config.apply_rope_fusion
                and not self.config.high_precision_rope
                and q_pos_emb is not None
                and k_pos_emb is not None
            ):
                query, key, _ = apply_rotary_pos_emb(
                    (query, key),
                    None,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    position_ids=position_ids,
                    mscale=None,
                    cp_group=self.pg_collection.cp,
                    sp_group=self.pg_collection.tp,
                )
            # elif self.config.apply_vision_rope:
            #     query, key = apply_rotary_pos_emb_vision(query,key,rotary_pos_cos,rotary_pos_sin)
            else:
                if q_pos_emb is not None:
                    # For sequence parallel, input is [S_sp, B, H, D] (time-major),
                    # so we need to set time_major=True for RoPE
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        None,
                        None,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        total_seq_len=total_seqlen_q,
                        position_ids=position_ids,
                        mscale=_yarn_get_concentration_factor_from_config(
                            self.config
                        ),
                        cp_group=self.pg_collection.cp,
                        sp_group=self.pg_collection.tp
                        if self.config.sequence_parallel
                        else None,
                    )

                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key,
                        k_pos_emb,
                        None,
                        None,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        total_seq_len=total_seqlen_kv,
                        position_ids=position_ids,
                        mscale=_yarn_get_concentration_factor_from_config(
                            self.config
                        ),
                        cp_group=self.pg_collection.cp,
                        sp_group=self.pg_collection.tp
                        if self.config.sequence_parallel
                        else None,
                    )

        if self.qk_rope_head_dim > 0 and self.qk_rope_head_dim < self.head_dim:
            query = paddle.concat([query, query_nope], axis=-1)
            key = paddle.concat([key, key_nope], axis=-1)

        # ==================================
        # core attention computation
        # ==================================

        # NOTE: For sequence parallel, the input is [seq, b, h],
        # transpose back to [b, seq, h] for attention computation
        # TODO: supports [seq, b, h] input in attention computation
        if self.config.sequence_parallel:
            query = query.transpose([1, 0, 2, 3]).contiguous()
            key = key.transpose([1, 0, 2, 3]).contiguous()
            value = value.transpose([1, 0, 2, 3]).contiguous()
            # Slice and adjust attn_mask_startend_row_indices for the local SP sequence
            # range. The full mask has shape [B, 1, S, 1] with absolute row indices.
            # Each SP rank processes key/query positions [tp_rank*L : (tp_rank+1)*L],
            # so we need the local slice with row indices adjusted to local space.
            if (
                attn_mask_startend_row_indices is not None
                and self.core_attention.context_parallel_size == 1
            ):
                # Skip this adjustment when CP is active, as DotProductAttention
                # expects the full global mask and handles CP splitting internally.
                local_seq = key.shape[1]  # S / tp_size after transpose
                if attn_mask_startend_row_indices.shape[2] != local_seq:
                    tp_rank = get_pg_rank(self.pg_collection.tp)
                    offset = tp_rank * local_seq
                    attn_mask_startend_row_indices = paddle.clip(
                        attn_mask_startend_row_indices[
                            :, :, offset : offset + local_seq, :
                        ]
                        - offset,
                        min=0,
                    ).astype(paddle.int32)

        if self.recompute_core_attention and self.training:
            core_attn_out = recompute(
                self.core_attention,
                query,
                key,
                value,
                attention_mask.clone() if attention_mask is not None else None,
                attn_mask_startend_row_indices.clone()
                if attn_mask_startend_row_indices is not None
                else None,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention,
            )
        else:
            # Static batching attention kernel.
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_startend_row_indices,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention
                and in_recompute,
                past_key_values=past_key_values,
                layer_idx=layer_idx,
                use_cache=use_cache,
            )
        # =================
        # Output. [b, sq, h]
        # =================

        if self.config.sequence_parallel:
            core_attn_out = core_attn_out.transpose([1, 0, 2]).contiguous()

        core_attn_out = self._post_core_attention_hook(core_attn_out)

        # Apply gated attention: gate the attention output before output projection
        gate_recompute = None
        if gate is not None:
            if self.recompute_gated_attn and self.training:
                gate_recompute = RecomputeWithoutOutput()
                core_attn_out = gate_recompute.recompute(
                    self._gate_apply,
                    core_attn_out,
                    gate,
                    preserve_rng_state=False,
                    share_grad_holder=True,
                )
            else:
                core_attn_out = self._gate_apply(core_attn_out, gate)

        if self.config.gpt_model_use_experimental_version and _LOG_LAYER_MD5:
            import logging

            _rank = paddle.distributed.get_rank()
            _ca_md5 = _md5(core_attn_out)
            logging.getLogger(__name__).info(
                f"[MD5 Probe PF] Rank={_rank} Layer={self.layer_number} core_attn_out MD5={_ca_md5} shape={list(core_attn_out.shape)}"
            )
        if (
            self.config.gpt_model_use_experimental_version
            and self.o_proj.bias is not None
            and self.config.tensor_model_parallel_size == 1
        ):
            orig_shape = core_attn_out.shape
            core_attn_out = core_attn_out.reshape([-1, core_attn_out.shape[-1]])
            # Use fused_linear to match EC's FusedLinear behavior (fused_gemm_epilogue)
            output = paddle.incubate.nn.functional.fused_linear(
                core_attn_out, self.o_proj.weight, self.o_proj.bias
            )
            output = output.reshape(
                [orig_shape[0], orig_shape[1], output.shape[-1]]
            )
            bias = None
        else:
            output, bias = self.o_proj(core_attn_out)

        if gate_recompute is not None:
            gate_recompute.discard_output_and_register_recompute(output)

        if self.config.gpt_model_use_experimental_version and _LOG_LAYER_MD5:
            _out = output
            _o_md5 = _md5(_out)
            logging.getLogger(__name__).info(
                f"[MD5 Probe PF] Rank={_rank} Layer={self.layer_number} attn_o_proj_out MD5={_o_md5} shape={list(_out.shape)}"
            )

        return output, bias

    def _gate_apply(self, core_attn_out, gate):
        """Apply gated attention: sigmoid(gate) * core_attn_out."""
        return core_attn_out * paddle.nn.functional.sigmoid(gate)

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8."""
        raise NotImplementedError(
            "set_for_recompute_input_layernorm is not implemented."
        )


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: SelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
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

        self.gated_attention = getattr(self.config, "gated_attention", False)
        gate_projection_size = (
            self.out_projection_size if self.gated_attention else 0
        )
        if not self.config.gpt_model_use_experimental_version:
            self.qkv_proj = build_spec_layer(
                sublayers_spec.qkv_proj,
                self.config.hidden_size,
                self.query_projection_size
                + self.key_projection_size
                + self.value_projection_size
                + gate_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.use_bias or self.config.attention_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_group=self.pg_collection.tp,
            )
        else:
            self.qkv_proj = build_spec_layer(
                sublayers_spec.qkv_proj,
                self.config.hidden_size,
                self.query_projection_size
                + self.key_projection_size
                + self.value_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.use_bias or self.config.attention_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_group=self.pg_collection.tp,
            )
            if self.gated_attention:
                self.gate_proj = build_spec_layer(
                    sublayers_spec.gate_proj,
                    self.config.hidden_size,
                    gate_projection_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=self.config.use_bias or self.config.attention_bias,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_group=self.pg_collection.tp,
                )

        # For per_layer qk_norm, norm operates on gathered (full) tensors,
        # so input_is_parallel should be False to avoid extra allreduce.
        if getattr(self.config, "qk_norm_type", "per_head") == "per_layer":
            norm_input_parallel = False
        else:
            norm_input_parallel = config.tensor_model_parallel_size > 1

        if sublayers_spec.q_norm is not None:
            if getattr(self.config, "qk_norm_type", "per_head") == "per_layer":
                q_norm_hidden_size = (
                    self.hidden_size_per_attention_head
                    * self.num_attention_heads
                )
            else:
                q_norm_hidden_size = self.hidden_size_per_attention_head
            self.q_norm = build_spec_layer(
                sublayers_spec.q_norm,
                hidden_size=q_norm_hidden_size,
                config=self.config,
                eps=self.config.rms_norm_eps,
                input_is_parallel=norm_input_parallel,
            )
        else:
            self.q_norm = None

        if sublayers_spec.k_norm is not None:
            if getattr(self.config, "qk_norm_type", "per_head") == "per_layer":
                k_norm_hidden_size = (
                    self.hidden_size_per_attention_head
                    * self.num_key_value_heads
                )
            else:
                k_norm_hidden_size = self.hidden_size_per_attention_head
            self.k_norm = build_spec_layer(
                sublayers_spec.k_norm,
                hidden_size=k_norm_hidden_size,
                config=self.config,
                eps=self.config.rms_norm_eps,
                input_is_parallel=norm_input_parallel,
            )
        else:
            self.k_norm = None

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, split_qkv=True
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`. If `split_qkv=False`, then
        the unsplit mixed_qkv tensor is returned.
        When gated_attention is enabled, also returns a gate tensor for output gating.
        """
        # Attention heads [b, sq, h] --> [b, sq, ng * group_dim]
        mixed_qkv, _ = self.qkv_proj(hidden_states)

        if self.config.gpt_model_use_experimental_version:
            if self.gated_attention:
                gate_output = self.gate_proj(hidden_states)
                gate = (
                    gate_output[0]
                    if isinstance(gate_output, tuple)
                    else gate_output
                )
            else:
                gate = None

        heads_per_group = (
            self.num_attention_heads_per_partition
            // self.num_query_groups_per_partition
        )
        q_dim = heads_per_group * self.hidden_size_per_attention_head

        # EC-compatible QKV split: EC uses non-interleaved layout [all_Q | all_K | all_V]
        # while PF default uses per-group interleaved layout [G0: Q,K,V | G1: Q,K,V | ...]
        if self.config.gpt_model_use_experimental_version:
            # EC-style: reshape to [b, sq, num_heads + 2*num_kv_heads, head_dim], split on axis=2
            num_heads = self.num_attention_heads_per_partition
            num_kv_heads = self.num_query_groups_per_partition
            mixed_qkv = mixed_qkv.reshape(
                *mixed_qkv.shape[:-1],
                num_heads + 2 * num_kv_heads,
                self.hidden_size_per_attention_head,
            )

            if not split_qkv:
                split_arg_list = [num_heads, num_kv_heads, num_kv_heads]
                return mixed_qkv, split_arg_list
            query, key, value = paddle.split(
                mixed_qkv, [num_heads, num_kv_heads, num_kv_heads], axis=2
            )
        else:
            if self.gated_attention:
                gate_dim = (
                    heads_per_group * self.value_hidden_size_per_attention_head
                )

            if self.gated_attention:
                # Per group: Q + Gate + K + V
                group_dim = (
                    q_dim
                    + gate_dim
                    + self.hidden_size_per_attention_head
                    + self.value_hidden_size_per_attention_head
                )
            else:
                # Per group: Q + K + V
                group_dim = (
                    q_dim
                    + self.hidden_size_per_attention_head
                    + self.value_hidden_size_per_attention_head
                )

            # [b, sq, hp] --> [b, sq, ng, group_dim]
            new_tensor_shape = (
                *mixed_qkv.shape[:-1],
                self.num_query_groups_per_partition,
                group_dim,
            )
            mixed_qkv = mixed_qkv.reshape(*new_tensor_shape)

            if self.gated_attention:
                split_arg_list = [
                    q_dim,
                    gate_dim,
                    self.hidden_size_per_attention_head,
                    self.value_hidden_size_per_attention_head,
                ]
            else:
                split_arg_list = [
                    q_dim,
                    self.hidden_size_per_attention_head,
                    self.value_hidden_size_per_attention_head,
                ]

            # Return unsplit mixed_qkv and split_arg_list
            if not split_qkv:
                return mixed_qkv, split_arg_list

            parts = paddle.split(mixed_qkv, split_arg_list, axis=3)

            if self.gated_attention:
                query, gate, key, value = parts
            else:
                query, key, value = parts
                gate = None

        if self.v_scale is not None:
            value = value * self.v_scale

        if getattr(self.config, "qk_norm_type", "per_head") == "per_layer" and (
            self.q_norm is not None or self.k_norm is not None
        ):
            # per_layer qk_norm: normalize across all heads jointly

            # Flatten to [b, sq, np * hn] / [b, sq, ng * hn]
            query = query.reshape(*query.shape[:2], -1)
            key = key.reshape(*key.shape[:2], -1)

            # TP gather: collect all TP shards so norm sees the full dimension
            enable_tp = get_pg_size(self.pg_collection.tp) > 1
            if enable_tp:
                query = gather_from_tensor_model_parallel_region(
                    query, group=self.pg_collection.tp
                )
                key = gather_from_tensor_model_parallel_region(
                    key, group=self.pg_collection.tp
                )

            if self.q_norm is not None:
                query = self.q_norm(query)
            if self.k_norm is not None:
                key = self.k_norm(key)

            # TP scatter: split back to per-rank shards
            if enable_tp:
                query = scatter_to_tensor_model_parallel_region(
                    query, group=self.pg_collection.tp
                )
                key = scatter_to_tensor_model_parallel_region(
                    key, group=self.pg_collection.tp
                )

            # Reshape to per-head layout [b, sq, np, hn] / [b, sq, ng, hn]
            query = query.reshape(
                query.shape[0],
                query.shape[1],
                -1,
                self.hidden_size_per_attention_head,
            )
            key = key.reshape(
                key.shape[0],
                key.shape[1],
                -1,
                self.hidden_size_per_attention_head,
            )
        else:
            # per_head qk_norm (default): reshape first, then normalize per head
            # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
            query = query.reshape(
                query.shape[0],
                query.shape[1],
                -1,
                self.hidden_size_per_attention_head,
            )

            if self.q_norm is not None:
                query = self.q_norm(query)

            if self.k_norm is not None:
                key = self.k_norm(key)

        if gate is not None:
            # [b, sq, ng, np/ng * hn] -> [b, sq, np * hn]
            gate = gate.reshape(*gate.shape[:2], -1)
            return query, key, value, gate

        return query, key, value

    def backward_dw(self) -> NoReturn:
        """Execute weight update operations"""
        self._backward_qkv_proj()
        self._backward_output_proj()

    def _backward_qkv_proj(self):
        """Update weights for QKV projection layer"""
        self.qkv_proj.backward_dw()

    def _backward_output_proj(self):
        """Update weights for output projection layer"""
        self.o_proj.backward_dw()


class SelfAttentionVHA(Attention):
    """VHA (Virtual Head Attention) self-attention with independent q/k/v/gate projections."""

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: SelfAttentionVHASublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
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

        self.gated_attention = getattr(self.config, "gated_attention", False)

        # VHA-specific projection sizes
        if self.is_swa:
            self.q_head_dim = self.config.swa_vha_q_lora_rank
        else:
            self.q_head_dim = self.config.vha_q_lora_rank
        self.query_projection_size = self.q_head_dim * (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.hidden_size_per_attention_head = self.head_dim
        self.num_attention_heads_per_partition = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.num_query_groups_per_partition = self.num_key_value_heads

        # VHA requires TP == 1
        world_size = get_pg_size(self.pg_collection.tp)
        assert world_size == 1, (
            "VHA attention currently requires tensor_model_parallel_size == 1"
        )

        # Independent projections
        self.q_proj = build_spec_layer(
            sublayers_spec.q_proj,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias or self.config.attention_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )
        # PLACEHOLDER_VHA_CLASS_CONTINUE
        self.k_proj = build_spec_layer(
            sublayers_spec.k_proj,
            self.config.hidden_size,
            self.key_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias or self.config.attention_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )
        self.v_proj = build_spec_layer(
            sublayers_spec.v_proj,
            self.config.hidden_size,
            self.value_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias or self.config.attention_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        if self.gated_attention and sublayers_spec.gate_proj is not None:
            self.gate_proj = build_spec_layer(
                sublayers_spec.gate_proj,
                self.config.hidden_size,
                self.out_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.use_bias or self.config.attention_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_group=self.pg_collection.tp,
            )
        else:
            self.gate_proj = None

        # QK norms
        if sublayers_spec.q_norm is not None:
            if getattr(self.config, "qk_norm_type", "per_head") == "per_layer":
                q_norm_hidden_size = self.head_dim * self.num_attention_heads
            else:
                q_norm_hidden_size = self.head_dim
            self.q_norm = build_spec_layer(
                sublayers_spec.q_norm,
                hidden_size=q_norm_hidden_size,
                config=self.config,
                eps=self.config.rms_norm_eps,
                input_is_parallel=False,
            )
        else:
            self.q_norm = None
        # PLACEHOLDER_VHA_CLASS_PART2

        if sublayers_spec.k_norm is not None:
            if getattr(self.config, "qk_norm_type", "per_head") == "per_layer":
                k_norm_hidden_size = self.head_dim * self.num_key_value_heads
            else:
                k_norm_hidden_size = self.head_dim
            self.k_norm = build_spec_layer(
                sublayers_spec.k_norm,
                hidden_size=k_norm_hidden_size,
                config=self.config,
                eps=self.config.rms_norm_eps,
                input_is_parallel=False,
            )
        else:
            self.k_norm = None

        # VHA parameters
        if self.q_head_dim == self.head_dim:
            eye = paddle.eye(self.head_dim)
            init_mats = paddle.stack(
                [
                    eye
                    + paddle.randn([self.head_dim, self.head_dim])
                    * (0.1 / math.sqrt(self.head_dim))
                    for _ in range(self.num_key_value_heads)
                ]
            )
        else:
            init_mats = []
            for _ in range(self.num_key_value_heads):
                mat = paddle.empty([self.q_head_dim, self.head_dim])
                nn.initializer.Orthogonal()(mat)
                mat = mat * math.sqrt(self.head_dim / self.q_head_dim)
                init_mats.append(mat)
            init_mats = paddle.stack(init_mats)
        self.vha_premix_weight = self.create_parameter(
            shape=[self.num_key_value_heads, self.q_head_dim, self.head_dim],
            default_initializer=nn.initializer.Assign(init_mats),
        )
        vha_postmix_rank = (
            self.config.swa_vha_postmix_rank
            if self.is_swa
            else self.config.vha_postmix_rank
        )
        if vha_postmix_rank is None:
            vha_postmix_rank = self.num_attention_heads // 4
        self.vha_postmix_U = self.create_parameter(
            shape=[self.num_attention_heads, vha_postmix_rank],
            default_initializer=nn.initializer.Normal(mean=0.0, std=0.01),
        )
        self.vha_postmix_V = self.create_parameter(
            shape=[self.num_attention_heads, vha_postmix_rank],
            default_initializer=nn.initializer.Constant(0.0),
        )

        vha_postmix_rank_val = vha_postmix_rank

    def _apply_vha_premix(self, query: Tensor) -> Tensor:
        # query: [b, sq, g, q_head_dim], premix: [nkv, q_head_dim, head_dim]
        # output: [b, sq, nkv*g, head_dim] = [b, sq, nh, head_dim]
        q_expanded = paddle.einsum(
            "btgr,krd->btkgd", query, self.vha_premix_weight
        )
        return q_expanded.reshape(
            [
                query.shape[0],
                query.shape[1],
                self.num_attention_heads,
                self.head_dim,
            ]
        )

    def _apply_vha_postmix(self, attn_out: Tensor) -> Tensor:
        mixed = attn_out.reshape(
            [
                attn_out.shape[0],
                attn_out.shape[1],
                self.num_attention_heads,
                self.v_head_dim,
            ]
        )
        z = paddle.einsum("bthd,hr->btrd", mixed, self.vha_postmix_U)
        delta = paddle.einsum("btrd,hr->bthd", z, self.vha_postmix_V)
        mixed = mixed + delta
        return mixed.reshape(
            [
                attn_out.shape[0],
                attn_out.shape[1],
                self.num_attention_heads * self.v_head_dim,
            ]
        )

    def _post_core_attention_hook(self, core_attn_out: Tensor) -> Tensor:
        return self._apply_vha_postmix(core_attn_out)

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, split_qkv=True
    ):
        """Derives query, key, value (and optionally gate) from hidden_states."""
        return self._get_qkv_vha(hidden_states)

    def _get_qkv_vha(self, hidden_states):
        query, _ = self.q_proj(hidden_states)  # [b, sq, g*q_head_dim]
        key, _ = self.k_proj(hidden_states)  # [b, sq, nkv*hd]
        value, _ = self.v_proj(hidden_states)  # [b, sq, nkv*v_hd]

        gate = None
        if self.gated_attention and self.gate_proj is not None:
            gate, _ = self.gate_proj(hidden_states)  # [b, sq, nh*v_hd]

        if os.environ.get("VHA_DEBUG"):
            import logging

            layer_type = "SWA" if self.is_swa else "Full"
            logging.getLogger(__name__).info(
                f"[VHA-Runtime] layer={self.layer_number} type={layer_type} | "
                f"q={list(query.shape)} k={list(key.shape)} v={list(value.shape)} "
                f"gate={list(gate.shape) if gate is not None else None} | "
                f"premix_w={list(self.vha_premix_weight.shape)} "
                f"postmix_U={list(self.vha_postmix_U.shape)} "
                f"postmix_V={list(self.vha_postmix_V.shape)}"
            )

        # Reshape query for premix: [b, sq, g, q_head_dim] where g = nh // nkv
        query = query.reshape(
            query.shape[0],
            query.shape[1],
            self.num_attention_heads // self.num_key_value_heads,
            self.q_head_dim,
        )
        query = self._apply_vha_premix(query)  # -> [b, sq, nh, hd]

        if self.v_scale is not None:
            value = value * self.v_scale

        if getattr(self.config, "qk_norm_type", "per_head") == "per_layer" and (
            self.q_norm is not None or self.k_norm is not None
        ):
            query = query.reshape(*query.shape[:2], -1)
            key = key.reshape(*key.shape[:2], -1)
            if self.q_norm is not None:
                query = self.q_norm(query)
            if self.k_norm is not None:
                key = self.k_norm(key)
            query = query.reshape(
                query.shape[0], query.shape[1], -1, self.head_dim
            )
            key = key.reshape(key.shape[0], key.shape[1], -1, self.head_dim)
        else:
            # per_head norm
            if self.q_norm is not None:
                query = self.q_norm(query)
            key = key.reshape(key.shape[0], key.shape[1], -1, self.head_dim)
            if self.k_norm is not None:
                key = self.k_norm(key)

        value = value.reshape(
            value.shape[0], value.shape[1], -1, self.v_head_dim
        )

        if gate is not None:
            gate = gate.reshape(*gate.shape[:2], -1)
            return query, key, value, gate

        return query, key, value

    def backward_dw(self) -> NoReturn:
        """Execute weight update operations."""
        self.q_proj.backward_dw()
        self.k_proj.backward_dw()
        self.v_proj.backward_dw()
        if self.gate_proj is not None:
            self.gate_proj.backward_dw()
        self.o_proj.backward_dw()


class CrossAttention(Attention):
    """Cross-attention layer class

    Cross-attention layer takes input with size [s, b, h] and context with size
    [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: CrossAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        if self.config.num_key_value_heads != self.config.num_attention_heads:
            raise ValueError(
                "Group query attention is not currently supported in cross attention."
            )
        assert self.query_projection_size == self.key_projection_size

        self.linear_q = build_spec_layer(
            sublayers_spec.linear_q,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv = build_spec_layer(
            sublayers_spec.linear_kv,
            self.config.hidden_size,
            2 * self.key_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias,
            skip_bias_add=False,
            is_expert=False,
        )

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states, split_qkv=True
    ):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        assert split_qkv, "split_qkv must be True for CrossAttention"
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = (
            *mixed_kv.size()[:-1],
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        # Attention head [b, sq, h] --> [b, sq, hp]
        query, _ = self.linear_q(hidden_states)

        # [b, sq, hp] --> [b, sq, np, hn]
        new_tensor_shape = (
            *query.size()[:-1],
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        return query, key, value
