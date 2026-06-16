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
import math
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from paddleformers.fleet.packed_seq_params import PackedSeqParams
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )

logger = logging.getLogger(__name__)

import paddle
from paddle import Tensor

from paddlefleet_ops.flash_mask_facade import (
    flash_attention,
    flashmask_attention,
)
from paddleformers.fleet.context_parallel_utils import flashmask_attention_cp
from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
from paddleformers.fleet.parallel_state import get_context_parallel_world_size
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.refined_recompute import (
    RefinedRcomputeFlashMaskAttention as rr_flashmask_attention,
    RefinedRcomputeFlashMaskCpAttention as rr_flashmask_attention_cp,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.layer import FleetLayer
from paddleformers.fleet.transformer.utils import (
    attention_mask_func,
    startend_row_indices_add_sliding_window,
)
from paddleformers.fleet.utils import divide


class DotProductAttention(FleetLayer):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        is_mtp_layer: bool = False,
        is_swa: bool = False,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        **kwargs,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.context_parallel_size = get_context_parallel_world_size()

        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now
        self.is_mtp_layer = is_mtp_layer
        self.is_swa = is_swa

        # k_channels and v_channels may differ from config.head_dim
        # Default to config.head_dim if not provided (standard attention)
        self.k_channels = (
            k_channels if k_channels is not None else self.config.head_dim
        )
        self.v_channels = (
            v_channels if v_channels is not None else self.config.head_dim
        )
        self.num_attention_heads = (
            num_attention_heads
            if num_attention_heads is not None
            else self.config.num_attention_heads
        )
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else self.config.num_key_value_heads
        )

        projection_size = self.k_channels * self.num_attention_heads

        # Per attention head and per partition values.
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp"]
            )
        else:
            assert hasattr(pg_collection, "tp"), (
                "DotProductAttention pg_collection must have tp process group"
            )

        world_size = (
            pg_collection.tp.world_size
            if pg_collection.tp is not None and pg_collection.tp.world_size >= 1
            else 1
        )
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(
            projection_size, self.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            self.num_attention_heads, world_size
        )
        self.num_query_groups_per_partition = divide(
            self.num_key_value_heads, world_size
        )

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(
                self.hidden_size_per_attention_head
            )
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.softmax_scale /= coeff

        if self.is_swa:
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        self.head_wise_swa_ratio = self.config.head_wise_swa_ratio
        self.sliding_window = sliding_window

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
            sliding_window=sliding_window,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = paddle.nn.Dropout(
            self.config.attention_dropout
            if attention_dropout is None
            else attention_dropout
        )

        softmax_type = self.config.softmax_type
        if (self.config.add_full_attention_sink_bias and not self.is_swa) or (
            self.config.add_swa_attention_sink_bias and self.is_swa
        ):
            softmax_type = "learnable"

        if softmax_type == "vanilla":
            self.softmax_offset = None
        elif softmax_type == "off-by-one":
            self.softmax_offset = paddle.zeros(
                self.num_attention_heads_per_partition
            )
        elif softmax_type == "learnable":
            self.softmax_offset = self.create_parameter(
                shape=[self.num_attention_heads_per_partition],
                dtype=self.config.params_dtype,
            )
            if config.perform_initialization:
                config.init_method(self.softmax_offset)
        else:
            raise ValueError("Softmax type not supported")
        self.rr_flashmask_attention_func = rr_flashmask_attention()
        self.rr_flashmask_attention_cp_func = rr_flashmask_attention_cp()

    def _ec_compatible_flash_attention(
        self, query, key, value, attn_mask_startend_row_indices=None
    ):
        """EC-compatible flash attention path for alignment mode.

        When startend_row_indices is provided (multi-doc packing), uses
        flashmask_attention with causal=True (matching EC behavior).
        Otherwise falls back to flash_attention with causal=True.
        """
        bsz, q_len, num_heads, q_head_dim = query.shape

        if attn_mask_startend_row_indices is not None:
            # flashmask path — matches EC's scaled_dot_product_attention
            if self.config.flashmask_use_varlen:
                flashmask_attention_func = partial(
                    flashmask_attention, use_varlen=True
                )
            else:
                flashmask_attention_func = flashmask_attention

            assert attn_mask_startend_row_indices.shape[-1] == 2
            attn_output = flashmask_attention_func(
                query.astype(value.dtype),
                key.astype(value.dtype),
                value,
                startend_row_indices=attn_mask_startend_row_indices,
                dropout=0.0,
                causal=False,  # EC uses causal=False with 2-col startend_row_indices
            )
        else:
            # simple causal path — no document boundaries
            attn_output, _ = flash_attention(
                query.astype(value.dtype),
                key.astype(value.dtype),
                value,
                dropout=0.0,
                causal=True,
                return_softmax=False,
            )

        attn_output = attn_output.reshape([bsz, q_len, -1])
        return attn_output

    def expand_attn_mask_startend_row_indices_for_cp(
        self, attn_mask_startend_row_indices, key
    ):
        """
        expand start_row_indice and end_row_indice
        """
        b, seq_len = key.shape[0], key.shape[1]
        seq_len = seq_len * self.context_parallel_size

        if attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = paddle.full(
                shape=[b, 1, seq_len, 1],
                fill_value=seq_len,
                dtype=paddle.int32,
            ).cuda()

        if attn_mask_startend_row_indices.shape[-1] == 1:
            b, k_heads, k_seqlen, _ = attn_mask_startend_row_indices.shape
            append_indices = paddle.to_tensor(
                np.arange(seq_len),
                dtype=attn_mask_startend_row_indices.dtype,
            ).cuda()
            append_indices = append_indices.reshape(1, 1, seq_len, 1)
            append_indices_expand = append_indices.expand(
                b, k_heads, k_seqlen, 1
            )
            attn_mask_startend_row_indices = paddle.concat(
                [attn_mask_startend_row_indices, append_indices_expand],
                axis=-1,
            )
        elif (
            attn_mask_startend_row_indices.shape[-1] == 2
            and self.config.experimental_dataflow
        ):
            # In EB dataflow, attn_mask_startend_row_indices.shape[-1] == 2
            # means attn_mask_startend_row_indices is ready, do not need to concat
            pass
        else:
            raise ValueError(
                "Invalid attention mask shape, when using context parallel, attn_mask_startend_row_indices.shape[-1] must be either 1 or 2"
            )
        return attn_mask_startend_row_indices

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_startend_row_indices: Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams | None = None,
        use_rr_flash_attention: bool = False,
        past_key_values=None,
        layer_idx=None,
        use_cache: bool = False,
        # DSA-specific parameters (ignored by DotProductAttention)
        x: Tensor | None = None,
        qr: Tensor | None = None,
        # fastdeploy specific parameters
        kv_compressed: paddle.Tensor = None,
        k_pos_emb: paddle.Tensor = None,
        q_absorbed: paddle.Tensor = None,
        v_b_proj_weight: paddle.Tensor = None,
    ):
        """Forward."""

        assert attention_bias is None, (
            "Attention bias is not supported for DotProductAttention."
        )
        assert not (
            use_rr_flash_attention and self.config.flashmask_use_varlen
        ), "flashmask_use_varlen does not support refined recompute now."

        use_eager = self.config._attn_implementation == "eager"

        if self.is_swa:
            assert not use_eager, (
                "SWA doesn't support _attn_implementation is eager"
            )

        if self.context_parallel_size > 1:
            assert packed_seq_params is None, (
                "Packed sequence is not supported by context_parallel_size > 1 now."
            )
            assert not self.config.flashmask_use_varlen, (
                "flashmask_use_varlen does not support context parallel now."
            )
            attn_mask_startend_row_indices = (
                self.expand_attn_mask_startend_row_indices_for_cp(
                    attn_mask_startend_row_indices, key
                )
            )
            assert (
                (
                    query.dtype == paddle.bfloat16
                    or query.dtype == paddle.float16
                )
                and attn_mask_startend_row_indices is not None
                and not use_eager
            )
        elif self.config.gpt_model_use_experimental_version:
            # EC-compatible flash attention path for alignment mode, only support non-cp
            return self._ec_compatible_flash_attention(
                query, key, value, attn_mask_startend_row_indices
            )

        bsz, q_len, num_heads, q_head_dim = query.shape
        v_head_dim = value.shape[-1]

        if use_eager and packed_seq_params is not None:
            raise ValueError(
                'packed_seq_params does not support _attn_implementation="eager"; '
                "please disable packed sequence inputs or use a fused attention implementation."
            )
        if packed_seq_params is not None:
            assert self.is_swa is False, "SWA doesn't support packed sequence"
            assert (
                query.dtype == paddle.bfloat16 or query.dtype == paddle.float16
            ), "attention only support fp16/bf16 when use packed_seq_params"

            if attn_mask_startend_row_indices is None:
                # Build flashmask startend_row_indices from cu_seqlens for block-diagonal
                # non-causal attention. Each token in segment i gets [end_i, total, 0, start_i],
                # so it attends only to tokens within its own segment.
                # This replaces the per-segment split + Python loop with a single FA call.
                cu_seqlens = packed_seq_params.cu_seqlens_kv
                seq_length = query.shape[1]
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                indices_per_segment = paddle.stack(
                    [
                        cu_seqlens[1:],  # col 0: lower_start = end_i
                        paddle.full_like(
                            cu_seqlens[1:], seq_length
                        ),  # col 1: lower_end   = total_seq
                        paddle.zeros_like(
                            cu_seqlens[:-1]
                        ),  # col 2: upper_start = 0
                        cu_seqlens[:-1],  # col 3: upper_end   = start_i
                    ],
                    axis=1,
                )  # [num_segments, 4]
                attn_mask_startend_row_indices = (
                    paddle.repeat_interleave(
                        indices_per_segment, lengths, axis=0
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )  # [1, 1, seq_len, 4]

            if use_rr_flash_attention:
                flashmask_attention_func = self.rr_flashmask_attention_func
            elif self.config.flashmask_use_varlen:
                flashmask_attention_func = partial(
                    flashmask_attention, use_varlen=True
                )
            else:
                flashmask_attention_func = flashmask_attention

            attn_output = flashmask_attention_func(
                query.astype(value.dtype),
                key.astype(value.dtype),
                value.astype(value.dtype),
                startend_row_indices=attn_mask_startend_row_indices,
                dropout=self.config.attention_dropout,
                causal=False,
            )
            attn_output = attn_output.reshape([bsz, q_len, -1])
            return attn_output
        if (
            (query.dtype == paddle.bfloat16 or query.dtype == paddle.float16)
            and attn_mask_startend_row_indices is None
            and not use_eager
        ):
            assert self.is_swa is False, (
                "SWA doesn't support scaled_dot_product_attention"
            )
            # KV cache support for inference
            if use_cache and past_key_values is not None:
                key, value = past_key_values.update(key, value, layer_idx)
                # During prefill (query_len > 1), is_causal=True handles causal masking.
                # During decode (query_len == 1), no causal mask needed; and KV length
                # = history + 1, so the original prefill attention_mask no longer matches
                # the extended KV length. Skip the mask in that case.
                is_causal = query.shape[1] > 1
                if query.shape[1] == 1:
                    attn_mask_kv = None
                else:
                    attn_mask_kv = attention_mask
            else:
                is_causal = True
                attn_mask_kv = attention_mask

            attn_output = paddle.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask_kv,
                self.config.attention_dropout,
                is_causal=is_causal,
            )

            attn_output = paddle.reshape(
                x=attn_output,
                shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]],
            )

            return attn_output

        elif (
            (query.dtype == paddle.bfloat16 or query.dtype == paddle.float16)
            and attn_mask_startend_row_indices is not None
            and not use_eager
        ):
            # Note:
            # attn_mask_startend_row_indices is not None for flashmask
            is_causal = attn_mask_type == AttnMaskType.causal
            if self.context_parallel_size > 1:
                flashmask_attention_func = (
                    self.rr_flashmask_attention_cp_func
                    if use_rr_flash_attention
                    else flashmask_attention_cp
                )
                is_causal = (
                    False  # only support non-causal for flashmask_attention_cp
                )
                assert attn_mask_startend_row_indices.shape[-1] == 2
            elif use_rr_flash_attention:
                flashmask_attention_func = self.rr_flashmask_attention_func
            elif self.config.flashmask_use_varlen:
                flashmask_attention_func = partial(
                    flashmask_attention, use_varlen=True
                )
            else:
                flashmask_attention_func = flashmask_attention

            # TODO(umiswing): move this padding to flash_mask_facade,
            # flash_mask_facade wrap the padding logic for fa/fm function call,
            # but it does not wrap the padding for rr now.
            # Handle MLA case where query/key head_dim != value head_dim
            # flashmask_attention requires head_dim_q == head_dim_v for backward pass
            need_value_padding = (
                use_rr_flash_attention and q_head_dim != v_head_dim
            )

            if need_value_padding:
                # Pad value to match query head_dim
                # value: [b, s, h, v_head_dim] -> [b, s, h, q_head_dim]
                bsz, seq_len, num_heads, _ = value.shape
                value_padding = paddle.zeros(
                    [bsz, seq_len, num_heads, q_head_dim - v_head_dim],
                    dtype=value.dtype,
                )
                value_padded = paddle.concat([value, value_padding], axis=-1)
            else:
                value_padded = value

            if self.sliding_window is not None:
                attn_mask_startend_row_indices = (
                    startend_row_indices_add_sliding_window(
                        attn_mask_startend_row_indices,
                        self.sliding_window,
                        self.head_wise_swa_ratio,
                        value.shape[2],
                    )
                )

            attn_output = flashmask_attention_func(
                query.astype(value.dtype),
                key.astype(value.dtype),
                value_padded.astype(value.dtype),
                startend_row_indices=attn_mask_startend_row_indices,
                dropout=self.config.attention_dropout,
                causal=is_causal,
            )

            if need_value_padding:
                # Truncate output back to original v_head_dim
                # attn_output: [b, s, h, q_head_dim] -> [b, s, h, v_head_dim]
                attn_output = attn_output[..., :v_head_dim]

            attn_output = attn_output.reshape([bsz, q_len, -1])

            return attn_output

        assert self.is_swa is False, (
            "SWA doesn't support scaled_dot_product_attention"
        )
        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [b, sk, ng, hn] -> [b, sk, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if (
            query.shape[2] != key.shape[2]
            and self.num_attention_heads_per_partition
            // self.num_query_groups_per_partition
            > 1
        ):
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition,
                dim=2,
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition,
                dim=2,
            )

        # [b, np, sq, sk]
        output_size = (
            query.shape[0],
            query.shape[2],
            query.shape[1],
            key.shape[1],
        )

        # [b, sq, np, hn] -> [b * np, sq, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use
        # simple strides to extract the queries.
        query = query.transpose([0, 2, 1, 3]).reshape(
            output_size[0] * output_size[1], output_size[2], -1
        )
        # [b, sk, np, hn] -> [b * np, hn, sk]
        key = key.transpose([0, 2, 3, 1]).reshape(
            output_size[0] * output_size[1], -1, output_size[3]
        )

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = paddle.empty(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query.dtype,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = paddle.baddbmm(
            matmul_input_buffer,
            query,
            key,
            beta=0.0,
            alpha=self.softmax_scale,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.reshape(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(
            attention_scores, attention_mask, self.softmax_offset
        )

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [b, sk, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.shape[0],
            value.shape[2],
            query.shape[1],
            value.shape[3],
        )

        # change view [b * np, sk, hn]
        value = value.transpose([0, 2, 1, 3]).reshape(
            output_size[0] * output_size[1], value.shape[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.reshape(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context = paddle.bmm(attention_probs, value)

        # change view [b, np, sq, hn]
        context = context.reshape(*output_size)

        # [b, np, sq, hn] --> [b, sq, np, hn]
        context = context.transpose([0, 2, 1, 3]).contiguous()

        # [b, sq, np, hn] --> [b, sq, hp]
        # use v_channels for output dimension (may differ from k_channels)
        new_context_shape = (
            *context.shape[:-2],
            self.hidden_size_per_partition // self.k_channels * self.v_channels,
        )
        context = context.reshape(*new_context_shape)

        return context
