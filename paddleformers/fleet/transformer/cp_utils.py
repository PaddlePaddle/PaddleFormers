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

"""
Context Parallelism (CP) communication primitives and index utilities.

Contiguous CP layout: rank r holds global positions [r*sq_local, (r+1)*sq_local).
All-gather along seq dim produces natural global order; reduce-scatter inverts it.

This module has no internal dependencies on other fleet.transformer modules,
so it can be safely imported by both csa_attention.py without circular imports.
"""

from __future__ import annotations

import paddle
from paddle import Tensor

# ===========================================================================
# Differentiable all-gather — delegates to ContextParallelAllGatherOp
# ===========================================================================
from paddleformers.fleet.context_parallel_utils import ContextParallelAllGatherOp


def all_gather_cp(x: Tensor, dim: int, group) -> Tensor:
    """Differentiable all-gather for contiguous CP.

    Delegates to ContextParallelAllGatherOp (mode='contiguous_allgather'),
    which uses NCCL reduce_scatter in backward for axis=0 (more efficient).
    """
    if group is None or group.nranks <= 1:
        return x
    return ContextParallelAllGatherOp.apply(x, dim, "contiguous_allgather")


# ===========================================================================
# CP-aware topk index generators
# ===========================================================================


def get_window_topk_idxs_cp(
    q_positions: Tensor,
    window_size: int,
    batch_size: int,
    sq_global: int,
) -> Tensor:
    """Sliding window indices using global q_positions.

    Args:
        q_positions: [sq_local] int64, global positions for this rank.
        window_size: sliding window size.
        batch_size: batch dimension.
        sq_global: global sequence length.

    Returns:
        [batch_size, sq_local, window_size] int32, -1 for invalid slots.
    """
    effective_window = min(window_size, sq_global)
    base = q_positions.unsqueeze(1)  # [sq_local, 1]
    offsets = paddle.arange(effective_window)  # [window_size]
    k_pos = (
        paddle.clip(base - effective_window + 1, min=0) + offsets
    )  # [sq_local, window_size]
    topk_idxs = paddle.where(k_pos > base, paddle.full_like(k_pos, -1), k_pos)
    return topk_idxs.unsqueeze(0).expand([batch_size, -1, -1]).cast("int32")


def get_compress_topk_idxs_cp(
    q_positions: Tensor,
    ratio: int,
    batch_size: int,
    offset: int,
    n_compressed_global: int,
) -> Tensor:
    """Static compressed topk indices using global q_positions (HCA path).

    Args:
        q_positions: [sq_local] global positions.
        ratio: compression ratio.
        batch_size: batch dimension.
        offset: kv_full offset for compressed positions (= sq_global).
        n_compressed_global: total compressed positions globally.

    Returns:
        [batch_size, sq_local, n_compressed_global] int32, -1 for invalid.
    """
    k_group_idx = paddle.arange(n_compressed_global)  # [n_comp]
    q_first_invalid = ((q_positions + 1) // ratio).unsqueeze(1)  # [sq_local, 1]
    invalid_mask = k_group_idx.unsqueeze(0) >= q_first_invalid
    matrix = paddle.where(
        invalid_mask,
        paddle.full([1], -1, dtype="int64"),
        k_group_idx.unsqueeze(0) + offset,
    )
    return matrix.unsqueeze(0).expand([batch_size, -1, -1]).cast("int32")


def map_compressed_topk_to_kv_full_cp(
    topk_indices_compressed: Tensor,
    q_positions: Tensor,
    ratio: int,
    offset: int,
) -> Tensor:
    """Map indexer topk indices to kv_full coordinates with CP-aware causal check.

    Args:
        topk_indices_compressed: [b, sq_local, topk_eff] compressed block ids.
        q_positions: [sq_local] global positions.
        ratio: compression ratio.
        offset: kv_full offset (= sq_global).

    Returns:
        [b, sq_local, topk_eff] int32 indices into kv_full, -1 for invalid.
    """
    n_valid = (
        ((q_positions + 1) // ratio)
        .unsqueeze(0)
        .unsqueeze(2)
        .cast(topk_indices_compressed.dtype)
    )  # [1, sq_local, 1], same dtype as input
    valid = (topk_indices_compressed >= 0) & (topk_indices_compressed < n_valid)
    return paddle.where(
        valid,
        topk_indices_compressed + offset,
        paddle.full_like(topk_indices_compressed, -1),
    )


def build_causal_mask_cp(
    q_positions: Tensor,
    n_compressed_global: int,
    ratio: int,
    batch_size: int,
) -> Tensor:
    """Build causal mask for CSA indexer with global positions.

    Args:
        q_positions: [sq_local] global positions.
        n_compressed_global: total compressed positions globally.
        ratio: compression ratio.
        batch_size: batch dimension.

    Returns:
        [batch_size, sq_local, n_compressed_global] float32, -inf for invalid.
    """
    compressed_ids = paddle.arange(n_compressed_global).unsqueeze(
        0
    )  # [1, n_comp]
    q_first_invalid = ((q_positions + 1) // ratio).unsqueeze(1)  # [sq_local, 1]
    mask = paddle.where(
        compressed_ids >= q_first_invalid,
        paddle.full([1], float("-inf"), dtype="float32"),
        paddle.zeros([1], dtype="float32"),
    )  # [sq_local, n_comp]
    return mask.unsqueeze(0).expand([batch_size, -1, -1])
