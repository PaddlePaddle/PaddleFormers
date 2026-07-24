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

import paddle

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@enable_compat_on_triton_kernel
@triton.jit
def balanced_gather_reorder_kernel(
    src_ptr,
    dst_ptr,
    N,
    chunk_size,
    inner_size,
    src_rank_stride,
    src_outer_stride,
    num_blocks_per_chunk,
    BLOCK_SIZE: tl.constexpr,
):
    """Reorder gathered buffer from [N ranks stacked on axis0] to balanced layout.

    grid: (num_blocks_per_chunk * 2 * N, outer_size, 1)

    src layout (gathered along axis=0): [N*S0, S1, ..., S_axis, ..., S_last]
      - logically viewed as [N, S0, ..., S_{axis-1}, S_axis, S_{axis+1}, ..., S_last]
      - src_rank_stride = S0 * S1 * ... * S_last (elements per rank)
      - src_outer_stride = S_axis * inner_size (elements per outer index)
    dst layout: [S0, ..., S_{axis-1}, 2*N*chunk_size, S_{axis+1}, ..., S_last]
      - dst_outer_stride = 2*N*chunk_size * inner_size
    """
    pid_x = tl.program_id(0)
    outer_idx = tl.program_id(1)

    chunk_idx = pid_x // num_blocks_per_chunk
    block_idx = pid_x % num_blocks_per_chunk

    is_start = chunk_idx < N
    src_rank = tl.where(is_start, chunk_idx, 2 * N - 1 - chunk_idx)
    src_axis_base = tl.where(is_start, 0, chunk_size)

    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < chunk_size * inner_size

    axis_local = offsets // inner_size
    inner_idx = offsets % inner_size

    src_offset = (
        src_rank * src_rank_stride
        + outer_idx * src_outer_stride
        + (src_axis_base + axis_local) * inner_size
        + inner_idx
    )

    dst_outer_stride = 2 * N * chunk_size * inner_size
    dst_offset = (
        outer_idx * dst_outer_stride
        + (chunk_idx * chunk_size + axis_local) * inner_size
        + inner_idx
    )

    data = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, data, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit
def balanced_scatter_reorder_kernel(
    src_ptr,
    dst_ptr,
    N,
    chunk_size,
    inner_size,
    src_outer_stride,
    dst_rank_stride,
    dst_outer_stride,
    num_blocks_per_chunk,
    BLOCK_SIZE: tl.constexpr,
):
    """Reorder balanced layout to per-rank [start_i, end_i] buffers for alltoall.

    grid: (num_blocks_per_chunk * 2 * N, outer_size, 1)

    src layout (balanced): [S0, ..., 2*N*chunk_size, ..., S_last]
      - src_outer_stride = 2*N*chunk_size * inner_size
    dst layout: [N, S0, ..., 2*chunk_size, ..., S_last] (contiguous buffer)
      - dst_rank_stride = outer_size * 2*chunk_size * inner_size
      - dst_outer_stride = 2*chunk_size * inner_size
    """
    pid_x = tl.program_id(0)
    outer_idx = tl.program_id(1)

    slot_idx = pid_x // num_blocks_per_chunk
    block_idx = pid_x % num_blocks_per_chunk

    rank_idx = slot_idx // 2
    is_start = (slot_idx % 2) == 0

    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < chunk_size * inner_size

    axis_local = offsets // inner_size
    inner_idx = offsets % inner_size

    # src: balanced layout [start_r0, start_r1, ..., end_rN-1, ..., end_r0]
    src_axis_pos = tl.where(
        is_start,
        rank_idx * chunk_size + axis_local,
        (2 * N - 1 - rank_idx) * chunk_size + axis_local,
    )
    src_offset = (
        outer_idx * src_outer_stride + src_axis_pos * inner_size + inner_idx
    )

    # dst: rank_idx's buffer with [start | end] along axis
    dst_axis_pos = tl.where(is_start, axis_local, chunk_size + axis_local)
    dst_offset = (
        rank_idx * dst_rank_stride
        + outer_idx * dst_outer_stride
        + dst_axis_pos * inner_size
        + inner_idx
    )

    data = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, data, mask=mask)
