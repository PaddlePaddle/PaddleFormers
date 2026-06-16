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

import math

import paddle
import triton
import triton.language as tl


@triton.jit
def _load_bounds(
    base_offset,
    k_offsets,
    load_mask,
    ptr_start_lt,
    ptr_end_lt,
    ptr_start_ut,
    ptr_end_ut,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    INT_MAX: tl.constexpr = 2147483647
    INT_MIN: tl.constexpr = -2147483648

    pad_lt = INT_MAX
    pad_ut = INT_MIN

    b_lts = tl.load(
        ptr_start_lt + base_offset + k_offsets, mask=load_mask, other=pad_lt
    )

    need_lte: tl.constexpr = (causal and mode == 2) or (
        not causal and mode == 4
    )

    if need_lte:
        b_lte = tl.load(
            ptr_end_lt + base_offset + k_offsets, mask=load_mask, other=pad_lt
        )
    else:
        b_lte = tl.full(b_lts.shape, pad_lt, dtype=tl.int32)

    if causal:
        b_uts = tl.full(b_lts.shape, pad_ut, dtype=tl.int32)
    else:
        if mode == 4:
            b_uts = tl.load(
                ptr_start_ut + base_offset + k_offsets,
                mask=load_mask,
                other=pad_ut,
            )
        else:
            b_uts = tl.full(b_lts.shape, pad_ut, dtype=tl.int32)

    need_ute: tl.constexpr = (not causal) and (mode == 2 or mode == 4)

    if need_ute:
        b_ute = tl.load(
            ptr_end_ut + base_offset + k_offsets, mask=load_mask, other=pad_ut
        )
    else:
        b_ute = tl.full(b_lts.shape, pad_ut, dtype=tl.int32)

    return b_lts, b_lte, b_uts, b_ute


@triton.jit
def _is_block_fully_masked(
    block_rows,
    lts_max,
    lte_min,
    uts_max,
    ute_min,
):
    # since we pass exact row indices now, use "<" for end
    in_lt = (block_rows[:, None] >= lts_max[None, :]) & (
        block_rows[:, None] < lte_min[None, :]
    )
    in_ut = (block_rows[:, None] >= uts_max[None, :]) & (
        block_rows[:, None] < ute_min[None, :]
    )

    mask = in_lt | in_ut
    return mask


@triton.jit
def check_fully_masked_state(
    mask_ptr_base_offset,
    k_offsets,
    k_load_mask,
    q_rows,
    ptrs_strict_lt_start,
    ptrs_strict_lt_end,
    ptrs_strict_ut_start,
    ptrs_strict_ut_end,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    fm_lts, fm_lte, fm_uts, fm_ute = _load_bounds(
        mask_ptr_base_offset,
        k_offsets,
        k_load_mask,
        ptrs_strict_lt_start,
        ptrs_strict_lt_end,
        ptrs_strict_ut_start,
        ptrs_strict_ut_end,
        causal=causal,
        mode=mode,
    )

    fm_geo = _is_block_fully_masked(
        q_rows,
        fm_lts,
        fm_lte,
        fm_uts,
        fm_ute,
    )
    fm_oob = ~k_load_mask[None, :]

    return fm_geo | fm_oob


@triton.jit
def _is_block_partially_masked(
    block_rows,
    lts_min,
    lte_max,
    uts_min,
    ute_max,
):
    # Logic: Overlap exists if Q is potentially inside [min_start, max_end)
    overlap_lt = (block_rows[:, None] < lte_max[None, :]) & (
        block_rows[:, None] >= lts_min[None, :]
    )
    overlap_ut = (block_rows[:, None] < ute_max[None, :]) & (
        block_rows[:, None] >= uts_min[None, :]
    )

    return overlap_lt | overlap_ut


@triton.jit
def check_partially_masked_state(
    mask_ptr_base_offset,
    k_offsets,
    k_load_mask,
    q_rows,
    ptrs_perm_lt_start,
    ptrs_perm_lt_end,
    ptrs_perm_ut_start,
    ptrs_perm_ut_end,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    pm_lts, pm_lte, pm_uts, pm_ute = _load_bounds(
        mask_ptr_base_offset,
        k_offsets,
        k_load_mask,
        ptrs_perm_lt_start,
        ptrs_perm_lt_end,
        ptrs_perm_ut_start,
        ptrs_perm_ut_end,
        causal=causal,
        mode=mode,
    )

    return _is_block_partially_masked(q_rows, pm_lts, pm_lte, pm_uts, pm_ute)


@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)

    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(
        tl.sum(tl.where(mask == 0, y, 0), 1)[:, None, :], shape
    ).to(y.dtype)
    right = tl.broadcast_to(
        tl.sum(tl.where(mask == 1, y, 0), 1)[:, None, :], shape
    ).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)

    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(
        bitwidth=x.dtype.primitive_bitwidth, signed=True
    )
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)

    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage

    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order

    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def bitonic_argsort_device(
    x, ids, n_dims: tl.constexpr, descending: tl.constexpr = tl.core.CONSTEXPR_0
):
    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(
            x, ids, i, 2 if i < n_dims else descending, n_dims
        )

    return x, ids


@triton.jit
def top_p_kernel(
    X_ptr,
    Out_ptr,
    stride_row,
    threshold_p,
    N_COLS,
    BLOCK_SIZE: tl.constexpr,
    NUM_DIMS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start_ptr = X_ptr + pid * stride_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask_load = offsets < N_COLS

    # Load with 0.0 padding to calculate correct sum
    x_raw = tl.load(row_start_ptr + offsets, mask=mask_load, other=0.0).to(
        tl.float32
    )
    row_sum = tl.sum(x_raw, axis=0)

    out_row_ptr = Out_ptr + pid * stride_row
    if row_sum == 0.0:
        tl.store(
            out_row_ptr + offsets,
            tl.zeros([BLOCK_SIZE], dtype=tl.int8),
            mask=mask_load,
        )
        return

    # Calculate the actual threshold value based on the sum
    actual_cutoff = row_sum * threshold_p

    padding_val = float("-inf")
    x_for_sort = tl.where(mask_load, x_raw, padding_val)
    ids = tl.arange(
        0, BLOCK_SIZE
    )  # Initialize indices [0, 1, ... BLOCK_SIZE-1]

    # Perform Bitonic Sort (Descending)
    # x_sorted: values from high to low
    # ids_sorted: original indices corresponding to those values
    x_sorted, ids_sorted = bitonic_argsort_device(
        x_for_sort, ids, NUM_DIMS, descending=1
    )

    cum_probs = tl.cumsum(x_sorted, axis=0)
    mask_keep = (cum_probs - x_sorted) < actual_cutoff

    # Force padding elements to be False (just in case)
    is_not_padding = x_sorted > padding_val
    mask_keep = mask_keep & is_not_padding

    # Scatter Write (Restore original order)
    mask_store = ids_sorted < N_COLS
    tl.store(out_row_ptr + ids_sorted, mask_keep.to(tl.int8), mask=mask_store)


def find_blocks_topp(x: paddle.Tensor, p: float):
    """
    Input:
        x: [b, h, m, n] float tensor (probabilities, unnormalized)
        p: float, threshold
    Output:
        mask: [b, h, m, n] bool tensor
    """
    original_shape = x.shape
    n = original_shape[-1]

    x_reshaped = x.reshape(-1, n).contiguous()
    B = x_reshaped.shape[0]  # Total number of rows

    block_size = triton.next_power_of_2(n)
    if block_size < 1:
        block_size = 1
    num_dims = int(math.log2(block_size))

    output_mask = paddle.empty(
        x_reshaped.shape, dtype=paddle.bool, device=x.device
    )

    grid = (B,)

    top_p_kernel[grid](
        x_reshaped,
        output_mask,
        x_reshaped.strides[0],
        p,
        n,
        BLOCK_SIZE=block_size,
        NUM_DIMS=num_dims,
    )

    return output_mask.reshape(original_shape)
