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
from dataclasses import dataclass

import paddle

# paddle.enable_compat()
import triton
import triton.language as tl

from .block_mask_utils import (
    check_fully_masked_state,
    check_partially_masked_state,
    find_blocks_topp,
)
from .index_utils import prepare_maxmin

LOG2E = 1.4426950408889634  # 1 / ln(2)


@triton.jit
def flashmask_apply(
    X,
    q_rows,
    base_offset,
    k_offsets,
    load_mask,
    lt_start_ptr,
    lt_end_ptr,
    ut_start_ptr,
    ut_end_ptr,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    INT_MAX: tl.constexpr = 2147483647
    INT_MIN: tl.constexpr = -2147483648

    pad_lt = INT_MAX
    pad_ut = INT_MIN

    lts = tl.load(lt_start_ptr + base_offset + k_offsets, mask=load_mask, other=pad_lt)
    if mode == 1:
        dense_mask = q_rows[:, None] >= lts[None, :]
    elif mode == 4:
        lte = tl.load(lt_end_ptr + base_offset + k_offsets, mask=load_mask, other=pad_lt)
        uts = tl.load(ut_start_ptr + base_offset + k_offsets, mask=load_mask, other=pad_ut)
        ute = tl.load(ut_end_ptr + base_offset + k_offsets, mask=load_mask, other=pad_ut)
        dense_mask = ((q_rows[:, None] >= lts[None, :]) & (q_rows[:, None] < lte[None, :])) | (
            (q_rows[:, None] >= uts[None, :]) & (q_rows[:, None] < ute[None, :])
        )
    else:
        if causal:
            lte = tl.load(
                lt_end_ptr + base_offset + k_offsets,
                mask=load_mask,
                other=pad_lt,
            )
            dense_mask = (q_rows[:, None] >= lts[None, :]) & (q_rows[:, None] < lte[None, :])
        else:
            ute = tl.load(
                ut_end_ptr + base_offset + k_offsets,
                mask=load_mask,
                other=pad_ut,
            )
            dense_mask = (q_rows[:, None] >= lts[None, :]) | (q_rows[:, None] < ute[None, :])

    X = (1.0 - dense_mask) * X  # set 0 for sum reduce
    return X, dense_mask


@triton.jit
def check_dense_contains_partial_stride(
    dense_flashmask,
    q_token_mask,
    k_token_mask,
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    dense_flashmask = tl.where(
        (q_token_mask[:, None] & k_token_mask[None, :]),
        dense_flashmask.to(tl.int32),
        tl.full([], 0, tl.int32),
    )
    mask_stride_cnt = dense_flashmask.reshape(BLOCK_SIZE // STRIDE, BLOCK_SIZE // STRIDE, STRIDE).sum(2)
    mask_stride_valid_cnt = k_token_mask.reshape(1, BLOCK_SIZE // STRIDE, STRIDE).to(tl.int32).sum(2)

    mask_stride_is_partial = (mask_stride_cnt > 0) & (mask_stride_cnt < mask_stride_valid_cnt)
    # return mask_stride_is_partial
    return tl.sum(mask_stride_is_partial.to(tl.int32)) > 0


@triton.jit
def gemm_fuse_softmax_causal(
    q,
    k,
    out,
    out_boundary_mask,
    # --- Mask Pointers ---
    lt_start_ptr,
    lt_end_ptr,
    ut_start_ptr,
    ut_end_ptr,
    lt_start_nstridemax,
    lt_start_nstridemin,
    lt_end_nstridemax,
    lt_end_nstridemin,
    ut_start_nstridemax,
    ut_start_nstridemin,
    ut_end_nstridemax,
    ut_end_nstridemin,
    # --- Params ---
    scale: float,
    seqlen_q: int,
    seqlen_k: int,
    num_q_blocks: int,
    num_k_blocks: int,
    N_STRIDES,
    STRIDE: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    HIDS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mode: tl.constexpr,
):
    i_block = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)

    ratio: tl.constexpr = BLOCK_SIZE // STRIDE
    G: tl.constexpr = HQ // H
    GIDS: tl.constexpr = HQ // HIDS

    i_hkv = i_h // G
    i_hid = i_h // GIDS

    # ================= 1. Coordinates Setup =================
    mask_ptr_base_bh_stride = i_b * N_STRIDES * HIDS + i_hid * N_STRIDES
    mask_ptr_base_bh_tokens = i_b * seqlen_k * HIDS + i_hid * seqlen_k

    # Load Q
    p_q = q + i_b * seqlen_q * HQ * K + (i_block * BLOCK_SIZE) * HQ * K + i_h * K
    p_q = p_q + tl.arange(0, ratio)[:, None] * (HQ * K * STRIDE) + tl.arange(0, K)[None, :] + HQ * K * (i_h % STRIDE)
    offs_tokens_q = tl.arange(0, ratio) * STRIDE + i_block * BLOCK_SIZE + (i_h % STRIDE)  # round-robin offset
    mask_q = offs_tokens_q < seqlen_q
    # mask_q = offs_tokens_q[:, None] < seqlen_q

    b_q = tl.load(p_q, mask=mask_q[:, None], other=0.0)
    b_q = (b_q * scale).to(b_q.dtype)

    # Softmax Accumulators
    m_i = tl.full([ratio], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([ratio], dtype=tl.float32)

    # Causal / FA3 Setup
    shift = seqlen_k - seqlen_q

    # k_safe_end: K blocks strictly to the left of the diagonal (Safe to compute fully)
    # Condition: k_block_end <= q_block_start + shift
    # (k + 1) * BLOCK <= i_block * BLOCK + shift
    k_safe_end = (i_block * BLOCK_SIZE + shift) // BLOCK_SIZE
    k_safe_end = min(num_k_blocks, max(0, k_safe_end))

    # k_valid_end: The last K block that intersects with the diagonal or Q block
    # Condition: k_block_start <= q_block_end_idx + shift
    # k * BLOCK <= ((i_block + 1) * BLOCK - 1) + shift
    k_valid_end = ((i_block + 1) * BLOCK_SIZE - 1 + shift) // BLOCK_SIZE + 1
    k_valid_end = min(num_k_blocks, max(k_safe_end, k_valid_end))

    p_k_base = k + i_b * seqlen_k * H * K + i_hkv * K
    offs_k_base = tl.arange(0, K)[:, None]
    offs_stride_k = tl.arange(0, ratio)
    offs_tokens_k = tl.arange(0, BLOCK_SIZE)

    # ================= 2. Loop 1: Statistics =================
    for iter in range(0, k_safe_end):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=True,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            # Load K & Compute Dot
            p_k = p_k_base + iter * BLOCK_SIZE * H * K + tl.arange(0, BLOCK_SIZE)[None, :] * H * K + offs_k_base
            b_k = tl.load(p_k)
            # CHANGE: NO REDUCE HERE
            # logits = tl.dot(b_q, b_k) # [ratio, BLOCK_SIZE]

            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=True,
                mode=mode,
            )

            real_partially_masked_stride_mask = (~fully_masked_stride_mask) & partially_masked_stride_mask
            if tl.sum(real_partially_masked_stride_mask) > 0:
                logits = tl.dot(b_q, b_k)  # [ratio, BLOCK_SIZE]
                curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE
                curr_token_load_mask = (iter * BLOCK_SIZE + offs_tokens_k) < seqlen_k
                X, dense_flashmask = flashmask_apply(
                    logits,
                    offs_tokens_q,
                    curr_token_offset,
                    offs_tokens_k,
                    curr_token_load_mask,
                    lt_start_ptr,
                    lt_end_ptr,
                    ut_start_ptr,
                    ut_end_ptr,
                    causal=True,
                    mode=mode,
                )

                # Reduce token logits to get stride score
                X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)
                fully_masked_by_fm = (dense_flashmask.reshape(ratio, ratio, STRIDE)).min(axis=2) == 1
                X = tl.where(fully_masked_by_fm, -1.0e6, X)
            else:
                # Reduce token logits to get stride score
                X = tl.dot(b_q, b_k.reshape(K, ratio, STRIDE).sum(2))

            X = tl.where(fully_masked_stride_mask, -1.0e6, X)

            # Update Stats
            m_local = tl.max(X, 1)
            m_new = tl.maximum(m_i, m_local)
            alpha = tl.math.exp2(m_i - m_new)
            X = X - m_new[:, None]
            l_local = tl.sum(tl.math.exp2(X), 1)
            l_i = l_i * alpha + l_local
            m_i = m_new

    for iter in range(k_safe_end, k_valid_end):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES
        # k_col_min = iter * BLOCK_SIZE

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=True,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            p_k = p_k_base + iter * BLOCK_SIZE * H * K + tl.arange(0, BLOCK_SIZE)[None, :] * H * K + offs_k_base
            mask_k = (tl.arange(0, BLOCK_SIZE)[None, :] + iter * BLOCK_SIZE) < seqlen_k
            b_k = tl.load(p_k, mask=mask_k, other=0.0)
            # b_k = b_k.reshape(K, ratio, STRIDE)
            # b_k = tl.sum(b_k, axis=2)
            logits = tl.dot(b_q, b_k)

            curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE
            curr_token_load_mask = (iter * BLOCK_SIZE + offs_tokens_k) < seqlen_k
            X, dense_flashmask = flashmask_apply(
                logits,
                offs_tokens_q,
                curr_token_offset,
                offs_tokens_k,
                curr_token_load_mask,
                lt_start_ptr,
                lt_end_ptr,
                ut_start_ptr,
                ut_end_ptr,
                causal=True,
                mode=mode,
            )
            global_offs_k = iter * BLOCK_SIZE + offs_tokens_k

            # Causal Condition: k_idx > q_idx + shift => Masked
            # Mask value: 0.0 (Identity for sum reduction of logits)
            causal_mask_token = global_offs_k[None, :] > (offs_tokens_q[:, None] + shift)
            X = tl.where(causal_mask_token, 0.0, X)

            # Reduce token logits to get stride score
            X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)
            # fully_masked_by_fm = dense_flashmask.reshape(ratio, ratio, STRIDE).min(axis=2) == 1
            causal_fuse_fm = causal_mask_token | dense_flashmask
            fully_masked_by_fm = causal_fuse_fm.reshape(ratio, ratio, STRIDE).min(axis=2) == 1
            X = tl.where(fully_masked_by_fm, -1.0e6, X)

            X = tl.where(fully_masked_stride_mask, -1.0e6, X)

            m_local = tl.max(X, 1)
            m_new = tl.maximum(m_i, m_local)
            alpha = tl.math.exp2(m_i - m_new)
            X = X - m_new[:, None]
            l_local = tl.sum(tl.math.exp2(X), 1)
            l_i = l_i * alpha + l_local
            m_i = m_new

    # ================= 3. Output Preparation =================
    l_i_inv = 1.0 / l_i

    stride_out_b = (HQ * num_q_blocks * num_k_blocks).to(tl.int64)
    stride_out_head = (num_q_blocks * num_k_blocks).to(tl.int64)
    stride_out_q = num_k_blocks.to(tl.int64)
    p_out = out + i_b * stride_out_b + i_h * stride_out_head + i_block * stride_out_q
    p_out_mask = out_boundary_mask + i_b * stride_out_b + i_h * stride_out_head + i_block * stride_out_q

    # ================= 4. Loop 2: Output (Exact Mirror) =================
    # 4.1 Non-Causal Blocks
    for iter in range(0, k_safe_end):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=True,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            # Load K & Compute Dot
            p_k = p_k_base + iter * BLOCK_SIZE * H * K + tl.arange(0, BLOCK_SIZE)[None, :] * H * K + offs_k_base
            b_k = tl.load(p_k)

            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=True,
                mode=mode,
            )

            real_partially_masked_stride_mask = (~fully_masked_stride_mask) & partially_masked_stride_mask

            if tl.sum(real_partially_masked_stride_mask) > 0:
                logits = tl.dot(b_q, b_k)  # [ratio, BLOCK_SIZE]

                curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE

                curr_token_load_mask = (iter * BLOCK_SIZE + offs_tokens_k) < seqlen_k

                X, dense_flashmask = flashmask_apply(
                    logits,
                    offs_tokens_q,
                    curr_token_offset,
                    offs_tokens_k,
                    curr_token_load_mask,
                    lt_start_ptr,
                    lt_end_ptr,
                    ut_start_ptr,
                    ut_end_ptr,
                    causal=True,
                    mode=mode,
                )

                # Reduce token logits to get stride score

                X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)
                fully_masked_by_fm = dense_flashmask.reshape(ratio, ratio, STRIDE).min(axis=2) == 1
                X = tl.where(fully_masked_by_fm, -1.0e6, X)

                has_partial = check_dense_contains_partial_stride(
                    dense_flashmask,
                    q_token_mask=mask_q,  # [ratio]
                    k_token_mask=curr_token_load_mask,  # [block_size]
                    BLOCK_SIZE=BLOCK_SIZE,
                    STRIDE=STRIDE,
                )
                tl.store(p_out_mask + iter, has_partial.to(tl.int8))

            else:
                X = tl.dot(b_q, b_k.reshape(K, ratio, STRIDE).sum(2))
                tl.store(p_out_mask + iter, tl.zeros([], dtype=tl.int8))

            X = tl.where(fully_masked_stride_mask, -1.0e6, X)

            # Normalization & Reduction
            X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
            X = tl.where(mask_q[:, None], X, 0)
            X = tl.where(m_i[:, None] < -1.0e5, 0, X)
            X = tl.sum(X, 1)  # Sum K-strides
            X = tl.sum(X, 0)  # Sum Q-tokens
            tl.store(p_out + iter, X.to(out.type.element_ty))

        else:
            tl.store(p_out + iter, tl.zeros([], dtype=out.type.element_ty))
            tl.store(p_out_mask + iter, tl.zeros([], dtype=tl.int8))

    # 4.2 Causal Block
    for iter in range(k_safe_end, k_valid_end):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio

        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=True,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            p_k = p_k_base + iter * BLOCK_SIZE * H * K + tl.arange(0, BLOCK_SIZE)[None, :] * H * K + offs_k_base

            mask_k = (tl.arange(0, BLOCK_SIZE)[None, :] + iter * BLOCK_SIZE) < seqlen_k

            b_k = tl.load(p_k, mask=mask_k, other=0.0)

            logits = tl.dot(b_q, b_k)
            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=True,
                mode=mode,
            )

            real_partially_masked_stride_mask = (~fully_masked_stride_mask) & partially_masked_stride_mask

            curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE

            curr_token_load_mask = (iter * BLOCK_SIZE + offs_tokens_k) < seqlen_k

            X, dense_flashmask = flashmask_apply(
                logits,
                offs_tokens_q,
                curr_token_offset,
                offs_tokens_k,
                curr_token_load_mask,
                lt_start_ptr,
                lt_end_ptr,
                ut_start_ptr,
                ut_end_ptr,
                causal=True,
                mode=mode,
            )

            global_offs_k = iter * BLOCK_SIZE + offs_tokens_k

            # Causal Condition: k_idx > q_idx + shift => Masked
            causal_mask_token = global_offs_k[None, :] > (offs_tokens_q[:, None] + shift)

            X = tl.where(causal_mask_token, 0.0, X)

            # Reduce token logits to get stride score
            X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)
            causal_fuse_fm = causal_mask_token | dense_flashmask
            fully_masked_by_fm = causal_fuse_fm.reshape(ratio, ratio, STRIDE).min(axis=2) == 1

            X = tl.where(fully_masked_by_fm, -1.0e6, X)
            has_partial = check_dense_contains_partial_stride(
                causal_fuse_fm,
                q_token_mask=mask_q,
                k_token_mask=curr_token_load_mask,
                BLOCK_SIZE=BLOCK_SIZE,
                STRIDE=STRIDE,
            )
            tl.store(p_out_mask + iter, has_partial.to(tl.int8))

            # Explicitly mask out fully masked stride blocks
            X = tl.where(fully_masked_stride_mask, -1.0e6, X)
            X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
            X = tl.where(m_i[:, None] < -1.0e5, 0, X)
            X = tl.where(mask_q[:, None], X, 0)
            X = tl.sum(X, 1)
            X = tl.sum(X, 0)
            tl.store(p_out + iter, X.to(out.type.element_ty))
        else:
            tl.store(p_out + iter, tl.zeros([], dtype=out.type.element_ty))
            tl.store(p_out_mask + iter, tl.zeros([], dtype=tl.int8))

    for iter in range(k_valid_end, num_k_blocks):
        tl.store(p_out + iter, tl.zeros([], dtype=out.type.element_ty))
        tl.store(p_out_mask + iter, tl.zeros([], dtype=tl.int8))


@triton.jit
def gemm_fuse_softmax_non_causal(
    q,
    k,
    out,
    out_boundary_mask,
    # --- Mask Pointers ---
    lt_start_ptr,
    lt_end_ptr,
    ut_start_ptr,
    ut_end_ptr,
    lt_start_nstridemax,
    lt_start_nstridemin,
    lt_end_nstridemax,
    lt_end_nstridemin,
    ut_start_nstridemax,
    ut_start_nstridemin,
    ut_end_nstridemax,
    ut_end_nstridemin,
    # --- Params ---
    scale: float,
    seqlen_q: int,
    seqlen_k: int,
    num_q_blocks: int,
    num_k_blocks: int,
    N_STRIDES,
    STRIDE: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    HIDS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mode: tl.constexpr,
):
    """
    Non-Causal (Bidirectional) Version:
    1. Loop over ALL K blocks (0 to num_k_blocks).
    2. No "Diagonal/Causal" check logic.
    3. Block Mask logic remains active (controlled by causal=False).
    """

    i_block = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)

    ratio: tl.constexpr = BLOCK_SIZE // STRIDE
    G: tl.constexpr = HQ // H
    GIDS: tl.constexpr = HQ // HIDS

    i_hkv = i_h // G
    i_hid = i_h // GIDS

    # ================= 1. Coordinates Setup =================

    mask_ptr_base_bh_stride = i_b * N_STRIDES * HIDS + i_hid * N_STRIDES
    mask_ptr_base_bh_tokens = i_b * seqlen_k * HIDS + i_hid * seqlen_k
    # Load Q (Round-Robin Sampling)

    p_q = q + i_b * seqlen_q * HQ * K + (i_block * BLOCK_SIZE) * HQ * K + i_h * K

    p_q = p_q + tl.arange(0, ratio)[:, None] * (HQ * K * STRIDE) + tl.arange(0, K)[None, :] + HQ * K * (i_h % STRIDE)

    offs_tokens_q = tl.arange(0, ratio) * STRIDE + i_block * BLOCK_SIZE + (i_h % STRIDE)

    mask_q = offs_tokens_q < seqlen_q
    b_q = tl.load(p_q, mask=mask_q[:, None], other=0.0)
    b_q = (b_q * scale).to(b_q.dtype)

    # Softmax Accumulators
    m_i = tl.full([ratio], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([ratio], dtype=tl.float32)

    # K Pointers Setup
    p_k_base = k + i_b * seqlen_k * H * K + i_hkv * K
    offs_k_base = tl.arange(0, K)[:, None]
    offs_stride_k = tl.arange(0, ratio)
    offs_tokens_k = tl.arange(0, BLOCK_SIZE)
    # ================= 2. Loop 1: Statistics =================
    # Iterate over ALL K blocks (No causal split)

    for iter in range(0, num_k_blocks):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        # [Check Fully Masked]
        # causal=False affects logic inside check (e.g. loads UT bounds)
        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=False,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            # Load K & Compute Dot
            p_k = p_k_base + iter * BLOCK_SIZE * H * K + tl.arange(0, BLOCK_SIZE)[None, :] * H * K + offs_k_base

            mask_k = (tl.arange(0, BLOCK_SIZE)[None, :] + iter * BLOCK_SIZE) < seqlen_k

            b_k = tl.load(p_k, mask=mask_k, other=0.0)
            # Compute Scores: [ratio, K] @ [K, BLOCK_SIZE] -> [ratio, BLOCK_SIZE]
            # logits = tl.dot(b_q, b_k)
            # [Check Partial Mask]

            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=False,
                mode=mode,
            )

            real_partially_masked_stride_mask = (~fully_masked_stride_mask) & partially_masked_stride_mask

            if tl.sum(real_partially_masked_stride_mask) > 0:
                logits = tl.dot(b_q, b_k)
                curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE
                curr_token_load_mask = (iter * BLOCK_SIZE + offs_tokens_k) < seqlen_k

                X, dense_flashmask = flashmask_apply(
                    logits,
                    offs_tokens_q,
                    curr_token_offset,
                    offs_tokens_k,
                    curr_token_load_mask,
                    lt_start_ptr,
                    lt_end_ptr,
                    ut_start_ptr,
                    ut_end_ptr,
                    causal=False,
                    mode=mode,
                )

                # Reduce token logits to get stride score
                X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)

                fully_masked_by_fm = dense_flashmask.reshape(ratio, ratio, STRIDE).min(axis=2) == 1
                X = tl.where(fully_masked_by_fm, -1.0e6, X)

            else:
                X = tl.dot(b_q, b_k.reshape(K, ratio, STRIDE).sum(2))

            # Explicitly mask out fully masked stride blocks
            X = tl.where(fully_masked_stride_mask, -1.0e6, X)

            # Update Stats
            m_local = tl.max(X, 1)
            m_new = tl.maximum(m_i, m_local)
            alpha = tl.math.exp2(m_i - m_new)
            X = X - m_new[:, None]
            l_local = tl.sum(tl.math.exp2(X), 1)
            l_i = l_i * alpha + l_local
            m_i = m_new

    # ================= 3. Output Preparation =================

    l_i_inv = 1.0 / l_i
    stride_out_b = (HQ * num_q_blocks * num_k_blocks).to(tl.int64)
    stride_out_head = (num_q_blocks * num_k_blocks).to(tl.int64)
    stride_out_q = num_k_blocks.to(tl.int64)

    p_out = out + i_b * stride_out_b + i_h * stride_out_head + i_block * stride_out_q

    p_out_mask = out_boundary_mask + i_b * stride_out_b + i_h * stride_out_head + i_block * stride_out_q

    # ================= 4. Loop 2: Output (Exact Mirror) =================

    for iter in range(0, num_k_blocks):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=False,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            p_k = p_k_base + iter * BLOCK_SIZE * H * K + tl.arange(0, BLOCK_SIZE)[None, :] * H * K + offs_k_base
            mask_k = (tl.arange(0, BLOCK_SIZE)[None, :] + iter * BLOCK_SIZE) < seqlen_k

            b_k = tl.load(p_k, mask=mask_k, other=0.0)

            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=False,
                mode=mode,
            )

            real_partially_masked_stride_mask = (~fully_masked_stride_mask) & partially_masked_stride_mask

            if tl.sum(real_partially_masked_stride_mask) > 0:
                logits = tl.dot(b_q, b_k)
                curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE

                curr_token_load_mask = (iter * BLOCK_SIZE + offs_tokens_k) < seqlen_k

                X, dense_flashmask = flashmask_apply(
                    logits,
                    offs_tokens_q,
                    curr_token_offset,
                    offs_tokens_k,
                    curr_token_load_mask,
                    lt_start_ptr,
                    lt_end_ptr,
                    ut_start_ptr,
                    ut_end_ptr,
                    causal=False,
                    mode=mode,
                )
                # Reduce token logits to get stride score
                X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)

                fully_masked_by_fm = dense_flashmask.reshape(ratio, ratio, STRIDE).min(axis=2) == 1
                X = tl.where(fully_masked_by_fm, -1.0e6, X)

                has_partial = check_dense_contains_partial_stride(
                    dense_flashmask,
                    q_token_mask=mask_q,
                    k_token_mask=curr_token_load_mask,
                    BLOCK_SIZE=BLOCK_SIZE,
                    STRIDE=STRIDE,
                )
                tl.store(p_out_mask + iter, has_partial.to(tl.int8))

            else:
                # Reduce token logits to get stride score
                X = tl.dot(b_q, b_k.reshape(K, ratio, STRIDE).sum(2))
                tl.store(p_out_mask + iter, tl.zeros([], dtype=tl.int8))

            X = tl.where(fully_masked_stride_mask, -1.0e6, X)

            # Normalization & Reduction
            X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
            X = tl.where(mask_q[:, None], X, 0)
            X = tl.where(m_i[:, None] < -1.0e5, 0, X)
            X = tl.sum(X, 1)  # Sum K-strides
            X = tl.sum(X, 0)  # Sum Q-tokens
            tl.store(p_out + iter, X.to(out.type.element_ty))

        else:
            tl.store(p_out + iter, tl.zeros([], dtype=out.type.element_ty))
            tl.store(p_out_mask + iter, tl.zeros([], dtype=tl.int8))


@dataclass(frozen=True)
class RawPtrs:
    # token-level: [B, HIDS, seqlen_q]
    lt_start: paddle.Tensor
    lt_end: paddle.Tensor
    ut_start: paddle.Tensor
    ut_end: paddle.Tensor


@dataclass(frozen=True)
class StrideMaxMinPtrs:
    # stride-level: [B, HIDS, n_strides]
    lt_start_max: paddle.Tensor
    lt_start_min: paddle.Tensor
    lt_end_max: paddle.Tensor
    lt_end_min: paddle.Tensor
    ut_start_max: paddle.Tensor
    ut_start_min: paddle.Tensor
    ut_end_max: paddle.Tensor
    ut_end_min: paddle.Tensor
    n_strides: int


def _require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


def _extract_raw_ptrs(
    startend_row_indices: paddle.Tensor,
    causal: bool,
) -> tuple[int, RawPtrs]:
    """
    startend_row_indices: [B, HIDS, seqlen_q, mode], mode in {1,2,4}
    - mode=1: only lt_start
    - mode=2:
        causal=True  -> (lt_start, lt_end)
        causal=False -> (lt_start, ut_end)
    - mode=4: (lt_start, lt_end, ut_start, ut_end)
    """
    mode = startend_row_indices.shape[-1]
    _require(mode in (1, 2, 4), f"Unsupported mode={mode}, expected 1/2/4")

    # 统一保证 contiguous
    x = startend_row_indices.contiguous()

    lt_start = x[..., 0].contiguous()

    lt_end = lt_start
    ut_start = lt_start
    ut_end = lt_start

    if mode == 2:
        if causal:
            lt_end = x[..., 1].contiguous()
        else:
            ut_end = x[..., 1].contiguous()
    elif mode == 4:
        lt_end = x[..., 1].contiguous()
        ut_start = x[..., 2].contiguous()
        ut_end = x[..., 3].contiguous()

    return mode, RawPtrs(lt_start=lt_start, lt_end=lt_end, ut_start=ut_start, ut_end=ut_end)


def _prepare_stride_maxmin_ptrs(
    raw: RawPtrs,
    mode: int,
    causal: bool,
    stride: int,
) -> StrideMaxMinPtrs:
    _require(stride > 0, "stride must be positive")

    lt_start_max, lt_start_min = prepare_maxmin(raw.lt_start, stride)
    n_strides = lt_start_max.shape[2]

    dummy_max = lt_start_max

    lt_end_max = lt_end_min = dummy_max
    ut_start_max = ut_start_min = dummy_max
    ut_end_max = ut_end_min = dummy_max

    if mode == 2:
        if causal:
            lt_end_max, lt_end_min = prepare_maxmin(raw.lt_end, stride)
        else:
            ut_end_max, ut_end_min = prepare_maxmin(raw.ut_end, stride)
    elif mode == 4:
        lt_end_max, lt_end_min = prepare_maxmin(raw.lt_end, stride)
        ut_start_max, ut_start_min = prepare_maxmin(raw.ut_start, stride)
        ut_end_max, ut_end_min = prepare_maxmin(raw.ut_end, stride)

    return StrideMaxMinPtrs(
        lt_start_max=lt_start_max,
        lt_start_min=lt_start_min,
        lt_end_max=lt_end_max,
        lt_end_min=lt_end_min,
        ut_start_max=ut_start_max,
        ut_start_min=ut_start_min,
        ut_end_max=ut_end_max,
        ut_end_min=ut_end_min,
        n_strides=n_strides,
    )


@paddle.use_compat_guard()
def rr_attn_estimate_triton_func(
    q: paddle.Tensor,
    k: paddle.Tensor,
    startend_row_indices: paddle.Tensor,
    stride: int = 8,
    causal: bool = True,
    threshold: float = 1.0,
) -> paddle.Tensor:
    """
    Returns:
      attn_sums: [B, HQ, ceil(seqlen_q/BS), ceil(seqlen_k/BS)]
      boundary_protection_mask: same shape, bool
    """
    _require(
        startend_row_indices.ndim == 4,
        "startend_row_indices must be [B, HIDS, seqlen_q, mode]",
    )

    bsz, q_len, num_q_heads, head_dim = q.shape
    bsz2, kv_len, num_kv_heads, _ = k.shape
    _require(bsz2 == bsz, "q/k batch size mismatch")

    _require(
        startend_row_indices.shape[0] == bsz,
        "startend_row_indices batch mismatch",
    )
    _require(
        startend_row_indices.shape[2] == kv_len,
        "startend_row_indices seqlen_k mismatch",
    )

    num_indices_heads = startend_row_indices.shape[1]
    _require(
        num_q_heads % num_kv_heads == 0,
        "MHA/GQA requires num_q_heads % num_kv_heads == 0",
    )
    _require(
        num_q_heads % num_indices_heads == 0,
        "Require num_q_heads % num_indices_heads == 0 for head mapping",
    )

    _require(
        startend_row_indices.place == q.place,
        "startend_row_indices must be on the same device as q",
    )
    _require(stride > 0, "stride must be positive")

    mode, raw = _extract_raw_ptrs(startend_row_indices, causal)
    stride_mm = _prepare_stride_maxmin_ptrs(raw, mode, causal, stride)

    # --- 5. Kernel Launch Setup ---
    BLOCK_SIZE = 128
    num_q_blocks = triton.cdiv(q_len, BLOCK_SIZE)
    num_k_blocks = triton.cdiv(kv_len, BLOCK_SIZE)

    attn_sums = paddle.empty(
        (bsz, num_q_heads, num_q_blocks, num_k_blocks),
        dtype=q.dtype,
    )

    boundary_protection_mask = paddle.empty(
        (bsz, num_q_heads, num_q_blocks, num_k_blocks),
        dtype=paddle.bool,
    )

    grid = (num_q_blocks, num_q_heads, bsz)

    scale = LOG2E / math.sqrt(head_dim) / stride

    kernel = gemm_fuse_softmax_causal if causal else gemm_fuse_softmax_non_causal

    kernel[grid](
        q,
        k,
        attn_sums,
        boundary_protection_mask,
        # raw pointers (token-level)
        raw.lt_start,
        raw.lt_end,
        raw.ut_start,
        raw.ut_end,
        # stride max/min pointers
        stride_mm.lt_start_max,
        stride_mm.lt_start_min,
        stride_mm.lt_end_max,
        stride_mm.lt_end_min,
        stride_mm.ut_start_max,
        stride_mm.ut_start_min,
        stride_mm.ut_end_max,
        stride_mm.ut_end_min,
        # meta
        scale=scale,
        seqlen_q=q_len,
        seqlen_k=kv_len,
        num_q_blocks=num_q_blocks,
        num_k_blocks=num_k_blocks,
        N_STRIDES=stride_mm.n_strides,
        STRIDE=stride,
        HQ=num_q_heads,
        H=num_kv_heads,
        HIDS=num_indices_heads,
        K=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        mode=mode,
    )

    return (
        attn_sums,
        boundary_protection_mask,
        find_blocks_topp(attn_sums, threshold),
    )
