#!/usr/bin/env python3

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
Fused Cross Entropy Triton Kernel。

基于在线 Softmax 算法，在单次扫描中同时完成 loss 计算和梯度计算，
避免 Paddle 原生实现中保存完整 softmax 中间张量带来的显存开销。

Multimax variant (`liger_cross_entropy_multimax_kernel`) additionally fuses
the learnable SegLU-style segmented modulation
    SegLU(x) = x + t0·max(r0-x,0) + t1·max(x-r1,0)
                + t2·max(r2-x,0)^2 + t3·max(x-r3,0)^2
into the same kernel: SegLU is applied in registers during both the lse pass
and the grad-write pass; per-row partial sums for grad_ranges/grad_ts are
accumulated in registers and atomic-added to global [4]-shape fp32 buffers.
This avoids materializing the four ReLU intermediates and SegLU output as
separate tensors, dropping per-chunk peak memory back to ~1× [C, V].
"""

import triton
import triton.language as tl

from ..triton_compat import enable_compat_on_triton_kernel


@enable_compat_on_triton_kernel
@triton.jit
def liger_cross_entropy_kernel(  # pragma: no cover - triton kernel body compiles to PTX, not python-instrumentable
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
):
    """计算交叉熵 loss，并可选地原地写回梯度。"""
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(
                X_ptr + X_offsets,
                mask=X_offsets < n_cols,
                other=float("-inf"),
            ).cast(tl.float32)

            X_block = tl.exp(X_block - m) / d
            X_block = tl.where(X_offsets != y, X_block, X_block - 1.0)

            if reduction == "mean":
                X_block = X_block / n_non_ignore

            tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    tl.debug_barrier()

    loss = lse - ori_X_y

    if reduction == "mean":
        loss = loss / n_non_ignore

    tl.store(loss_ptr, loss)


@enable_compat_on_triton_kernel
@triton.jit
def liger_cross_entropy_multimax_kernel(  # pragma: no cover - triton kernel body compiles to PTX, not python-instrumentable
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    # Multimax learnable params, passed as fp32 scalars (compiled in once
    # per chunk; ranges/ts are tiny [4] tensors so the host->device sync
    # to extract them is negligible vs the chunk's GEMM/CE cost).
    r0,
    r1,
    r2,
    r3,
    t0,
    t1,
    t2,
    t3,
    # Per-chunk fp32 [4] grad accumulators for ranges/ts.
    grad_r_ptr,
    grad_t_ptr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
    HAS_MULTIMAX_GRADIENTS: tl.constexpr,
):
    """Liger CE kernel with fused SegLU activation + closed-form SegLU backward.

    Forward:  L = lse(SegLU(X)) - SegLU(X)[y]   (per row)
    Backward: writes grad_x = dL/dX in place into X_ptr; atomic-adds the
              per-row partial sums for grad_ranges and grad_ts into the
              fp32 [4] global buffers grad_r_ptr / grad_t_ptr.

    SegLU is computed in registers from the loaded X block, so no extra HBM
    traffic relative to the no-multimax kernel.
    """
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    # Apply SegLU to the y-th element (loss target) once.
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    my0 = tl.maximum(r0 - ori_X_y, 0.0)
    my1 = tl.maximum(ori_X_y - r1, 0.0)
    my2 = tl.maximum(r2 - ori_X_y, 0.0)
    my3 = tl.maximum(ori_X_y - r3, 0.0)
    seglu_X_y = ori_X_y + t0 * my0 + t1 * my1 + t2 * my2 * my2 + t3 * my3 * my3

    # Pass 1: online lse over SegLU(X).
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        in_bounds = X_offsets < n_cols
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=in_bounds,
            other=0.0,  # finite filler; we re-mask SegLU output to -inf below
        ).cast(tl.float32)
        m0_b = tl.maximum(r0 - X_block, 0.0)
        m1_b = tl.maximum(X_block - r1, 0.0)
        m2_b = tl.maximum(r2 - X_block, 0.0)
        m3_b = tl.maximum(X_block - r3, 0.0)
        seglu_b = (
            X_block
            + t0 * m0_b
            + t1 * m1_b
            + t2 * m2_b * m2_b
            + t3 * m3_b * m3_b
        )
        # Padded lanes contribute -inf to lse so they vanish in exp().
        seglu_b = tl.where(in_bounds, seglu_b, float("-inf"))
        block_max = tl.max(seglu_b)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(seglu_b - m_new))
        m = m_new

    lse = m + tl.log(d)

    if HAS_GRADIENTS or HAS_MULTIMAX_GRADIENTS:
        # Per-row scalar accumulators for grad_ranges and grad_ts.
        gr0 = 0.0
        gr1 = 0.0
        gr2 = 0.0
        gr3 = 0.0
        gt0 = 0.0
        gt1 = 0.0
        gt2 = 0.0
        gt3 = 0.0

        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            in_bounds = X_offsets < n_cols
            X_block = tl.load(
                X_ptr + X_offsets,
                mask=in_bounds,
                other=0.0,
            ).cast(tl.float32)
            # Recompute SegLU forward in registers (free vs HBM reload).
            m0_b = tl.maximum(r0 - X_block, 0.0)
            m1_b = tl.maximum(X_block - r1, 0.0)
            m2_b = tl.maximum(r2 - X_block, 0.0)
            m3_b = tl.maximum(X_block - r3, 0.0)
            seglu_b = (
                X_block
                + t0 * m0_b
                + t1 * m1_b
                + t2 * m2_b * m2_b
                + t3 * m3_b * m3_b
            )
            # Softmax over SegLU output, then subtract one-hot at target.
            grad_out = tl.exp(seglu_b - m) / d
            grad_out = tl.where(X_offsets != y, grad_out, grad_out - 1.0)
            if reduction == "mean":
                grad_out = grad_out / n_non_ignore
            # Padded lanes must contribute zero to grads (both stored grad
            # and the param-grad reductions).
            grad_out = tl.where(in_bounds, grad_out, 0.0)

            if HAS_GRADIENTS:
                # SegLU backward: d_out/d_x = 1 - t0*1{r0>x} + t1*1{x>r1}
                #                              - 2*t2*max(r2-x,0) + 2*t3*max(x-r3,0)
                mask0 = (m0_b > 0.0).to(tl.float32)
                mask1 = (m1_b > 0.0).to(tl.float32)
                dx_dx = (
                    1.0
                    - t0 * mask0
                    + t1 * mask1
                    - 2.0 * t2 * m2_b
                    + 2.0 * t3 * m3_b
                )
                grad_x = grad_out * dx_dx
                tl.store(X_ptr + X_offsets, grad_x, mask=in_bounds)

            if HAS_MULTIMAX_GRADIENTS:
                # Per-row partial sums for grad_ts and grad_ranges.
                #   d_out/d_t0 = m0,   d_out/d_t1 = m1
                #   d_out/d_t2 = m2^2, d_out/d_t3 = m3^2
                #   d_out/d_r0 =  t0*1{r0>x},  d_out/d_r1 = -t1*1{x>r1}
                #   d_out/d_r2 =  2*t2*m2,     d_out/d_r3 = -2*t3*m3
                mm0 = (m0_b > 0.0).to(tl.float32)
                mm1 = (m1_b > 0.0).to(tl.float32)
                gt0 += tl.sum(grad_out * m0_b)
                gt1 += tl.sum(grad_out * m1_b)
                gt2 += tl.sum(grad_out * m2_b * m2_b)
                gt3 += tl.sum(grad_out * m3_b * m3_b)
                gr0 += t0 * tl.sum(grad_out * mm0)
                gr1 += -t1 * tl.sum(grad_out * mm1)
                gr2 += 2.0 * t2 * tl.sum(grad_out * m2_b)
                gr3 += -2.0 * t3 * tl.sum(grad_out * m3_b)

        if HAS_MULTIMAX_GRADIENTS:
            # One atomic_add per row per scalar (8 total). Race-free across
            # programs and well below kernel runtime.
            tl.atomic_add(grad_r_ptr + 0, gr0)
            tl.atomic_add(grad_r_ptr + 1, gr1)
            tl.atomic_add(grad_r_ptr + 2, gr2)
            tl.atomic_add(grad_r_ptr + 3, gr3)
            tl.atomic_add(grad_t_ptr + 0, gt0)
            tl.atomic_add(grad_t_ptr + 1, gt1)
            tl.atomic_add(grad_t_ptr + 2, gt2)
            tl.atomic_add(grad_t_ptr + 3, gt3)

    tl.debug_barrier()

    loss = lse - seglu_X_y

    if reduction == "mean":
        loss = loss / n_non_ignore

    tl.store(loss_ptr, loss)
