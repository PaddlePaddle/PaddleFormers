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
Triton RMSNorm forward + backward kernels.

Features:
- Tuned for small dim (<= 1024), e.g. QK Norm.
- Supports strided input: works directly on split/slice tensors, no contiguous copy.
- Deterministic backward reduce: split into 2 kernels, atomic-free and efficient.
"""

import paddle

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@enable_compat_on_triton_kernel
@triton.jit
def rms_norm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    Invvar_ptr,
    stride_x_row: tl.constexpr,
    N1: tl.constexpr,
    actual_n2: tl.constexpr,  # actual normalize dim size
    BLOCK_N2: tl.constexpr,  # power of 2, >= actual_n2
    eps: tl.constexpr,
):
    """Forward kernel."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N2)
    mask = cols < actual_n2

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for row_idx in range(pid, N1, num_programs):
        x_offset = row_idx * stride_x_row
        y_offset = row_idx * actual_n2

        x = tl.load(X_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / actual_n2
        invvar = 1.0 / tl.sqrt(var + eps)

        y = x * invvar * w

        tl.store(Y_ptr + y_offset + cols, y, mask=mask)
        tl.store(Invvar_ptr + row_idx, invvar)


@enable_compat_on_triton_kernel
@triton.jit
def rms_norm_bwd_dx_kernel(
    DY_ptr,  # grad [n1, n2]
    X_ptr,  # forward input [n1, n2]
    W_ptr,  # weight [n2]
    Invvar_ptr,  # 1/rms [n1]
    DX_ptr,  # dx output [n1, n2]
    PartDW_ptr,  # dγ partial sum [NUM_PROGRAMS, BLOCK_N2]
    stride_x_row: tl.constexpr,
    N1: tl.constexpr,
    actual_n2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):
    """
    Backward kernel 1: compute dx and accumulate dγ partial sum.

    dx_ij = invvar_i * (dy_ij * w_j - x_ij * invvar_i^2 * dot_i / N2)
      where dot_i = sum_j(dy_ij * w_j * x_ij)

    part_dw[pid, j] = sum_{rows owned by pid}(dy_ij * x_ij * invvar_i)
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N2)
    mask = cols < actual_n2

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # dγ partial sum accumulator, in registers
    part_dw = tl.zeros([BLOCK_N2], dtype=tl.float32)

    for row_idx in range(pid, N1, num_programs):
        dy_offset = row_idx * actual_n2
        x_offset = row_idx * stride_x_row
        dx_offset = row_idx * actual_n2

        dy = tl.load(DY_ptr + dy_offset + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)
        invvar = tl.load(Invvar_ptr + row_idx)

        # dot_i = sum_j(dy_ij * w_j * x_ij)
        dy_w = dy * w
        dot = tl.sum(dy_w * x, axis=0)

        # dx_ij = invvar * (dy_ij * w_j - x_ij * invvar^2 * dot / N2)
        dx = invvar * (dy_w - x * (invvar * invvar) * (dot / actual_n2))

        tl.store(DX_ptr + dx_offset + cols, dx, mask=mask)

        # accumulate dγ local sum: dy_ij * x_ij * invvar_i
        part_dw += dy * x * invvar

    # output partial sum; each program outputs its own row
    tl.store(PartDW_ptr + pid * BLOCK_N2 + cols, part_dw, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit
def rms_norm_bwd_dw_partial_kernel(
    PartDW_ptr,  # [NUM_PARTS, BLOCK_N2] input
    TmpDW_ptr,  # [grid, BLOCK_N2] output (one row per program)
    NUM_PARTS: tl.constexpr,
    actual_n2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):
    """
    Backward kernel 2a: parallel reduce partial sums across programs.
    Each program strides through part_dw rows, accumulates in registers, outputs one row.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N2)
    mask = cols < actual_n2
    acc = tl.zeros([BLOCK_N2], dtype=tl.float32)

    for i in range(pid, NUM_PARTS, num_progs):
        acc += tl.load(PartDW_ptr + i * BLOCK_N2 + cols, mask=mask, other=0.0)

    tl.store(TmpDW_ptr + pid * BLOCK_N2 + cols, acc, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit
def rms_norm_bwd_dw_final_kernel(
    TmpDW_ptr,  # [NUM_REDUCE, BLOCK_N2] input
    DW_ptr,  # [n2] final output
    NUM_REDUCE,
    actual_n2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):
    """
    Backward kernel 2b: single-program final reduce.
    NUM_REDUCE is small (~64), trivial.
    """
    cols = tl.arange(0, BLOCK_N2)
    mask = cols < actual_n2
    dw = tl.zeros([BLOCK_N2], dtype=tl.float32)

    for i in range(NUM_REDUCE):
        dw += tl.load(TmpDW_ptr + i * BLOCK_N2 + cols, mask=mask, other=0.0)

    tl.store(DW_ptr + cols, dw.to(DW_ptr.dtype.element_ty), mask=mask)


class RMSNormFusionTriton(paddle.autograd.PyLayer):
    """Triton RMSNorm with autograd support."""

    @staticmethod
    def forward(ctx, x, weight, epsilon=1e-6):
        """forward"""
        orig_shape = x.shape
        n2 = x.shape[-1]
        block_n2 = triton.next_power_of_2(n2)
        n1 = 1
        for s in orig_shape[:-1]:
            n1 *= s

        if x.ndim >= 2:
            stride_x_row = x.stride()[x.ndim - 2]
        else:
            stride_x_row = n2

        y = paddle.empty(orig_shape, dtype=x.dtype)
        invvar = paddle.empty([n1], dtype=paddle.float32)

        ROWS_PER_PROG = 128
        num_programs = min(n1, max(1, (n1 + ROWS_PER_PROG - 1) // ROWS_PER_PROG))

        rms_norm_fwd_kernel[(num_programs,)](
            x,
            weight,
            y,
            invvar,
            stride_x_row,
            n1,
            n2,
            BLOCK_N2=block_n2,
            eps=epsilon,
            num_warps=1 if block_n2 <= 256 else 4,
        )

        ctx.save_for_backward(x, weight, invvar)
        ctx.n1 = n1
        ctx.n2 = n2
        ctx.block_n2 = block_n2
        ctx.stride_x_row = stride_x_row
        ctx.num_programs = num_programs
        return y

    @staticmethod
    def backward(ctx, dy):
        """backward"""
        x, weight, invvar = ctx.saved_tensor()
        n1 = ctx.n1
        n2 = ctx.n2
        block_n2 = ctx.block_n2
        stride_x_row = ctx.stride_x_row
        num_programs = ctx.num_programs

        dx = paddle.empty(dy.shape, dtype=dy.dtype)

        # partial sums buffer: [num_programs, block_n2]
        part_dw = paddle.empty([num_programs, block_n2], dtype=paddle.float32)

        # Kernel 1: dx + partial dγ
        rms_norm_bwd_dx_kernel[(num_programs,)](
            dy,
            x,
            weight,
            invvar,
            dx,
            part_dw,
            stride_x_row,
            n1,
            n2,
            BLOCK_N2=block_n2,
            num_warps=1 if block_n2 <= 256 else 4,
        )

        # Kernel 2a: parallel reduce of partial sums
        NUM_REDUCE = 64
        tmp_dw = paddle.empty([NUM_REDUCE, block_n2], dtype=paddle.float32)
        rms_norm_bwd_dw_partial_kernel[(NUM_REDUCE,)](
            part_dw,
            tmp_dw,
            num_programs,
            n2,
            BLOCK_N2=block_n2,
        )

        # Kernel 2b: single-program final reduce
        dw = paddle.empty([n2], dtype=weight.dtype)
        rms_norm_bwd_dw_final_kernel[(1,)](
            tmp_dw,
            dw,
            NUM_REDUCE,
            n2,
            BLOCK_N2=block_n2,
        )

        return dx, dw
