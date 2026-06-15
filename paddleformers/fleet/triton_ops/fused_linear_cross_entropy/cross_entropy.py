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
"""

import triton
import triton.language as tl

from ..triton_compat import enable_compat_on_triton_kernel


@enable_compat_on_triton_kernel
@triton.jit
def liger_cross_entropy_kernel(
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
