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
element_mul_kernel：用于 backward 中对梯度做 grad_output 缩放。

当 grad_output != 1.0（非 mean reduction 或自定义 loss 权重）时，
将各梯度张量逐元素乘以标量 grad_output。
"""

import triton
import triton.language as tl

from ..triton_compat import enable_compat_on_triton_kernel


@enable_compat_on_triton_kernel
@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    原地将 X 的每行所有元素乘以标量 grad_output。
    """
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride
    grad_output = tl.load(grad_output_ptr)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(
            X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols
        )
