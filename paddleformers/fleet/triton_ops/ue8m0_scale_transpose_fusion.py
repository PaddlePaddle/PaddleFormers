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

"""Triton kernel for converting fused stack UE8M0 scales to transpose layout."""

import paddle

from paddleformers.fleet.triton_ops.utils import is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@triton.jit
def _fuse_stack_ue8m0_scale_transpose_kernel(
    scale_ptr,
    out_ptr,
    num_experts: tl.constexpr,
    m: tl.constexpr,
    k: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col_group = tl.program_id(1)

    k_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_col_group * 4 + tl.arange(0, 4)

    expert_idx = k_offsets // k
    local_k = k_offsets - expert_idx * k
    k_block = local_k // 128
    m_blocks = col_offsets

    k_block_group = k_block // 4
    k_block_inner = k_block - k_block_group * 4
    src_offsets = (expert_idx[:, None] * m + m_blocks[None, :] * 128) * (
        k // 512
    ) + k_block_group[:, None]
    packed_scale = tl.load(
        scale_ptr + src_offsets, mask=expert_idx[:, None] < num_experts, other=0
    )
    exp = (packed_scale >> (k_block_inner[:, None] * 8)) & 0xFF
    packed = tl.sum(exp << (tl.arange(0, 4)[None, :] * 8), axis=1)

    out_offsets = k_offsets * (m // 512) + pid_col_group
    tl.store(out_ptr + out_offsets, packed, mask=k_offsets < num_experts * k)


class FuseStackUe8m0ScaleTransposeTriton(paddle.autograd.PyLayer):
    """Convert fuse_stack_fp8_quant UE8M0 scale to transpose-quant scale layout."""

    @staticmethod
    def forward(ctx, scale, num_experts, m, k):
        assert scale.dtype == paddle.int32, (
            f"scale must be int32, got {scale.dtype}"
        )
        assert len(scale.shape) == 2, f"scale must be 2-D, got {scale.shape}"
        assert num_experts >= 0, (
            f"num_experts must be non-negative, got {num_experts}"
        )
        assert m >= 0, f"m must be non-negative, got {m}"
        assert k >= 0, f"k must be non-negative, got {k}"
        assert m % 512 == 0, (
            f"m must be divisible by 512 for packed UE8M0 scale, got {m}"
        )
        assert k % 512 == 0, (
            f"k must be divisible by 512 for packed UE8M0 scale, got {k}"
        )
        assert scale.shape == [num_experts * m, k // 512], (
            f"scale shape must be {[num_experts * m, k // 512]}, got {scale.shape}"
        )

        out = paddle.empty([num_experts * k, m // 512], dtype=paddle.int32)
        if num_experts == 0 or m == 0 or k == 0:
            return out

        block_rows = 128
        grid = (triton.cdiv(num_experts * k, block_rows), m // 512)
        _fuse_stack_ue8m0_scale_transpose_kernel[grid](
            scale,
            out,
            num_experts,
            m,
            k,
            BLOCK_ROWS=block_rows,
            num_warps=4,
        )
        return out

    @staticmethod
    def backward(ctx, out_grad):
        return None, None, None, None


def fuse_stack_ue8m0_scale_transpose(scale, num_experts, m, k):
    """Convert non-transposed fused stack UE8M0 scale to transposed scale layout."""
    return FuseStackUe8m0ScaleTransposeTriton.apply(scale, num_experts, m, k)
