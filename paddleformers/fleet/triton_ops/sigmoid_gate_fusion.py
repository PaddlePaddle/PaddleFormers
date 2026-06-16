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
Triton sigmoid gate forward + backward kernels.

Features:
- High performance by combining sigmoid and gate in one pass.
- Numerically precise sigmoid to match Paddle eager mode exactly.
- Supports fp16, bf16 and fp32 with deterministic backward.
"""

import paddle

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _sigmoid_precise(x):
    """
    sigmoid(x) = 1 / (1 + exp(-x))
    Note: to match paddle eager, use precise exp with rn div.
    """
    exp_neg = libdevice.exp(-x)
    return libdevice.div_rn(1.0, 1.0 + exp_neg)


@enable_compat_on_triton_kernel
@triton.jit
def fused_sigmoid_gate_fwd_kernel(
    attn_out_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    out = attn_out * sigmoid(gate)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    attn_out = tl.load(attn_out_ptr + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    act_out = _sigmoid_precise(gate.to(tl.float32)).to(gate.dtype)
    out = attn_out * act_out

    tl.store(out_ptr + offsets, out, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit
def fused_sigmoid_gate_bwd_kernel(
    out_grad_ptr,
    attn_out_ptr,
    gate_ptr,
    attn_out_grad_ptr,
    gate_grad_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    d_attn = dout * sigmoid(gate)
    d_gate = dout * attn_out * (1 - sigmoid(gate)) * sigmoid(gate)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    out_grad = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0)
    attn_out = tl.load(attn_out_ptr + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    act_out = _sigmoid_precise(gate.to(tl.float32)).to(gate.dtype)
    one = tl.full((BLOCK_SIZE,), 1, act_out.dtype)

    attn_out_grad = act_out * out_grad
    gate_grad = out_grad * attn_out * (one - act_out) * act_out

    tl.store(attn_out_grad_ptr + offsets, attn_out_grad, mask=mask)
    tl.store(gate_grad_ptr + offsets, gate_grad, mask=mask)


class SigmoidGateFusionTriton(paddle.autograd.PyLayer):
    """Triton sigmoid gate with autograd support."""

    @staticmethod
    def forward(ctx, attn_out, gate):
        """forward"""
        assert attn_out.shape == gate.shape, (
            f"attn_out and gate must have the same shape, but got {attn_out.shape} and {gate.shape}"
        )
        assert attn_out.dtype == gate.dtype, (
            f"attn_out and gate must have the same dtype, but got {attn_out.dtype} and {gate.dtype}"
        )
        assert attn_out.dtype in (
            paddle.float16,
            paddle.bfloat16,
            paddle.float32,
        ), f"Unsupported dtype for sigmoid gate: {attn_out.dtype}"

        out = paddle.empty_like(attn_out)
        n_elements = attn_out.size
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)

        fused_sigmoid_gate_fwd_kernel[grid](
            attn_out,
            gate,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
        )

        ctx.save_for_backward(attn_out, gate)
        ctx.n_elements = n_elements
        ctx.block_size = block_size
        return out

    @staticmethod
    def backward(ctx, out_grad):
        """backward"""
        attn_out, gate = ctx.saved_tensor()
        attn_out_grad = paddle.empty_like(attn_out)
        gate_grad = paddle.empty_like(gate)
        grid = (triton.cdiv(ctx.n_elements, ctx.block_size),)

        fused_sigmoid_gate_bwd_kernel[grid](
            out_grad,
            attn_out,
            gate,
            attn_out_grad,
            gate_grad,
            ctx.n_elements,
            BLOCK_SIZE=ctx.block_size,
        )

        return attn_out_grad, gate_grad
