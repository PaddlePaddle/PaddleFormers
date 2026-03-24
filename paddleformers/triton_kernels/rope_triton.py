# Copyright (c) 2025, Tri Dao.
# Ported from Flash Attention: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
# Original code licensed under BSD-3-Clause license.

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

"""Triton Rotary Position Embedding for PaddlePaddle."""

import paddle

from ..utils.log import logger

# NOTE: Currently, global enabling is NOT supported.
# paddle.enable_compat(scope={"triton"})

try:
    import triton
    import triton.language as tl
except:
    logger.warning("Triton is not installed" "Please run 'python -m pip install triton>=3.1' to install Triton.")


IS_TRITON_IN_PADDLE_AVAILABLE = False
try:
    import use_triton_in_paddle

    use_triton_in_paddle.make_triton_compatible_with_paddle()
    IS_TRITON_IN_PADDLE_AVAILABLE = True
except:
    logger.warning(
        "Triton is installed, but not yet compatible with Paddle. "
        "Please run 'python -m pip install use-triton-in-paddle' to enable Triton support in Paddle."
    )


@triton.jit
def _rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    # Dimensions
    seqlen,
    nheads,
    seqlen_ro,
    # Strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    ROTARY_DIM: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Triton kernel for rotary position embedding.
    Ported directly from Flash Attention's rotary_kernel.

    Grid: (cdiv(nheads, BLOCK_H), cdiv(seqlen, BLOCK_M), batch)

    cos/sin shape: [seqlen, rotary_dim/2]

    Forward:
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos

    Backward (conjugate=True, sin = -sin):
        o0 = x0 * cos + x1 * sin
        o1 = x0 * (-sin) + x1 * cos = -x0 * sin + x1 * cos
    """
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2

    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    # Apply batch offset
    X = X + pid_batch * stride_x_batch
    OUT = OUT + pid_batch * stride_out_batch

    # Early exit if out of bounds
    if pid_m * BLOCK_M >= seqlen:
        return

    # Head and sequence indices
    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, BLOCK_K // 2)

    # Load cos and sin: [seqlen, rotary_dim/2]
    COS = COS + (rm[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    SIN = SIN + (rm[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)

    if CONJUGATE:
        sin = -sin

    # Compute pointers for X and OUT
    X = X + (
        rh[:, None, None] * stride_x_nheads
        + rm[None, :, None] * stride_x_seqlen
        + rk_half[None, None, :] * stride_x_headdim
    )
    OUT = OUT + (
        rh[:, None, None] * stride_out_nheads
        + rm[None, :, None] * stride_out_seqlen
        + rk_half[None, None, :] * stride_out_headdim
    )

    mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk_half[None, None, :] < ROTARY_DIM_HALF)

    # Load first half (x0) and second half (x1)
    x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask, other=0.0).to(tl.float32)

    # Apply rotation (same formula as Flash Attention)
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos

    # Store results
    tl.store(OUT, o0, mask=mask)
    tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)


def apply_rotary(
    x: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    conjugate: bool = False,
) -> paddle.Tensor:
    """
    Apply rotary position embedding to input tensor.
    Matches Flash Attention's apply_rotary interface.

    Args:
        x: (batch, seqlen, nheads, headdim) or (seqlen, nheads, headdim)
        cos: (seqlen_ro, rotary_dim/2)
        sin: (seqlen_ro, rotary_dim/2)
        conjugate: If True, negate sin (used for backward pass)

    Returns:
        out: Same shape and dtype as x
    """
    orig_dtype = x.dtype
    is_3d = x.ndim == 3
    if is_3d:
        x = x.unsqueeze(0)

    batch, seqlen, nheads, headdim = x.shape
    seqlen_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2

    assert sin.shape == cos.shape
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    out = paddle.empty_like(x)

    # Copy non-rotary dimensions if needed
    if rotary_dim < headdim:
        out[..., rotary_dim:] = x[..., rotary_dim:]

    BLOCK_M = 8 if rotary_dim <= 128 else 4
    BLOCK_H = 2

    grid = (triton.cdiv(nheads, BLOCK_H), triton.cdiv(seqlen, BLOCK_M), batch)

    _rotary_kernel[grid](
        out,
        x,
        cos,
        sin,
        seqlen,
        nheads,
        seqlen_ro,
        out.strides[0],
        out.strides[1],
        out.strides[2],
        out.strides[3],
        x.strides[0],
        x.strides[1],
        x.strides[2],
        x.strides[3],
        ROTARY_DIM=rotary_dim,
        CONJUGATE=conjugate,
        BLOCK_H=BLOCK_H,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )

    # Convert back to original dtype
    out = out.astype(orig_dtype)

    if is_3d:
        out = out.squeeze(0)

    return out


class ApplyRotaryEmb(paddle.autograd.PyLayer):
    """
    Autograd wrapper for rotary position embedding.
    Matches Flash Attention's ApplyRotaryEmb.
    """

    @staticmethod
    def forward(ctx, x, cos, sin):
        """
        Forward pass.

        Args:
            x: (batch, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim/2)
        """
        out = apply_rotary(x, cos, sin, conjugate=False)
        ctx.save_for_backward(cos, sin)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass uses conjugate rotation (sin = -sin).
        """
        cos, sin = ctx.saved_tensor()
        grad_x = apply_rotary(grad_output, cos, sin, conjugate=True)
        return grad_x, None, None


def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary embedding with autograd support.
    Matches Flash Attention's apply_rotary_emb interface.

    Args:
        x: (batch, seqlen, nheads, headdim) or (seqlen, nheads, headdim)
        cos, sin: (seqlen, rotary_dim/2)

    Returns:
        out: Same shape and dtype as x
    """
    return ApplyRotaryEmb.apply(x, cos, sin)


class ApplyRotaryPosEmbVision(paddle.autograd.PyLayer):
    """
    Apply rotary position embedding to q and k tensors together.
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        """
        Forward pass for q and k.

        Args:
            q, k: (batch, seqlen, nheads, headdim) or (seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim/2)
        """
        q_out = apply_rotary(q, cos, sin, conjugate=False)
        k_out = apply_rotary(k, cos, sin, conjugate=False)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        """
        Backward pass for q and k.
        """
        cos, sin = ctx.saved_tensor()
        grad_q_out = apply_rotary(grad_q, cos, sin, conjugate=True)
        grad_k_out = apply_rotary(grad_k, cos, sin, conjugate=True)
        return grad_q_out, grad_k_out, None, None


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """
    Apply rotary position embedding to q and k tensors.

    Args:
        q, k: (batch, seqlen, nheads, headdim) or (seqlen, nheads, headdim)
               Any dtype (bf16/fp16/fp32), output preserves input dtype
        cos, sin: (seqlen, rotary_dim/2)

    Returns:
        q_out, k_out: Same shape and dtype as input
    """
    return ApplyRotaryPosEmbVision.apply(q, k, cos, sin)


def _run_once(fn):
    """Decorator that caches the result of a function after first call."""
    result = None
    has_run = False

    def wrapper(*args, **kwargs):
        nonlocal result, has_run
        if not has_run:
            result = fn(*args, **kwargs)
            has_run = True
        return result

    return wrapper


@_run_once
def _check_triton_available(*args, **kwargs):
    """Check if triton is available and version >= 3.0.0"""
    try:
        import triton

        version = getattr(triton, "__version__")
        major = int(version.split(".")[0])
        return major >= 3
    except ImportError:
        return False


apply_rotary_pos_emb_vision.is_available = (
    lambda *args, **kwargs: _check_triton_available(*args, **kwargs) and IS_TRITON_IN_PADDLE_AVAILABLE
)
