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
High performance in-place RoPE for DSv4 hybrid attention.

This is a hand-specialised fusion of the call

    t_pe = t[..., nope_dim:]
    t_pe = _apply_rotary_pos_emb_bshd(
        t_pe, freqs,
        mscale=mscale,
        rotary_interleaved=False,
        multi_latent_attention=True,
        inverse=inverse,
        mla_output_remove_interleaving=True,
    )
    out = paddle.concat([t[..., :nope_dim], t_pe], axis=-1)

Features:
- Binary equal to Paddle eager mode by using triton asm.
- Coalesced memory access and tuned block size for best performance.
- Generically supports q/kv and o (inverse rope) in one kernel.
"""

import paddle

from .utils import is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


def _get_block_h(nheads: int) -> int:
    """Largest power-of-2 dividing nheads, capped at 32."""
    block_h = min(32, triton.next_power_of_2(nheads))
    while block_h > 1 and nheads % block_h != 0:
        block_h //= 2
    return block_h


@triton.jit
def _mul_round_bf16(a, b):
    """Compute (a * b) in fp32, then round to bf16 via inline PTX.

    Forces an explicit `cvt.rn.bf16.f32` after the multiply so the Triton
    compiler cannot fuse it into an FFMA with the surrounding add/sub. This
    is what gives us bit-exact parity with eager Paddle, which stores each
    intermediate to bf16 memory between elementwise ops.
    """
    return tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;",
        "=h,r",
        [(a.to(tl.float32) * b.to(tl.float32))],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rope_mla_inplace_fwd_kernel(
    T,
    COS,
    SIN,
    nope_dim,
    pe_dim: tl.constexpr,
    head_num: tl.constexpr,
    stride_x_seq,
    stride_x_nheads,
    BLOCK_H: tl.constexpr,
):
    """Forward: rotate t[..., nope_dim:] in place (interleaved in/out)."""
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1).to(tl.int64)

    # COS/SIN are pre-broadcast to [B*S, pe_dim] (contiguous), so each token
    # index maps directly to a row regardless of whether freqs originally had
    # B=1 or B>1, and regardless of slicing along the seq axis.
    half: tl.constexpr = pe_dim // 2

    cos_left = tl.load(COS + pid_m * pe_dim + tl.arange(0, half))
    sin_left = tl.load(SIN + pid_m * pe_dim + tl.arange(0, half))
    cos_right = tl.load(COS + pid_m * pe_dim + half + tl.arange(0, half))
    sin_right = tl.load(SIN + pid_m * pe_dim + half + tl.arange(0, half))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, half)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, half)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, half)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, half)

    # Pointer to the start of this token's (head_block_first) row, then advance
    # past the nope channels to land on the rope slice.
    T = T + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
    # Offsets into the rope slice: [BLOCK_H, pe_dim] with last dim STRIDE=1.
    # We deliberately load the whole pe_dim contiguously instead of poking
    # at 2k / 2k+1 with stride-2 offsets — Triton's lowering for stride-2
    # int64 offsets has historically been flaky (extra sector requests, no
    # vectorization), and explicit contiguous loads compile down to
    # `ld.global.v4.b32` which is the theoretical optimum for bf16.
    flat_off = (
        tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads.to(tl.int64)
        + nope_dim
        + tl.arange(0, pe_dim)[None, :]
    )
    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num

    # One contiguous load per program, then de-interleave in registers.
    x = tl.load(T + flat_off, mask=head_mask)  # [BLOCK_H, pe_dim] bf16
    x = tl.reshape(x, (BLOCK_H, half, 2))
    x_1, x_2 = tl.split(x)  # both [BLOCK_H, half]

    # ---- bit-exact match with eager Paddle path -------------------------
    # Eager runs this as three separate elementwise ops: each `a*b` and
    # each `+`/`-` lands in bf16 memory before the next op reads it. We
    # mirror that by keeping every intermediate as bf16 and forcing a
    # store/reload through a hand-written __nv_bf16 round, which the
    # Triton/PTX compiler cannot fuse into FFMA.
    y_left = _mul_round_bf16(x_1, cos_left).to(tl.float32) - _mul_round_bf16(
        x_2, sin_left
    ).to(tl.float32)
    y_right = _mul_round_bf16(x_2, cos_right).to(tl.float32) + _mul_round_bf16(
        x_1, sin_right
    ).to(tl.float32)

    # Re-interleave (mla_output_remove_interleaving=True writes back to the
    # same 2k / 2k+1 positions) and store in one contiguous write.
    y = tl.join(y_left, y_right)  # [BLOCK_H, half, 2]
    y = tl.reshape(y, (BLOCK_H, pe_dim))
    tl.store(T + flat_off, y, mask=head_mask)


@triton.jit
def _rope_mla_inplace_bwd_kernel(
    DO,
    COS,
    SIN,
    nope_dim,
    pe_dim: tl.constexpr,
    head_num: tl.constexpr,
    stride_x_seq,
    stride_x_nheads,
    BLOCK_H: tl.constexpr,
):
    """Backward: transform grad in place by the transpose of the forward 2x2."""
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1).to(tl.int64)

    half: tl.constexpr = pe_dim // 2

    cos_left = tl.load(COS + pid_m * pe_dim + tl.arange(0, half))
    sin_left = tl.load(SIN + pid_m * pe_dim + tl.arange(0, half))
    cos_right = tl.load(COS + pid_m * pe_dim + half + tl.arange(0, half))
    sin_right = tl.load(SIN + pid_m * pe_dim + half + tl.arange(0, half))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, half)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, half)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, half)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, half)

    DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
    flat_off = (
        tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads.to(tl.int64)
        + nope_dim
        + tl.arange(0, pe_dim)[None, :]
    )
    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num

    g = tl.load(DO + flat_off, mask=head_mask)  # [BLOCK_H, pe_dim] bf16
    g = tl.reshape(g, (BLOCK_H, half, 2))
    g1, g2 = tl.split(g)  # both [BLOCK_H, half]

    # Same FFMA-blocking trick as forward (see `_mul_round_bf16`): each
    # product is independently rounded to bf16 before the add, matching
    # eager Paddle's elementwise op-by-op execution.
    dx_1 = _mul_round_bf16(g1, cos_left).to(tl.float32) + _mul_round_bf16(
        g2, sin_right
    ).to(tl.float32)
    dx_2 = _mul_round_bf16(g2, cos_right).to(tl.float32) - _mul_round_bf16(
        g1, sin_left
    ).to(tl.float32)

    dx = tl.join(dx_1, dx_2)  # [BLOCK_H, half, 2]
    dx = tl.reshape(dx, (BLOCK_H, pe_dim))
    tl.store(DO + flat_off, dx, mask=head_mask)


class RoPEMLAInplaceFusion(paddle.autograd.PyLayer):
    """PyLayer wrapping the in-place fwd/bwd kernels."""

    @staticmethod
    def forward(ctx, t, cos, sin, nope_dim, pe_dim, clone_input):
        # Clone input if the upstream depends on it.
        t = t.clone() if clone_input else t
        assert t.stride(-1) == 1
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        B, S, H, D = t.shape
        assert D == nope_dim + pe_dim
        assert pe_dim % 4 == 0
        assert cos.shape[-1] == pe_dim
        assert sin.shape[-1] == pe_dim

        # Flatten BS for the kernel (view, no copy on contiguous tensors).
        t_flat = t.reshape([B * S, H, D])
        BLOCK_H = _get_block_h(H)
        assert H % BLOCK_H == 0, (
            f"head_num must be divisible by BLOCK_H ({BLOCK_H}), got {H}"
        )

        grid = (B * S, triton.cdiv(H, BLOCK_H))
        _rope_mla_inplace_fwd_kernel[grid](
            t_flat,
            cos,
            sin,
            nope_dim,
            pe_dim,
            H,
            t_flat.stride(0),
            t_flat.stride(1),
            BLOCK_H,
        )

        ctx.save_for_backward(cos, sin)
        ctx.nope_dim = nope_dim
        ctx.pe_dim = pe_dim
        ctx.block_h = BLOCK_H
        ctx.shape = (B, S, H, D)
        # Return the reshape-back view; storage is identical to input t.
        return t

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        B, S, H, D = ctx.shape
        # Run in place on grad; nope channels are passed through unchanged.
        grad_flat = grad.contiguous().reshape([B * S, H, D])
        BLOCK_H = ctx.block_h
        grid = (B * S, triton.cdiv(H, BLOCK_H))
        _rope_mla_inplace_bwd_kernel[grid](
            grad_flat,
            cos,
            sin,
            ctx.nope_dim,
            ctx.pe_dim,
            H,
            grad_flat.stride(0),
            grad_flat.stride(1),
            BLOCK_H,
        )
        return grad, None, None  # (t, cos, sin)


def fused_apply_mla_rope_inplace(
    t: paddle.Tensor,
    freqs: paddle.Tensor,
    nope_dim: int,
    mscale: float = 1.0,
    inverse: bool = False,
    clone_input: bool = False,
) -> paddle.Tensor:
    """In-place RoPE on t's trailing rope channels.

    Specialised for DSv4 hybrid attention's MLA pe path:
      bshd, multi_latent_attention=True, mla_output_remove_interleaving=True,
      rotary_interleaved=False, mscale=1.0, no SP/CP, no THD,
      high_precision_rope=False (cos/sin computed in fp32, cast to t.dtype
      to match the unfused path's bf16 arithmetic).

    `t` is generic — both q and k/v can be passed through, since the only
    assumption is that the last `pe_dim` channels carry the interleaved
    rope pairs.

    Args:
        t: [B, S, H, nope_dim + pe_dim], contiguous, bf16 (or fp16/fp32).
            Mutated in place.
        freqs: [B, S, 1, pe_dim], fp32 angle tensor. May be non-contiguous.
        nope_dim: number of leading nope channels left untouched.
        mscale: scaling factor for rotary embedding.
        inverse: if True, apply the inverse rotation (used by the
            inv_rope post-attention canonicalisation step).
        clone_input: if True, clone the input t before applying rope.

    Returns:
        t (same storage as the input). Channels [..., :nope_dim] are
        unchanged; channels [..., nope_dim:] are rotated.
    """
    # Check t
    assert t.is_contiguous(), (
        "fused_apply_mla_rope_inplace requires t to be contiguous, "
        f"got shape={t.shape} strides={t.strides}"
    )
    B, S, H, D = t.shape
    pe_dim = freqs.shape[-1]
    assert D > pe_dim, f"t last dim {D} must exceed rope pe_dim {pe_dim}"
    nope_dim_check = D - pe_dim
    assert nope_dim == nope_dim_check, (
        f"nope_dim {nope_dim} mismatches D-pe_dim {nope_dim_check}"
    )
    assert t.dtype == paddle.bfloat16, (
        f"fused_apply_mla_rope_inplace is designed for bf16, got {t.dtype}"
    )

    # Check freqs
    assert freqs.dim() == 4 and freqs.shape[2] == 1, (
        f"freqs must be [B,S,1,D]; got {freqs.shape}"
    )
    B_f, S_f, _, D_f = freqs.shape
    assert S_f == S and D_f == pe_dim, (
        f"freqs [B,S,1,D]={freqs.shape} mismatches t [B,S]=[{B},{S}], "
        f"pe_dim={pe_dim}"
    )
    assert B_f == 1 or B_f == B, f"freqs B {B_f} must be 1 or {B}"
    if B_f < B:
        freqs = freqs.broadcast_to([B, S, 1, pe_dim])

    # Compute cos/sin on the (possibly non-contiguous / broadcast) freqs.
    # For inverse=True, mirror the eager `sin_ = -sin_` step (rope_utils.py).
    # Negating after the bf16 cast is equivalent to negating before (bf16
    # negate is a sign-bit flip), so both orders produce identical bytes.
    cos = (paddle.cos(freqs) * mscale).to(t.dtype)
    sin = (paddle.sin(freqs) * mscale).to(t.dtype)
    if inverse:
        sin = -sin
    return RoPEMLAInplaceFusion.apply(
        t, cos, sin, nope_dim, pe_dim, clone_input
    )
