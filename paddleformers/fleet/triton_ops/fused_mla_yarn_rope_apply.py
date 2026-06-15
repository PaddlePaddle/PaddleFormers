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

import paddle

from .utils import is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


def _get_block_h(nheads):
    """Compute a fixed BLOCK_H based on nheads for deterministic results.

    Picks the largest power-of-2 that divides nheads (capped at 128)
    to maximize per-block parallelism while ensuring exact divisibility.
    """
    block_h = min(128, triton.next_power_of_2(nheads))
    # Find largest power-of-2 <= block_h that divides nheads
    while block_h > 1 and nheads % block_h != 0:
        block_h //= 2
    return block_h


@triton.jit
def _get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size):
    token_idx = -1
    this_seq_len = 0
    seq_idx = 0
    last_cum_seqlen = tl.load(cu_seqlens) // cp_size
    while seq_idx < seq_num:
        cur_cum_seqlen = tl.load(cu_seqlens + seq_idx + 1) // cp_size
        if token_idx == -1 and cur_cum_seqlen > pid_m:
            token_idx = pid_m - last_cum_seqlen
            this_seq_len = cur_cum_seqlen - last_cum_seqlen
        last_cum_seqlen = cur_cum_seqlen
        seq_idx += 1
    if cp_size > 1:
        if token_idx < this_seq_len // 2:
            token_idx = token_idx + cp_rank * this_seq_len // 2
        else:
            token_idx = (token_idx - this_seq_len // 2) + (2 * cp_size - cp_rank - 1) * this_seq_len // 2
    return token_idx


@triton.jit
def rotary_fwd_q_kernel(
    Q,
    COS,
    SIN,
    qk_head_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    seq_len,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel of the forward pass for applying YARN RoPE to MLA's query.
    This kernel inplace modifies the input tensor Q.

    Input:
        Q: [batch_size, seq_len, head_num, qk_head_dim + emb_dim] (bshd, viewed as flat)
            or [total_seq_len, head_num, qk_head_dim + emb_dim] (thd)
        COS/SIN: [max_seq_len, emb_dim]

        seq_len: sequence length for bshd format (not used for thd)
        seq_num: number of sequences for thd format, not used for bshd
        cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1).to(tl.int64)

    if cu_seqlens_q is None:
        # bshd: pid_m = b * seq_len + s  →  token_idx = s
        token_idx = pid_m % seq_len
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads.to(tl.int64) + qk_head_dim
    mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    x_2_off = x_1_off + 1
    x_1 = tl.load(Q + x_1_off, mask=mask)
    x_2 = tl.load(Q + x_2_off, mask=mask)

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right

    x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
    x_right_off = x_left_off + emb_dim // 2
    tl.store(Q + x_left_off, x_left, mask=mask)
    tl.store(Q + x_right_off, x_right, mask=mask)


@triton.jit
def rotary_bwd_q_kernel(
    DO,
    COS,
    SIN,
    qk_head_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    seq_len,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel of the backward pass for applying YARN RoPE to MLA's query.
    This kernel inplace modifies the input tensor DO.

    Input:
        DO: [batch_size, seq_len, head_num, qk_head_dim + emb_dim] (bshd, viewed as flat)
            or [total_seq_len, head_num, qk_head_dim + emb_dim] (thd)
        COS/SIN: [max_seq_len, emb_dim]

        seq_len, seq_num, and cu_seqlens_q are the same as in the forward pass
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1).to(tl.int64)

    if cu_seqlens_q is None:
        token_idx = pid_m % seq_len
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx.to(tl.int64) * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx.to(tl.int64) * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx.to(tl.int64) * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx.to(tl.int64) * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads.to(tl.int64) + qk_head_dim
    mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
    x_right_off = x_left_off + emb_dim // 2
    x_left = tl.load(DO + x_left_off, mask=mask)
    x_right = tl.load(DO + x_right_off, mask=mask)

    x_1 = x_left * cos_left + x_right * sin_right
    x_2 = -x_left * sin_left + x_right * cos_right

    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    x_2_off = x_1_off + 1
    tl.store(DO + x_1_off, x_1, mask=mask)
    tl.store(DO + x_2_off, x_2, mask=mask)


class ApplyMLARotaryEmbQ(paddle.autograd.PyLayer):
    """
    Autograd function for applying YARN RoPE to MLA's query.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        cos,
        sin,
        qk_head_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        rotary_interleaved=False,
    ):
        """
        Forward function for ApplyMLARotaryEmbQ.

        Args:
            q: [batch_size, seq_len, head_num, qk_head_dim + emb_dim] (bshd)
                or [total_seq_len, head_num, qk_head_dim + emb_dim] (thd)
            cos/sin: [max_seq_len, 1, 1, emb_dim]
            cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
            rotary_interleaved: whether to apply RoPE interleaved, only supports False for now
        """
        assert not rotary_interleaved
        q = q.contiguous().clone()
        batch_size = None
        max_seqlen = None
        seq_num = None
        assert cu_seqlens_q is None, "THD is not supported for rope fusion now."
        if cu_seqlens_q is None:
            # bshd: [batch, seq, nhead, dim]
            batch_size, max_seqlen, nheads, headdim = q.shape
            q = q.reshape(-1, nheads, headdim)
            total_seqlen = q.shape[0]
        else:
            # thd
            total_seqlen, nheads, headdim = q.shape
            seq_num = len(cu_seqlens_q) - 1
            # Two-step reshape (via an intermediate different shape) creates a
            # non-leaf view, allowing inplace kernel modification when the input
            # requires gradient — consistent with the bshd path which uses
            # q.view(-1, nheads, headdim) to achieve the same effect.
        assert q.stride(-1) == 1
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        assert headdim == qk_head_dim + emb_dim
        assert emb_dim % 4 == 0
        assert cos.shape[-1] == emb_dim
        assert sin.shape[-1] == emb_dim
        BLOCK_H = _get_block_h(nheads)
        assert nheads % BLOCK_H == 0, f"head_num must be divisible by BLOCK_H ({BLOCK_H}), but got {nheads}"

        grid = (total_seqlen, triton.cdiv(nheads, BLOCK_H))
        rotary_fwd_q_kernel[grid](
            q,
            cos,
            sin,
            qk_head_dim,
            emb_dim,
            nheads,
            max_seqlen,
            seq_num,
            cu_seqlens_q,
            q.stride(0),
            q.stride(1),
            cp_rank,
            cp_size,
            BLOCK_H,
        )
        ctx.save_for_backward(cos, sin)
        ctx.qk_head_dim = qk_head_dim
        ctx.emb_dim = emb_dim
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.rotary_interleaved = rotary_interleaved
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.max_seqlen = max_seqlen
        ctx.batch_size = batch_size
        ctx.block_h = BLOCK_H
        if cu_seqlens_q is None:
            q = q.reshape(batch_size, max_seqlen, nheads, headdim)
        return q

    @staticmethod
    def backward(ctx, grad):
        """
        Backward function for ApplyMLARotaryEmbQ.

        Args:
            grad: [batch_size, seq_len, head_num, qk_head_dim + emb_dim] (bshd)
                or [total_seq_len, head_num, qk_head_dim + emb_dim] (thd)
        """
        cos, sin = ctx.saved_tensors
        grad = grad.contiguous().clone()
        batch_size = None
        max_seqlen = None
        seq_num = None
        if ctx.cu_seqlens_q is None:
            batch_size, max_seqlen, nheads, headdim = grad.shape
            grad = grad.reshape(-1, nheads, headdim)
            total_seqlen = grad.shape[0]
        else:
            seq_num = len(ctx.cu_seqlens_q) - 1
            total_seqlen, nheads, headdim = grad.shape
        assert grad.stride(-1) == 1

        BLOCK_H = ctx.block_h
        grid = (total_seqlen, triton.cdiv(nheads, BLOCK_H))
        rotary_bwd_q_kernel[grid](
            grad,
            cos,
            sin,
            ctx.qk_head_dim,
            ctx.emb_dim,
            nheads,
            max_seqlen,
            seq_num,
            ctx.cu_seqlens_q,
            grad.stride(0),
            grad.stride(1),
            ctx.cp_rank,
            ctx.cp_size,
            BLOCK_H,
        )
        if ctx.cu_seqlens_q is None:
            grad = grad.reshape(batch_size, max_seqlen, nheads, headdim)

        if ctx.cu_seqlens_q is None:
            return grad, None, None  # q, cos, sin
        else:
            return grad, None, None, None  # q, cos, sin, cu_seqlens_q


def fused_apply_mla_rope_for_q(
    t: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    qk_head_dim: int,
    emb_dim: int,
    cu_seqlens_q: paddle.Tensor | None = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
):
    """
    Fused function for applying YARN RoPE to MLA's query.
    This function inplace modifies the input tensor t.
    Along the last dimension of t, the last emb_dim elements are applied with RoPE.
    The first qk_head_dim elements are not modified.
    It is an experimental feature and may change in future versions.
    It supports bshd and thd input formats.

    For the notations below, seq_len is the length of the sequence per batch for bshd format,
    total_seq_len is the total length of the sequences for thd format.
    max_seq_len is the maximum length of the sequences in the input tensor.

    Args:
        t: [batch_size, seq_len, head_num, qk_head_dim + emb_dim] (bshd)
            or [total_seq_len, head_num, qk_head_dim + emb_dim] (thd)
        cos/sin: [max_seq_len, 1, 1, emb_dim]
        cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
        rotary_interleaved: whether to apply RoPE interleaved, only supports False for now

    Returns:
        t: inplace modified input tensor
    """
    return ApplyMLARotaryEmbQ.apply(
        t,
        cos,
        sin,
        qk_head_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        rotary_interleaved,
    )


@triton.jit
def rotary_fwd_kv_kernel(
    KV,
    K_POS_EMB,
    O_KEY,
    O_VALUE,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    head_num: tl.constexpr,
    seq_len,
    seq_num,
    cu_seqlens_kv,
    stride_kv_seq,
    stride_kv_nheads,
    stride_emb_seq,
    stride_k_seq,
    stride_k_nheads,
    stride_v_seq,
    stride_v_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Triton kernel of the forward pass for applying YARN RoPE to MLA's key and value.
    It splits the input tensor KV into key and value,
    and concatenates the processed RoPE to the key.

    Input:
        KV: [batch_size, seq_len, head_num, k_dim + v_dim] (bshd, viewed as flat)
            or [total_seq_len, head_num, k_dim + v_dim] (thd)
        K_POS_EMB: [batch_size, seq_len, emb_dim] (bshd) or [total_seq_len, emb_dim] (thd)
        COS/SIN: [max_seq_len, emb_dim]

        seq_len: sequence length for bshd format (not used for thd)
        seq_num: number of sequences for thd format, not used for bshd
        cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format

    Output:
        O_KEY: [batch_size, seq_len, head_num, emb_dim + k_dim] (bshd)
            or [total_seq_len, head_num, emb_dim + k_dim] (thd)
        O_VALUE: [batch_size, seq_len, head_num, v_dim] (bshd)
            or [total_seq_len, head_num, v_dim] (thd)
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1).to(tl.int64)

    if cu_seqlens_kv is None:
        token_idx = pid_m % seq_len
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx.to(tl.int64) * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx.to(tl.int64) * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx.to(tl.int64) * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx.to(tl.int64) * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    KV_ptr = KV + pid_m * stride_kv_seq + pid_head * BLOCK_H * stride_kv_nheads
    kv_off = tl.arange(0, BLOCK_H)[:, None] * stride_kv_nheads.to(tl.int64)
    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    k_range = tl.arange(0, BLOCK_K)[None, :]
    v_range = tl.arange(0, BLOCK_V)[None, :]
    k_valid = k_range < k_dim
    v_valid = v_range < v_dim
    k_in_off = kv_off + k_range
    v_in_off = kv_off + k_dim + v_range
    k = tl.load(KV_ptr + k_in_off, mask=head_mask & k_valid, other=0.0)
    v = tl.load(KV_ptr + v_in_off, mask=head_mask & v_valid, other=0.0)

    K_ptr = O_KEY + pid_m * stride_k_seq + pid_head * BLOCK_H * stride_k_nheads
    V_ptr = O_VALUE + pid_m * stride_v_seq + pid_head * BLOCK_H * stride_v_nheads

    k_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads.to(tl.int64) + k_range
    v_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_v_nheads.to(tl.int64) + v_range
    tl.store(K_ptr + k_out_off, k, mask=head_mask & k_valid)
    tl.store(V_ptr + v_out_off, v, mask=head_mask & v_valid)

    EMB = K_POS_EMB + pid_m * stride_emb_seq
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    x_1 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2)
    x_2 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2 + 1)

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right
    x_left = x_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    x_right = x_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    x_left_off = (
        tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads.to(tl.int64) + k_dim + tl.arange(0, emb_dim // 2)[None, :]
    )
    x_right_off = x_left_off + emb_dim // 2
    tl.store(K_ptr + x_left_off, x_left, mask=head_mask)
    tl.store(K_ptr + x_right_off, x_right, mask=head_mask)


@triton.jit
def rotary_bwd_kv_kernel(
    dK,
    dV,
    dKV,
    dEMB,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    head_num: tl.constexpr,
    seq_len,
    seq_num,
    cu_seqlens_kv,
    stride_dk_seq,
    stride_dk_nheads,
    stride_dv_seq,
    stride_dv_nheads,
    stride_dkv_seq,
    stride_dkv_nheads,
    stride_demb_seq,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Triton kernel of the backward pass for applying YARN RoPE to MLA's key and value.

    Input:
        dK: [batch_size, seq_len, head_num, emb_dim + k_dim] (bshd, viewed as flat)
            or [total_seq_len, head_num, emb_dim + k_dim] (thd)
        dV: [batch_size, seq_len, head_num, v_dim] (bshd)
            or [total_seq_len, head_num, v_dim] (thd)
        COS/SIN: [max_seq_len, emb_dim]

        seq_len, seq_num, and cu_seqlens_kv are the same as in the forward pass

    Output:
        dKV: [batch_size, seq_len, head_num, k_dim + v_dim] (bshd)
            or [total_seq_len, head_num, k_dim + v_dim] (thd)
        dEMB: [batch_size, seq_len, emb_dim] (bshd) or [total_seq_len, emb_dim] (thd)
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1).to(tl.int64)

    if cu_seqlens_kv is None:
        token_idx = pid_m % seq_len
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    dKV_ptr = dKV + pid_m * stride_dkv_seq + pid_head * BLOCK_H * stride_dkv_nheads
    dkv_off = tl.arange(0, BLOCK_H)[:, None] * stride_dkv_nheads.to(tl.int64)
    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    k_range = tl.arange(0, BLOCK_K)[None, :]
    v_range = tl.arange(0, BLOCK_V)[None, :]
    k_valid = k_range < k_dim
    v_valid = v_range < v_dim
    dk_out_off = dkv_off + k_range
    dv_out_off = dkv_off + k_dim + v_range

    dK_ptr = dK + pid_m * stride_dk_seq + pid_head * BLOCK_H * stride_dk_nheads
    dV_ptr = dV + pid_m * stride_dv_seq + pid_head * BLOCK_H * stride_dv_nheads
    dk_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads.to(tl.int64) + k_range
    dv_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dv_nheads.to(tl.int64) + v_range
    dk = tl.load(dK_ptr + dk_in_off, mask=head_mask & k_valid, other=0.0)
    dv = tl.load(dV_ptr + dv_in_off, mask=head_mask & v_valid, other=0.0)
    tl.store(dKV_ptr + dk_out_off, dk, mask=head_mask & k_valid)
    tl.store(dKV_ptr + dv_out_off, dv, mask=head_mask & v_valid)

    if pid_head == 0:
        x_left_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        x_right_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        for i in tl.static_range(triton.cdiv(head_num, BLOCK_H)):
            dK_ptr = dK + pid_m * stride_dk_seq.to(tl.int64) + i * BLOCK_H * stride_dk_nheads
            x_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads.to(tl.int64) + k_dim
            mask = (i * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
            x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
            x_right_off = x_left_off + emb_dim // 2
            x_left = tl.load(dK_ptr + x_left_off, mask=mask, other=0.0)
            x_right = tl.load(dK_ptr + x_right_off, mask=mask, other=0.0)
            x_left_accum += x_left
            x_right_accum += x_right
        x_left_accum = tl.sum(x_left_accum, axis=0)
        x_right_accum = tl.sum(x_right_accum, axis=0)
        # Keep in float32 for the cos/sin multiply to avoid bf16 precision loss;
        # cast to output dtype only at store time.

        cos_left = tl.load(COS + token_idx.to(tl.int64) * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx.to(tl.int64) * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx.to(tl.int64) * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx.to(tl.int64) * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

        x_1 = x_left_accum * cos_left + x_right_accum * sin_right
        x_2 = -x_left_accum * sin_left + x_right_accum * cos_right
        dEMB_ptr = dEMB + pid_m * stride_demb_seq
        tl.store(
            dEMB_ptr + tl.arange(0, emb_dim // 2) * 2,
            x_1.to(dEMB.dtype.element_ty),
        )
        tl.store(
            dEMB_ptr + tl.arange(0, emb_dim // 2) * 2 + 1,
            x_2.to(dEMB.dtype.element_ty),
        )


class ApplyMLARotaryEmbKV(paddle.autograd.PyLayer):
    """
    Autograd function for applying YARN RoPE to MLA's key and value.
    """

    @staticmethod
    def forward(
        ctx,
        kv,
        k_pos_emb,
        cos,
        sin,
        emb_dim,
        k_dim,
        v_dim,
        cu_seqlens_kv,
        cp_rank,
        cp_size,
        rotary_interleaved=False,
    ):
        """
        Forward function for ApplyMLARotaryEmbKV.

        Args:
            kv: [batch_size, seq_len, head_num, k_dim + v_dim] (bshd)
                or [total_seq_len, head_num, k_dim + v_dim] (thd)
            k_pos_emb: [batch_size, seq_len, 1, emb_dim] (bshd)
                or [total_seq_len, 1, emb_dim] (thd)
            cos/sin: [max_seq_len, 1, 1, emb_dim]
            cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format
            rotary_interleaved: whether to apply RoPE interleaved, only supports False for now
        """
        assert not rotary_interleaved
        batch_size = None
        max_seqlen = None
        seq_num = None
        if cu_seqlens_kv is None:
            # bshd: [batch, seq, nhead, dim]
            batch_size, max_seqlen, nheads, headdim = kv.shape
            kv = kv.contiguous().reshape(-1, nheads, headdim)
            k_pos_emb = k_pos_emb.contiguous().reshape(-1, emb_dim)
            total_seqlen = kv.shape[0]
        else:
            # thd
            seq_num = len(cu_seqlens_kv) - 1
            total_seqlen, nheads, headdim = kv.shape
        assert headdim == k_dim + v_dim
        assert kv.stride(-1) == 1
        assert k_pos_emb.stride(-1) == 1
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        assert emb_dim % 4 == 0
        BLOCK_H = _get_block_h(nheads)
        BLOCK_K = triton.next_power_of_2(k_dim)
        BLOCK_V = triton.next_power_of_2(v_dim)

        o_key = kv.new_empty(total_seqlen, nheads, emb_dim + k_dim)
        o_value = kv.new_empty(total_seqlen, nheads, v_dim)

        grid = (total_seqlen, triton.cdiv(nheads, BLOCK_H))
        rotary_fwd_kv_kernel[grid](
            kv,
            k_pos_emb,
            o_key,
            o_value,
            cos,
            sin,
            emb_dim,
            k_dim,
            v_dim,
            nheads,
            max_seqlen,
            seq_num,
            cu_seqlens_kv,
            kv.stride(0),
            kv.stride(1),
            k_pos_emb.stride(0),
            o_key.stride(0),
            o_key.stride(1),
            o_value.stride(0),
            o_value.stride(1),
            cp_rank,
            cp_size,
            BLOCK_H,
            BLOCK_K,
            BLOCK_V,
        )
        ctx.save_for_backward(cos, sin)
        ctx.rotary_interleaved = rotary_interleaved
        ctx.emb_dim = emb_dim
        ctx.k_dim = k_dim
        ctx.v_dim = v_dim
        ctx.cu_seqlens_kv = cu_seqlens_kv
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.max_seqlen = max_seqlen
        ctx.batch_size = batch_size
        ctx.block_h = BLOCK_H
        if cu_seqlens_kv is None:
            o_key = o_key.reshape(batch_size, max_seqlen, nheads, emb_dim + k_dim)
            o_value = o_value.reshape(batch_size, max_seqlen, nheads, v_dim)
        return o_key, o_value

    @staticmethod
    def backward(ctx, dk, dv):
        """
        Backward function for ApplyMLARotaryEmbKV.

        Args:
            dk: [batch_size, seq_len, head_num, emb_dim + k_dim] (bshd)
                or [total_seq_len, head_num, emb_dim + k_dim] (thd)
            dv: [batch_size, seq_len, head_num, v_dim] (bshd)
                or [total_seq_len, head_num, v_dim] (thd)
        """
        cos, sin = ctx.saved_tensors
        batch_size = None
        max_seqlen = None
        seq_num = None
        if ctx.cu_seqlens_kv is None:
            # bshd
            batch_size, max_seqlen, nheads, _ = dk.shape
            dk = dk.contiguous().reshape(-1, nheads, ctx.emb_dim + ctx.k_dim)
            dv = dv.contiguous().reshape(-1, nheads, ctx.v_dim)
            total_seqlen = dk.shape[0]
        else:
            # thd
            seq_num = len(ctx.cu_seqlens_kv) - 1
            total_seqlen, nheads, _ = dk.shape
        assert dk.stride(-1) == 1
        assert dv.stride(-1) == 1

        d_kv = dk.new_empty(total_seqlen, nheads, ctx.k_dim + ctx.v_dim)
        d_emb = dk.new_empty(total_seqlen, 1, ctx.emb_dim)

        BLOCK_H = ctx.block_h
        BLOCK_K = triton.next_power_of_2(ctx.k_dim)
        BLOCK_V = triton.next_power_of_2(ctx.v_dim)
        grid = (total_seqlen, triton.cdiv(nheads, BLOCK_H))
        rotary_bwd_kv_kernel[grid](
            dk,
            dv,
            d_kv,
            d_emb,
            cos,
            sin,
            ctx.emb_dim,
            ctx.k_dim,
            ctx.v_dim,
            nheads,
            max_seqlen,
            seq_num,
            ctx.cu_seqlens_kv,
            dk.stride(0),
            dk.stride(1),
            dv.stride(0),
            dv.stride(1),
            d_kv.stride(0),
            d_kv.stride(1),
            d_emb.stride(0),
            ctx.cp_rank,
            ctx.cp_size,
            BLOCK_H,
            BLOCK_K,
            BLOCK_V,
        )
        if ctx.cu_seqlens_kv is None:
            d_kv = d_kv.reshape(batch_size, max_seqlen, nheads, ctx.k_dim + ctx.v_dim)
            d_emb = d_emb.reshape(batch_size, max_seqlen, 1, ctx.emb_dim)

        if ctx.cu_seqlens_kv is None:
            return d_kv, d_emb, None, None  # kv, k_pos_emb, cos, sin
        else:
            return (
                d_kv,
                d_emb,
                None,
                None,
                None,
            )  # kv, k_pos_emb, cos, sin, cu_seqlens_kv


def fused_apply_mla_rope_for_kv(
    kv: paddle.Tensor,
    k_pos_emb: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    emb_dim: int,
    k_dim: int,
    v_dim: int,
    cu_seqlens_kv: paddle.Tensor | None = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
):
    """
    Fused function for applying YARN RoPE to MLA's key and value.
    It splits the input tensor kv into key and value,
    and concatenates the processed RoPE to the key.

    For the notations below, seq_len is the length of sequence per batch for bshd format,
    total_seq_len is the total length of the sequences for thd format.
    max_seq_len is the maximum length of the sequences in the input tensor.

    Args:
        kv: [batch_size, seq_len, head_num, k_dim + v_dim] (bshd)
            or [total_seq_len, head_num, k_dim + v_dim] (thd)
        k_pos_emb: [batch_size, seq_len, 1, emb_dim] (bshd)
            or [total_seq_len, 1, emb_dim] (thd)
        cos/sin: [max_seq_len, 1, 1, emb_dim]
        cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format
        rotary_interleaved: whether to apply RoPE interleaved, only supports False for now

    Returns:
        key: [batch_size, seq_len, head_num, emb_dim + k_dim] (bshd)
            or [total_seq_len, head_num, emb_dim + k_dim] (thd)
        value: [batch_size, seq_len, head_num, v_dim] (bshd)
            or [total_seq_len, head_num, v_dim] (thd)
    """
    return ApplyMLARotaryEmbKV.apply(
        kv,
        k_pos_emb,
        cos,
        sin,
        emb_dim,
        k_dim,
        v_dim,
        cu_seqlens_kv,
        cp_rank,
        cp_size,
        rotary_interleaved,
    )
