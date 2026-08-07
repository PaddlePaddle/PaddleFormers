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
Fused Triton permute for the Ulysses all-to-all.

For the production context-parallel path (batch_dim_idx=0), the reshape+
transpose+.contiguous() permutes around the all-to-all are pure data movement
whose cost (two physical copies) rivals the communication itself. The permute
only reorders the leading dims (batch, seq, rank) while the trailing (head,
dim) stay contiguous, so each element-block [Hloc*D] is a contiguous->
contiguous copy with a remapped base offset. A single Triton kernel fuses
reshape+transpose+contiguous into one coalesced copy, and the post output
reuses the send buffer so peak memory stays at 1x send + 1x recv (lower than
the reference permute path, which needs an extra .contiguous() buffer).

Only batch_dim_idx=0 with a 4-D input is fused; callers fall back to the
reference reshape/permute path for other layouts. Both are bit-exact.
"""

import paddle
from paddle import distributed as dist

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@enable_compat_on_triton_kernel
@triton.jit
def _seq2head_pre_kernel(  # pragma: no cover
    x_ptr, out_ptr, P, b, Sloc, Hloc, D, SEG, BLOCK: tl.constexpr
):
    # seq->head pre: send[dst, bb, s, hl, dd] = x[bb, s, dst*Hloc+hl, dd].
    # grid = (P*b*Sloc, cdiv(SEG, BLOCK)); .to(int64) guards >2^31 offsets.
    pid = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    s = pid % Sloc
    tmp = pid // Sloc
    bb = tmp % b
    dst = tmp // b
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SEG
    H = P * Hloc
    # offs walks [Hloc, D] contiguously; src head = dst*Hloc + offs//D.
    src_off = ((bb * Sloc + s) * H + dst * Hloc) * D + offs
    val = tl.load(x_ptr + src_off, mask=mask, other=0.0)
    dst_base = (((dst * b + bb) * Sloc + s) * Hloc) * D
    tl.store(out_ptr + dst_base + offs, val, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit
def _seq2head_post_kernel(  # pragma: no cover
    recv_ptr, out_ptr, P, b, Sloc, Hloc, D, SEG, BLOCK: tl.constexpr
):
    # seq->head post: out[bb, src*Sloc+s, hl, dd] = recv[src, bb, s, hl, dd].
    pid = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    s = pid % Sloc
    tmp = pid // Sloc
    bb = tmp % b
    src = tmp // b
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SEG
    src_base = (((src * b + bb) * Sloc + s) * Hloc) * D
    val = tl.load(recv_ptr + src_base + offs, mask=mask, other=0.0)
    seq = src * Sloc + s
    dst_base = ((bb * (P * Sloc) + seq) * Hloc) * D
    tl.store(out_ptr + dst_base + offs, val, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit
def _head2seq_pre_kernel(  # pragma: no cover
    g_ptr, out_ptr, P, b, Sloc, Hloc, D, SEG, BLOCK: tl.constexpr
):
    # head->seq pre: send[dst, bb, s, hl, dd] = g[bb, dst*Sloc+s, hl, dd].
    pid = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    s = pid % Sloc
    tmp = pid // Sloc
    bb = tmp % b
    dst = tmp // b
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SEG
    gseq = dst * Sloc + s
    src_off = ((bb * (P * Sloc) + gseq) * Hloc) * D + offs
    val = tl.load(g_ptr + src_off, mask=mask, other=0.0)
    dst_base = (((dst * b + bb) * Sloc + s) * Hloc) * D
    tl.store(out_ptr + dst_base + offs, val, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit
def _head2seq_post_kernel(  # pragma: no cover
    recv_ptr, out_ptr, P, b, Sloc, Hloc, D, SEG, BLOCK: tl.constexpr
):
    # head->seq post: out[bb, s, src*Hloc+hl, dd] = recv[src, bb, s, hl, dd].
    pid = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    s = pid % Sloc
    tmp = pid // Sloc
    bb = tmp % b
    src = tmp // b
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SEG
    src_base = (((src * b + bb) * Sloc + s) * Hloc) * D
    val = tl.load(recv_ptr + src_base + offs, mask=mask, other=0.0)
    hl = offs // D
    dd = offs % D
    H = P * Hloc
    dst_off = ((bb * Sloc + s) * H + src * Hloc + hl) * D + dd
    tl.store(out_ptr + dst_off, val, mask=mask)


def ulysses_alltoall_fused_supported(_scatter_idx, batch_dim_idx, input):
    """Whether the fused kernel path covers this layout.

    Only batch_dim_idx=0 with a 4-D input is fused; other layouts must fall
    back to the reference reshape/permute all-to-all. scatter_idx does not
    affect support (both seq->head and head->seq are handled), so it is
    accepted for signature symmetry but unused here.
    """
    return batch_dim_idx == 0 and len(input.shape) == 4


def ulysses_single_all_to_all_fused(input, scatter_idx, group):
    """Fused seq<->head all-to-all for batch_dim_idx=0.

    scatter_idx >= 2: seq->head, input [b, Sloc, H, D] -> [b, P*Sloc, H//P, D].
    scatter_idx <  2: head->seq, input [b, P*Sloc, Hloc, D] -> [b, Sloc, P*Hloc, D].
    Bit-exact with the reference permute path; out reuses the send buffer so the
    peak stays at 1x send + 1x recv (2x a single tensor).
    """
    P = dist.get_world_size(group)
    input = input.contiguous()

    if scatter_idx >= 2:
        b, Sloc, H, D = input.shape
        if H % P != 0:
            raise ValueError(
                f"Number of heads ({H}) must be divisible by world size ({P})"
            )
        Hloc = H // P
        pre_kernel = _seq2head_pre_kernel
        post_kernel = _seq2head_post_kernel
        out_shape = [b, P * Sloc, Hloc, D]
    else:
        b, Nglob, Hloc, D = input.shape
        if Nglob % P != 0:
            raise ValueError(
                f"Global sequence length ({Nglob}) must be divisible by "
                f"world size ({P})"
            )
        Sloc = Nglob // P
        H = P * Hloc
        pre_kernel = _head2seq_pre_kernel
        post_kernel = _head2seq_post_kernel
        out_shape = [b, Sloc, H, D]

    SEG = Hloc * D
    BLOCK = min(triton.next_power_of_2(SEG), 4096)
    grid = (P * b * Sloc, triton.cdiv(SEG, BLOCK))

    send = paddle.empty([P, b, Sloc, Hloc, D], dtype=input.dtype)
    pre_kernel[grid](input, send, P, b, Sloc, Hloc, D, SEG, BLOCK=BLOCK)

    recv = paddle.empty_like(send)
    dist.stream.alltoall_single(
        recv.reshape([-1]),
        send.reshape([-1]),
        group=group,
        use_calc_stream=True,
    )

    # send is no longer needed after comm; reuse its memory as out (same numel,
    # post reads recv and writes out so there is no aliasing hazard).
    out = send.reshape(out_shape)
    post_kernel[grid](recv, out, P, b, Sloc, Hloc, D, SEG, BLOCK=BLOCK)
    return out
