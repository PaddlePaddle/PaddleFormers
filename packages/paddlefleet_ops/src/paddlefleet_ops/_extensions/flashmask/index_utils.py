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

import paddle
import triton
import triton.language as tl


@triton.jit
def scan_maxmin_chunked(
    input_ptr,
    output_max_ptr,
    output_min_ptr,
    seqlen,
    num_chunks,
    chunk_size: tl.constexpr,
    BN: tl.constexpr,
):
    INT_MAX: tl.constexpr = 2147483647
    INT_MIN: tl.constexpr = -2147483648

    i_tile = tl.program_id(0)
    i_bh = tl.program_id(1)

    p_tile = i_tile * BN + tl.arange(0, BN)
    mask_tile = p_tile < seqlen
    b_tile = tl.load(input_ptr + i_bh * seqlen + p_tile, mask=mask_tile)

    b_omax = tl.where(mask_tile, b_tile, INT_MIN).reshape((BN // chunk_size, chunk_size))
    b_omax = tl.max(b_omax, axis=1)

    b_omin = tl.where(mask_tile, b_tile, INT_MAX).reshape((BN // chunk_size, chunk_size))
    b_omin = tl.min(b_omin, axis=1)

    offs_out = tl.arange(0, BN // chunk_size) + i_tile * (BN // chunk_size)
    mask_out = offs_out < num_chunks
    tl.store(output_max_ptr + i_bh * num_chunks + offs_out, b_omax, mask=mask_out)
    tl.store(output_min_ptr + i_bh * num_chunks + offs_out, b_omin, mask=mask_out)


def prepare_maxmin(input: paddle.Tensor, chunk_size: int) -> tuple[paddle.Tensor, paddle.Tensor]:
    bsz, num_heads, seq_len = input.shape
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    output_max = paddle.empty([bsz, num_heads, num_chunks], dtype=paddle.int32)
    output_min = paddle.empty([bsz, num_heads, num_chunks], dtype=paddle.int32)

    BN = 512
    grid = ((seq_len + BN - 1) // BN, bsz * num_heads)
    scan_maxmin_chunked[grid](
        input,
        output_max,
        output_min,
        seq_len,
        num_chunks,
        chunk_size=chunk_size,
        BN=BN,
    )

    return output_max, output_min
