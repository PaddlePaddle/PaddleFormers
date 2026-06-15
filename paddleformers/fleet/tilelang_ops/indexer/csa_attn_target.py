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

# TileLang target-probability kernel for DeepSeek V4 CSA indexer loss.
#
# 2-pass online softmax with GEMM-based matmul. Key is [B, S_comp, D] (shared
# across heads per MLA design).
#
# Pass 1: online softmax — stream over topk blocks, maintain running max & sum
#          via the rescaling trick (2x global memory read vs 3x for 3-pass).
# Pass 2: recompute exp(logits - max) / sum and reduce over heads.

import paddle
import tilelang
from tilelang import language as T


def _tilelang_dtype(tensor):
    if tensor.dtype == paddle.bfloat16:
        return "bfloat16"
    if tensor.dtype == paddle.float16:
        return "float16"
    raise TypeError(f"TileLang CSA attention target expects bf16/fp16 inputs, got {tensor.dtype}")


def _shape(tensor):
    return tuple(tensor.shape)


def _require_tensor(name, tensor):
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"{name} must be a paddle.Tensor, got {type(tensor)!r}")


def _require_contiguous(name, tensor):
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_interface_inputs(
    query,
    key_comp,
    topk_indices,
    block_I,
    num_stages,
):
    for name, tensor in (
        ("query", query),
        ("key_comp", key_comp),
        ("topk_indices", topk_indices),
    ):
        _require_tensor(name, tensor)
        _require_contiguous(name, tensor)
    if query.ndim != 4:
        raise ValueError(f"query must have shape [B, S, H, D], got {_shape(query)}")
    if key_comp.ndim != 3:
        raise ValueError(f"key_comp must have shape [B, S_comp, D], got {_shape(key_comp)}")
    if topk_indices.ndim != 3:
        raise ValueError(f"topk_indices must have shape [B, S, topk], got {_shape(topk_indices)}")

    batch, seq_len, heads, dim = _shape(query)
    batch_k, _, dim_k = _shape(key_comp)
    batch_i, seq_len_i, topk_effective = _shape(topk_indices)
    if batch != batch_k or batch != batch_i:
        raise ValueError(
            f"batch mismatch: query={_shape(query)}, key_comp={_shape(key_comp)}, topk_indices={_shape(topk_indices)}"
        )
    if seq_len != seq_len_i:
        raise ValueError(f"sequence mismatch: query={_shape(query)}, topk_indices={_shape(topk_indices)}")
    if dim != dim_k:
        raise ValueError(f"dim mismatch: query={_shape(query)}, key_comp={_shape(key_comp)}")
    if dim & (dim - 1):
        raise ValueError(f"dim must be a power of 2, got {dim}")
    if heads <= 0:
        raise ValueError(f"heads must be positive, got {heads}")
    if heads > 64 and heads % 64 != 0:
        raise ValueError(f"heads must be a multiple of 64 when heads > 64, got {heads}")
    if topk_effective <= 0:
        raise ValueError("topk_indices last dimension must be positive")
    if int(block_I) <= 0:
        raise ValueError(f"block_I must be positive, got {block_I}")
    if int(num_stages) != 0:
        raise ValueError(f"num_stages must be 0, got {num_stages}")


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def tl_csa_attn_target_reducesum(
    heads: int,
    dim: int,
    topk: int,
    block_I: int = 32,
    dtype: str = "bfloat16",
    num_stages: int = 0,
    num_threads: int = 128,
):
    assert num_stages == 0
    assert dim == tilelang.math.next_power_of_2(dim)
    assert topk % block_I == 0
    assert heads > 0

    if heads > 64:
        assert heads % 64 == 0
        replicate_h = heads // 64
    else:
        replicate_h = 1
    padded_h = max(tilelang.math.next_power_of_2(heads), 16)
    h_per_block = padded_h if replicate_h == 1 else 64

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_comp = T.dynamic("seq_len_comp")

    FP32 = "float"
    INT32 = "int32"

    query_shape = [batch, seq_len, heads, dim]
    key_shape = [batch, seq_len_comp, dim]
    topk_indices_shape = [batch, seq_len, topk]
    partial_shape = [batch, seq_len, replicate_h, topk]

    @T.prim_func
    def target_kernel(
        Query: T.Tensor(query_shape, dtype),
        KeyComp: T.Tensor(key_shape, dtype),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        SoftmaxScale: T.Tensor([1], FP32),
        PartialReduceSum: T.Tensor(partial_shape, FP32),
    ):
        with T.Kernel(seq_len * replicate_h, batch, threads=num_threads) as (
            bx,
            by,
        ):
            s_i = bx if replicate_h == 1 else bx // replicate_h
            r_i = bx % replicate_h
            h_base = 0 if replicate_h == 1 else r_i * 64

            query_shared = T.alloc_shared([h_per_block, dim], dtype=dtype)
            key_shared = T.alloc_shared([block_I, dim], dtype=dtype)
            indices_shared = T.alloc_shared([block_I], dtype=INT32)
            safe_indices_shared = T.alloc_shared([block_I], dtype=INT32)

            logits = T.alloc_fragment([h_per_block, block_I], dtype=FP32)
            row_max = T.alloc_fragment([h_per_block], dtype=FP32)
            block_max = T.alloc_fragment([h_per_block], dtype=FP32)
            row_sum = T.alloc_fragment([h_per_block], dtype=FP32)
            block_sum = T.alloc_fragment([h_per_block], dtype=FP32)
            reduce_sum = T.alloc_fragment([block_I], dtype=FP32)

            # Load query tile
            for h_i, d_i in T.Parallel(h_per_block, dim):
                query_shared[h_i, d_i] = T.if_then_else(
                    h_base + h_i < heads,
                    Query[by, s_i, h_base + h_i, d_i],
                    0,
                )
            T.sync_threads()

            # Pass 1: online softmax — compute row_max and row_sum in one pass
            T.fill(row_max, -T.infinity(FP32))
            T.fill(row_sum, 0)
            num_blocks = T.ceildiv(topk, block_I)
            for block_idx in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block_I):
                    indices_shared[i] = TopkIndices[by, s_i, block_idx * block_I + i]
                    safe_indices_shared[i] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        indices_shared[i],
                        0,
                    )
                T.sync_threads()

                for i, d_i in T.Parallel(block_I, dim):
                    key_shared[i, d_i] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        KeyComp[by, safe_indices_shared[i], d_i],
                        0,
                    )
                T.sync_threads()

                T.gemm(
                    query_shared,
                    key_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                for h_i, i in T.Parallel(h_per_block, block_I):
                    logits[h_i, i] = T.if_then_else(
                        ((h_base + h_i) < heads) & (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        logits[h_i, i] * SoftmaxScale[0],
                        -T.infinity(FP32),
                    )
                # Online update: rescale running sum then incorporate new block
                T.fill(block_max, -T.infinity(FP32))
                T.reduce_max(logits, block_max, dim=1, clear=False)
                for h_i in T.Parallel(h_per_block):
                    row_sum[h_i] *= T.exp(row_max[h_i] - T.max(row_max[h_i], block_max[h_i]))
                    row_max[h_i] = T.max(row_max[h_i], block_max[h_i])
                for h_i, i in T.Parallel(h_per_block, block_I):
                    logits[h_i, i] = T.exp(logits[h_i, i] - row_max[h_i])
                T.fill(block_sum, 0)
                T.reduce_sum(logits, block_sum, dim=1, clear=False)
                for h_i in T.Parallel(h_per_block):
                    row_sum[h_i] += block_sum[h_i]

            # Pass 2: compute normalized probs and reduce over heads
            for block_idx in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block_I):
                    indices_shared[i] = TopkIndices[by, s_i, block_idx * block_I + i]
                    safe_indices_shared[i] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        indices_shared[i],
                        0,
                    )
                T.sync_threads()

                for i, d_i in T.Parallel(block_I, dim):
                    key_shared[i, d_i] = T.if_then_else(
                        (indices_shared[i] >= 0) & (indices_shared[i] < seq_len_comp),
                        KeyComp[by, safe_indices_shared[i], d_i],
                        0,
                    )
                T.sync_threads()

                T.gemm(
                    query_shared,
                    key_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                for h_i, i in T.Parallel(h_per_block, block_I):
                    logits[h_i, i] = T.if_then_else(
                        ((h_base + h_i) < heads)
                        & (indices_shared[i] >= 0)
                        & (indices_shared[i] < seq_len_comp)
                        & (row_sum[h_i] > 0),
                        T.exp(logits[h_i, i] * SoftmaxScale[0] - row_max[h_i]) / row_sum[h_i],
                        0,
                    )
                T.fill(reduce_sum, 0)
                T.reduce_sum(logits, reduce_sum, dim=0, clear=False)
                T.copy(
                    reduce_sum,
                    PartialReduceSum[
                        by,
                        s_i,
                        r_i,
                        block_idx * block_I : block_idx * block_I + block_I,
                    ],
                )

    return target_kernel


def csa_attn_target_reducesum_interface(
    query,
    key_comp,
    topk_indices,
    softmax_scale: float,
    block_I: int = 32,
    num_stages: int = 0,
    num_threads: int = 128,
):
    _validate_interface_inputs(
        query,
        key_comp,
        topk_indices,
        block_I,
        num_stages,
    )

    batch, seq_len, heads, dim = query.shape
    topk_effective = topk_indices.shape[-1]

    padded_topk = (topk_effective + block_I - 1) // block_I * block_I
    if padded_topk != topk_effective:
        pad = paddle.full(
            [batch, seq_len, padded_topk - topk_effective],
            -1,
            dtype=topk_indices.dtype,
        )
        topk_indices = paddle.concat([topk_indices, pad], axis=-1).contiguous()

    replicate_h = heads // 64 if heads > 64 else 1
    kernel = tl_csa_attn_target_reducesum(
        heads=heads,
        dim=dim,
        topk=padded_topk,
        block_I=block_I,
        dtype=_tilelang_dtype(query),
        num_stages=num_stages,
        num_threads=num_threads,
    )
    partial = paddle.empty(
        [batch, seq_len, replicate_h, padded_topk],
        dtype="float32",
    )
    scale = paddle.full([1], float(softmax_scale), dtype="float32")
    kernel(query, key_comp, topk_indices, scale, partial)
    valid = topk_indices[:, :, :topk_effective] >= 0
    target = partial[:, :, :, :topk_effective].sum(axis=2)
    target = paddle.where(valid, target, paddle.zeros_like(target))
    target = target / target.sum(axis=-1, keepdim=True).clip(min=1e-10)
    return paddle.where(valid, target, paddle.zeros_like(target))
