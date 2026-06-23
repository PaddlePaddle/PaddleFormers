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

"""Shared helpers for CSA sparse-attn backends (tilelang and cudnn)."""

import paddle


def prepare_inputs(q, kv, attn_sink, topk_idxs):
    """Validate shapes and normalize dtypes shared by all sparse-attn kernels.

    ``topk_idxs`` is cast to int32 and ``attn_sink`` to float32; ``q``/``kv``
    keep their original dtype.
    """
    if len(q.shape) != 4:
        raise ValueError(f"q must have shape [B, S, H, D], got {q.shape}")
    if len(kv.shape) != 3:
        raise ValueError(f"kv must have shape [B, S_kv, D], got {kv.shape}")
    if len(topk_idxs.shape) != 3:
        raise ValueError(
            f"topk_idxs must have shape [B, S, topk], got {topk_idxs.shape}"
        )

    if topk_idxs.dtype != paddle.int32:
        topk_idxs = topk_idxs.cast("int32")

    if attn_sink.dtype != paddle.float32:
        attn_sink = attn_sink.cast("float32")
    return q, kv, attn_sink, topk_idxs


def _local_to_global_flat(local_idxs, seqlen_kv: int):
    """Convert local per-batch indices to Paddle batch-first flat indices.

    Follows the Paddle layout used by this module: tensors are flattened from
    ``[B, S, ...]`` to ``[B * S, ...]`` in batch-first row order. The global KV
    index is ``batch_id * seqlen_kv + local`` for valid entries and ``-1``
    otherwise.

    Args:
        local_idxs: ``(B, S, topk)`` int, values in ``[0, seqlen_kv)`` or -1.
        seqlen_kv: KV sequence length per batch.

    Returns:
        ``(B * S, topk)`` int32.
    """
    b, sq, topk = local_idxs.shape
    idxs_flat = local_idxs.reshape([b * sq, topk])
    valid = idxs_flat >= 0
    batch_ids = (
        paddle.arange(b, dtype=idxs_flat.dtype)
        .unsqueeze(1)
        .expand([b, sq])
        .reshape([b * sq])
    )
    batch_offsets = (batch_ids * seqlen_kv).unsqueeze(1)
    return paddle.where(valid, idxs_flat + batch_offsets, idxs_flat).cast(
        "int32"
    )
