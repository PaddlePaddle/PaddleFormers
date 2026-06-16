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
import paddle.nn.functional as F

from . import sparse_mqa_fwd

try:
    from paddlefleet_ops.flash_mla import (
        flash_mla_sparse_fwd as _flash_mla_sparse_fwd,
    )
except (ImportError, RuntimeError):
    _flash_mla_sparse_fwd = None


def _prepare_inputs(q, kv, attn_sink, topk_idxs):
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


def sparse_attn(q, kv, attn_sink, topk_idxs, sm_scale=None, backend="tilelang"):
    q, kv, attn_sink, topk_idxs = _prepare_inputs(q, kv, attn_sink, topk_idxs)
    if backend == "cudnn":
        out, lse, _ = flash_mla_sparse_attn(
            q, kv, attn_sink, topk_idxs, sm_scale=sm_scale
        )
    else:
        out, lse = sparse_mqa_fwd.sparse_mqa_fwd_interface(
            q, kv, attn_sink, topk_idxs, sm_scale=sm_scale
        )
    if not isinstance(out, paddle.Tensor) or not isinstance(lse, paddle.Tensor):
        raise RuntimeError(
            f"TileLang must return Paddle tensors, got output={type(out)!r}, lse={type(lse)!r}. "
            "Ensure paddle.enable_compat(scope={'tilelang'}) runs before import tilelang."
        )
    return out, lse


def _get_topk_alignment() -> int:
    """Minimum ``TopK`` alignment required by the current GPU architecture.

    * SM90 : dual-warpgroup loop steps by 2 blocks → ``2 * B_TOPK = 128``
    * SM100: single-pipeline loop steps by 1 block → ``B_TOPK`` (64 for
      head64, 128 for head128). DSA uses ``D = 512`` which maps to the
      head64 kernel path → 64.
    """
    sm = paddle.cuda.get_device_capability()
    if sm[0] >= 10:
        return 64
    return 128


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


def flash_mla_sparse_attn(
    q, kv, attn_sink, topk_idxs, sm_scale=None, indexer_topk: int = 0
):
    if _flash_mla_sparse_fwd is None:
        raise RuntimeError("flash_mla is not available")

    b, sq, h, d = q.shape
    _, skv, _ = kv.shape
    topk = topk_idxs.shape[-1]

    q_flat = q.reshape([b * sq, h, d])
    kv_flat = kv.reshape([b * skv, d])
    global_idxs = _local_to_global_flat(topk_idxs, skv)

    topk_align = _get_topk_alignment()
    topk_padded = (topk + topk_align - 1) // topk_align * topk_align
    if topk_padded != topk:
        global_idxs = F.pad(global_idxs, (0, topk_padded - topk), value=-1)

    res = _flash_mla_sparse_fwd(
        q_flat,
        kv_flat.unsqueeze(1),
        global_idxs.unsqueeze(1),
        sm_scale,
        d_v=d,
        attn_sink=attn_sink,
        topk_length=None,
        indexer_topk=indexer_topk,
    )
    if indexer_topk > 0:
        out_flat, _max_logits, lse, lse_indexer = res
        lse_indexer = lse_indexer.reshape([b, sq, h])
    else:
        out_flat, _max_logits, lse = res
        lse_indexer = None
    return out_flat.reshape([b, sq, h, d]), lse.reshape([b, sq, h]), lse_indexer
