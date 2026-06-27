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

"""FlashMLA sparse forward kernel for the "cudnn" CSA sparse-attn backend.

This is the forward half of the "cudnn" backend (the backward half lives in
``csa_sparse_attn_bwd_cudnn.py``). It wraps the FlashMLA sparse prefill kernel
and the index/alignment helpers it needs.
"""

import paddle
import paddle.nn.functional as F

from paddleformers.fleet.fusions.csa_sparse_attn_utils import _local_to_global_flat

try:
    from paddlefleet_ops.flash_mla import (
        flash_mla_sparse_fwd as _flash_mla_sparse_fwd,
    )
except (ImportError, RuntimeError):
    _flash_mla_sparse_fwd = None


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
