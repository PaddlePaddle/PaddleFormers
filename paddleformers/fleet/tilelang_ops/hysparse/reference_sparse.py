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

"""Naive Paddle reference (散算子) for the HySparse block-sparse MQA gather
attention over the *absorbed-MLA* shared latent.

This is the correctness ground truth for :mod:`block_sparse_mqa_tl`. It mirrors
the DSA operator ``block_sparse_mqa_attention_dsa`` semantics exactly:

* **Single shared K/V latent.** ``shared_key_sq [B, S_kv, Dk]`` is the one
  shared head. The *key* used for the ``q·k`` logit is the full ``Dk`` (e.g.
  576); the *value* used for ``p·v`` is its leading ``kv_lora_rank`` slice
  ``Dv`` (e.g. 512). So ``Dk != Dv`` in general.
* **Document-relative block gather.** A selected block id ``blk`` spans absolute
  key columns ``[bos + blk*block_B, bos + (blk+1)*block_B)`` with ``bos`` the
  query's document start (``valid_range[..., 0]``). ``blk < 0`` = padding slot.
  A gathered column is masked out if ``col >= eos`` (``valid_range[..., 1]``).
* **Attention sink.** An optional per-head learnable sink logit competes as a
  virtual key column in the softmax denominator (pre-scaled logit, added in the
  same units as the scaled ``q·k``). ``None`` -> plain (sinkless) softmax.
* **Output** ``[B, S, H*Dv]``.

Written for readability / correctness, not speed. Fully differentiable in
``query``, ``shared_key_sq`` and ``attn_sink`` for autograd gradient checks.
"""

import paddle

NEG_INF = float("-inf")


def _range_mask(valid_range, seq_len_kv):
    """Boolean key mask [B, 1, S, S_kv] from valid_range [B, S, 2] ([bos,eos))."""
    bos = valid_range[..., 0].unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
    eos = valid_range[..., 1].unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
    col = paddle.arange(seq_len_kv, dtype=valid_range.dtype)
    col = col.reshape([1, 1, 1, seq_len_kv])  # [1, 1, 1, S_kv]
    return (col >= bos) & (col < eos)  # [B, 1, S, S_kv]


def _selected_key_mask(indices, valid_range, seq_len_kv, block_B):
    """Boolean [B, S, S_kv]: key column selected by this query's block ids.

    Block ids are **document-relative**: block ``j`` of a query spans key
    columns ``[bos + j*block_B, bos + (j+1)*block_B)`` where ``bos`` is the
    query's document start (``valid_range[..., 0]``). A column is selected iff
    it lies at or after ``bos`` and its relative block id is among the query's
    (valid, ``>= 0``) selected ids.

    indices: [B, S, nsel] int block ids; -1 marks an invalid/padding slot.
    """
    col = paddle.arange(seq_len_kv, dtype="int64").reshape([1, 1, seq_len_kv])
    bos = valid_range[..., 0:1].astype("int64")  # [B, S, 1]
    rel = col - bos  # [B, S, S_kv]
    col_block = paddle.where(  # relative block id of each key column
        rel >= 0, rel // block_B, paddle.full_like(rel, -1)
    )  # [B, S, S_kv]
    idx = indices.astype("int64").unsqueeze(-2)  # [B, S, 1, nsel]
    col_block_e = col_block.unsqueeze(-1)  # [B, S, S_kv, 1]
    hit = (col_block_e == idx) & (idx >= 0)  # [B, S, S_kv, nsel]
    return hit.any(axis=-1)  # [B, S, S_kv]


def ref_block_sparse_mqa(
    query,
    shared_key_sq,
    shared_block_indices,
    valid_range,
    sm_scale=None,
    block_B=64,
    kv_lora_rank=512,
    attn_sink=None,
):
    """Naive fp32 reference for the block-sparse MQA gather over the shared
    latent.

    Args:
        query:                [B, S, H, Dk] query (H heads). Dk = kv_lora_rank +
                              rope (e.g. 576).
        shared_key_sq:        [B, S_kv, Dk] single shared K/V latent. Key = full
                              Dk; value = leading ``kv_lora_rank`` slice (Dv).
        shared_block_indices: [B, S, nsel] int document-relative block ids
                              (-1 = padding).
        valid_range:          [B, S, 2] int per-query [bos, eos).
        sm_scale:             softmax scale; defaults to Dk**-0.5.
        block_B:              block size in tokens.
        kv_lora_rank:         value dim Dv (leading slice of the latent).
        attn_sink:            [H] fp32 per-head learnable sink logit, or None for
                              plain (sinkless) softmax.

    Returns:
        out: [B, S, H*Dv].
    """
    b, s, h, dk = query.shape
    s_kv = shared_key_sq.shape[1]
    dv = kv_lora_rank
    if sm_scale is None:
        sm_scale = dk**-0.5

    qb = query.transpose([0, 2, 1, 3]).astype("float32")  # [B, H, S, Dk]
    k = shared_key_sq.astype("float32")  # [B, S_kv, Dk]
    v = shared_key_sq[..., :dv].astype("float32")  # [B, S_kv, Dv]
    kb = k.unsqueeze(1)  # [B, 1, S_kv, Dk]
    vb = v.unsqueeze(1)  # [B, 1, S_kv, Dv]

    logits = paddle.matmul(qb, kb, transpose_y=True) * sm_scale  # [B,H,S,S_kv]

    range_m = _range_mask(valid_range, s_kv)  # [B, 1, S, S_kv]
    sel_m = _selected_key_mask(
        shared_block_indices, valid_range, s_kv, block_B
    ).unsqueeze(1)  # [B, 1, S, S_kv]
    mask = range_m & sel_m  # [B, 1, S, S_kv]

    # Stable masked softmax. The row max is taken over the valid (masked-in)
    # logits only; empty rows (no valid key) get a finite ``m_safe`` so no NaN
    # leaks into the (unused) output or its gradient.
    masked_logits = paddle.where(
        mask, logits, paddle.full_like(logits, NEG_INF)
    )
    m = masked_logits.max(axis=-1, keepdim=True)  # [B, H, S, 1]
    m_safe = paddle.where(paddle.isfinite(m), m, paddle.zeros_like(m))
    w = paddle.where(
        mask, paddle.exp(logits - m_safe), paddle.zeros_like(logits)
    )  # [B, H, S, S_kv]
    denom = w.sum(axis=-1, keepdim=True)  # [B, H, S, 1]
    if attn_sink is not None:
        # Virtual sink column: a per-head pre-scaled logit competing in the same
        # softmax denominator (same units as the scaled q·k). Sinkless (None)
        # drops the term entirely -> plain softmax.
        sink_h = attn_sink.astype("float32").reshape([1, h, 1, 1])  # [1,H,1,1]
        denom = denom + paddle.exp(sink_h - m_safe)
    denom_safe = paddle.where(
        denom > 0, denom, paddle.ones_like(denom)
    )  # empty row -> 0/1 = 0
    probs = w / denom_safe  # [B, H, S, S_kv]

    out = paddle.matmul(probs, vb.expand([b, h, s_kv, dv]))  # [B, H, S, Dv]
    out = out.transpose([0, 2, 1, 3]).reshape([b, s, h * dv])
    return out


def make_causal_valid_range(seq_len, batch=1, doc_lengths=None):
    """Build valid_range [B, S, 2] for a causal (+ optional document) mask.

    Args:
        seq_len:     total sequence length S (== S_kv).
        batch:       batch size B.
        doc_lengths: optional list of packed document lengths (sum == seq_len).
                     Each token's ``bos`` is its document start; ``eos = t+1``
                     (causal). ``None`` -> a single document.

    Returns:
        valid_range: [B, S, 2] int32.
    """
    pos = paddle.arange(seq_len, dtype="int32")
    eos = pos + 1  # causal: attend up to and including self
    if doc_lengths is None:
        bos = paddle.zeros([seq_len], dtype="int32")
    else:
        assert sum(doc_lengths) == seq_len
        starts = []
        cur = 0
        for dl in doc_lengths:
            starts += [cur] * dl
            cur += dl
        bos = paddle.to_tensor(starts, dtype="int32")
    vr = paddle.stack([bos, eos], axis=-1)  # [S, 2]
    return vr.unsqueeze(0).expand([batch, seq_len, 2]).contiguous()


def build_random_block_indices(valid_range, topk, block_B, seq_len_kv, seed=0):
    """Build random-but-valid document-relative block ids for testing.

    For each query token, ranks the document-relative blocks whose start column
    ``bos + j*block_B`` falls before ``eos`` by a random score and keeps the top
    ``topk`` (padding empty slots with -1 when a query has fewer valid blocks
    than ``topk``). This exercises the gather + ``-1`` padding + causal tail
    masking paths.

    Returns:
        indices: [B, S, topk] int32 (document-relative block ids; -1 padding).
    """
    paddle.seed(seed)
    b, s, _ = valid_range.shape
    bos = valid_range[..., 0:1].astype("int64")  # [B, S, 1]
    eos = valid_range[..., 1:2].astype("int64")  # [B, S, 1]
    n_blk = (seq_len_kv + block_B - 1) // block_B
    j = paddle.arange(n_blk, dtype="int64").reshape([1, 1, n_blk])
    start = bos + j * block_B  # [B, S, n_blk] absolute block start col
    valid = start < eos  # [B, S, n_blk]
    score = paddle.rand([b, s, n_blk])
    score = paddle.where(valid, score, paddle.full_like(score, -1.0))
    k = min(topk, n_blk)
    _, idx = paddle.topk(score, k=k, axis=-1)  # [B, S, k]
    valid_i = valid.astype("int32")
    picked_valid = paddle.take_along_axis(valid_i, idx, axis=-1) > 0  # [B,S,k]
    idx = paddle.where(
        picked_valid, idx.astype("int32"), paddle.full_like(idx, -1, "int32")
    )
    if k < topk:
        pad = paddle.full([b, s, topk - k], -1, dtype="int32")
        idx = paddle.concat([idx, pad], axis=-1)
    return idx.contiguous()
