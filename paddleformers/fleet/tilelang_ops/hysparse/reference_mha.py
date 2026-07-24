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

"""Naive Paddle reference (散算子) for the independent MHA block-score attention.

Ground-truth for :mod:`block_score_mha`: standard full attention with H
INDEPENDENT heads (per-head K/V, NOT MQA) that additionally emits per-(query,
key-block) max SCALED logits used for TopK block selection.

Masking (causal + document) is expressed through ``valid_range`` ``[B, S, 2]``
giving, per query token, the half-open valid key column range ``[bos, eos)``.

``ref_block_score_attn_mha`` returns ``block_logit`` storing the SCALED
per-block max logit (``sm_scale * max q.k`` over the block's valid columns;
``-inf`` if none), matching the kernel and the current
``pipeline.block_scores_from_logit`` (``exp(block_logit - lse)``).
"""

import paddle
import paddle.nn.functional as F

NEG_INF = float("-inf")


def _range_mask(valid_range, seq_len_kv):
    """Boolean key mask [B, 1, S, S_kv] from valid_range [B, S, 2]."""
    bos = valid_range[..., 0].unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
    eos = valid_range[..., 1].unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
    col = paddle.arange(seq_len_kv, dtype=valid_range.dtype)
    col = col.reshape([1, 1, 1, seq_len_kv])  # [1, 1, 1, S_kv]
    return (col >= bos) & (col < eos)  # [B, 1, S, S_kv]


def _to_bhsd(x):
    """[B, S, H, D] -> [B, H, S, D]."""
    return x.transpose([0, 2, 1, 3])


def ref_block_score_attn_mha(q, k, v, valid_range, sm_scale=None, block_B=64):
    """Block-score attention reference (MHA): full attention with H independent
    heads (per-head K/V) plus block-max SCALED logits.

    Args:
        q:           [B, S, H, D] query.
        k:           [B, S_kv, H, D] key (per head).
        v:           [B, S_kv, H, D_v] value (per head).
        valid_range: [B, S, 2] int, per-query [bos, eos) valid key columns.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size for the emitted block scores.

    Returns:
        out:         [B, S, H, D_v] attention output.
        lse:         [B, S, H] natural log-sum-exp of the scaled masked logits.
        block_logit: [B, H, S, num_blocks] SCALED per-block max logit
                     (num_blocks = ceil(S_kv / block_B)); -inf for blocks with
                     no valid key. Block coordinates are DOCUMENT-relative.
    """
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    d_v = v.shape[-1]
    if sm_scale is None:
        sm_scale = d**-0.5

    qb = _to_bhsd(q).astype("float32")  # [B, H, S, D]
    kb = _to_bhsd(k).astype("float32")  # [B, H, S_kv, D]
    vb = _to_bhsd(v).astype("float32")  # [B, H, S_kv, D_v]

    logits = paddle.matmul(qb, kb, transpose_y=True) * sm_scale  # [B,H,S,S_kv]
    mask = _range_mask(valid_range, s_kv)  # [B,1,S,S_kv]
    neg = paddle.full_like(logits, NEG_INF)
    logits = paddle.where(mask, logits, neg)

    # Rows with no valid key -> 0 output, -inf lse (matches the kernel).
    row_has_key = mask.any(axis=-1, keepdim=True)  # [B,1,S,1]
    lse = paddle.logsumexp(logits, axis=-1)  # [B,H,S]
    probs = F.softmax(logits, axis=-1)  # [B,H,S,S_kv]
    probs = paddle.where(
        row_has_key.expand_as(probs), probs, paddle.zeros_like(probs)
    )
    out = paddle.matmul(probs, vb)  # [B,H,S,D_v]

    # SCALED block-max logit with DOCUMENT-relative block coordinates: block j
    # of a query spans key columns [bos + j*block_B, bos + (j+1)*block_B). The
    # per-block value is max over that block's valid columns of the scaled logit
    # (sm_scale * q.k); -inf if the block holds no valid key.
    num_blocks = (s_kv + block_B - 1) // block_B
    col = paddle.arange(s_kv, dtype="int64").reshape([1, 1, s_kv])  # [1,1,S_kv]
    bos = valid_range[..., 0:1].astype("int64")  # [B, S, 1]
    rel = col - bos  # [B, S, S_kv] col position relative to doc start
    rel_id = paddle.where(  # relative block id; -1 for cols before doc start
        rel >= 0, rel // block_B, paddle.full_like(rel, -1)
    )  # [B, S, S_kv]
    rel_id = rel_id.unsqueeze(1)  # [B, 1, S, S_kv]
    block_logit_list = []
    for j in range(num_blocks):
        hit = (rel_id == j) & mask  # [B, 1, S, S_kv] valid cols in block j
        neg_l = paddle.full_like(logits, NEG_INF)
        masked = paddle.where(hit, logits, neg_l)  # scaled logit or -inf
        bmax = masked.max(axis=-1)  # [B, H, S]
        # paddle.max over an all -inf row returns the float32 lowest (~-3.4e38),
        # not -inf; force blocks with no valid key to true -inf so masked blocks
        # read back as score 0 (matches the kernel's pre-filled -inf).
        any_hit = hit.any(axis=-1)  # [B, 1, S] -> broadcast over H
        neg_inf = paddle.full_like(bmax, NEG_INF)
        bmax = paddle.where(any_hit.expand_as(bmax), bmax, neg_inf)
        block_logit_list.append(bmax)  # [B, H, S]
    block_logit = paddle.stack(block_logit_list, axis=-1)  # [B,H,S,num_blocks]

    out = out.transpose([0, 2, 1, 3])  # back to [B,S,H,D_v]
    lse = lse.transpose([0, 2, 1])  # [B,S,H]
    return out.astype(q.dtype), lse, block_logit


def make_causal_valid_range(seq_len, batch=1, doc_lengths=None):
    """Build valid_range [B, S, 2] for causal (+ optional document) masking.

    Args:
        seq_len:     total sequence length S (== S_kv).
        batch:       batch size B.
        doc_lengths: optional list of document lengths (packed along S). If
                     given, sum must equal seq_len; each token's bos is its
                     document start. If None, a single document is assumed.

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
