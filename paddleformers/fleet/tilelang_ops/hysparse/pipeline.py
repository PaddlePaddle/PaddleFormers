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

"""HySparse block-TopK selection.

:func:`select_topk_blocks` recovers eq.(3) block scores from the scaled
per-block max logits emitted by the block-score attention forward, aggregates
them across heads by a group-wise **maximum** (shared block selection), masks
blocks that hold no causal/document-valid key, and TopK-selects per-query block
indices. The block scores feed a non-differentiable TopK, so no autograd graph
is built for the selection.
"""

import paddle


def block_scores_from_logit(block_logit, lse):
    """Recover eq.(3) block-max probability scores on the host.

    The block-score attention forward (FA4-fused, :mod:`block_score_fa4`) emits
    the **scaled** per-block max logit (``softmax_scale * q.k``, the exact value
    fed into softmax); this turns it into the eq.(3) softmax probability
    ``exp(block_logit - lse)``. The scale already lives inside ``block_logit``,
    so no ``sm_scale`` multiply is applied here.

    Args:
        block_logit: [B, H, S, num_blocks] scaled per-block max logit (-inf if
            fully masked), as returned by the forward.
        lse:         [B, S, H] natural-log LSE from the forward.

    Returns:
        [B, H, S, num_blocks] block-max softmax probabilities in [0, 1];
        fully-masked blocks are 0.
    """
    lse_bhs = lse.transpose([0, 2, 1]).unsqueeze(-1)  # [B,H,S,1]
    scaled = block_logit.astype("float32") - lse_bhs
    scores = paddle.exp(scaled)
    # exp(-inf) already 0, but guard any nan from (-inf)-(-inf) style edge.
    scores = paddle.where(
        paddle.isfinite(scores), scores, paddle.zeros_like(scores)
    )
    return scores


def _valid_block_mask(valid_range, num_blocks, block_B):
    """Boolean [B, S, num_blocks]: relative block j holds >=1 valid key.

    Block ids are **document-relative**: block j of a query spans key columns
    ``[bos + j*block_B, bos + (j+1)*block_B)``. It holds at least one valid key
    iff its start still lies inside the query's valid range, i.e.
    ``bos + j*block_B < eos``.
    """
    bos = valid_range[..., 0:1].astype("int64")  # [B, S, 1]
    eos = valid_range[..., 1:2].astype("int64")  # [B, S, 1]
    j = paddle.arange(num_blocks, dtype="int64").reshape([1, 1, num_blocks])
    start = bos + j * block_B  # absolute column where relative block j starts
    return start < eos  # [B, S, num_blocks]


def select_topk_blocks(
    block_logit, lse, valid_range, topk, block_B, head_agg="max"
):
    """Select per-query TopK key blocks from block-score attention outputs.

    Args:
        block_logit: [B, H, S, num_blocks] scaled per-block max logit.
        lse:         [B, S, H] natural-log LSE from block-score attention.
        valid_range: [B, S, 2] int, per-query [bos, eos) valid key columns.
        topk:        number of blocks to select per query token.
        block_B:     key block size.
        head_agg:    how to aggregate block scores across heads so the whole
                     query group shares one selection. ``"max"`` (paper eq. 3
                     group-wise maximum) or ``"sum"``.

    Returns:
        indices: [B, S, topk] int32 selected block ids, shared across heads;
                 slots beyond the number of valid blocks are -1. The width is
                 always ``topk`` even when ``topk > num_blocks`` (the extra
                 slots are -1 padding), keeping the shape contract stable.
    """
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    b, h, s, num_blocks = block_logit.shape
    # Block selection is a hard, non-differentiable TopK. Detach the score
    # inputs so no autograd graph is built for it: block_logit / lse come out of
    # the differentiable attention PyLayer, and without detaching they drag topk
    # into backward, where topk_grad dereferences the (int) index gradient and
    # segfaults.
    block_logit = block_logit.detach()
    lse = lse.detach()
    scores = block_scores_from_logit(block_logit, lse)  # [B,H,S,nb]
    # aggregate across heads (block selection shared across the query group)
    if head_agg == "max":
        agg = scores.max(axis=1)  # [B, S, num_blocks]
    elif head_agg == "sum":
        agg = scores.sum(axis=1)  # [B, S, num_blocks]
    else:
        raise ValueError(f"unknown head_agg={head_agg!r}")

    valid = _valid_block_mask(valid_range, num_blocks, block_B)  # [B,S,nb]
    neg = paddle.full_like(agg, -1.0)
    agg = paddle.where(valid, agg, neg)  # invalid blocks pushed to the bottom

    k = min(topk, num_blocks)
    top_val, top_idx = paddle.topk(agg, k=k, axis=-1)  # [B, S, k]
    # slots that landed on an invalid block (negative score) -> -1
    top_idx = paddle.where(top_val >= 0, top_idx, paddle.full_like(top_idx, -1))
    top_idx = top_idx.astype("int32")
    if k < topk:
        # honour the promised [B, S, topk] width when topk exceeds the number
        # of blocks; the surplus slots are -1 padding (already ignored by the
        # gather kernel and the reference).
        pad = paddle.full([b, s, topk - k], -1, dtype="int32")
        top_idx = paddle.concat([top_idx, pad], axis=-1)
    return top_idx.contiguous()
