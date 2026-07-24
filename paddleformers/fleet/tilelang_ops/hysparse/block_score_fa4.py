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

"""FA4-fused full block-score attention (HySparse full layers).

The fused FlashAttention-v4 sm100 forward kernel (``paddlefleet_ops.flash_mask``)
now optionally emits the HySparse per-(query, key-block) max logit alongside the
attention output, computed inside the softmax epilogue (see the ``has_block_logit``
path in ``flash_fwd_sm100.py``). This wrapper allocates the block-logit buffer,
runs the fast FA4 path, and returns an ``(out, lse, block_logit)`` triple -- at
~FA4 speed and without an extra HBM Q/K re-read for the block-max reduction.

Semantics:
* ``out``         ``[B, S, H, D_v]`` attention output (softmax-scaled).
* ``lse``         ``[B, S, H]`` natural-log LSE (transposed from FA4's ``[B, H, S]``).
* ``block_logit`` ``[B, H, S, num_blocks]`` max of the **scaled** attention
  logit (``softmax_scale * q.k``, the exact value fed into softmax) over each
  key-block's valid columns; fully-masked / never-visited blocks stay ``-inf``.
  Storing the scaled logit puts every head on one head-independent scale, so a
  downstream ``block_logit - LSE`` yields log(max attention weight in the
  block). Convert to the eq.(3) probability with
  :func:`pipeline.block_scores_from_logit`.

Masking: ``causal=True`` (single-document causal). Document masks flow through
FA4's flashmask ``startend_row_indices`` (passed straight through); because the
block-max reduce reads the score fragment *after* ``mask_fn`` runs, any mask FA4
supports is honoured automatically.
"""

import paddle

from paddlefleet_ops.flash_mask.cute.flashmask_utils import FlashMaskInfoPaddle
from paddlefleet_ops.flash_mask.cute.interface import (
    _flash_attn_bwd,
    _flash_attn_fwd,
)


class _BlockScoreFA4Attn(paddle.autograd.PyLayer):
    """Differentiable FA4-fused full block-score attention.

    ``out`` carries gradient (FA4 fwd + bwd kernels); ``lse`` and the in-place
    ``block_logit`` buffer are non-differentiable (they feed the hard TopK block
    selection). Runs the fast fused FA4 sm100 path.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        block_logit,
        block_bos,
        causal,
        sm_scale,
        startend_row_indices,
        block_B,
    ):
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=causal,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            block_logit=block_logit,
            block_size=block_B,
            block_bos=block_bos,
            pack_gqa=False,
        )
        # save_for_backward only accepts Tensors; startend_row_indices is
        # optional (None disables flashmask), so stash whether it is present
        # and only save it when it is a real Tensor.
        ctx.has_mask = startend_row_indices is not None
        if ctx.has_mask:
            ctx.save_for_backward(q, k, v, startend_row_indices, out, lse)
        else:
            ctx.save_for_backward(q, k, v, out, lse)
        ctx.needs_grad = (
            not q.stop_gradient,
            not k.stop_gradient,
            not v.stop_gradient,
        )
        ctx.mark_non_differentiable(lse)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        return out, lse

    @staticmethod
    def backward(ctx, dout, *_):
        if ctx.has_mask:
            q, k, v, startend_row_indices, out, lse = ctx.saved_tensor()
        else:
            q, k, v, out, lse = ctx.saved_tensor()
            startend_row_indices = None
        flashmask_info = None
        if startend_row_indices is not None:
            flashmask_info = FlashMaskInfoPaddle(
                startend_row_indices=startend_row_indices,
                is_causal=ctx.causal,
            )
        dq, dk, dv, _ = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout.contiguous(),
            lse,
            flashmask_info,
            softmax_scale=ctx.sm_scale,
            causal=ctx.causal,
            deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                "FLAGS_cudnn_deterministic"
            ],
        )
        # One grad slot per forward Tensor input, in order:
        #   q, k, v, block_logit, block_bos [, startend_row_indices].
        # block_logit / block_bos are non-differentiable buffers -> None;
        # startend_row_indices (only a Tensor input when has_mask) -> None.
        gq, gk, gv = ctx.needs_grad
        grads = [
            dq if gq else None,
            dk if gk else None,
            dv if gv else None,
            None,  # block_logit
            None,  # block_bos
        ]
        if ctx.has_mask:
            grads.append(None)  # startend_row_indices
        return tuple(grads)


def block_score_fa4_attn_fwd(
    q,
    k,
    v,
    valid_range=None,
    sm_scale=None,
    block_B=64,
    causal=True,
    startend_row_indices=None,
):
    """FA4-fused MHA block-score forward.

    Args:
        q:           [B, S, H, D] bf16 query.
        k:           [B, S_kv, H, D] bf16 key.
        v:           [B, S_kv, H, D_v] bf16 value.
        valid_range: [B, S, 2] int32 per-query [bos, eos). The ``bos`` column is
            threaded into the kernel as the per-query document start so the fused
            block-max buckets key columns by DOCUMENT-relative block
            (``floor((col - bos) / block_B)``). This makes packed (bos>0)
            block selection bit-identical to running each document alone (bos=0)
            -- "pack-equivalence". When ``None`` the kernel uses bos=0 (single
            document, absolute == relative). FA4 still does the actual masking
            via ``causal`` and/or ``startend_row_indices``.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size (must divide FA4's n_block_size=128).
        causal:      apply causal masking (single-document).
        startend_row_indices: optional flashmask document-mask indices; when
            provided FA4 runs its flashmask path and the block-max honours it.

    Returns:
        (out [B,S,H,D_v], lse [B,S,H], block_logit [B,H,S,num_blocks]).
        num_blocks = ceil(S_kv / block_B). Block coordinates are DOCUMENT-relative
        (relative to each query's ``bos`` from ``valid_range``), matching the
        HySparse pipeline / TopK selection convention.
    """
    assert q.is_contiguous(), "q must be contiguous"
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    assert list(k.shape) == [b, s_kv, h, d], (
        f"k must be [B, S_kv, H, D] matching q; got k {k.shape}, q {q.shape}"
    )
    d_v = v.shape[-1]
    assert list(v.shape) == [b, s_kv, h, d_v], (
        f"v must be [B, S_kv, H, D_v]; got {v.shape}"
    )
    if sm_scale is None:
        sm_scale = d**-0.5

    num_blocks = (s_kv + block_B - 1) // block_B
    # Pre-fill -inf: FA4 only writes the key-blocks its (causal/mask) iteration
    # actually visits; skipped blocks must read back as -inf so their host-side
    # block-score is 0. The buffer is mutated in place by the kernel.
    block_logit = paddle.full(
        [b, h, s, num_blocks], float("-inf"), dtype="float32"
    )

    # Per-query document start (bos) for document-relative block bucketing.
    # valid_range[..., 0] is the bos of each query row; None => single-document
    # (bos=0), which makes relative bucketing degenerate to the absolute one.
    if valid_range is not None:
        assert list(valid_range.shape) == [b, s, 2], (
            f"valid_range must be [B={b}, S={s}, 2]; got {valid_range.shape}"
        )
        block_bos = valid_range[..., 0].contiguous().astype("int32")
    else:
        block_bos = paddle.zeros([b, s], dtype="int32")

    out, lse = _BlockScoreFA4Attn.apply(
        q,
        k,
        v,
        block_logit,
        block_bos,
        causal,
        float(sm_scale),
        startend_row_indices,
        block_B,
    )

    # FA4 LSE is [B, H, S]; the HySparse pipeline expects [B, S, H].
    lse = lse.transpose([0, 2, 1]).contiguous()
    return out, lse, block_logit
