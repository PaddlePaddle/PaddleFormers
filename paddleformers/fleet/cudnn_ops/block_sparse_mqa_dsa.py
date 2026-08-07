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

"""DSA (FlashMLA sparse fwd + cuDNN DSA bwd) backend for the HySparse
block-sparse MQA gather branch.

DeepSeek-v4's CSA sparse attention (FlashMLA sparse forward + cuDNN DSA
backward) natively handles the absorbed-MQA D=576 query / D_v=512 value
single-shared-head layout. This module bridges the HySparse *block* selection
onto that *token*-level DSA path:

1. **Block -> token index expansion.** Each selected document-relative block
   ``j`` (spanning key cols ``[bos + j*block_B, bos + (j+1)*block_B)``) is
   expanded into its ``block_B`` absolute token columns. ``block_B == 64`` equals
   the SM100 TopK alignment, so one block == one DSA tile chunk (no padding).
2. **Causal/doc masking folded into the index list.** Any expanded column with
   ``col >= eos`` (or belonging to a ``-1`` padding block) is set to ``-1``,
   which DSA treats as invalid -- reproducing the ``valid_range [bos, eos)``
   semantics without a kernel argument.
3. **Sinkless softmax via a very-negative sink (default).** HySparse is sinkless;
   DSA always applies an attention sink. Passing ``sink = -1e30`` per head makes
   ``exp(sink - m) -> 0``, recovering plain softmax. The same sink tensor is used
   in forward and backward and its gradient is discarded. When a learnable
   per-head ``attn_sink`` is supplied instead (attention-sink / softmax
   off-by-one bias), its logits fill the real heads, padded heads stay ``-1e30``,
   and the sink gradient is computed analytically in the backward (the SM100
   cuDNN DSA op allocates ``d_sink`` but never fills it -- it returns zeros --
   so we derive ``d_sink[h] = -sum_{b,s}(p_sink * Delta)`` from the saved LSE /
   out / dO instead).
4. **K==V-unified latent.** DSA takes one ``kv_full`` tensor whose value is its
   leading ``kv_lora_rank`` slice. ``shared_key_sq [B, S, 576]`` already has
   value == leading 512, so it is passed directly and its 576-wide gradient is
   the (combined) gradient w.r.t. the shared latent.

FlashMLA sparse fwd only supports ``h_q == 64`` on SM100, so query heads are
zero-padded up to 64 when ``H < 64`` (padded heads receive zero output gradient
and contribute no KV gradient).
"""

import os
from functools import lru_cache

import paddle

_DSA_HEADS = 64  # FlashMLA sparse fwd only supports h_q == 64 on SM100.
_NEG_SINK = -1e30  # sink so large-negative that exp(sink - m) underflows to 0.


@lru_cache(maxsize=1)
def is_dsa_available() -> bool:
    """Whether the FlashMLA sparse fwd + cuDNN DSA bwd path can run here.

    The DSA fwd/bwd kernels are only implemented for SM100+ (Blackwell); there
    is no eager fallback below it. Probe the actual PaddleFleet ops dependencies
    once per process. Avoid importing the standalone ``cudnn`` package here:
    under Paddle's torch proxy its module discovery can recursively enter
    ``find_spec`` and hang the first attention forward.
    """
    try:
        import paddlefleet_ops

        from paddleformers.fleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn

        if paddle.device.cuda.get_device_capability()[0] < 10:
            return False
        if (
            not paddlefleet_ops.is_flash_mla_available()
            or csa_sparse_attn_fwd_cudnn._flash_mla_sparse_fwd is None
        ):
            return False
        if not paddlefleet_ops.is_cudnn_frontend_available():
            return False
    except (ImportError, RuntimeError, AttributeError):
        return False
    return True


def _expand_blocks_to_token_indices(indices, valid_range, block_B):
    """Expand doc-relative block ids to per-token key-column indices.

    Args:
        indices:     [B, S, topk] int, document-relative block ids (-1 padding).
        valid_range: [B, S, 2] int, per-query ``[bos, eos)`` valid key columns.
        block_B:     block size in tokens.

    Returns:
        [B, S, topk * block_B] int32 absolute key columns; entries whose column
        is ``>= eos`` or that belong to a ``-1`` padding block are set to -1.

    The whole computation is a pure integer index construction and MUST NOT
    carry an autograd graph. Under full-layer recompute, ``indices`` /
    ``valid_range`` are recomputed with grad tracking enabled, so the trailing
    ``paddle.where(...).astype("int32")`` would otherwise build a stray
    Where/Cast grad chain. Passing that grad-tracked integer tensor into the
    ``_BlockSparseDSA`` PyLayer registers a backward edge to an orphan
    ``CastGradNode`` which the engine then schedules with an empty grad holder
    (ref_cnt 0) -> ``cast()`` on an undefined tensor -> segfault. Building the
    indices under ``no_grad`` (and returning a detached tensor) removes the
    stray grad nodes entirely.
    """
    with paddle.no_grad():
        b, s, topk = indices.shape
        bos = valid_range[..., 0:1].astype("int64")  # [B, S, 1]
        eos = valid_range[..., 1:2].astype("int64")  # [B, S, 1]

        blk = indices.astype("int64")  # [B, S, topk]
        start = bos + blk * block_B  # [B, S, topk] absolute col of block start
        offs = paddle.arange(block_B, dtype="int64").reshape([1, 1, 1, block_B])
        cols = start.unsqueeze(-1) + offs  # [B, S, topk, block_B]
        cols = cols.reshape([b, s, topk * block_B])

        blk_invalid = (
            (blk < 0).unsqueeze(-1).expand([b, s, topk, block_B])
        ).reshape([b, s, topk * block_B])
        # cols >= bos always holds (blk >= 0 => start >= bos, offs >= 0); only
        # the tail past eos needs masking, plus columns from -1 padding blocks.
        col_invalid = cols >= eos  # eos broadcasts over the last dim
        invalid = paddle.logical_or(blk_invalid, col_invalid)
        neg = paddle.full_like(cols, -1)
        token_indices = paddle.where(invalid, neg, cols).astype("int32")
    token_indices.stop_gradient = True
    return token_indices


class _BlockSparseDSA(paddle.autograd.PyLayer):
    """FlashMLA sparse forward + cuDNN DSA backward for absorbed MQA.

    forward inputs:
        query:          [B, S, H, Dk] (Dk=576)
        shared_key_sq:  [B, S, Dk] single shared K/V latent (value = leading Dv)
        token_indices:  [B, S, L] int32 per-batch-local key cols (-1 invalid),
                        already block-expanded + doc/causal masked.
        sm_scale, kv_lora_rank (Dv), num_heads (real H)
    outputs: out [B, S, H * Dv] (differentiable in query and shared_key_sq).
    """

    @staticmethod
    def forward(
        ctx,
        query,
        shared_key_sq,
        token_indices,
        sm_scale,
        d_v,
        num_heads,
        attn_sink=None,
    ):
        from paddleformers.fleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
            flash_mla_sparse_attn,
        )

        b, s, h, dk = query.shape
        ctx.num_heads = num_heads
        ctx.d_v = d_v
        ctx.sm_scale = float(sm_scale)
        ctx.query_dtype = query.dtype
        ctx.kv_dtype = shared_key_sq.dtype

        # Pad query heads up to the DSA-supported h_q == 64. The FlashMLA
        # sparse backend fixes h_q at _DSA_HEADS (sink is [_DSA_HEADS]); it can
        # only handle h <= _DSA_HEADS by zero-padding the head dim. h > 64 is
        # unsupported and must be rejected here rather than failing deep in the
        # CUDA op with an opaque shape error.
        if h > _DSA_HEADS:
            raise ValueError(
                f"HySparse DSA sparse attention supports at most {_DSA_HEADS} "
                f"query heads per rank, but got {h}. Reduce "
                "num_attention_heads / swa_num_attention_heads (per-rank after "
                f"TP) to <= {_DSA_HEADS}."
            )
        if h < _DSA_HEADS:
            pad = paddle.zeros([b, s, _DSA_HEADS - h, dk], dtype=query.dtype)
            q_pad = paddle.concat([query, pad], axis=2)
        else:
            q_pad = query

        # Attention sink over the DSA-fixed 64 heads. When ``attn_sink`` is None
        # HySparse is sinkless: a per-head ``-1e30`` makes ``exp(sink - m) -> 0``,
        # recovering plain softmax and discarding the (unused) sink gradient.
        # When a learnable ``attn_sink [num_heads]`` is supplied, its logits fill
        # the real heads and the padded heads keep ``-1e30`` (they contribute no
        # output / gradient). The sink gradient is routed back to the parameter.
        ctx.learnable_sink = attn_sink is not None
        if attn_sink is None:
            sink = paddle.full([_DSA_HEADS], _NEG_SINK, dtype="float32")
        else:
            assert list(attn_sink.shape) == [num_heads], (
                f"attn_sink must be [num_heads={num_heads}]; got "
                f"{attn_sink.shape}"
            )
            sink_real = attn_sink.cast("float32")
            if h < _DSA_HEADS:
                sink_pad = paddle.full(
                    [_DSA_HEADS - h], _NEG_SINK, dtype="float32"
                )
                sink = paddle.concat([sink_real, sink_pad], axis=0)
            else:
                sink = sink_real
            sink = sink.contiguous()
        kv = shared_key_sq  # [B, S, Dk] value = leading d_v

        out, lse, _ = flash_mla_sparse_attn(
            q_pad,
            kv,
            sink,
            token_indices,
            sm_scale=ctx.sm_scale,
            d_v=d_v,
        )  # out [B, S, 64, d_v], lse [B, S, 64]

        ctx.save_for_backward(q_pad, kv, out, lse, token_indices, sink)
        ctx.needs_grad = (
            not query.stop_gradient,
            not shared_key_sq.stop_gradient,
            ctx.learnable_sink and not attn_sink.stop_gradient,
        )
        out_h = out[:, :, :h, :].contiguous()  # drop padded heads
        return out_h.reshape([b, s, h * d_v])

    @staticmethod
    def backward(ctx, grad_output):
        from paddleformers.fleet.cudnn_ops import csa_sparse_attn_bwd_cudnn
        from paddleformers.fleet.fusions.csa_sparse_attn_utils import (
            _local_to_global_flat,
        )

        q_pad, kv, out, lse, token_indices, sink = ctx.saved_tensor()
        b, s, hpad, dk = q_pad.shape
        d_v = ctx.d_v
        h = ctx.num_heads
        _, skv, _ = kv.shape

        # Re-pad the incoming grad back to hpad heads (padded heads get 0 grad,
        # so they contribute nothing to dq / dkv).
        grad_output = grad_output.reshape([b, s, h, d_v])
        if h < hpad:
            gpad = paddle.zeros([b, s, hpad - h, d_v], dtype=grad_output.dtype)
            do = paddle.concat([grad_output, gpad], axis=2)
        else:
            do = grad_output
        do = do.contiguous()

        q_flat = q_pad.reshape([b * s, hpad, dk])
        o_flat = out.reshape([b * s, hpad, d_v])
        do_flat = do.reshape([b * s, hpad, d_v])
        kv_flat = kv.reshape([b * skv, dk])
        gidx_flat = _local_to_global_flat(token_indices, skv)

        # dq/dkv softmax normalization for the finite-sink absorbed-MQA path.
        #
        # The forward output was formed with the sink competing in the softmax
        # denominator: p_k = exp(l_k - lse_full), lse_full = logaddexp(lse_kv,
        # sink). But the forward kernel returns a KV-only ``lse`` (the sink is
        # excluded), and the cuDNN DSA backward's ``d_qk != d_v`` branch (the
        # absorbed-MQA Dk=576 / Dv=512 layout used here) consumes the passed LSE
        # verbatim -- it does NOT fold the sink into the denominator itself.
        # Feeding it the KV-only LSE therefore overestimates every p_k for a
        # finite sink and corrupts dq (confirmed: packed finite-sink dQ cos
        # 0.976 vs the dense reference). Fix: for a finite (learnable) sink on
        # this Dk!=Dv path, pass a sink-inclusive LSE and neutralize the sink
        # argument (a -1e30 sink can no longer double-count in the kernel), so
        # p_k matches the forward exactly.
        #
        # Sinkless keeps the KV-only ``lse`` and the -1e30 ``sink`` untouched
        # (logaddexp(lse, -1e30) == lse), so that path is bit-for-bit unchanged.
        # The analytic d_sink below intentionally keeps using the original
        # KV-only ``lse`` (it re-derives lse_full from it).
        lse_bwd = lse
        sink_bwd = sink
        # [ablation gate] HYSPARSE_DSA_FINITE_SINK_FIX=0 disables the finite-sink
        # LSE correction to reproduce the exact online (v2) DSA backward behavior
        # (KV-only LSE fed verbatim). Default on (=1). Root-cause confirmation only.
        _dsa_finite_sink_fix = (
            os.environ.get("HYSPARSE_DSA_FINITE_SINK_FIX", "1") != "0"
        )
        if _dsa_finite_sink_fix and ctx.learnable_sink and dk != d_v:
            lse_bwd = paddle.logaddexp(
                lse.astype("float32"),
                sink.astype("float32").reshape([1, 1, hpad]),
            ).astype(lse.dtype)
            sink_bwd = paddle.full([hpad], _NEG_SINK, dtype="float32")
        lse_flat = lse_bwd.reshape([b * s, hpad])

        dq_flat, dkv_flat, _d_sink_unused = csa_sparse_attn_bwd_cudnn(
            q_flat,
            kv_flat,
            o_flat,
            do_flat,
            lse_flat,
            sink_bwd,
            gidx_flat,
            softmax_scale=ctx.sm_scale,
        )

        gq, gk, gsink = ctx.needs_grad
        dq = None
        if gq:
            dq = dq_flat.reshape([b, s, hpad, dk])[:, :, :h, :].contiguous()
            dq = dq.cast(ctx.query_dtype)
        dkv = None
        if gk:
            dkv = dkv_flat.reshape([b, s, dk]).cast(ctx.kv_dtype)
        d_attn_sink = None
        if gsink:
            # The cuDNN DSA backward (SM100) allocates ``d_sink`` but its kernel
            # never populates it -- it always returns zeros. So compute the sink
            # gradient analytically here from the saved forward tensors.
            #
            # For a virtual sink logit ``s_h`` competing in the softmax denom
            # ``Z = sum_k exp(logit_k) + exp(s_h)``, weight ``p_k = exp(l_k)/Z``
            # and sink mass ``p_sink = exp(s_h)/Z``. Since
            # ``d p_k / d s_h = -p_k * p_sink`` and ``out = sum_k p_k v_k``:
            #   d out / d s_h = -p_sink * out
            #   d_sink[h] = sum_{b,s}( dO . (d out / d s_h) )
            #             = -sum_{b,s}( p_sink * (dO . out) )
            #             = -sum_{b,s}( p_sink * Delta )
            # with ``Delta[b,s,h] = sum_dv(out * dO)``. The forward LSE is
            # KV-only (excludes the sink), so the full log-denominator is
            # ``logaddexp(lse_kv, s_h)`` and ``p_sink = exp(s_h - lse_full)``.
            out_h = out[:, :, :h, :].astype("float32")
            do_h = do[:, :, :h, :].astype("float32")
            delta = (out_h * do_h).sum(axis=-1)  # [b, s, h]
            sink_real = sink[:h].astype("float32").reshape([1, 1, h])
            lse_h = lse[:, :, :h].astype("float32")
            lse_full = paddle.logaddexp(lse_h, sink_real)
            p_sink = paddle.exp(sink_real - lse_full)  # [b, s, h]
            d_attn_sink = (-(delta * p_sink).sum(axis=[0, 1])).contiguous()
            d_attn_sink = d_attn_sink.cast("float32")

        # One returned grad per **tensor** input, in order. Non-tensor inputs
        # (sm_scale, d_v, num_heads) occupy no slot. ``attn_sink`` occupies a
        # slot only when it was passed as a tensor (sinkless -> None -> no slot),
        # so the returned count is 3 (sinkless) or 4 (learnable sink).
        grads = [dq, dkv, None]  # query, shared_key_sq, token_indices
        if ctx.learnable_sink:
            grads.append(d_attn_sink)
        return tuple(grads)


def block_sparse_mqa_attention_dsa(
    query,
    shared_key_sq,
    shared_block_indices,
    valid_range,
    sm_scale=None,
    block_B=64,
    kv_lora_rank=512,
    attn_sink=None,
):
    """HySparse block-sparse gather attention over the absorbed MQA latent.

    Args:
        query:                [B, S, H, Dk] (Dk = kv_lora_rank + rope, e.g. 576).
        shared_key_sq:        [B, S, Dk] shared K/V latent; value = leading
                              ``kv_lora_rank`` slice.
        shared_block_indices: [B, S, topk] int, document-relative selected block
                              ids (-1 padding), from ``select_topk_blocks``.
        valid_range:          [B, S, 2] int, per-query ``[bos, eos)``.
        sm_scale:             softmax scale (defaults to ``Dk ** -0.5``).
        block_B:              block size in tokens (must equal the DSA TopK
                              alignment, 64 on SM100).
        kv_lora_rank:         value dim ``Dv`` (leading slice of the latent).
        attn_sink:            [H] fp32 per-head learnable attention-sink logit,
                              or ``None`` for HySparse's default sinkless softmax
                              (a ``-1e30`` sink whose gradient is discarded).

    Returns:
        ``(out, None)`` where ``out`` is ``[B, S, H * kv_lora_rank]`` and carries
        gradient to ``query`` and ``shared_key_sq`` (and ``attn_sink`` when a
        learnable sink is supplied). The second element keeps the ``(out, lse)``
        tuple shape of the consumer call site; DSA does not surface a
        differentiable lse here.

    ``kv_lora_rank`` < 512 (e.g. ernielite's 448): the FlashMLA sparse kernel
    hard-requires ``d_v == 512`` and ``d_qk in {512, 576}``. We map the smaller
    latent onto that fixed layout by zero-padding the *value* region up to 512:
    the latent ``[value(kv_lora_rank) | rope]`` is re-laid as
    ``[value(kv_lora_rank) | zeros(512 - kv_lora_rank) | rope]`` for both query
    and key. The dot-product score is unchanged (the inserted zeros contribute
    nothing), the kernel value becomes the leading 512 (= real value padded with
    zeros), and the output's leading ``kv_lora_rank`` columns are sliced back out.
    The pad/slice run outside the PyLayer so autograd routes the gradients back
    to the original ``query`` / ``shared_key_sq`` shapes automatically.
    """
    if sm_scale is None:
        sm_scale = query.shape[-1] ** -0.5
    token_indices = _expand_blocks_to_token_indices(
        shared_block_indices, valid_range, block_B
    )

    b, s, num_heads, d_qk = query.shape
    pad_v = 512 - kv_lora_rank
    if pad_v > 0:
        # Re-lay [value | rope] -> [value | zeros | rope] so value == leading 512.
        q_val, q_rope = query[..., :kv_lora_rank], query[..., kv_lora_rank:]
        k_val = shared_key_sq[..., :kv_lora_rank]
        k_rope = shared_key_sq[..., kv_lora_rank:]
        zq = paddle.zeros([b, s, num_heads, pad_v], dtype=query.dtype)
        zk = paddle.zeros([b, s, pad_v], dtype=shared_key_sq.dtype)
        query_p = paddle.concat([q_val, zq, q_rope], axis=-1)
        key_p = paddle.concat([k_val, zk, k_rope], axis=-1)
        eff_d_v = 512
    elif pad_v < 0:
        raise ValueError(
            f"HySparse DSA supports kv_lora_rank <= 512, got {kv_lora_rank}."
        )
    else:
        query_p, key_p, eff_d_v = query, shared_key_sq, kv_lora_rank

    out = _BlockSparseDSA.apply(
        query_p,
        key_p,
        token_indices,
        float(sm_scale),
        int(eff_d_v),
        int(num_heads),
        attn_sink,
    )

    if pad_v > 0:
        # Drop the padded value columns: keep the real leading kv_lora_rank.
        out = out.reshape([b, s, num_heads, eff_d_v])[..., :kv_lora_rank]
        out = out.reshape([b, s, num_heads * kv_lora_rank])
    return out, None
