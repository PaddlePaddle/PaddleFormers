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

"""Single-card tests for FA4 attention-sink (learnable_sink) support.

Covers PR "Support FA4 sink." (ab8c450) on the non-CP paths:
  1. Low-level cute ``flashmask_attention(..., learnable_sink=)`` fwd+bwd vs an
     inline fp32 golden (trainable sink / sink=None / GQA / dsink dtype).
  2. Facade ``paddlefleet_ops.flash_mask_facade.flashmask_attention`` (+2 lines:
     forwards learnable_sink), exercising its two quirks (clone() on
     startend_row_indices; lse reshape only valid for nheads==1).
  3. ``DotProductAttention`` softmax_offset construction (vanilla/off-by-one/
     learnable + sink-bias promotion) and the fwd/bwd sink branch.
  4. Refined-recompute (rr) FlashMask attention with learnable_sink, driven
     through ``recompute`` to force both forward passes, vs the non-rr path.
"""

import math
import unittest

import numpy as np
import paddle

from paddlefleet_ops import is_flash_mask_available

# Force FA4 (cute) backend for the whole module: sink only exists there.
paddle.set_flags({"FLAGS_flash_attn_version": 4})

DTYPE = paddle.bfloat16
SEED = 2026

# FA4 attention-sink lives only in the cute backend, which is built solely for
# compute capability >= 10 (sm100/Blackwell). Elsewhere the cute kernels are
# absent and the facade falls back to a sink-less backend, so skip these tests.
_SINK_AVAILABLE = is_flash_mask_available()
_SKIP_REASON = (
    "FA4 attention-sink requires the cute backend (sm100, capability >= 10)"
)


def _startend_row_indices(batch_size, seq_len, causal):
    """Build a [b, 1, seq_len, 2] int32 startend_row_indices tensor.

    causal=True  -> start=0,       end=arange(1, seq+1)  (lower-triangular)
    causal=False -> start=seq_len, end=0                 (full attention)

    For the non-causal 2-col case the columns are (down_start, up_end):
    rows >= down_start and rows < up_end are masked. "Mask nothing" (full
    attention) is therefore down_start=seq_len, up_end=0 -- matching the repo's
    generate_non_causal_mask. (start=0, end=seq_len would mask EVERY row.)
    """
    if causal:
        start = np.zeros((batch_size, 1, seq_len, 1), dtype=np.int32)
        end = np.arange(1, seq_len + 1, dtype=np.int32).reshape(
            1, 1, seq_len, 1
        )
        end = np.broadcast_to(end, (batch_size, 1, seq_len, 1))
    else:
        start = np.full((batch_size, 1, seq_len, 1), seq_len, dtype=np.int32)
        end = np.zeros((batch_size, 1, seq_len, 1), dtype=np.int32)
    indices = np.concatenate([start, end], axis=-1)
    return paddle.to_tensor(indices, dtype=paddle.int32)


def _attn_bias_from_indices(startend_row_indices, seqlen_q, nheads, causal):
    """Dense additive bias (-inf masked) from a 2-col startend_row_indices.

    Mirrors generate_startend_row_indices.startend_row_indices_to_attn_bias for
    the causal-2col / non-causal-2col cases used in these tests. Returns a fp32
    tensor broadcastable to [b, nheads, seqlen_q, seqlen_k].
    """
    bz, num_head, seqlen_k, bound_num = startend_row_indices.shape
    assert nheads % num_head == 0
    idx = startend_row_indices.numpy()
    m = np.zeros((bz, num_head, seqlen_q, seqlen_k), dtype=np.float32)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seqlen_k):
                downstart = int(idx[bi, hi, j, 0])
                if causal:
                    downend = int(idx[bi, hi, j, 1])
                    m[bi, hi, downstart:downend, j] = -np.inf
                    # bottom-right aligned causal mask
                    top = max(0, j - (seqlen_k - seqlen_q))
                    m[bi, hi, :top, j] = -np.inf
                else:
                    upend = int(idx[bi, hi, j, 1])
                    m[bi, hi, downstart:, j] = -np.inf
                    m[bi, hi, :upend, j] = -np.inf
    m = np.repeat(m, nheads // num_head, axis=1)
    return paddle.to_tensor(m, dtype=paddle.float32)


def attention_ref_with_sink(q, k, v, attn_bias, learnable_sink):
    """fp32 reference for flashmask + attention-sink.

    q,k,v: [b, s, h, d] (GQA: h_kv may be < h_q). attn_bias: additive [-inf] mask
    broadcastable to [b, h_q, sq, sk]. learnable_sink: [h_q] per-q-head logit
    (un-scaled) competing only in the softmax denominator, or None.

    sink formula:
        sink   = sink[h].reshape(1, h, 1, 1)            # fp32, un-scaled logit
        row_max= max(scores.max(-1, keepdim), sink)
        denom  = exp(scores-row_max).sum(-1, keepdim) + exp(sink-row_max)
        attn   = exp(scores-row_max) / denom
        out    = attn @ v
    """
    qf = q.astype("float32").transpose([0, 2, 1, 3])  # b h s d
    kf = k.astype("float32").transpose([0, 2, 1, 3])
    vf = v.astype("float32").transpose([0, 2, 1, 3])

    h_q = qf.shape[1]
    g = h_q // kf.shape[1]
    if g > 1:
        kf = paddle.repeat_interleave(kf, g, axis=1)
        vf = paddle.repeat_interleave(vf, g, axis=1)

    d = qf.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    scores = paddle.matmul(qf * softmax_scale, kf, transpose_y=True)

    if attn_bias is not None:
        bias = attn_bias
        if bias.shape[1] != h_q:
            bias = paddle.repeat_interleave(bias, h_q // bias.shape[1], axis=1)
        scores = scores + bias
        all_inf = (bias == -np.inf).all(axis=-1, keepdim=True)
        scores = paddle.where(all_inf, paddle.full_like(scores, -1e30), scores)
    else:
        all_inf = None

    row_max = scores.max(axis=-1, keepdim=True)
    if learnable_sink is not None:
        sink = learnable_sink.astype("float32").reshape([1, h_q, 1, 1])
        row_max = paddle.maximum(row_max, sink)
        exp_sink = paddle.exp(sink - row_max)
    else:
        exp_sink = 0.0

    exp_scores = paddle.exp(scores - row_max)
    denom = exp_scores.sum(axis=-1, keepdim=True) + exp_sink
    attention = exp_scores / denom
    if all_inf is not None:
        attention = paddle.where(
            all_inf, paddle.zeros_like(attention), attention
        )

    out = paddle.matmul(attention.astype(vf.dtype), vf)
    out = out.transpose([0, 2, 1, 3])  # b s h d
    return out


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestCuteFlashmaskSink(unittest.TestCase):
    """Low-level cute flashmask_attention numerical correctness with sink."""

    def _run(
        self,
        batch_size,
        seq_len,
        nheads,
        nheads_kv,
        d,
        causal,
        use_sink,
    ):
        from paddlefleet_ops.flash_mask.cute.interface import (
            flashmask_attention,
        )

        paddle.seed(SEED)
        np.random.seed(SEED)

        q_ref = paddle.randn([batch_size, seq_len, nheads, d], dtype=DTYPE)
        k_ref = paddle.randn([batch_size, seq_len, nheads_kv, d], dtype=DTYPE)
        v_ref = paddle.randn([batch_size, seq_len, nheads_kv, d], dtype=DTYPE)
        for t in (q_ref, k_ref, v_ref):
            t.stop_gradient = False

        q, k, v = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]
        for t in (q, k, v):
            t.stop_gradient = False

        if use_sink:
            sink_ref = paddle.randn([nheads], dtype=DTYPE)
            sink_ref.stop_gradient = False
            sink = sink_ref.detach().clone()
            sink.stop_gradient = False
        else:
            sink_ref = None
            sink = None

        startend_row_indices = _startend_row_indices(
            batch_size, seq_len, causal
        )
        attn_bias = _attn_bias_from_indices(
            startend_row_indices, seq_len, nheads, causal
        )

        out_ref = attention_ref_with_sink(
            q_ref, k_ref, v_ref, attn_bias, sink_ref
        )

        out = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=startend_row_indices,
            causal=causal,
            learnable_sink=sink,
        )

        # Forward tolerance: 2x the bf16->fp32 round-trip noise of the reference.
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        max_diff = (out - out_ref).abs().max().item()
        self.assertLessEqual(
            max_diff,
            2e-2 + fwd_atol,
            f"fwd max diff {max_diff} too large (atol={fwd_atol})",
        )

        # Backward
        g = paddle.randn(out.shape, dtype=out.dtype)
        out.backward(g.clone())
        out_ref.backward(g.clone())

        for name, a, b in (
            ("dq", q.grad, q_ref.grad),
            ("dk", k.grad, k_ref.grad),
            ("dv", v.grad, v_ref.grad),
        ):
            atol = 2 * (b + 0.3 - 0.3 - b).abs().max().item()
            diff = (a - b).abs().max().item()
            self.assertLessEqual(
                diff, 5e-2 + atol, f"{name} max diff {diff} too large"
            )

        if use_sink:
            self.assertIsNotNone(sink.grad)
            # Kernel computes dsink in fp32 then casts back to sink dtype (bf16).
            self.assertEqual(sink.grad.dtype, DTYPE)
            self.assertEqual(list(sink.grad.shape), [nheads])

    def test_causal_with_trainable_sink(self):
        self._run(2, 256, 4, 4, 128, causal=True, use_sink=True)

    def test_noncausal_with_trainable_sink(self):
        self._run(2, 256, 4, 4, 128, causal=False, use_sink=True)

    def test_sink_none_matches_plain_softmax(self):
        self._run(2, 256, 4, 4, 128, causal=True, use_sink=False)

    def test_gqa_with_sink(self):
        # nheads_kv < nheads exercises the GQA repeat path with sink.
        self._run(2, 256, 8, 2, 128, causal=True, use_sink=True)

    def test_small_head_dim_sink(self):
        self._run(2, 128, 4, 4, 64, causal=False, use_sink=True)

    def test_fixed_sink_backward_returns_dsink_slot(self):
        # A FIXED (stop_gradient=True) bf16 sink is a valid forward input, but
        # FlashMaskFunc.backward chooses its return arity from
        # ``learnable_sink is None`` alone (interface.py:1770-1772) -- it does
        # NOT consult stop_gradient. So a non-None fixed sink makes backward
        # return the 4-tuple (dq, dk, dv, dsink); Paddle's PyLayer then rejects
        # it because the sink forward input has stop_gradient=True and its slot
        # must be None. We assert the forward is still numerically correct and
        # that backward raises this ValueError, documenting the limitation.
        from paddlefleet_ops.flash_mask.cute.interface import (
            flashmask_attention,
        )

        paddle.seed(SEED)
        np.random.seed(SEED)
        b, s, h, d = 2, 256, 4, 128
        q = paddle.randn([b, s, h, d], dtype=DTYPE)
        k = paddle.randn([b, s, h, d], dtype=DTYPE)
        v = paddle.randn([b, s, h, d], dtype=DTYPE)
        for t in (q, k, v):
            t.stop_gradient = False

        # off-by-one == a fixed all-zeros sink logit (stop_gradient=True).
        sink = paddle.zeros([h], dtype=DTYPE)
        sink.stop_gradient = True

        idx = _startend_row_indices(b, s, causal=True)
        attn_bias = _attn_bias_from_indices(idx, s, h, causal=True)
        out_ref = attention_ref_with_sink(q, k, v, attn_bias, sink)

        out = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=idx,
            causal=True,
            learnable_sink=sink,
        )
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        self.assertLessEqual(
            (out - out_ref).abs().max().item(), 2e-2 + fwd_atol
        )

        with self.assertRaises(ValueError):
            out.backward(paddle.randn(out.shape, dtype=out.dtype))

    def test_sink_dtype_assert(self):
        # The cute kernel asserts learnable_sink is bf16; fp32 must raise.
        from paddlefleet_ops.flash_mask.cute.interface import (
            flashmask_attention,
        )

        paddle.seed(SEED)
        b, s, h, d = 1, 64, 2, 64
        q = paddle.randn([b, s, h, d], dtype=DTYPE)
        k = paddle.randn([b, s, h, d], dtype=DTYPE)
        v = paddle.randn([b, s, h, d], dtype=DTYPE)
        sink_fp32 = paddle.zeros([h], dtype=paddle.float32)
        idx = _startend_row_indices(b, s, causal=True)
        with self.assertRaises(AssertionError):
            flashmask_attention(
                q,
                k,
                v,
                startend_row_indices=idx,
                causal=True,
                learnable_sink=sink_fp32,
            )


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestFacadeFlashmaskSink(unittest.TestCase):
    """paddlefleet_ops.flash_mask_facade.flashmask_attention sink forwarding.

    The PR adds learnable_sink to the facade signature and forwards it to the
    cute kernel. Note two facade quirks exercised here:
      - startend_row_indices.clone() is called unconditionally -> must pass a
        non-None tensor.
      - return_softmax_lse reshapes lse to [bsz, q_len] -> only valid nheads==1.
    """

    def _run(self, nheads, nheads_kv, use_sink, return_lse=False, causal=False):
        from paddlefleet_ops.flash_mask_facade import flashmask_attention

        paddle.seed(SEED)
        b, s, d = 2, 128, 128
        q = paddle.randn([b, s, nheads, d], dtype=DTYPE)
        k = paddle.randn([b, s, nheads_kv, d], dtype=DTYPE)
        v = paddle.randn([b, s, nheads_kv, d], dtype=DTYPE)
        for t in (q, k, v):
            t.stop_gradient = False

        sink = None
        if use_sink:
            sink = paddle.randn([nheads], dtype=DTYPE)
            sink.stop_gradient = False

        idx = _startend_row_indices(b, s, causal)
        out = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=idx,
            causal=causal,
            return_softmax_lse=return_lse,
            learnable_sink=sink,
        )
        if return_lse:
            out, lse = out
            self.assertEqual(list(lse.shape), [b, s])
        self.assertEqual(list(out.shape), [b, s, nheads, d])

        # Reference + numerical check (facade should be a thin wrapper).
        attn_bias = _attn_bias_from_indices(idx, s, nheads, causal)
        out_ref = attention_ref_with_sink(q, k, v, attn_bias, sink)
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        diff = (out - out_ref).abs().max().item()
        self.assertLessEqual(diff, 2e-2 + fwd_atol)

        out.sum().backward()
        self.assertIsNotNone(q.grad)
        if use_sink:
            self.assertIsNotNone(sink.grad)

    def test_facade_with_sink(self):
        self._run(nheads=4, nheads_kv=4, use_sink=True)

    def test_facade_sink_none(self):
        self._run(nheads=4, nheads_kv=4, use_sink=False)

    def test_facade_lse_single_head(self):
        # lse reshape [bsz, q_len] only works for nheads == 1 (facade quirk 2).
        self._run(nheads=1, nheads_kv=1, use_sink=True, return_lse=True)

    def test_facade_gqa_with_sink(self):
        self._run(nheads=8, nheads_kv=2, use_sink=True)


def _make_config(
    softmax_type="vanilla",
    add_full_attention_sink_bias=False,
    add_swa_attention_sink_bias=False,
    num_attention_heads=4,
    head_dim=128,
    hidden_size=512,
):
    """TransformerConfig for DotProductAttention sink tests (bf16, no CP)."""
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )

    config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    config.head_dim = head_dim
    config.num_key_value_heads = num_attention_heads
    config.softmax_scale = None
    config.use_bias = False
    config.context_parallel_size = 1
    config.apply_query_key_layer_scaling = False
    config.sliding_window = None
    config.window_attn_skip_freq = None
    config.fp16 = False
    config.bf16 = True
    config.masked_softmax_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.softmax_type = softmax_type
    config.add_full_attention_sink_bias = add_full_attention_sink_bias
    config.add_swa_attention_sink_bias = add_swa_attention_sink_bias
    # learnable softmax_offset is created with config.params_dtype; the cute
    # kernel requires bf16 sink, so params_dtype must be bf16.
    config.params_dtype = paddle.bfloat16
    config.perform_initialization = False
    config.flashmask_use_varlen = False
    config.experimental_dataflow = False
    return config


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestDotProductAttentionSinkInit(unittest.TestCase):
    """softmax_offset construction across softmax_type / sink-bias promotion."""

    def _build(self, **cfg_kwargs):
        from paddleformers.fleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddleformers.fleet.transformer.enums import AttnMaskType

        config = _make_config(**cfg_kwargs)
        return DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

    def test_vanilla_offset_none(self):
        attn = self._build(softmax_type="vanilla")
        self.assertIsNone(attn.softmax_offset)

    def test_offbyone_zeros(self):
        attn = self._build(softmax_type="off-by-one")
        self.assertIsNotNone(attn.softmax_offset)
        self.assertEqual(
            list(attn.softmax_offset.shape),
            [attn.num_attention_heads_per_partition],
        )
        self.assertTrue(bool((attn.softmax_offset == 0).all().item()))

    def test_learnable_is_parameter(self):
        attn = self._build(softmax_type="learnable")
        self.assertIsNotNone(attn.softmax_offset)
        # A create_parameter() result is trainable (stop_gradient False).
        self.assertFalse(attn.softmax_offset.stop_gradient)
        self.assertEqual(attn.softmax_offset.dtype, paddle.bfloat16)

    def test_full_attention_sink_bias_promotes_to_learnable(self):
        # add_full_attention_sink_bias + non-SWA -> promoted to learnable.
        attn = self._build(
            softmax_type="vanilla",
            add_full_attention_sink_bias=True,
        )
        self.assertIsNotNone(attn.softmax_offset)
        self.assertFalse(attn.softmax_offset.stop_gradient)


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestDotProductAttentionSinkForward(unittest.TestCase):
    """Full fwd/bwd through the flashmask sink branch of DotProductAttention."""

    def _run(self, softmax_type, **cfg_kwargs):
        from paddleformers.fleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddleformers.fleet.transformer.enums import AttnMaskType

        paddle.seed(SEED)
        config = _make_config(softmax_type=softmax_type, **cfg_kwargs)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        b, s = 2, 64
        h = config.num_attention_heads
        d = config.head_dim
        q = paddle.randn([b, s, h, d], dtype=DTYPE)
        k = paddle.randn([b, s, h, d], dtype=DTYPE)
        v = paddle.randn([b, s, h, d], dtype=DTYPE)
        for t in (q, k, v):
            t.stop_gradient = False

        idx = _startend_row_indices(b, s, causal=True)
        out = attn(
            query=q,
            key=k,
            value=v,
            attention_mask=None,
            attn_mask_startend_row_indices=idx,
            attn_mask_type=AttnMaskType.causal,
        )
        self.assertEqual(list(out.shape), [b, s, h * d])

        out.sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

        if softmax_type == "learnable":
            self.assertIsNotNone(attn.softmax_offset.grad)
            self.assertEqual(attn.softmax_offset.grad.dtype, paddle.bfloat16)

    def test_forward_vanilla(self):
        self._run("vanilla")

    def test_forward_learnable_sink(self):
        self._run("learnable")

    def test_forward_offbyone(self):
        # off-by-one builds an fp32 zeros offset; the cute kernel asserts bf16,
        # so this path is expected to raise on the fa4 sink branch.
        with self.assertRaises(AssertionError):
            self._run("off-by-one")


@unittest.skipUnless(_SINK_AVAILABLE, _SKIP_REASON)
class TestRefinedRecomputeFlashMaskSink(unittest.TestCase):
    """Refined-recompute (rr) non-CP path with learnable_sink.

    The rr FlashMask attention keys off ``framework._dygraph_tracer()._has_grad``
    to pick between two forward passes: the first (``_has_grad`` False) runs the
    real cute kernel under no_grad and stashes tensors; the second (``_has_grad``
    True) rebuilds the graph via ``FlashMaskAttnFunctor``, whose custom backward
    computes the grads. We drive both passes manually so the test exercises the
    rr functor's fwd/bwd directly, and compare against the non-rr cute
    ``flashmask_attention`` sink path.
    """

    def _run(self, causal, use_sink, sink_trainable=True):
        from paddle import framework

        from paddlefleet_ops.flash_mask.cute.interface import (
            flashmask_attention,
        )
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        paddle.seed(SEED)
        np.random.seed(SEED)
        b, s, h, d = 2, 256, 4, 128

        q_ref = paddle.randn([b, s, h, d], dtype=DTYPE)
        k_ref = paddle.randn([b, s, h, d], dtype=DTYPE)
        v_ref = paddle.randn([b, s, h, d], dtype=DTYPE)
        for t in (q_ref, k_ref, v_ref):
            t.stop_gradient = False

        q, k, v = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]
        for t in (q, k, v):
            t.stop_gradient = False

        if use_sink:
            sink_ref = paddle.randn([h], dtype=DTYPE)
            sink_ref.stop_gradient = not sink_trainable
            sink = sink_ref.detach().clone()
            sink.stop_gradient = not sink_trainable
        else:
            sink_ref = sink = None

        idx = _startend_row_indices(b, s, causal)

        # Non-rr reference through the plain cute flashmask_attention.
        out_ref = flashmask_attention(
            q_ref,
            k_ref,
            v_ref,
            startend_row_indices=idx,
            causal=causal,
            learnable_sink=sink_ref,
        )

        # Drive the rr two-pass mechanism MANUALLY (no recompute) so the test
        # exercises FlashMaskAttnFunctor's fwd/bwd directly. The first call runs
        # with _has_grad False (stash pass, no grad tracked); the second runs
        # with _has_grad True (graph-rebuild pass) so `out` carries the functor
        # grad node and out.backward() invokes the rr custom backward directly.
        rr_attn = RefinedRcomputeFlashMaskAttention()
        tracer = framework._dygraph_tracer()
        prev_has_grad = tracer._has_grad

        tracer._has_grad = False
        try:
            rr_attn(q, k, v, idx, causal=causal, learnable_sink=sink)
        finally:
            tracer._has_grad = prev_has_grad

        tracer._has_grad = True
        try:
            out = rr_attn(q, k, v, idx, causal=causal, learnable_sink=sink)
        finally:
            tracer._has_grad = prev_has_grad
        self.assertEqual(list(out.shape), [b, s, h, d])

        # rr should match the non-rr sink path exactly (same kernel).
        max_diff = (out - out_ref).abs().max().item()
        self.assertLessEqual(
            max_diff, 1e-2, f"rr fwd max diff {max_diff} vs non-rr too large"
        )

        # A fixed (stop_gradient) sink makes the non-rr flashmask_attention
        # PyLayer return a dsink slot that Paddle rejects on backward (a known
        # limitation of that op, covered by test_fixed_sink_backward_returns_
        # dsink_slot). So only run the reference backward when the sink is not a
        # fixed tensor; the rr backward is always exercised below.
        ref_backward_safe = not (use_sink and not sink_trainable)

        g = paddle.randn(out.shape, dtype=out.dtype)
        out.backward(g.clone())
        if ref_backward_safe:
            out_ref.backward(g.clone())
            for name, a, ref in (
                ("dq", q.grad, q_ref.grad),
                ("dk", k.grad, k_ref.grad),
                ("dv", v.grad, v_ref.grad),
            ):
                self.assertIsNotNone(a, f"{name} grad is None")
                diff = (a - ref).abs().max().item()
                self.assertLessEqual(
                    diff, 2e-2, f"rr {name} max diff {diff} vs non-rr too large"
                )
        else:
            # Still assert the rr backward produced q/k/v grads.
            for name, a in (("dq", q.grad), ("dk", k.grad), ("dv", v.grad)):
                self.assertIsNotNone(a, f"{name} grad is None")

        if use_sink and sink_trainable:
            self.assertIsNotNone(sink.grad)
            self.assertEqual(sink.grad.dtype, DTYPE)
            self.assertEqual(list(sink.grad.shape), [h])
            sink_diff = (sink.grad - sink_ref.grad).abs().max().item()
            self.assertLessEqual(
                sink_diff,
                2e-2,
                f"rr dsink max diff {sink_diff} vs non-rr too large",
            )
        elif use_sink and not sink_trainable:
            # Fixed sink is stop_gradient -> rr returns no sink grad.
            self.assertIsNone(sink.grad)

    def test_rr_causal_trainable_sink(self):
        self._run(causal=True, use_sink=True)

    def test_rr_noncausal_trainable_sink(self):
        self._run(causal=False, use_sink=True)

    def test_rr_sink_none(self):
        self._run(causal=True, use_sink=False)

    def test_rr_fixed_sink_no_grad(self):
        self._run(causal=True, use_sink=True, sink_trainable=False)

    def test_rr_non_fa4_sink_raises(self):
        # Sink is only supported on the fa_version==4 cute backend; forcing v3
        # must make the rr entry point reject a non-None sink.
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        paddle.seed(SEED)
        b, s, h, d = 2, 128, 4, 128
        q = paddle.randn([b, s, h, d], dtype=DTYPE)
        idx = _startend_row_indices(b, s, causal=True)
        sink = paddle.randn([h], dtype=DTYPE)
        rr_attn = RefinedRcomputeFlashMaskAttention()

        old = paddle.get_flags(["FLAGS_flash_attn_version"])[
            "FLAGS_flash_attn_version"
        ]
        paddle.set_flags({"FLAGS_flash_attn_version": 3})
        try:
            with self.assertRaises(NotImplementedError):
                rr_attn.forward(q, q, q, idx, learnable_sink=sink)
        finally:
            paddle.set_flags({"FLAGS_flash_attn_version": old})


if __name__ == "__main__":
    unittest.main()
