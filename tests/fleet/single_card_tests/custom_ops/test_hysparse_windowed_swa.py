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

"""Unit tests for the HySparse windowed MQA flash attention
(:mod:`paddleformers.fleet.tilelang_ops.hysparse.windowed_mqa_attn` fwd/bwd) and the
sliding-window attention wrapper it powers
(:mod:`paddleformers.fleet.tilelang_ops.hysparse.swa_attn.sliding_window_mqa_attention`).

The windowed kernel is a fused single-shared-K/V-head (MQA) flash attention
whose ``valid_range [B, S, 2]`` half-open ``[bos, eos)`` expresses causal +
document + sliding-window masking. We verify it against a differentiable dense
masked reference (allowed col iff ``bos <= col < eos``) for both the forward
output and the autograd grads, and exercise the ``sliding_window_mqa_attention``
autograd wrapper (including its ``sm_scale=None`` default).
"""

import math
import os
import sys
import unittest

import paddle

# Test-only precision metrics live one directory up (single_card_tests/).
_TESTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)
from _hysparse_metrics import assert_close

from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
    make_causal_valid_range,
)

paddle.enable_compat(scope={"tilelang"}, silent=True)

_NEG_INF = float("-inf")


def _skip_if_no_cuda(tc):
    if not paddle.device.is_compiled_with_cuda():
        tc.skipTest("CUDA build of Paddle required")
    if paddle.device.cuda.device_count() == 0:
        tc.skipTest("no CUDA device available")


def _window_allow_mask(valid_range, s_kv):
    """Bool [B, S, S_kv]: col allowed iff ``bos <= col < eos``."""
    import numpy as np

    vr = valid_range.numpy()
    b, s, _ = vr.shape
    allow = np.zeros([b, s, s_kv], dtype=bool)
    for bi in range(b):
        for i in range(s):
            bos, eos = int(vr[bi, i, 0]), int(vr[bi, i, 1])
            lo, hi = max(0, bos), min(s_kv, eos)
            if hi > lo:
                allow[bi, i, lo:hi] = True
    return paddle.to_tensor(allow)


def _ref_masked_attn(q, k, v, allow, sm_scale, attn_sink=None):
    """Differentiable dense masked MQA attention reference.

    q [B,S,H,D] fp32 leaf, k [B,Skv,D], v [B,Skv,Dv], allow [B,S,Skv] bool.
    ``attn_sink`` [H] fp32, when given, adds a virtual sink column to the
    softmax denominator (attention sink / off-by-one): the sink logit competes
    in the row max and its ``exp(sink - m)`` mass reduces every real weight so
    they sum to < 1.
    """
    logits = paddle.einsum("bshd,bkd->bshk", q, k) * sm_scale
    neg = paddle.full_like(logits, _NEG_INF)
    logits = paddle.where(allow.unsqueeze(2), logits, neg)
    row_has = allow.any(axis=-1)
    m = logits.max(axis=-1, keepdim=True)  # [B,S,H,1]
    if attn_sink is not None:
        sink = attn_sink.reshape([1, 1, -1, 1]).astype("float32")
        m = paddle.maximum(m, sink)
    m = paddle.where(paddle.isfinite(m), m, paddle.zeros_like(m))
    p = paddle.exp(logits - m)
    denom = p.sum(axis=-1, keepdim=True)
    if attn_sink is not None:
        denom = denom + paddle.exp(
            attn_sink.reshape([1, 1, -1, 1]).astype("float32") - m
        )
    denom = paddle.where(denom > 0, denom, paddle.ones_like(denom))
    p = p / denom
    out = paddle.einsum("bshk,bkc->bshc", p, v)
    return out * row_has.astype("float32").unsqueeze(-1).unsqueeze(-1)


def _sliding_window_valid_range(b, s, window):
    """Causal sliding-window ``[max(0, t-W+1), t+1)`` valid_range [B,S,2]."""
    t = paddle.arange(s, dtype="int32")
    bos = (
        paddle.clip(t - window + 1, min=0).reshape([1, s, 1]).expand([b, s, 1])
    )
    eos = (t + 1).reshape([1, s, 1]).expand([b, s, 1])
    return paddle.concat([bos, eos], axis=-1).contiguous()


def _rand_inputs(b, s, h, d, dv, s_kv, seed=7):
    paddle.seed(seed)
    q = paddle.randn([b, s, h, d], dtype="float32")
    k = paddle.randn([b, s_kv, d], dtype="float32")
    v = paddle.randn([b, s_kv, dv], dtype="float32")
    return q, k, v


def _cos(a, b):
    import numpy as np

    af = a.astype("float32").numpy().reshape(-1)
    bf = b.astype("float32").numpy().reshape(-1)
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12
    return float(np.dot(af, bf) / denom)


class TestWindowedMQAForward(unittest.TestCase):
    BLOCK_B = 64

    def test_full_causal_matches_dense(self):
        _skip_if_no_cuda(self)
        from paddleformers.fleet.tilelang_ops.hysparse.windowed_mqa_attn import (
            windowed_mqa_attn_fwd,
        )

        b, s, h, d, dv, s_kv = 1, 192, 8, 64, 64, 192
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv)
        # full causal window == whole prefix.
        vr = _sliding_window_valid_range(b, s, window=s)
        sm_scale = 1.0 / math.sqrt(d)
        out, lse = windowed_mqa_attn_fwd(
            q.astype("bfloat16"),
            k.astype("bfloat16"),
            v.astype("bfloat16"),
            vr,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
        )
        allow = _window_allow_mask(vr, s_kv)
        ref = _ref_masked_attn(q, k, v, allow, sm_scale)
        self.assertEqual(list(out.shape), [b, s, h, dv])
        self.assertEqual(list(lse.shape), [b, s, h])
        assert_close(self, "out", out, ref, min_cos=0.995, max_rel_l2=6e-2)

    def test_sliding_window_matches_dense(self):
        _skip_if_no_cuda(self)
        from paddleformers.fleet.tilelang_ops.hysparse.windowed_mqa_attn import (
            windowed_mqa_attn_fwd,
        )

        b, s, h, d, dv, s_kv = 1, 192, 4, 64, 64, 192
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv, seed=13)
        vr = _sliding_window_valid_range(b, s, window=48)
        sm_scale = 1.0 / math.sqrt(d)
        out, _ = windowed_mqa_attn_fwd(
            q.astype("bfloat16"),
            k.astype("bfloat16"),
            v.astype("bfloat16"),
            vr,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
        )
        allow = _window_allow_mask(vr, s_kv)
        ref = _ref_masked_attn(q, k, v, allow, sm_scale)
        assert_close(self, "out", out, ref, min_cos=0.995, max_rel_l2=6e-2)

    def test_asymmetric_dk_dv(self):
        # absorbed-MLA-like head: D (key/query) != D_v (value). D=128, D_v=64.
        _skip_if_no_cuda(self)
        from paddleformers.fleet.tilelang_ops.hysparse.windowed_mqa_attn import (
            windowed_mqa_attn_fwd,
        )

        b, s, h, d, dv, s_kv = 1, 128, 4, 128, 64, 128
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv, seed=17)
        vr = _sliding_window_valid_range(b, s, window=s)
        sm_scale = 1.0 / math.sqrt(d)
        out, _ = windowed_mqa_attn_fwd(
            q.astype("bfloat16"),
            k.astype("bfloat16"),
            v.astype("bfloat16"),
            vr,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
        )
        allow = _window_allow_mask(vr, s_kv)
        ref = _ref_masked_attn(q, k, v, allow, sm_scale)
        self.assertEqual(list(out.shape), [b, s, h, dv])
        assert_close(self, "out", out, ref, min_cos=0.995, max_rel_l2=6e-2)


class TestSlidingWindowWrapper(unittest.TestCase):
    BLOCK_B = 64

    def test_grads_match_reference(self):
        _skip_if_no_cuda(self)
        from paddleformers.fleet.tilelang_ops.hysparse.swa_attn import (
            sliding_window_mqa_attention,
        )

        b, s, h, d, dv, s_kv = 1, 128, 8, 64, 64, 128
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv, seed=5)
        vr = _sliding_window_valid_range(b, s, window=64)
        sm_scale = 1.0 / math.sqrt(d)

        qk = q.astype("bfloat16").detach()
        kk = k.astype("bfloat16").detach()
        vk = v.astype("bfloat16").detach()
        qk.stop_gradient = False
        kk.stop_gradient = False
        vk.stop_gradient = False
        out, _ = sliding_window_mqa_attention(
            qk, kk, vk, vr, sm_scale=sm_scale, block_B=self.BLOCK_B
        )
        g = paddle.randn(out.shape, dtype="float32").astype("bfloat16")
        out.backward(g)

        qr = q.detach()
        kr = k.detach()
        vr_ = v.detach()
        qr.stop_gradient = False
        kr.stop_gradient = False
        vr_.stop_gradient = False
        allow = _window_allow_mask(vr, s_kv)
        ref = _ref_masked_attn(qr, kr, vr_, allow, sm_scale)
        ref.backward(g.astype("float32"))

        assert_close(
            self, "dQ", qk.grad, qr.grad, min_cos=0.99, max_rel_l2=7e-2
        )
        assert_close(
            self, "dK", kk.grad, kr.grad, min_cos=0.99, max_rel_l2=1.3e-1
        )
        assert_close(
            self, "dV", vk.grad, vr_.grad, min_cos=0.99, max_rel_l2=1.3e-1
        )

    def test_grads_ragged_seqlen(self):
        # Regression for the windowed-bwd out-of-bounds guard: when
        # seq_len % block_M != 0 the last query tile (grid = ceildiv(S, BM))
        # spans rows past seq_len. The backward must zero-fill that ragged
        # tail (per-row guarded Q/dO load) instead of reading OOB. Use S=130
        # (block_M auto-fits to 64 -> 130 % 64 = 2, a 62-row ragged tail).
        _skip_if_no_cuda(self)
        from paddleformers.fleet.tilelang_ops.hysparse.swa_attn import (
            sliding_window_mqa_attention,
        )

        b, s, h, d, dv, s_kv = 1, 130, 8, 64, 64, 130
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv, seed=23)
        vr = _sliding_window_valid_range(b, s, window=48)
        sm_scale = 1.0 / math.sqrt(d)

        qk = q.astype("bfloat16").detach()
        kk = k.astype("bfloat16").detach()
        vk = v.astype("bfloat16").detach()
        qk.stop_gradient = False
        kk.stop_gradient = False
        vk.stop_gradient = False
        out, _ = sliding_window_mqa_attention(
            qk, kk, vk, vr, sm_scale=sm_scale, block_B=self.BLOCK_B
        )
        g = paddle.randn(out.shape, dtype="float32").astype("bfloat16")
        out.backward(g)

        qr = q.detach()
        kr = k.detach()
        vr_ = v.detach()
        qr.stop_gradient = False
        kr.stop_gradient = False
        vr_.stop_gradient = False
        allow = _window_allow_mask(vr, s_kv)
        ref = _ref_masked_attn(qr, kr, vr_, allow, sm_scale)
        ref.backward(g.astype("float32"))

        # grads must be finite (no OOB garbage) and match the dense reference.
        import numpy as np

        for name, got in (("dq", qk.grad), ("dk", kk.grad), ("dv", vk.grad)):
            self.assertTrue(
                np.isfinite(got.astype("float32").numpy()).all(),
                f"{name} has non-finite entries (OOB read?)",
            )
        assert_close(
            self, "dQ", qk.grad, qr.grad, min_cos=0.99, max_rel_l2=7e-2
        )
        assert_close(
            self, "dK", kk.grad, kr.grad, min_cos=0.99, max_rel_l2=1.3e-1
        )
        assert_close(
            self, "dV", vk.grad, vr_.grad, min_cos=0.99, max_rel_l2=1.3e-1
        )

    def test_wrapper_default_sm_scale(self):
        _skip_if_no_cuda(self)
        from paddleformers.fleet.tilelang_ops.hysparse.swa_attn import (
            sliding_window_mqa_attention,
        )

        b, s, h, d, dv, s_kv = 1, 64, 4, 64, 64, 64
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv, seed=9)
        vr = _sliding_window_valid_range(b, s, window=s)
        qk = q.astype("bfloat16")
        qk.stop_gradient = False
        # sm_scale=None -> defaults to q.shape[-1] ** -0.5 in the wrapper.
        out, lse = sliding_window_mqa_attention(
            qk,
            k.astype("bfloat16"),
            v.astype("bfloat16"),
            vr,
            block_B=self.BLOCK_B,
        )
        self.assertEqual(list(out.shape), [b, s, h, dv])
        self.assertTrue(lse.stop_gradient)
        out.backward(paddle.ones_like(out))
        self.assertIsNotNone(qk.grad)


class TestWindowedMQAAttnSink(unittest.TestCase):
    """Attention-sink (softmax off-by-one bias) support on the SWA MQA path."""

    BLOCK_B = 64

    def test_neg_sink_matches_sinkless(self):
        # A very-negative sink must reproduce the plain sinkless forward
        # bit-for-bit: exp(sink - m) underflows to 0, so the denominator is
        # unchanged. Guards the "None path == -1e30 path" invariant.
        _skip_if_no_cuda(self)
        from paddleformers.fleet.tilelang_ops.hysparse.windowed_mqa_attn import (
            windowed_mqa_attn_fwd,
        )

        b, s, h, d, dv, s_kv = 1, 96, 8, 64, 64, 96
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv, seed=11)
        vr = _sliding_window_valid_range(b, s, window=48)
        sm_scale = 1.0 / math.sqrt(d)
        qb, kb, vb = (
            q.astype("bfloat16"),
            k.astype("bfloat16"),
            v.astype("bfloat16"),
        )

        out_none, _ = windowed_mqa_attn_fwd(
            qb, kb, vb, vr, sm_scale=sm_scale, block_B=self.BLOCK_B
        )
        neg_sink = paddle.full([h], -1e30, dtype="float32")
        out_neg, _ = windowed_mqa_attn_fwd(
            qb,
            kb,
            vb,
            vr,
            attn_sink=neg_sink,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
        )
        import numpy as np

        np.testing.assert_array_equal(
            out_none.astype("float32").numpy(),
            out_neg.astype("float32").numpy(),
        )

    def test_forward_matches_sink_reference(self):
        # A finite learnable sink must match the dense masked reference that
        # folds the sink into the softmax denominator.
        _skip_if_no_cuda(self)
        from paddleformers.fleet.tilelang_ops.hysparse.windowed_mqa_attn import (
            windowed_mqa_attn_fwd,
        )

        b, s, h, d, dv, s_kv = 1, 128, 8, 64, 64, 128
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv, seed=13)
        vr = _sliding_window_valid_range(b, s, window=64)
        sm_scale = 1.0 / math.sqrt(d)
        # Mixed-magnitude sinks: some heads sink-heavy, some near-zero.
        paddle.seed(31)
        attn_sink = paddle.randn([h], dtype="float32")

        out, _ = windowed_mqa_attn_fwd(
            q.astype("bfloat16"),
            k.astype("bfloat16"),
            v.astype("bfloat16"),
            vr,
            attn_sink=attn_sink,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
        )
        allow = _window_allow_mask(vr, s_kv)
        ref = _ref_masked_attn(q, k, v, allow, sm_scale, attn_sink=attn_sink)
        assert_close(self, "out", out, ref, min_cos=0.99, max_rel_l2=6e-2)

    def test_sink_grad_matches_reference(self):
        # d(attn_sink) from the fused backward must match autograd on the dense
        # sink reference (and q/k/v grads stay correct with a live sink).
        _skip_if_no_cuda(self)
        from paddleformers.fleet.tilelang_ops.hysparse.swa_attn import (
            sliding_window_mqa_attention,
        )

        b, s, h, d, dv, s_kv = 1, 128, 8, 64, 64, 128
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv, seed=17)
        vr = _sliding_window_valid_range(b, s, window=64)
        sm_scale = 1.0 / math.sqrt(d)

        qk = q.astype("bfloat16").detach()
        kk = k.astype("bfloat16").detach()
        vk = v.astype("bfloat16").detach()
        paddle.seed(41)
        sink_k = paddle.randn([h], dtype="float32").detach()
        for t in (qk, kk, vk, sink_k):
            t.stop_gradient = False
        out, _ = sliding_window_mqa_attention(
            qk,
            kk,
            vk,
            vr,
            attn_sink=sink_k,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
        )
        g = paddle.randn(out.shape, dtype="float32").astype("bfloat16")
        out.backward(g)

        qr = q.detach()
        kr = k.detach()
        vr_ = v.detach()
        sink_r = sink_k.detach().astype("float32")
        for t in (qr, kr, vr_, sink_r):
            t.stop_gradient = False
        allow = _window_allow_mask(vr, s_kv)
        ref = _ref_masked_attn(qr, kr, vr_, allow, sm_scale, attn_sink=sink_r)
        ref.backward(g.astype("float32"))

        self.assertIsNotNone(sink_k.grad)
        assert_close(
            self, "dQ", qk.grad, qr.grad, min_cos=0.99, max_rel_l2=7e-2
        )
        assert_close(
            self,
            "dSink",
            sink_k.grad,
            sink_r.grad,
            min_cos=0.99,
            max_rel_l2=8e-2,
        )


class TestPackedMultiDocBackward(unittest.TestCase):
    """Packed multi-document (nonzero-bos) fwd+bwd precision on the SWA path.

    A single packed sequence of documents ``[40, 88, 133, 27]`` (S=288) with
    per-document causal masking (``bos`` = doc start, nonzero for docs 2..4)
    exercises the doc-boundary masking in both directions. We compare the fused
    kernel's output and its dQ/dK/dV (and dSink, for the learnable-sink case)
    against the differentiable dense masked reference using the shared metrics
    helper, which prints max-abs / max-rel / RMSE / cosine / allclose for each.
    """

    BLOCK_B = 64
    DOC_LENS = [40, 88, 133, 27]

    def _run(self, attn_sink_seed=None):
        from paddleformers.fleet.tilelang_ops.hysparse.swa_attn import (
            sliding_window_mqa_attention,
        )

        b, h, d, dv = 1, 8, 64, 64
        s = sum(self.DOC_LENS)
        s_kv = s
        q, k, v = _rand_inputs(b, s, h, d, dv, s_kv, seed=101)
        vr = make_causal_valid_range(s, batch=b, doc_lengths=self.DOC_LENS)
        sm_scale = 1.0 / math.sqrt(d)

        qk = q.astype("bfloat16").detach()
        kk = k.astype("bfloat16").detach()
        vk = v.astype("bfloat16").detach()
        leaves = [qk, kk, vk]
        sink_k = None
        if attn_sink_seed is not None:
            paddle.seed(attn_sink_seed)
            sink_k = paddle.randn([h], dtype="float32").detach()
            leaves.append(sink_k)
        for t in leaves:
            t.stop_gradient = False
        out, _ = sliding_window_mqa_attention(
            qk,
            kk,
            vk,
            vr,
            attn_sink=sink_k,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
        )
        g = paddle.randn(out.shape, dtype="float32").astype("bfloat16")
        out.backward(g)

        qr = q.detach()
        kr = k.detach()
        vr_ = v.detach()
        ref_leaves = [qr, kr, vr_]
        sink_r = None
        if sink_k is not None:
            sink_r = sink_k.detach().astype("float32")
            ref_leaves.append(sink_r)
        for t in ref_leaves:
            t.stop_gradient = False
        allow = _window_allow_mask(vr, s_kv)
        ref = _ref_masked_attn(qr, kr, vr_, allow, sm_scale, attn_sink=sink_r)
        ref.backward(g.astype("float32"))

        assert_close(self, "out", out, ref, min_cos=0.99, max_rel_l2=6e-2)
        assert_close(
            self, "dQ", qk.grad, qr.grad, min_cos=0.99, max_rel_l2=7e-2
        )
        assert_close(
            self, "dK", kk.grad, kr.grad, min_cos=0.99, max_rel_l2=1.3e-1
        )
        assert_close(
            self, "dV", vk.grad, vr_.grad, min_cos=0.99, max_rel_l2=1.3e-1
        )
        if sink_k is not None:
            self.assertIsNotNone(sink_k.grad)
            assert_close(
                self,
                "dSink",
                sink_k.grad,
                sink_r.grad,
                min_cos=0.99,
                max_rel_l2=8e-2,
            )

    def test_packed_multidoc_backward_sinkless(self):
        _skip_if_no_cuda(self)
        self._run(attn_sink_seed=None)

    def test_packed_multidoc_backward_learnable_sink(self):
        _skip_if_no_cuda(self)
        self._run(attn_sink_seed=53)


if __name__ == "__main__":
    unittest.main()
