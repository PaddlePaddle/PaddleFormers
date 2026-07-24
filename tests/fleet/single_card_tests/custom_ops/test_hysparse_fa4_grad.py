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

"""Backward / autograd coverage for the FA4-fused full block-score attention
(:mod:`paddleformers.fleet.tilelang_ops.hysparse.block_score_fa4`).

The consistency test (``test_hysparse_fa4_topk_consistency``) only drives the
forward path. Here we exercise the ``_BlockScoreFA4Attn`` PyLayer *backward*
(FA4 sm100 bwd kernel) and the ``sm_scale=None`` default, verifying the
attention-output gradient against a plain dense-attention reference (the
``block_logit`` / ``lse`` outputs are non-differentiable and carry no grad).

FA4 block-score fusion runs only on SM 10.x (Blackwell); the test skips
otherwise.
"""

import math
import os
import sys
import unittest

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

# The reusable precision metrics helper lives at the single_card_tests root
# (sibling of this ``custom_ops`` package). Put that root on sys.path so the
# module is importable regardless of the cwd the test runner uses.
_TESTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)
from _hysparse_metrics import assert_close

from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
    make_causal_valid_range,
)

_NEG_INF = float("-inf")


def _sm100_or_skip(tc):
    if not paddle.device.is_compiled_with_cuda():
        tc.skipTest("CUDA build of Paddle required")
    if paddle.device.cuda.device_count() == 0:
        tc.skipTest("no CUDA device available")
    if paddle.device.cuda.get_device_capability()[0] != 10:
        tc.skipTest("FA4 block-score fusion requires SM 10.x (Blackwell)")


def _ref_causal_attn(q, k, v, sm_scale):
    """Differentiable dense causal MHA reference. q/k/v [B,S,H,D] fp32."""
    b, s, h, d = q.shape
    sk = k.shape[1]
    qf = q.transpose([0, 2, 1, 3])
    kf = k.transpose([0, 2, 1, 3])
    vf = v.transpose([0, 2, 1, 3])
    logits = paddle.matmul(qf, kf, transpose_y=True) * sm_scale  # [B,H,S,Sk]
    row = paddle.arange(s).reshape([s, 1])
    col = paddle.arange(sk).reshape([1, sk])
    masked = (col > row + (sk - s)).reshape([1, 1, s, sk])
    logits = paddle.where(masked, paddle.full_like(logits, _NEG_INF), logits)
    p = paddle.nn.functional.softmax(logits, axis=-1)
    out = paddle.matmul(p, vf)  # [B,H,S,D]
    return out.transpose([0, 2, 1, 3])  # [B,S,H,D]


def _cos(a, b):
    import numpy as np

    af = a.astype("float32").numpy().reshape(-1)
    bf = b.astype("float32").numpy().reshape(-1)
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12
    return float(np.dot(af, bf) / denom)


def _doc_bounds(doc_lens):
    """[(start, end)] cumulative document boundaries for the packed sequence."""
    bounds, off = [], 0
    for L in doc_lens:
        bounds.append((off, off + L))
        off += L
    return bounds


def _multidoc_startend_row_indices(doc_lens, h):
    """flashmask LTS [1, H, S, 1] int32: mask query rows >= doc_end per key col.

    Combined with FA4's built-in causal (rows < col masked), each key column c
    in document [ds, de) is visible exactly to rows [c, de) -- a block-diagonal
    (per-document) causal mask.
    """
    s = sum(doc_lens)
    lts = paddle.zeros([s], dtype="int32")
    for ds, de in _doc_bounds(doc_lens):
        lts[ds:de] = de
    return lts.reshape([1, 1, s, 1]).expand([1, h, s, 1]).contiguous()


def _packed_causal_mask(doc_lens):
    """Bool [S, S]: col c visible to row r iff same document and c <= r.

    This is the exact block-diagonal causal mask that FA4's ``causal`` +
    per-document ``startend_row_indices`` produce for the packed sequence, so it
    is the reference masking for the differentiable dense backward check.
    """
    import numpy as np

    s = sum(doc_lens)
    m = np.zeros([s, s], dtype=bool)
    for ds, de in _doc_bounds(doc_lens):
        for r in range(ds, de):
            m[r, ds : r + 1] = True
    return paddle.to_tensor(m)


def _ref_packed_attn(q, k, v, mask, sm_scale):
    """Differentiable dense masked MHA reference (fp32). q/k/v [B,S,H,D];
    ``mask`` [S,S] bool selects the visible (row, col) pairs."""
    b, s, h, d = q.shape
    sk = k.shape[1]
    qf = q.transpose([0, 2, 1, 3])
    kf = k.transpose([0, 2, 1, 3])
    vf = v.transpose([0, 2, 1, 3])
    logits = paddle.matmul(qf, kf, transpose_y=True) * sm_scale  # [B,H,S,Sk]
    mm = mask.reshape([1, 1, s, sk])
    logits = paddle.where(mm, logits, paddle.full_like(logits, _NEG_INF))
    p = paddle.nn.functional.softmax(logits, axis=-1)
    out = paddle.matmul(p, vf)  # [B,H,S,D]
    return out.transpose([0, 2, 1, 3])  # [B,S,H,D]


class TestBlockScoreFA4Backward(unittest.TestCase):
    def test_backward_matches_dense_reference(self):
        _sm100_or_skip(self)
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
        )

        b, s, h, d = 1, 256, 8, 64
        paddle.seed(2026)
        q = paddle.randn([b, s, h, d], dtype="bfloat16")
        k = paddle.randn([b, s, h, d], dtype="bfloat16")
        v = paddle.randn([b, s, h, d], dtype="bfloat16")
        sm_scale = 1.0 / math.sqrt(d)

        qf = q.detach()
        kf = k.detach()
        vf = v.detach()
        qf.stop_gradient = False
        kf.stop_gradient = False
        vf.stop_gradient = False
        out, lse, block_logit = block_score_fa4_attn_fwd(
            qf, kf, vf, sm_scale=sm_scale, block_B=64, causal=True
        )
        g = paddle.randn(out.shape, dtype="float32").astype("bfloat16")
        out.backward(g)

        qr = q.astype("float32").detach()
        kr = k.astype("float32").detach()
        vr = v.astype("float32").detach()
        qr.stop_gradient = False
        kr.stop_gradient = False
        vr.stop_gradient = False
        ref = _ref_causal_attn(qr, kr, vr, sm_scale)
        ref.backward(g.astype("float32"))

        # Cosine floor + magnitude-sensitive rel-L2 ceiling: cosine alone is
        # scale-invariant and would miss a constant-factor gradient scale bug.
        assert_close(
            self,
            "fa4_dense_dq",
            qf.grad,
            qr.grad,
            min_cos=0.99,
            max_rel_l2=6e-3,
        )
        assert_close(
            self,
            "fa4_dense_dk",
            kf.grad,
            kr.grad,
            min_cos=0.99,
            max_rel_l2=6e-3,
        )
        assert_close(
            self,
            "fa4_dense_dv",
            vf.grad,
            vr.grad,
            min_cos=0.99,
            max_rel_l2=6e-3,
        )

    def test_default_sm_scale(self):
        _sm100_or_skip(self)
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
        )

        b, s, h, d = 1, 128, 4, 64
        paddle.seed(7)
        q = paddle.randn([b, s, h, d], dtype="bfloat16")
        k = paddle.randn([b, s, h, d], dtype="bfloat16")
        v = paddle.randn([b, s, h, d], dtype="bfloat16")
        # sm_scale=None -> defaults to d ** -0.5.
        out_default, _, _ = block_score_fa4_attn_fwd(
            q, k, v, block_B=64, causal=True
        )
        out_explicit, _, _ = block_score_fa4_attn_fwd(
            q, k, v, sm_scale=d**-0.5, block_B=64, causal=True
        )
        self.assertEqual(list(out_default.shape), [b, s, h, d])
        # Kernel-vs-kernel: default sm_scale must reproduce the explicit one.
        # Guard magnitude too so a scale drift between the two paths can't hide
        # behind a scale-invariant cosine.
        assert_close(
            self,
            "default_sm_scale_out",
            out_default,
            out_explicit,
            min_cos=0.999,
            max_rel_l2=1e-3,
        )

    def test_packed_multidoc_backward(self):
        # Packed [40, 88, 133, 27] documents (all unaligned to block_B) with a
        # per-document causal flashmask. The FA4 fwd+bwd dQ/dK/dV must match a
        # differentiable dense reference under the exact block-diagonal mask.
        _sm100_or_skip(self)
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
        )

        doc_lens = [40, 88, 133, 27]  # sum = 288
        s = sum(doc_lens)
        h, d = 8, 64
        sm_scale = 1.0 / math.sqrt(d)
        paddle.seed(2026)
        q = paddle.randn([1, s, h, d], dtype="bfloat16")
        k = paddle.randn([1, s, h, d], dtype="bfloat16")
        v = paddle.randn([1, s, h, d], dtype="bfloat16")

        valid_range = make_causal_valid_range(s, batch=1, doc_lengths=doc_lens)
        startend = _multidoc_startend_row_indices(doc_lens, h)

        qf = q.detach()
        kf = k.detach()
        vf = v.detach()
        qf.stop_gradient = False
        kf.stop_gradient = False
        vf.stop_gradient = False
        out, _, _ = block_score_fa4_attn_fwd(
            qf,
            kf,
            vf,
            valid_range=valid_range,
            sm_scale=sm_scale,
            block_B=64,
            causal=True,
            startend_row_indices=startend,
        )
        g = paddle.randn(out.shape, dtype="float32").astype("bfloat16")
        out.backward(g)

        mask = _packed_causal_mask(doc_lens)
        qr = q.astype("float32").detach()
        kr = k.astype("float32").detach()
        vr = v.astype("float32").detach()
        qr.stop_gradient = False
        kr.stop_gradient = False
        vr.stop_gradient = False
        ref = _ref_packed_attn(qr, kr, vr, mask, sm_scale)
        ref.backward(g.astype("float32"))

        assert_close(
            self,
            "fa4_packed_dq",
            qf.grad,
            qr.grad,
            min_cos=0.99,
            max_rel_l2=6e-3,
        )
        assert_close(
            self,
            "fa4_packed_dk",
            kf.grad,
            kr.grad,
            min_cos=0.99,
            max_rel_l2=6e-3,
        )
        assert_close(
            self,
            "fa4_packed_dv",
            vf.grad,
            vr.grad,
            min_cos=0.99,
            max_rel_l2=6e-3,
        )


if __name__ == "__main__":
    unittest.main()
