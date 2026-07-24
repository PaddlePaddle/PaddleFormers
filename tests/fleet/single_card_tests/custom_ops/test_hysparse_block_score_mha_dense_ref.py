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

"""Independent dense-reference fwd + bwd coverage for the TileLang MHA
block-score attention oracle (:mod:`block_score_mha`), the scorer used when
``hy_sparse_use_tilelang=true``.

This is the pytest-suite counterpart of the standalone
``scratch_tilelang_tests/test_block_score_mha.py`` script: it cross-checks the
TileLang forward (``out`` / ``lse`` / scaled ``block_logit``) and backward
(``dq`` / ``dk`` / ``dv`` via BOTH the explicit ``block_score_mha_bwd_interface``
and the ``PyLayer`` autograd wiring) against the naive fp32 Paddle reference
``ref_block_score_attn_mha`` (散算子). All gradient checks use a RANDOM
cotangent (not a uniform ``ones`` grad), and every comparison is scored with
MAGNITUDE-sensitive metrics -- cosine floor + dtype-aware allclose + an explicit
relative-L2 / norm-ratio bound -- so a systematic scale error cannot hide behind
a high cosine. TopK block-index consistency through the production
``select_topk_blocks`` pipeline is also checked.

Runs on any TileLang-capable CUDA GPU; skips gracefully otherwise.
"""

import os
import sys
import unittest

import numpy as np
import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

# Reusable precision metrics helper at the single_card_tests root.
_TESTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)
from _hysparse_metrics import assert_close, compute_metrics


def _tl_unavailable_reason():
    if not paddle.device.is_compiled_with_cuda():
        return "CUDA build of Paddle required"
    if paddle.device.cuda.device_count() == 0:
        return "no CUDA device available"
    try:
        from paddleformers.fleet.tilelang_ops.hysparse.block_score_mha import (
            block_score_mha_attn_fwd,  # noqa: F401
        )
    except (ImportError, RuntimeError):
        return "TileLang MHA block-score op import failed"
    return None


_SKIP_REASON = None


def _skip_if_no_tl(tc):
    global _SKIP_REASON
    if _SKIP_REASON is None:
        _SKIP_REASON = _tl_unavailable_reason() or ""
    if _SKIP_REASON:
        tc.skipTest(_SKIP_REASON)


def _rel_l2(got, ref):
    """(relative-L2 error, ||got||/||ref|| norm ratio) over finite entries."""
    g = got.astype("float32").numpy().reshape(-1)
    r = ref.astype("float32").numpy().reshape(-1)
    fin = np.isfinite(g) & np.isfinite(r)
    g, r = g[fin], r[fin]
    rn = float(np.linalg.norm(r))
    gn = float(np.linalg.norm(g))
    rel = float(np.linalg.norm(g - r) / (rn + 1e-12))
    ratio = gn / (rn + 1e-12)
    return rel, ratio


class TestBlockScoreMHADenseRef(unittest.TestCase):
    """TileLang MHA block-score oracle vs naive fp32 dense reference."""

    BLOCK_B = 64

    # ----- builders -------------------------------------------------------
    def _make(self, b, s, h, d, d_v, doc_lengths, seed):
        from paddleformers.fleet.tilelang_ops.hysparse.reference_mha import (
            make_causal_valid_range,
        )

        paddle.seed(seed)
        np.random.seed(seed)
        # 0.5 scale keeps logits in a sane range so bf16 rounding, not softmax
        # saturation, governs the comparison.
        q = (paddle.randn([b, s, h, d]) * 0.5).cast("bfloat16")
        k = (paddle.randn([b, s, h, d]) * 0.5).cast("bfloat16")
        v = (paddle.randn([b, s, h, d_v]) * 0.5).cast("bfloat16")
        vr = make_causal_valid_range(s, batch=b, doc_lengths=doc_lengths)
        sm = d**-0.5
        # RANDOM cotangent (not ones): exercises every output channel's grad.
        dout = (paddle.randn([b, s, h, d_v]) * 0.5).cast("bfloat16")
        return q, k, v, vr, sm, dout

    def _leaf(self, t, dtype=None):
        x = t.detach().clone()
        if dtype is not None:
            x = x.astype(dtype)
        x.stop_gradient = False
        return x

    # ----- core case ------------------------------------------------------
    def _run_case(self, tag, b, s, h, d, d_v, doc_lengths, seed=0):
        from paddleformers.fleet.tilelang_ops.hysparse.block_score_mha import (
            block_score_mha_attn_fwd,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.block_score_mha_bwd import (
            block_score_mha_bwd_interface,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.reference_mha import (
            ref_block_score_attn_mha,
        )

        q, k, v, vr, sm, dout = self._make(b, s, h, d, d_v, doc_lengths, seed)
        if doc_lengths is not None:
            self.assertGreater(int(vr[0, :, 0].max()), 0)  # nonzero bos

        # ---- forward: TileLang kernel ----
        qk, kk, vk = self._leaf(q), self._leaf(k), self._leaf(v)
        out_k, lse_k, blk_k = block_score_mha_attn_fwd(
            qk, kk, vk, valid_range=vr, sm_scale=sm, block_B=self.BLOCK_B
        )
        # ---- forward: fp32 dense reference (higher-precision ground truth) ----
        qr = self._leaf(q, "float32")
        kr = self._leaf(k, "float32")
        vr_l = self._leaf(v, "float32")
        out_r, lse_r, blk_r = ref_block_score_attn_mha(
            qr, kr, vr_l, vr, sm_scale=sm, block_B=self.BLOCK_B
        )
        dout_f = dout.astype("float32")

        assert_close(
            self,
            f"{tag}:fwd_out",
            out_k,
            out_r,
            min_cos=0.99,
            require_allclose=True,
        )
        self._check_magnitude(f"{tag}:fwd_out", out_k, out_r, 5e-2)
        assert_close(
            self,
            f"{tag}:fwd_lse",
            lse_k,
            lse_r,
            min_cos=0.999,
            max_rel_l2=1e-7,
        )
        self._check_block_logit(f"{tag}:block_logit", blk_k, blk_r)

        # ---- TopK block indices via the production pipeline ----
        topk = 3
        self._check_topk(f"{tag}:topk", blk_k, lse_k, blk_r, lse_r, vr, topk)

        # ---- backward via explicit interface (deterministic) ----
        dq_k, dk_k, dv_k = block_score_mha_bwd_interface(
            qk,
            kk,
            vk,
            out_k,
            dout,
            lse_k,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
        )
        dq_r, dk_r, dv_r = paddle.grad(
            outputs=[out_r], inputs=[qr, kr, vr_l], grad_outputs=[dout_f]
        )
        for nm, gk, gr in (
            ("dq", dq_k, dq_r),
            ("dk", dk_k, dk_r),
            ("dv", dv_k, dv_r),
        ):
            assert_close(self, f"{tag}:bwd_{nm}", gk, gr, min_cos=0.99)
            self._check_magnitude(f"{tag}:bwd_{nm}", gk, gr, 8e-2)

        # ---- backward via the PyLayer autograd wiring ----
        qp, kp, vp = self._leaf(q), self._leaf(k), self._leaf(v)
        out_p, _, _ = block_score_mha_attn_fwd(
            qp, kp, vp, valid_range=vr, sm_scale=sm, block_B=self.BLOCK_B
        )
        (out_p.astype("float32") * dout.astype("float32")).sum().backward()
        for nm, gp, gr in (
            ("dq", qp.grad, dq_r),
            ("dk", kp.grad, dk_r),
            ("dv", vp.grad, dv_r),
        ):
            self.assertIsNotNone(gp, f"{tag}:pylayer_{nm} grad is None")
            assert_close(self, f"{tag}:pylayer_{nm}", gp, gr, min_cos=0.99)
            self._check_magnitude(f"{tag}:pylayer_{nm}", gp, gr, 8e-2)

    # ----- magnitude-sensitive checks -------------------------------------
    def _check_magnitude(self, name, got, ref, rel_tol):
        """Relative-L2 bound + norm-ratio near 1 (catches scale errors a cosine
        floor alone would miss). Prints rel-L2 / RMSE / norm-ratio for EVERY
        case so packed multi-doc and Dk!=Dv magnitudes are visible, not just
        asserted."""
        rel, ratio = _rel_l2(got, ref)
        m = compute_metrics(got, ref)
        print(
            f"    [{name}] rel_l2={rel:.3e} rmse={m.rmse:.3e} "
            f"norm_ratio={ratio:.4f}"
        )
        self.assertLessEqual(
            rel, rel_tol, f"{name}: rel-L2 {rel:.3e} > {rel_tol:g}"
        )
        self.assertLess(
            abs(ratio - 1.0),
            max(rel_tol, 5e-2),
            f"{name}: norm ratio {ratio:.4f} far from 1",
        )

    def _check_block_logit(self, name, got, ref):
        """Scaled block-max logit: -inf mask pattern must match; finite entries
        within a scaled tolerance."""
        g = got.astype("float32")
        r = ref.astype("float32")
        big_neg = -1e30
        g = paddle.where(g < big_neg, paddle.full_like(g, float("-inf")), g)
        r = paddle.where(r < big_neg, paddle.full_like(r, float("-inf")), r)
        g_inf = paddle.logical_not(paddle.isfinite(g))
        r_inf = paddle.logical_not(paddle.isfinite(r))
        pattern_mismatch = int((g_inf != r_inf).astype("int32").sum().item())
        self.assertEqual(
            pattern_mismatch, 0, f"{name}: -inf mask pattern mismatch"
        )
        fin = paddle.isfinite(g) & paddle.isfinite(r)
        gf = paddle.where(fin, g, paddle.zeros_like(g))
        rf = paddle.where(fin, r, paddle.zeros_like(r))
        m = compute_metrics(gf, rf)
        print(f"    [{name}] finite max_abs={m.max_abs:.3e} rmse={m.rmse:.3e}")
        self.assertLessEqual(
            m.max_abs, 6e-2, f"{name}: block_logit max_abs {m.max_abs:.3e}"
        )

    def _check_topk(self, name, k_logit, k_lse, r_logit, r_lse, vr, topk):
        """select_topk_blocks picks a per-row block SET; compare kernel vs ref
        as sets (bf16 ties/order may differ)."""
        from paddleformers.fleet.tilelang_ops.hysparse.pipeline import (
            select_topk_blocks,
        )

        ik = select_topk_blocks(k_logit, k_lse, vr, topk, self.BLOCK_B).numpy()
        ir = select_topk_blocks(r_logit, r_lse, vr, topk, self.BLOCK_B).numpy()
        b, s, _ = ik.shape
        mism = 0
        for bi in range(b):
            for si in range(s):
                sk = {int(x) for x in ik[bi, si] if x >= 0}
                sr = {int(x) for x in ir[bi, si] if x >= 0}
                if sk != sr:
                    mism += 1
        total = b * s
        print(f"    [{name}] topk mismatched rows={mism}/{total}")
        self.assertLessEqual(
            mism,
            max(1, int(0.01 * total)),
            f"{name}: {mism}/{total} TopK rows differ kernel-vs-ref",
        )

    # ----- test cases -----------------------------------------------------
    def test_single_doc_b1(self):
        _skip_if_no_tl(self)
        self._run_case("single_b1", 1, 256, 8, 256, 256, None, seed=1)

    def test_multi_doc_packed_ragged(self):
        # Packed docs, all unaligned to BLOCK_B=64 -> nonzero bos downstream.
        _skip_if_no_tl(self)
        self._run_case(
            "multidoc", 1, 288, 8, 256, 256, [40, 88, 133, 27], seed=3
        )

    def test_dk_ne_dv(self):
        # D (qk) != D_v (value) path.
        _skip_if_no_tl(self)
        self._run_case("dk_ne_dv", 1, 192, 8, 128, 64, [80, 112], seed=4)

    def test_online_h64_d256(self):
        # Production-scale full-layer SCORER dims: this op is real per-head MHA
        # over the DECOMPRESSED q/k/v (multi_latent_attention.py:830 passes
        # query/key [B,S,H,Dk] + value [B,S,H,Dv]). For ernielite_layer43:
        # H=num_attention_heads=64, D=qk_nope(192)+qk_rope(64)=256, D_v=
        # v_head_dim=256. This is the scorer's per-head head_dim -- NOT the
        # sparse-gather shared-key latent Dk (=kv_lora_rank+rope=512/576), which
        # is a single MQA latent head exercised in the mqa_tl precision file.
        _skip_if_no_tl(self)
        self._run_case(
            "online_h64", 1, 224, 64, 256, 256, [50, 100, 74], seed=5
        )


if __name__ == "__main__":
    unittest.main()
