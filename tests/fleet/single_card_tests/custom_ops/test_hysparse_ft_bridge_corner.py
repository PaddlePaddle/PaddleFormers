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

"""HySparse **FT bridge** corner-case validation: FA4 block scorer ->
``select_topk_blocks`` -> TileLang ``block_sparse_mqa_attention_tl`` gather.

The single artifact that crosses the bridge is the per-query TopK **block-index
tensor**. The FA4 scorer runs on the DECOMPRESSED-MHA full-attention head
(independent heads, ``head_dim = qk_nope+qk_rope = 256``); the downstream gather
runs on the ABSORBED-MLA MQA latent (``Dk = kv_lora_rank+rope``, one shared K/V
head). Their query/key tensors are DIFFERENT objects with different dims -- only
the document-relative block coordinates are shared. This file therefore never
assumes the scorer ``q`` equals the gather ``q``; it feeds INDEPENDENT absorbed
tensors into the gather and only reuses the scorer-produced ``indices``.

What each check proves (all strict-fp32-referenced or exact):

* **TopK document-relative validity** -- every selected block id is either ``-1``
  padding or a real block whose start column ``bos + blk*block_B`` lies before
  ``eos`` (holds at least one causal, same-document key), and ``blk < nb``.
* **Repeatability** -- re-running scorer+selection on identical inputs yields
  bit-identical indices; the saturation case also bounds TileLang atomic-gradient
  drift across two identical gather forward/backward executions.
* **TopK correctness** -- indices agree (set-wise) with an independent eager fp32
  scatter scorer over the same relative coordinates.
* **Gather precision** -- forward + random-cotangent backward of the TileLang
  gather (fed the bridge indices) match the fp32 ``ref_block_sparse_mqa``
  autograd ground truth on cosine AND magnitude-sensitive rel-L2.

Corner coverage: exact online dims (scorer H64/D256, gather H64/Dk512/Dv448),
packed very-short docs incl. length-1 and BLOCK_B-unaligned, natural ``-1``
padding (topk > available blocks) and explicit all-``-1`` gather rows, sink
extremes (zero / large-positive / ``-1e30`` disabled), and large-logit
saturation.

Requires FA4 (SM10.x/Blackwell) for the scorer and a CUDA GPU for TileLang;
skips gracefully otherwise. No production code is touched.
"""

import os
import sys
import unittest

import numpy as np
import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

_TESTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)
from _hysparse_metrics import assert_close, compute_metrics

_NEG_INF = float("-inf")
_SKIP_REASON = None


def _ft_unavailable_reason():
    if not paddle.device.is_compiled_with_cuda():
        return "CUDA build of Paddle required"
    if paddle.device.cuda.device_count() == 0:
        return "no CUDA device available"
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        return (
            f"FA4 block scorer requires SM 10.x (Blackwell); got SM {major}.x"
        )
    try:
        from paddleformers.fleet.tilelang_ops.hysparse import (  # noqa: F401
            block_score_fa4_attn_fwd,
            select_topk_blocks,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (  # noqa: F401
            block_sparse_mqa_attention_tl,
        )
    except (ImportError, RuntimeError) as exc:
        return f"hysparse FT bridge import failed: {exc}"
    return None


def _skip_if_no_ft(tc):
    global _SKIP_REASON
    if _SKIP_REASON is None:
        _SKIP_REASON = _ft_unavailable_reason() or ""
    if _SKIP_REASON:
        tc.skipTest(_SKIP_REASON)


def _num_blocks(seqlen_k, block_B):
    return (seqlen_k + block_B - 1) // block_B


def _eager_scorer_block_logit_lse(q, k, valid_range, sm_scale, block_B):
    """Document-relative fp32 scatter reference for the FA4 MHA block scorer.

    q, k: [B, S, H, D] decompressed-MHA (independent heads). Block ``j`` of row
    ``r`` covers relative key columns ``[bos_r + j*block_B, bos_r+(j+1)*block_B)``
    restricted to causally-valid same-document keys (``bos_r <= c < eos_r``).
    Returns (block_logit [B,H,S,nb] scaled per-block max logit, lse [B,S,H]) --
    the independent model whose TopK the bridge indices are checked against.
    """
    b, s, h, d = q.shape
    sk = k.shape[1]
    nb = _num_blocks(sk, block_B)
    qf = q.astype("float32").transpose([0, 2, 1, 3])  # [B,H,S,D]
    kf = k.astype("float32").transpose([0, 2, 1, 3])  # [B,H,Sk,D]
    logits = paddle.matmul(qf, kf, transpose_y=True) * sm_scale  # [B,H,S,Sk]

    bos = valid_range[0, :, 0].astype("int64").reshape([s, 1])  # [S,1]
    eos = valid_range[0, :, 1].astype("int64").reshape([s, 1])  # [S,1]
    cols = paddle.arange(sk, dtype="int64").reshape([1, sk])
    valid = (cols >= bos) & (cols < eos)  # [S, Sk]
    rel_block = (cols - bos) // block_B  # [S, Sk] (meaningful where valid)

    neg = paddle.full_like(logits, _NEG_INF)
    vmask = valid.reshape([1, 1, s, sk])
    masked_all = paddle.where(vmask, logits, neg)
    lse_bhs = paddle.logsumexp(masked_all, axis=-1)  # [B,H,S]
    lse = lse_bhs.transpose([0, 2, 1]).contiguous()  # [B,S,H]

    block_logit = paddle.full([b, h, s, nb], _NEG_INF, dtype="float32")
    for j in range(nb):
        cm = (valid & (rel_block == j)).reshape([1, 1, s, sk])
        mj = paddle.where(cm, logits, neg)
        block_logit[:, :, :, j] = mj.max(axis=-1)
    return block_logit, lse


def _topk_set_mismatch(idx_a, idx_b):
    """Per-query set mismatch count between two [B,S,topk] index tensors."""
    a = paddle.sort(idx_a.astype("int64"), axis=-1)
    b = paddle.sort(idx_b.astype("int64"), axis=-1)
    return int((a != b).astype("int32").sum().item()), int(a.numel().item())


class TestHySparseFTBridgeCorner(unittest.TestCase):
    """FA4 scorer -> select_topk_blocks -> TileLang gather corner tests."""

    BLOCK_B = 64

    # ---- stage 1: the bridge (scorer -> indices) --------------------------
    def _scorer_indices(self, q_s, k_s, v_s, valid_range, topk, sm_scale):
        """Run the FA4 fused scorer and return (indices, block_logit, lse)."""
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
            select_topk_blocks,
        )

        _, lse, block_logit = block_score_fa4_attn_fwd(
            q_s,
            k_s,
            v_s,
            valid_range=valid_range,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            causal=True,
        )
        idx = select_topk_blocks(
            block_logit, lse, valid_range, topk, self.BLOCK_B
        )
        return idx, block_logit, lse

    def _assert_indices_valid(self, tag, idx, valid_range, s_kv):
        """Every id is -1 padding or a document-relative block holding a valid
        key: 0 <= blk < nb AND bos + blk*BLOCK_B < eos. No duplicate valid id in
        a query row (shared selection must not gather the same block twice)."""
        nb = _num_blocks(s_kv, self.BLOCK_B)
        idx_np = idx.numpy()
        vr = valid_range.numpy()
        b, s, topk = idx_np.shape
        for bi in range(b):
            for i in range(s):
                bos, eos = int(vr[bi, i, 0]), int(vr[bi, i, 1])
                row = idx_np[bi, i]
                valid_ids = [int(x) for x in row if x >= 0]
                self.assertEqual(
                    len(valid_ids),
                    len(set(valid_ids)),
                    f"{tag}: duplicate block id in row {i}: {row.tolist()}",
                )
                for blk in valid_ids:
                    self.assertLess(
                        blk, nb, f"{tag}: block {blk} >= nb={nb} row {i}"
                    )
                    start = bos + blk * self.BLOCK_B
                    self.assertLess(
                        start,
                        eos,
                        f"{tag}: block {blk} start {start} >= eos {eos} row {i}",
                    )
                    self.assertLessEqual(
                        start,
                        i,
                        f"{tag}: block {blk} starts after query row {i}",
                    )

    # ---- stage 2: the gather (indices -> attention) -----------------------
    @staticmethod
    def _make_sink(mode, h, seed):
        """Per-head sink logit [H] fp32 for the requested extreme, or None."""
        if mode is None:
            return None
        if mode == "zero":
            return np.zeros([h], dtype="float32")
        if mode == "pos":  # large positive -> sink dominates the softmax denom
            return np.full([h], 5.0, dtype="float32")
        if mode == "neginf":  # disabled sink -> must match sinkless
            return np.full([h], -1e30, dtype="float32")
        if mode == "rand":
            rng = np.random.RandomState(seed + 5)
            return (rng.randn(h) * 0.5).astype("float32")
        raise ValueError(f"unknown sink mode {mode!r}")

    def _gather_check(
        self,
        tag,
        idx,
        valid_range,
        b,
        s,
        h,
        dk,
        dv,
        sink_mode,
        seed,
        qk_scale=0.5,
        check_repeatability=False,
    ):
        """Feed the bridge ``idx`` into the TileLang gather on INDEPENDENT
        absorbed-latent tensors and score fwd + random-cotangent bwd against the
        fp32 ``ref_block_sparse_mqa`` autograd ground truth."""
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
            block_sparse_mqa_attention_tl,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
            ref_block_sparse_mqa,
        )

        rng = np.random.RandomState(seed)
        q_np = (rng.randn(b, s, h, dk) * qk_scale).astype("float32")
        k_np = (rng.randn(b, s, dk) * qk_scale).astype("float32")
        sm = dk**-0.5
        sink_np = self._make_sink(sink_mode, h, seed)

        # ---- fp32 autograd reference (uses the SAME bridge indices) ----
        qr = paddle.to_tensor(q_np, dtype="float32", stop_gradient=False)
        kr = paddle.to_tensor(k_np, dtype="float32", stop_gradient=False)
        sink_r = (
            paddle.to_tensor(sink_np, dtype="float32", stop_gradient=False)
            if sink_np is not None
            else None
        )
        out_r = ref_block_sparse_mqa(
            qr,
            kr,
            idx,
            valid_range,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=dv,
            attn_sink=sink_r,
        )
        g_np = (rng.randn(*out_r.shape) * 0.5).astype("float32")  # random dO
        out_r.backward(paddle.to_tensor(g_np, dtype="float32"))
        dq_r, dkv_r = qr.grad, kr.grad
        ds_r = sink_r.grad if sink_r is not None else None

        # ---- TileLang kernel (bf16 autograd) ----
        qk = paddle.to_tensor(q_np, dtype="bfloat16", stop_gradient=False)
        kk = paddle.to_tensor(k_np, dtype="bfloat16", stop_gradient=False)
        sink_k = (
            paddle.to_tensor(sink_np, dtype="float32", stop_gradient=False)
            if sink_np is not None
            else None
        )
        out_k, second = block_sparse_mqa_attention_tl(
            qk,
            kk,
            idx,
            valid_range,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=dv,
            attn_sink=sink_k,
        )
        self.assertIsNone(second)
        out_k.backward(paddle.to_tensor(g_np, dtype="bfloat16"))
        dq_k, dkv_k = qk.grad, kk.grad
        ds_k = sink_k.grad if sink_k is not None else None

        self.assertTrue(bool(paddle.isfinite(out_k.astype("float32")).all()))
        assert_close(
            self, f"{tag}:out", out_k, out_r, min_cos=0.99, max_rel_l2=6e-2
        )
        assert_close(
            self, f"{tag}:dq", dq_k, dq_r, min_cos=0.99, max_rel_l2=7e-2
        )
        assert_close(
            self, f"{tag}:dkv", dkv_k, dkv_r, min_cos=0.99, max_rel_l2=1.3e-1
        )
        if sink_np is not None:
            self.assertIsNotNone(ds_k)
            if sink_mode == "neginf":
                # Disabled sink: both gradients are exactly the zero vector;
                # cosine is undefined for two zeros, so compare values directly.
                self.assertTrue(bool(paddle.equal_all(ds_k, ds_r)))
            else:
                assert_close(
                    self,
                    f"{tag}:dsink",
                    ds_k,
                    ds_r,
                    min_cos=0.99,
                    max_rel_l2=8e-2,
                )

        if check_repeatability:
            qk2 = paddle.to_tensor(q_np, dtype="bfloat16", stop_gradient=False)
            kk2 = paddle.to_tensor(k_np, dtype="bfloat16", stop_gradient=False)
            sink_k2 = (
                paddle.to_tensor(sink_np, dtype="float32", stop_gradient=False)
                if sink_np is not None
                else None
            )
            out_k2, _ = block_sparse_mqa_attention_tl(
                qk2,
                kk2,
                idx,
                valid_range,
                sm_scale=sm,
                block_B=self.BLOCK_B,
                kv_lora_rank=dv,
                attn_sink=sink_k2,
            )
            out_k2.backward(paddle.to_tensor(g_np, dtype="bfloat16"))
            self.assertTrue(
                np.array_equal(
                    out_k.astype("float32").numpy(),
                    out_k2.astype("float32").numpy(),
                )
            )
            for name, got, repeat, max_drift in (
                ("dq", dq_k, qk2.grad, 1e-5),
                ("dkv", dkv_k, kk2.grad, 5e-5),
            ):
                drift = compute_metrics(repeat, got)
                print(f"[{tag}:repeat_{name}] rel_l2={drift.rel_l2:.3e}")
                self.assertLessEqual(drift.rel_l2, max_drift)
            if sink_k is not None:
                drift = compute_metrics(sink_k2.grad, ds_k)
                print(f"[{tag}:repeat_dsink] rel_l2={drift.rel_l2:.3e}")
                self.assertLessEqual(drift.rel_l2, 1e-5)
        return out_k

    # ---- full FT bridge orchestration -------------------------------------
    def _run_bridge(
        self, tag, s, doc_lens, topk, sink_mode, seed, scorer_qk_scale=1.0
    ):
        """End-to-end bridge: scorer (H64/D256) -> TopK -> gather (Dk512/Dv448).

        Validates index validity + exact repeatability + eager-ref correctness,
        then the gather forward/backward precision on the bridge indices.
        """
        from paddleformers.fleet.tilelang_ops.hysparse import select_topk_blocks
        from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
            make_causal_valid_range,
        )

        h, d_score = 64, 256  # exact online full-attn scorer head
        vr = make_causal_valid_range(s, batch=1, doc_lengths=doc_lens)
        if doc_lens is not None:
            self.assertGreater(int(vr[0, :, 0].max()), 0)  # real packed bos>0
        sm_score = d_score**-0.5

        paddle.seed(seed)
        q_s = (
            (paddle.randn([1, s, h, d_score]) * scorer_qk_scale)
            .astype("bfloat16")
            .contiguous()
        )
        k_s = (
            (paddle.randn([1, s, h, d_score]) * scorer_qk_scale)
            .astype("bfloat16")
            .contiguous()
        )
        v_s = paddle.randn([1, s, h, d_score]).astype("bfloat16").contiguous()

        idx, bl, lse = self._scorer_indices(q_s, k_s, v_s, vr, topk, sm_score)

        # block_logit / lse must never carry NaN or +inf (only -inf = masked).
        self.assertFalse(bool(paddle.isnan(bl).any()))
        self.assertFalse(bool((bl == float("inf")).any()))
        self.assertTrue(bool(paddle.isfinite(lse).all()))

        # exact repeatability of the scorer -> selection on identical inputs.
        idx2, bl2, _ = self._scorer_indices(q_s, k_s, v_s, vr, topk, sm_score)
        self.assertTrue(
            bool(paddle.equal_all(idx, idx2)),
            f"{tag}: TopK indices not bit-repeatable across runs",
        )
        self.assertTrue(
            bool(paddle.equal_all(bl, bl2)),
            f"{tag}: block_logit not bit-repeatable across runs",
        )

        # document-relative validity of every selected id.
        self._assert_indices_valid(tag, idx, vr, s)

        # correctness: agree (set-wise) with the independent eager fp32 scorer.
        bl_ref, lse_ref = _eager_scorer_block_logit_lse(
            q_s, k_s, vr, sm_score, self.BLOCK_B
        )
        idx_ref = select_topk_blocks(bl_ref, lse_ref, vr, topk, self.BLOCK_B)
        mism, total = _topk_set_mismatch(idx, idx_ref)
        self.assertLessEqual(
            mism,
            max(1, int(0.01 * total)),
            f"{tag}: bridge TopK differs from eager fp32 ref {mism}/{total}",
        )

        # gather precision on the bridge indices (independent absorbed latent).
        out_k = self._gather_check(
            tag,
            idx,
            vr,
            1,
            s,
            h,
            512,
            448,
            sink_mode,
            seed,
            check_repeatability=scorer_qk_scale > 1.0,
        )
        print(
            f"\n[{tag}] docs={doc_lens} topk={topk} sink={sink_mode} "
            f"scorer_scale={scorer_qk_scale} TopK_ref_mism={mism}/{total}"
        )
        return idx, out_k

    # ---- test cases -------------------------------------------------------
    def test_bridge_single_doc_sinkless(self):
        _skip_if_no_ft(self)
        self._run_bridge(
            "single_sinkless",
            s=256,
            doc_lens=None,
            topk=6,
            sink_mode=None,
            seed=101,
        )

    def test_bridge_packed_shortdocs_len1_unaligned(self):
        # Packed very-short docs including a length-1 doc and BLOCK_B-unaligned
        # lengths. topk=8 > blocks available in the short docs -> natural -1
        # padding slots flow across the bridge into the gather.
        _skip_if_no_ft(self)
        self._run_bridge(
            "packed_len1",
            s=256,
            doc_lens=[1, 37, 5, 90, 1, 122],
            topk=8,
            sink_mode="rand",
            seed=102,
        )

    def test_bridge_sink_zero(self):
        _skip_if_no_ft(self)
        self._run_bridge(
            "sink_zero",
            s=256,
            doc_lens=[88, 168],
            topk=6,
            sink_mode="zero",
            seed=103,
        )

    def test_bridge_sink_large_positive(self):
        # Large positive sink dominates the softmax denominator -> near-zero
        # gather output; kernel and fp32 ref must still agree.
        _skip_if_no_ft(self)
        self._run_bridge(
            "sink_pos",
            s=256,
            doc_lens=[128, 128],
            topk=6,
            sink_mode="pos",
            seed=104,
        )

    def test_bridge_sink_neg_inf(self):
        # -1e30 sink == disabled; gather must reproduce the sinkless result and
        # emit an exactly-zero sink gradient.
        _skip_if_no_ft(self)
        self._run_bridge(
            "sink_neginf",
            s=256,
            doc_lens=[96, 160],
            topk=6,
            sink_mode="neginf",
            seed=105,
        )

    def test_bridge_large_logit_saturation(self):
        # Scale scorer q/k by 8x so its logits saturate the softmax (near one-hot):
        # block_logit must stay finite and TopK stay valid + repeatable. The
        # downstream gather then validates those selected indices at normal scale.
        _skip_if_no_ft(self)
        self._run_bridge(
            "saturation",
            s=256,
            doc_lens=[100, 156],
            topk=6,
            sink_mode="rand",
            seed=106,
            scorer_qk_scale=8.0,
        )

    def test_gather_all_neg_one_rows(self):
        # Explicit all--1 gather rows (no block selected): the op must emit an
        # exactly-zero output for those query rows (no key + finite sink -> 0),
        # matching the fp32 reference, without NaN.
        _skip_if_no_ft(self)
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
            block_sparse_mqa_attention_tl,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
            build_random_block_indices,
            make_causal_valid_range,
        )

        b, s, h, dk, dv, topk = 1, 192, 64, 512, 448, 6
        vr = make_causal_valid_range(s, batch=b, doc_lengths=[64, 128])
        idx = build_random_block_indices(vr, topk, self.BLOCK_B, s, seed=207)
        # Force the first 8 query rows of each doc to select NOTHING (all -1).
        idx_np = idx.numpy()
        idx_np[0, 0:8, :] = -1
        idx_np[0, 64:72, :] = -1
        idx = paddle.to_tensor(idx_np, dtype="int32")
        self._assert_indices_valid("all_neg1", idx, vr, s)

        rng = np.random.RandomState(207)
        q_np = (rng.randn(b, s, h, dk) * 0.5).astype("float32")
        k_np = (rng.randn(b, s, dk) * 0.5).astype("float32")
        sink_np = (rng.randn(h) * 0.5).astype("float32")
        sm = dk**-0.5

        from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
            ref_block_sparse_mqa,
        )

        qr = paddle.to_tensor(q_np, dtype="float32", stop_gradient=False)
        kr = paddle.to_tensor(k_np, dtype="float32", stop_gradient=False)
        sink_r = paddle.to_tensor(sink_np, dtype="float32", stop_gradient=False)
        out_r = ref_block_sparse_mqa(
            qr,
            kr,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=dv,
            attn_sink=sink_r,
        )
        g_np = (rng.randn(*out_r.shape) * 0.5).astype("float32")
        out_r.backward(paddle.to_tensor(g_np, dtype="float32"))

        qk = paddle.to_tensor(q_np, dtype="bfloat16", stop_gradient=False)
        kk = paddle.to_tensor(k_np, dtype="bfloat16", stop_gradient=False)
        sink_k = paddle.to_tensor(sink_np, dtype="float32", stop_gradient=False)
        out_k, _ = block_sparse_mqa_attention_tl(
            qk,
            kk,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=dv,
            attn_sink=sink_k,
        )
        out_k.backward(paddle.to_tensor(g_np, dtype="bfloat16"))

        out_k32 = out_k.astype("float32").reshape([b, s, h, dv])
        self.assertTrue(bool(paddle.isfinite(out_k32).all()))
        # empty rows -> exactly zero (kernel and ref), no NaN leak.
        empty_rows = list(range(0, 8)) + list(range(64, 72))
        nonempty_rows = [i for i in range(s) if i not in set(empty_rows)]
        empty = out_k32.numpy()[0, empty_rows]
        self.assertEqual(float(np.abs(empty).max()), 0.0)
        empty_ref = out_r.reshape([b, s, h, dv]).numpy()[0, empty_rows]
        self.assertEqual(float(np.abs(empty_ref).max()), 0.0)
        # forward matches everywhere (ref forward is finite on empty rows).
        assert_close(
            self, "all_neg1:out", out_k, out_r, min_cos=0.99, max_rel_l2=6e-2
        )

        # The kernel emits an exactly-zero, finite dq for empty rows. The fp32
        # ``ref_block_sparse_mqa`` autograd, by contrast, yields NaN dq on a
        # fully-masked row (its ``reduce_max`` over an all -inf slice feeds an
        # inf-inf into the softmax backward) -- a reference limitation, NOT a
        # kernel defect. So verify the kernel's empty-row grad directly and
        # compare kernel-vs-ref dq only on the NON-empty rows.
        dq_k = qk.grad.astype("float32").numpy()
        dq_r = qr.grad.astype("float32").numpy()
        self.assertTrue(np.isfinite(dq_k[0, empty_rows]).all())
        self.assertEqual(float(np.abs(dq_k[0, empty_rows]).max()), 0.0)
        assert_close(
            self,
            "all_neg1:dq(nonempty)",
            dq_k[0, nonempty_rows],
            dq_r[0, nonempty_rows],
            min_cos=0.99,
            max_rel_l2=7e-2,
        )


if __name__ == "__main__":
    unittest.main()
