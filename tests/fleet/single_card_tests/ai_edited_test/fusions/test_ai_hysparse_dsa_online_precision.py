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

"""Independent numerical re-audit of the HySparse block-sparse DSA backend
(:func:`paddleformers.fleet.cudnn_ops.block_sparse_mqa_attention_dsa`).

This suite is deliberately SELF-CONTAINED: it does NOT reuse the repo's
``reference_sparse.ref_block_sparse_mqa`` or the shared ``_hysparse_metrics``
helper. The dense masked-softmax MQA reference, the packed valid_range / block
index generators, and the amplitude metrics are all re-derived here from
scratch, so a bug shared between the production op and the repo reference cannot
hide the defect from both.

Coverage that the existing ``test_hysparse_mqa_gather_dsa.py`` does NOT provide:

* **Online ernielite dims** ``Dk=512, Dv=448, H=64`` (kv_lora_rank=448 from
  ``ernielite_layer43_*`` configs). Dv=448 < 512 exercises the value-padding
  re-lay path (``pad_v = 512 - 448 = 64``, ``[value | zeros | rope]``) inside
  ``block_sparse_mqa_attention_dsa`` -- a code path the existing tests, which
  only run Dv=512 (pad_v=0), never touch.
* **Random cotangent** backward (``(out * g).sum().backward()`` with a random
  ``g``) instead of the all-ones cotangent from ``out.sum()``. This probes the
  full backward Jacobian rather than a single column-sum direction.
* **Amplitude-sensitive metrics**: max-abs, RMSE, relative-L2
  (``||got-ref||_2 / ||ref||_2``) and dtype-aware allclose, reported for every
  compared tensor -- not just cosine (which is scale-blind).
* **dSink** comparison against an autograd reference for the learnable sink.
* **sinkless / very-negative-sink** equivalence and a finite-negative sink.

Skips gracefully unless FlashMLA sparse fwd + cuDNN DSA bwd on an SM100+ GPU are
available. Does NOT modify the production wrapper.
"""

import math
import unittest

import numpy as np
import paddle

try:
    from paddleformers.fleet.cudnn_ops.block_sparse_mqa_dsa import (
        block_sparse_mqa_attention_dsa,
        is_dsa_available,
    )

    _HAS_DSA = is_dsa_available()
except (ImportError, RuntimeError, AttributeError):
    _HAS_DSA = False

BLOCK_B = 64  # SM100 TopK alignment == one DSA tile chunk.


# --------------------------------------------------------------------------- #
# Independent amplitude metrics (self-contained; no shared helper).           #
# --------------------------------------------------------------------------- #
def _flat_f64(x):
    if x is None:
        raise AssertionError("tensor is None (missing gradient?)")
    return (
        np.ascontiguousarray(x.astype("float64").numpy())
        .reshape(-1)
        .astype(np.float64)
    )


def _metrics(got, ref, eps=1e-30):
    """max-abs, RMSE, relative-L2, cosine over finite entries + allclose."""
    g = _flat_f64(got)
    r = _flat_f64(ref)
    assert g.shape == r.shape, f"shape {g.shape} vs {r.shape}"
    fin = np.isfinite(g) & np.isfinite(r)
    n_bad = int((~fin).sum())
    gf, rf = g[fin], r[fin]
    diff = gf - rf
    max_abs = float(np.abs(diff).max()) if gf.size else 0.0
    rmse = float(np.sqrt(np.mean(diff * diff))) if gf.size else 0.0
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(rf) + eps))
    denom = float(np.linalg.norm(gf) * np.linalg.norm(rf)) + eps
    cos = float(np.dot(gf, rf) / denom) if gf.size else 1.0
    # bf16-grade dtype-aware allclose on the finite entries.
    allclose = bool(np.allclose(gf, rf, rtol=1e-2, atol=2e-2) and n_bad == 0)
    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "rel_l2": rel_l2,
        "cos": cos,
        "allclose": allclose,
        "n_nonfinite": n_bad,
    }


def _report(name, m):
    print(
        f"[{name}] max_abs={m['max_abs']:.4e} rmse={m['rmse']:.4e} "
        f"rel_l2={m['rel_l2']:.4e} cos={m['cos']:.6f} "
        f"allclose={m['allclose']} nonfinite={m['n_nonfinite']}",
        flush=True,
    )
    return m


# --------------------------------------------------------------------------- #
# Independent test-data builders (self-contained).                            #
# --------------------------------------------------------------------------- #
def _causal_valid_range(doc_lens):
    """[1, S, 2] int32 [bos, eos) for a packed causal multi-doc sequence."""
    bos, eos = [], []
    cur = 0
    for dl in doc_lens:
        for t in range(dl):
            bos.append(cur)
            eos.append(cur + t + 1)  # causal: attend up to & including self
        cur += dl
    vr = paddle.to_tensor(np.stack([bos, eos], axis=-1), dtype="int32")
    return vr.unsqueeze(0).contiguous()  # [1, S, 2]


def _random_block_indices(valid_range, topk, block_B, seed):
    """[1, S, topk] int32 doc-relative block ids (-1 padding).

    For each query, the doc-relative blocks whose absolute start
    ``bos + j*block_B`` falls before ``eos`` are candidates; keep a random topk
    (padding with -1 when fewer valid blocks than topk exist).
    """
    rng = np.random.RandomState(seed)
    vr = valid_range.numpy()[0]  # [S, 2]
    s = vr.shape[0]
    n_blk = (int(vr[:, 1].max()) + block_B - 1) // block_B
    out = np.full([s, topk], -1, dtype=np.int32)
    for i in range(s):
        bos, eos = int(vr[i, 0]), int(vr[i, 1])
        valid = [j for j in range(n_blk) if bos + j * block_B < eos]
        if not valid:
            continue
        scores = rng.rand(len(valid))
        order = np.argsort(-scores)
        pick = [valid[o] for o in order[:topk]]
        out[i, : len(pick)] = pick
    return paddle.to_tensor(out, dtype="int32").unsqueeze(0).contiguous()


# --------------------------------------------------------------------------- #
# Independent dense masked-softmax MQA reference (fp64, fully differentiable). #
# --------------------------------------------------------------------------- #
def _ref_attn(q_f, k_f, idx, vr, sm_scale, block_B, dv, sink=None):
    """Dense masked-softmax MQA over the exact selected-block column set.

    q_f [B,S,H,Dk] fp64, k_f [B,S_kv,Dk] fp64. Key = full Dk; value = leading
    ``dv``. A key column ``c`` is attended by query ``i`` iff ``bos<=c<eos`` and
    ``(c-bos)//block_B`` is one of query i's (valid, >=0) selected block ids. An
    optional per-head ``sink`` adds a virtual (value-less) softmax column.
    """
    b, s, h, dk = q_f.shape
    s_kv = k_f.shape[1]
    col = paddle.arange(s_kv, dtype="int64").reshape([1, 1, s_kv])
    bos = vr[..., 0:1].astype("int64")  # [B,S,1]
    eos = vr[..., 1:2].astype("int64")  # [B,S,1]
    rel = col - bos  # [B,S,S_kv]
    col_block = paddle.where(
        rel >= 0, rel // block_B, paddle.full_like(rel, -1)
    )
    idx64 = idx.astype("int64").unsqueeze(-2)  # [B,S,1,topk]
    hit = (col_block.unsqueeze(-1) == idx64) & (idx64 >= 0)  # [B,S,S_kv,topk]
    sel = hit.any(axis=-1)  # [B,S,S_kv]
    in_range = (col >= bos) & (col < eos)  # [B,S,S_kv]
    allow = sel & in_range  # [B,S,S_kv]

    logits = paddle.einsum("bshd,bkd->bshk", q_f, k_f) * sm_scale
    allow_e = allow.unsqueeze(2)  # [B,S,1,S_kv]
    neg = paddle.full_like(logits, float("-inf"))
    masked = paddle.where(allow_e, logits, neg)
    m = masked.max(axis=-1, keepdim=True)  # [B,S,H,1]
    if sink is not None:
        m = paddle.maximum(m, sink.reshape([1, 1, h, 1]).astype("float64"))
    m = paddle.where(paddle.isfinite(m), m, paddle.zeros_like(m))
    w = paddle.where(allow_e, paddle.exp(logits - m), paddle.zeros_like(logits))
    denom = w.sum(axis=-1, keepdim=True)  # [B,S,H,1]
    if sink is not None:
        denom = denom + paddle.exp(
            sink.reshape([1, 1, h, 1]).astype("float64") - m
        )
    denom = paddle.where(denom > 0, denom, paddle.ones_like(denom))
    p = w / denom
    v = k_f[..., :dv].unsqueeze(1)  # [B,1,S_kv,dv]
    out = paddle.einsum("bshk,bnkc->bshc", p, v)  # [B,S,H,dv]
    row_has = allow.any(axis=-1).astype("float64")  # [B,S]
    out = out * row_has.unsqueeze(-1).unsqueeze(-1)
    return out.reshape([b, s, h * dv])


# --------------------------------------------------------------------------- #
# Runners.                                                                     #
# --------------------------------------------------------------------------- #
def _run_dsa(q_bf16, k_bf16, idx, vr, sm, dv, cotangent, sink=None):
    qd = q_bf16.detach().clone()
    kd = k_bf16.detach().clone()
    qd.stop_gradient = False
    kd.stop_gradient = False
    sink_d = None
    if sink is not None:
        sink_d = sink.detach().clone()
        sink_d.stop_gradient = False
    out, _ = block_sparse_mqa_attention_dsa(
        qd,
        kd,
        idx,
        vr,
        sm_scale=sm,
        block_B=BLOCK_B,
        kv_lora_rank=dv,
        attn_sink=sink_d,
    )
    (out.astype("float32") * cotangent).sum().backward()
    return (
        out,
        qd.grad,
        kd.grad,
        (sink_d.grad if sink_d is not None else None),
    )


def _run_ref(q_bf16, k_bf16, idx, vr, sm, dv, cotangent, sink=None):
    # Reference sees the EXACT bf16 input values (upcast to fp64), isolating
    # kernel error from input quantization.
    qr = q_bf16.detach().astype("float64")
    kr = k_bf16.detach().astype("float64")
    qr.stop_gradient = False
    kr.stop_gradient = False
    sink_r = None
    if sink is not None:
        sink_r = sink.detach().astype("float64")
        sink_r.stop_gradient = False
    out = _ref_attn(qr, kr, idx, vr, sm, BLOCK_B, dv, sink=sink_r)
    (out * cotangent.astype("float64")).sum().backward()
    return (
        out,
        qr.grad,
        kr.grad,
        (sink_r.grad if sink_r is not None else None),
    )


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "DSA precision audit requires CUDA"
)
@unittest.skipUnless(
    _HAS_DSA, "requires FlashMLA sparse fwd + cuDNN DSA backward (SM100+)"
)
class TestDsaOnlinePrecision(unittest.TestCase):
    # Online ernielite (kv_lora_rank=448) absorbed-MQA DSA dims.
    DK = 512  # per-head query/key score dim = kv_lora_rank(448) + rope(64)
    DV = 448  # value dim (leading kv_lora_rank slice); pad_v = 512-448 = 64
    H = 64
    DOC_LENS = [40, 88, 133, 27]  # packed, all unaligned to BLOCK_B (sum 288)

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_device("gpu:0")
        except Exception as exc:
            raise unittest.SkipTest(f"gpu:0 unavailable: {exc}")

    def _make(self, dk, dv, h, topk, seed):
        s = sum(self.DOC_LENS)
        paddle.seed(seed)
        q = paddle.randn([1, s, h, dk]).cast("bfloat16")
        kf = paddle.randn([1, s, dk]).cast("bfloat16")
        vr = _causal_valid_range(self.DOC_LENS)
        idx = _random_block_indices(vr, topk, BLOCK_B, seed=seed + 1)
        sm = 1.0 / math.sqrt(dk)
        g = paddle.randn([1, s, h * dv]).cast("float32")  # random cotangent
        return q, kf, vr, idx, sm, g

    def test_online_dims_sinkless_random_cotangent(self):
        """Dv=448 value-pad path, packed multi-doc, sinkless, random
        cotangent: forward out + dQ + dKV must match the fp64 reference."""
        q, kf, vr, idx, sm, g = self._make(
            self.DK, self.DV, self.H, topk=4, seed=7
        )
        self.assertGreater(int(vr[0, :, 0].max()), 0)  # packed (bos>0) present
        out_d, dq_d, dkv_d, _ = _run_dsa(q, kf, idx, vr, sm, self.DV, g)
        out_r, dq_r, dkv_r, _ = _run_ref(q, kf, idx, vr, sm, self.DV, g)
        mo = _report("online_sinkless_out", _metrics(out_d, out_r))
        mq = _report("online_sinkless_dQ", _metrics(dq_d, dq_r))
        mk = _report("online_sinkless_dKV", _metrics(dkv_d, dkv_r))
        self.assertEqual(
            list(out_d.shape), [1, sum(self.DOC_LENS), self.H * self.DV]
        )
        for tag, m in (("out", mo), ("dQ", mq), ("dKV", mk)):
            self.assertEqual(m["n_nonfinite"], 0, f"{tag} non-finite")
            self.assertGreater(m["cos"], 0.99, f"{tag} cos {m['cos']}")
            self.assertLess(m["rel_l2"], 5e-2, f"{tag} rel_l2 {m['rel_l2']}")

    def test_online_dims_finite_sink_random_cotangent(self):
        """Dv=448 value-pad path, packed multi-doc, FINITE per-head learnable
        sink, random cotangent: compares out / dQ / dKV / dSink.

        REGRESSION GATE for the previously-confirmed finite-sink dQ defect: on
        PACKED (nonzero-bos) sequences the DSA finite-sink backward dQ used to
        diverge from the dense reference (cos ~0.976) because the cuDNN DSA
        ``d_qk != d_v`` backward consumed a KV-only LSE that omitted the sink
        from the softmax denominator. The production fix (block_sparse_mqa_dsa
        backward: pass a sink-inclusive ``logaddexp(lse, sink)`` and neutralize
        the sink to -1e30 on the Dk!=Dv finite-sink path) makes dQ match. This
        strict dQ assertion locks that fix in; independent verification here
        gives cos ~0.999995, rel_l2 ~3e-3. Forward / dKV / dSink also match.
        """
        q, kf, vr, idx, sm, g = self._make(
            self.DK, self.DV, self.H, topk=4, seed=21
        )
        paddle.seed(29)
        sink = paddle.randn([self.H], dtype="float32") * 0.5
        out_d, dq_d, dkv_d, ds_d = _run_dsa(
            q, kf, idx, vr, sm, self.DV, g, sink=sink
        )
        out_r, dq_r, dkv_r, ds_r = _run_ref(
            q, kf, idx, vr, sm, self.DV, g, sink=sink
        )
        mo = _report("online_sink_out", _metrics(out_d, out_r))
        mk = _report("online_sink_dKV", _metrics(dkv_d, dkv_r))
        ms = _report("online_sink_dSink", _metrics(ds_d, ds_r))
        mq = _report("online_sink_dQ", _metrics(dq_d, dq_r))  # gate

        self.assertGreater(mo["cos"], 0.99, "forward out must match")
        self.assertLess(mo["rel_l2"], 5e-2)
        self.assertGreater(mk["cos"], 0.99, "dKV must match")
        self.assertIsNotNone(ds_d)
        self.assertGreater(ms["cos"], 0.99, "dSink must match")
        # STRICT dQ gate for the known finite-sink packed defect.
        self.assertGreater(
            mq["cos"],
            0.99,
            f"finite-sink packed dQ diverges (cos={mq['cos']:.6f}, "
            f"max_abs={mq['max_abs']:.4e}, rmse={mq['rmse']:.4e}, "
            f"rel_l2={mq['rel_l2']:.4e}) -- known DSA backend defect (task #1)",
        )

    def test_no_pad_dims_finite_sink_control(self):
        """Control at Dv=512 (pad_v=0), packed, finite sink, random cotangent.
        Confirms the finite-sink dQ fix holds for the native Dv=512 layout too
        (the divergence was tied to the Dk!=Dv backward LSE, not the value-pad
        re-lay), so both the padded (Dv=448) and native (Dv=512) paths match."""
        q, kf, vr, idx, sm, g = self._make(576, 512, self.H, topk=4, seed=41)
        paddle.seed(43)
        sink = paddle.randn([self.H], dtype="float32") * 0.5
        out_d, dq_d, dkv_d, ds_d = _run_dsa(q, kf, idx, vr, sm, 512, g, sink)
        out_r, dq_r, dkv_r, ds_r = _run_ref(q, kf, idx, vr, sm, 512, g, sink)
        _report("nopad_sink_out", _metrics(out_d, out_r))
        _report("nopad_sink_dKV", _metrics(dkv_d, dkv_r))
        _report("nopad_sink_dSink", _metrics(ds_d, ds_r))
        mq = _report("nopad_sink_dQ", _metrics(dq_d, dq_r))
        # Same strict gate for the native layout (fix must hold here too).
        self.assertGreater(
            mq["cos"],
            0.99,
            f"nopad finite-sink dQ diverges (cos={mq['cos']:.6f})",
        )

    def test_sinkless_equals_very_negative_sink(self):
        """attn_sink=None (production sinkless -> internal -1e30) must equal an
        explicit very-negative per-head sink in BOTH forward and gradients."""
        q, kf, vr, idx, sm, g = self._make(
            self.DK, self.DV, self.H, topk=4, seed=55
        )
        neg = paddle.full([self.H], -1e30, dtype="float32")
        out_none, dq_none, dkv_none, _ = _run_dsa(
            q, kf, idx, vr, sm, self.DV, g, sink=None
        )
        out_neg, dq_neg, dkv_neg, ds_neg = _run_dsa(
            q, kf, idx, vr, sm, self.DV, g, sink=neg
        )
        mo = _report("sinkless_vs_neg_out", _metrics(out_none, out_neg))
        mq = _report("sinkless_vs_neg_dQ", _metrics(dq_none, dq_neg))
        mk = _report("sinkless_vs_neg_dKV", _metrics(dkv_none, dkv_neg))
        self.assertGreater(mo["cos"], 0.999999)
        self.assertLess(mo["max_abs"], 1e-3)
        self.assertGreater(mq["cos"], 0.999999)
        self.assertGreater(mk["cos"], 0.999999)
        # The -1e30 sink mass underflows: its dSink must be ~0.
        self.assertIsNotNone(ds_neg)
        self.assertLess(float(paddle.abs(ds_neg).max()), 1e-3)

    def test_finite_negative_sink_matches_reference(self):
        """A finite NEGATIVE sink (weaker than sinkless) still matches the
        dense reference in forward and dKV."""
        q, kf, vr, idx, sm, g = self._make(
            self.DK, self.DV, self.H, topk=4, seed=67
        )
        sink = paddle.full([self.H], -3.0, dtype="float32")
        out_d, _, dkv_d, ds_d = _run_dsa(
            q, kf, idx, vr, sm, self.DV, g, sink=sink
        )
        out_r, _, dkv_r, ds_r = _run_ref(
            q, kf, idx, vr, sm, self.DV, g, sink=sink
        )
        mo = _report("negsink_out", _metrics(out_d, out_r))
        mk = _report("negsink_dKV", _metrics(dkv_d, dkv_r))
        ms = _report("negsink_dSink", _metrics(ds_d, ds_r))
        self.assertGreater(mo["cos"], 0.99)
        self.assertLess(mo["rel_l2"], 5e-2)
        self.assertGreater(mk["cos"], 0.99)
        self.assertGreater(ms["cos"], 0.99)


if __name__ == "__main__":
    unittest.main()
