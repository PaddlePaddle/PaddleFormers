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

"""Packed multi-document backward coverage for the HySparse block-sparse MQA
TileLang oracle (:func:`block_sparse_mqa_attention_tl`).

The forward/consistency of the TileLang oracle is covered elsewhere; here we
exercise its *backward* on a packed [40, 88, 133, 27] document sequence (all
document lengths unaligned to ``block_B``, so downstream rows carry a NONZERO
``bos``). Gradients dQ / dKV / d_sink are validated against the shared naive
``ref_block_sparse_mqa`` (散算子) over the exact same document-relative block
coordinates, for both the sinkless and a finite learnable-sink softmax. A final
packed-vs-solo check proves the query gradient of a document's rows is identical
whether the document runs alone (``bos=0``) or packed behind a prefix.

Beyond the all-ones ``out.sum()`` cotangent, ``test_packed_backward_random_
cotangent`` drives the backward with a RANDOM ``dOut`` (uniform cotangents can
mask per-element sign/scale cancellations), and ``test_online_dims_h64`` runs
the full DSV4-online sparse-gather scale (``H=64``, ``Dk=576``, ``Dv=512``),
forcing backward head-tiling at the large latent width. Every gradient
comparison enforces a magnitude-sensitive relative-Frobenius ceiling
(``max_rel_l2``) on top of the scale-blind cosine floor.

Runs on any TileLang-capable CUDA GPU; skips gracefully otherwise.
"""

import math
import os
import sys
import unittest

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

# Reusable precision metrics helper at the single_card_tests root.
_TESTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)
from _hysparse_metrics import assert_close

from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
    build_random_block_indices,
    make_causal_valid_range,
    ref_block_sparse_mqa,
)


def _tl_unavailable_reason():
    if not paddle.device.is_compiled_with_cuda():
        return "CUDA build of Paddle required"
    if paddle.device.cuda.device_count() == 0:
        return "no CUDA device available"
    try:
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
            block_sparse_mqa_attention_tl,  # noqa: F401
        )
    except (ImportError, RuntimeError):
        return "TileLang block-sparse MQA op import failed"
    return None


_SKIP_REASON = None


def _skip_if_no_tl(tc):
    global _SKIP_REASON
    if _SKIP_REASON is None:
        _SKIP_REASON = _tl_unavailable_reason() or ""
    if _SKIP_REASON:
        tc.skipTest(_SKIP_REASON)


class TestBlockSparseMQATLMultiDocGrad(unittest.TestCase):
    """Packed [40, 88, 133, 27] backward for the TileLang oracle."""

    BLOCK_B = 64
    Dk = 576
    Dv = 512
    DOC_LENS = [40, 88, 133, 27]  # sum = 288, all unaligned to BLOCK_B

    def _make(self, h, topk, doc_lens, seed=7):
        s = sum(doc_lens)
        paddle.seed(seed)
        q = paddle.randn([1, s, h, self.Dk]).cast("bfloat16")
        kf = paddle.randn([1, s, self.Dk]).cast("bfloat16")
        vr = make_causal_valid_range(s, batch=1, doc_lengths=doc_lens)
        idx = build_random_block_indices(
            vr, topk, self.BLOCK_B, s, seed=seed + 1
        )
        sm = 1.0 / math.sqrt(self.Dk)
        return q, kf, vr, idx, sm

    def _forward_tl(self, q, kf, idx, vr, sm, sink=None):
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
            block_sparse_mqa_attention_tl,
        )

        out, _ = block_sparse_mqa_attention_tl(
            q,
            kf,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=sink,
        )
        return out

    def _run_tl(self, q, kf, idx, vr, sm, sink=None, cotangent=None):
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
            block_sparse_mqa_attention_tl,
        )

        qd = q.detach().clone()
        kd = kf.detach().clone()
        qd.stop_gradient = False
        kd.stop_gradient = False
        sink_d = None
        if sink is not None:
            sink_d = sink.detach().clone()
            sink_d.stop_gradient = False
        out, _ = block_sparse_mqa_attention_tl(
            qd,
            kd,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=sink_d,
        )
        if cotangent is None:
            out.sum().backward()
        else:
            out.backward(cotangent)
        g_sink = sink_d.grad if sink_d is not None else None
        return out, qd.grad, kd.grad, g_sink

    def _run_ref(self, q, kf, idx, vr, sm, sink=None, cotangent=None):
        qr = q.detach().cast("float32")
        kr = kf.detach().cast("float32")
        qr.stop_gradient = False
        kr.stop_gradient = False
        sink_r = None
        if sink is not None:
            sink_r = sink.detach().clone()
            sink_r.stop_gradient = False
        out = ref_block_sparse_mqa(
            qr,
            kr,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=sink_r,
        )
        if cotangent is None:
            out.sum().backward()
        else:
            out.backward(cotangent.cast("float32"))
        g_sink = sink_r.grad if sink_r is not None else None
        return out, qr.grad, kr.grad, g_sink

    def test_packed_backward_sinkless(self):
        _skip_if_no_tl(self)
        h, topk = 4, 4
        q, kf, vr, idx, sm = self._make(h, topk, self.DOC_LENS, seed=7)
        self.assertGreater(int(vr[0, :, 0].max()), 0)  # nonzero bos present
        out_d, dq_d, dkv_d, _ = self._run_tl(q, kf, idx, vr, sm)
        out_r, dq_r, dkv_r, _ = self._run_ref(q, kf, idx, vr, sm)
        assert_close(
            self, "tl_packed_out", out_d, out_r, min_cos=0.99, max_rel_l2=2e-2
        )
        assert_close(
            self, "tl_packed_dq", dq_d, dq_r, min_cos=0.99, max_rel_l2=2e-2
        )
        assert_close(
            self, "tl_packed_dkv", dkv_d, dkv_r, min_cos=0.99, max_rel_l2=2e-2
        )

    def test_packed_backward_finite_sink(self):
        _skip_if_no_tl(self)
        h, topk = 4, 4
        q, kf, vr, idx, sm = self._make(h, topk, self.DOC_LENS, seed=21)
        paddle.seed(29)
        # Moderate sink magnitude keeps the softmax mass sane so the dq/dkv
        # comparison tests the sink math rather than bf16 rounding noise.
        sink = paddle.randn([h], dtype="float32") * 0.5
        out_d, dq_d, dkv_d, ds_d = self._run_tl(q, kf, idx, vr, sm, sink=sink)
        out_r, dq_r, dkv_r, ds_r = self._run_ref(q, kf, idx, vr, sm, sink=sink)
        assert_close(
            self,
            "tl_packed_sink_out",
            out_d,
            out_r,
            min_cos=0.99,
            max_rel_l2=2e-2,
        )
        assert_close(
            self, "tl_packed_sink_dq", dq_d, dq_r, min_cos=0.99, max_rel_l2=2e-2
        )
        assert_close(
            self,
            "tl_packed_sink_dkv",
            dkv_d,
            dkv_r,
            min_cos=0.99,
            max_rel_l2=2e-2,
        )
        self.assertIsNotNone(ds_d)
        assert_close(
            self,
            "tl_packed_sink_dsink",
            ds_d,
            ds_r,
            min_cos=0.99,
            max_rel_l2=2e-2,
        )

    def test_neg_sink_matches_sinkless(self):
        # A very-negative sink must reproduce the plain (sinkless) softmax
        # output: exp(sink - m) -> 0 leaves the denominator unchanged.
        _skip_if_no_tl(self)
        h, topk = 4, 4
        q, kf, vr, idx, sm = self._make(h, topk, self.DOC_LENS, seed=33)
        neg_sink = paddle.full([h], -1e30, dtype="float32")
        out_none = self._forward_tl(q, kf, idx, vr, sm)
        out_neg = self._forward_tl(q, kf, idx, vr, sm, sink=neg_sink)
        assert_close(
            self, "tl_negsink_vs_sinkless", out_neg, out_none, min_cos=0.999
        )

    def test_packed_backward_random_cotangent(self):
        # Backward driven by a RANDOM dOut instead of the all-ones out.sum()
        # cotangent: the uniform direction can mask per-element sign/scale
        # cancellations in dQ/dKV, a random cotangent exercises the true
        # per-position gradient. Runs both sinkless and a finite sink.
        _skip_if_no_tl(self)
        h, topk = 4, 4
        for tag, sink in (("nosink", None), ("sink", True)):
            q, kf, vr, idx, sm = self._make(h, topk, self.DOC_LENS, seed=51)
            s = sum(self.DOC_LENS)
            paddle.seed(53)
            cot = paddle.randn([1, s, h * self.Dv]).cast("bfloat16")
            sk = None
            if sink:
                sk = paddle.randn([h], dtype="float32") * 0.5
            out_d, dq_d, dkv_d, ds_d = self._run_tl(
                q, kf, idx, vr, sm, sink=sk, cotangent=cot
            )
            out_r, dq_r, dkv_r, ds_r = self._run_ref(
                q, kf, idx, vr, sm, sink=sk, cotangent=cot
            )
            assert_close(
                self,
                f"tl_randcot_{tag}_dq",
                dq_d,
                dq_r,
                min_cos=0.99,
                max_rel_l2=2e-2,
            )
            assert_close(
                self,
                f"tl_randcot_{tag}_dkv",
                dkv_d,
                dkv_r,
                min_cos=0.99,
                max_rel_l2=2e-2,
            )
            if sk is not None:
                assert_close(
                    self,
                    f"tl_randcot_{tag}_dsink",
                    ds_d,
                    ds_r,
                    min_cos=0.99,
                    max_rel_l2=2e-2,
                )

    def test_online_dims_h64(self):
        # DSV4-online sparse-gather scale: H=64 query heads over the absorbed
        # MLA latent (Dk=576=kv_lora_rank512+rope64, Dv=512). H=64 fills PH on
        # the fwd M dim and forces backward head-tiling (_fit_block_h) at the
        # large Dk=576 latent width. Packed ragged docs + random cotangent +
        # a finite sink -- the full online numeric surface in one shot.
        _skip_if_no_tl(self)
        h, topk = 64, 4
        q, kf, vr, idx, sm = self._make(h, topk, self.DOC_LENS, seed=61)
        s = sum(self.DOC_LENS)
        paddle.seed(63)
        cot = paddle.randn([1, s, h * self.Dv]).cast("bfloat16")
        sink = paddle.randn([h], dtype="float32") * 0.5
        out_d, dq_d, dkv_d, ds_d = self._run_tl(
            q, kf, idx, vr, sm, sink=sink, cotangent=cot
        )
        out_r, dq_r, dkv_r, ds_r = self._run_ref(
            q, kf, idx, vr, sm, sink=sink, cotangent=cot
        )
        assert_close(
            self, "tl_h64_out", out_d, out_r, min_cos=0.99, max_rel_l2=2e-2
        )
        assert_close(
            self, "tl_h64_dq", dq_d, dq_r, min_cos=0.99, max_rel_l2=2e-2
        )
        assert_close(
            self, "tl_h64_dkv", dkv_d, dkv_r, min_cos=0.99, max_rel_l2=2e-2
        )
        assert_close(
            self, "tl_h64_dsink", ds_d, ds_r, min_cos=0.99, max_rel_l2=2e-2
        )

    def test_packed_vs_solo_dq_equivalence(self):
        # A document run alone (bos=0) and the SAME document packed behind a
        # prefix must yield identical query gradients for the document's rows:
        # block ids are document-relative and eos is per-document causal, so the
        # gather sees the same key columns (shifted by the prefix).
        _skip_if_no_tl(self)
        h, topk = 4, 4
        prefix_len, doc_len = 91, 133  # both unaligned to BLOCK_B
        sm = 1.0 / math.sqrt(self.Dk)

        paddle.seed(101)
        q_doc = paddle.randn([1, doc_len, h, self.Dk]).cast("bfloat16")
        k_doc = paddle.randn([1, doc_len, self.Dk]).cast("bfloat16")
        q_pre = paddle.randn([1, prefix_len, h, self.Dk]).cast("bfloat16")
        k_pre = paddle.randn([1, prefix_len, self.Dk]).cast("bfloat16")

        vr_solo = make_causal_valid_range(doc_len, batch=1)
        idx_doc = build_random_block_indices(
            vr_solo, topk, self.BLOCK_B, doc_len, seed=202
        )
        _, dq_solo, _, _ = self._run_tl(q_doc, k_doc, idx_doc, vr_solo, sm)

        total = prefix_len + doc_len
        q_pack = paddle.concat([q_pre, q_doc], axis=1)
        k_pack = paddle.concat([k_pre, k_doc], axis=1)
        vr_pack = make_causal_valid_range(
            total, batch=1, doc_lengths=[prefix_len, doc_len]
        )
        vr_pre = make_causal_valid_range(prefix_len, batch=1)
        idx_pre = build_random_block_indices(
            vr_pre, topk, self.BLOCK_B, prefix_len, seed=303
        )
        idx_pack = paddle.concat([idx_pre, idx_doc], axis=1).contiguous()
        _, dq_pack, _, _ = self._run_tl(q_pack, k_pack, idx_pack, vr_pack, sm)

        dq_pack_doc = dq_pack[:, prefix_len:, :, :]
        assert_close(
            self, "tl_pack_vs_solo_dq", dq_pack_doc, dq_solo, min_cos=0.99
        )


if __name__ == "__main__":
    unittest.main()
