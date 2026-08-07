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

"""Augmented precision coverage for the HySparse block-sparse MQA TileLang
oracle (:func:`block_sparse_mqa_attention_tl`), the sparse gather used when
``hy_sparse_use_tilelang=true``.

The existing ``test_hysparse_block_sparse_mqa_tl_multidoc_grad`` validates the
packed multi-doc backward but only under a UNIFORM (``out.sum()`` == all-ones)
cotangent and at the small H=4 / Dk=576 config. This file adds the coverage
that a uniform cotangent cannot reach:

* **Random cotangent.** ``out.backward(g)`` with a random ``g`` per head/token,
  so a per-channel sign/scale bug in ``dV = P^T dO`` (which a ones cotangent can
  mask by summing symmetric terms) is exposed. All of dq / dkv / d_sink are
  scored against the fp32 ``ref_block_sparse_mqa`` autograd gradient.
* **Online-scale dims.** ``H=64`` query heads at the two production absorbed-MLA
  latent shapes: ``Dk=576/Dv=512`` (csa configs, ``kv_lora_rank=512``) and
  ``Dk=512/Dv=448`` (ernielite_layer43, ``kv_lora_rank=448`` + rope 64).
* **Finite learnable sink** and **packed ragged docs** combined with the random
  cotangent.

Metrics are MAGNITUDE-sensitive: a cosine floor plus a relative-L2 / norm-ratio
bound, so a systematic scale error cannot pass on cosine alone. (Per-element
``max_rel`` is intentionally NOT gated: near-zero reference entries make it
meaningless for these summed bf16 gradients.)

NOTE on requested dims: the team asked for ``Dk=256, Dv=448``, but the op
requires ``Dv <= Dk`` (the value is the leading ``Dv`` slice of the ``Dk`` shared
latent), so ``Dk=256`` with ``Dv=448`` is not a valid latent shape. The real
``kv_lora_rank=448`` online config is ``Dk=512`` (448 + 64 rope) / ``Dv=448``,
which is what is exercised here.

Runs on any TileLang-capable CUDA GPU; skips gracefully otherwise.
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


def _rel_l2(got, ref):
    """(relative-L2 error, ||got||/||ref|| norm ratio) over finite entries."""
    g = got.astype("float32").numpy().reshape(-1)
    r = ref.astype("float32").numpy().reshape(-1)
    fin = np.isfinite(g) & np.isfinite(r)
    g, r = g[fin], r[fin]
    rn = float(np.linalg.norm(r))
    rel = float(np.linalg.norm(g - r) / (rn + 1e-12))
    ratio = float(np.linalg.norm(g)) / (rn + 1e-12)
    return rel, ratio


class TestBlockSparseMQATLPrecisionAug(unittest.TestCase):
    """Random-cotangent + online-dim precision for the TileLang MQA oracle."""

    BLOCK_B = 64

    def _build(self, b, s, h, dk, topk, doc_lengths, seed):
        from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
            build_random_block_indices,
            make_causal_valid_range,
        )

        paddle.seed(seed)
        np.random.seed(seed)
        q = (np.random.randn(b, s, h, dk) * 0.5).astype("float32")
        kf = (np.random.randn(b, s, dk) * 0.5).astype("float32")
        vr = make_causal_valid_range(s, batch=b, doc_lengths=doc_lengths)
        idx = build_random_block_indices(
            vr, topk, self.BLOCK_B, s, seed=seed + 1
        )
        return q, kf, vr, idx

    def _mag(self, name, got, ref, rel_tol):
        """Cosine (via assert_close) already run; add rel-L2 + RMSE + ratio."""
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
            max(rel_tol, 6e-2),
            f"{name}: norm ratio {ratio:.4f} far from 1",
        )

    def _confirm_sink_rounding(
        self, tag, ds_k, ds_r, q_np, k_np, idx, vr, sm, sink_np, g_np, dv
    ):
        """Prove the kernel dSink discrepancy is bf16 rounding, not a formula
        bug.

        The kernel folds the sink as a virtual softmax column and computes its
        gradient analytically on host from bf16 ``delta``/``p_sink``. Its error
        vs the fp32 autograd reference (``ds_r``) is only trustworthy as a
        *precision* claim if it sits at the intrinsic bf16 noise level. We
        measure that level directly: re-run the SAME fp32 reference math but
        with q / k / dO pre-quantized to bf16 (input + cotangent rounding, no
        kernel involved) and compare its dSink to the fp32 ground truth. If the
        kernel error is within a small multiple of this pure-rounding floor, the
        residual is bf16 accumulation, not a formula defect.
        """
        from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
            ref_block_sparse_mqa,
        )

        def _bf16q(a):  # fp32 -> bf16 -> fp32: intrinsic bf16 rounding
            t = paddle.to_tensor(a, dtype="float32").astype("bfloat16")
            return t.astype("float32")

        qq = _bf16q(q_np)
        kq = _bf16q(k_np)
        qq.stop_gradient = False
        kq.stop_gradient = False
        # sink itself is fp32 in BOTH kernel and ref, so it is not quantized.
        sq = paddle.to_tensor(sink_np, dtype="float32", stop_gradient=False)
        oq = ref_block_sparse_mqa(
            qq,
            kq,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=dv,
            attn_sink=sq,
        )
        oq.backward(_bf16q(g_np))
        ds_bf16in = sq.grad
        floor_rel, _ = _rel_l2(ds_bf16in, ds_r)
        kern_rel, _ = _rel_l2(ds_k, ds_r)
        print(
            f"    [{tag}:dsink] kernel_rel_l2={kern_rel:.3e} "
            f"bf16_noise_floor_rel_l2={floor_rel:.3e}"
        )
        # Same order as the pure-rounding floor => rounding, not a formula bug.
        # (Kernel adds bf16 atomic accumulation on top of the two input/dO
        # roundings the floor captures, so allow a modest multiplier.)
        self.assertLessEqual(
            kern_rel,
            max(6.0 * floor_rel, 1e-2),
            f"{tag}:dsink kernel err {kern_rel:.3e} >> bf16 floor "
            f"{floor_rel:.3e}: not explained by rounding",
        )

    def _run(
        self,
        tag,
        b,
        s,
        h,
        dk,
        dv,
        topk,
        doc_lengths,
        sink_scale,
        seed=0,
        rand_cotangent=True,
        sink_value=None,
    ):
        """One end-to-end case: random-cotangent backward, kernel vs fp32 ref.

        ``sink_scale=None`` -> sinkless; otherwise a finite learnable sink.
        ``sink_value`` supplies a constant sink (for example ``-1e30``) instead
        of the default ``randn(H) * sink_scale``. ``rand_cotangent`` picks a
        per-element random grad_output vs the all-ones baseline.
        """
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
            block_sparse_mqa_attention_tl,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
            ref_block_sparse_mqa,
        )

        q_np, k_np, vr, idx = self._build(b, s, h, dk, topk, doc_lengths, seed)
        sm = dk**-0.5
        if doc_lengths is not None:
            self.assertGreater(int(vr[0, :, 0].max()), 0)  # nonzero bos
        sink_np = None
        if sink_value is not None:
            sink_np = np.full([h], sink_value, dtype="float32")
        elif sink_scale is not None:
            paddle.seed(seed + 5)
            sink_np = (np.random.randn(h) * sink_scale).astype("float32")

        # ---- fp32 reference (autograd ground truth) ----
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
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=dv,
            attn_sink=sink_r,
        )
        if rand_cotangent:
            g_np = (np.random.randn(*out_r.shape) * 0.5).astype("float32")
        else:
            g_np = np.ones(out_r.shape, dtype="float32")
        out_r.backward(paddle.to_tensor(g_np, dtype="float32"))
        dq_r, dkv_r = qr.grad, kr.grad
        ds_r = sink_r.grad if sink_r is not None else None

        # ---- TileLang kernel (bf16, autograd) ----
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
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=dv,
            attn_sink=sink_k,
        )
        self.assertIsNone(second)
        out_k.backward(paddle.to_tensor(g_np, dtype="bfloat16"))
        dq_k, dkv_k = qk.grad, kk.grad
        ds_k = sink_k.grad if sink_k is not None else None

        # out is [B, S, H*Dv]; compare against reshaped ref.
        assert_close(self, f"{tag}:out", out_k, out_r, min_cos=0.99)
        self._mag(f"{tag}:out", out_k, out_r, 5e-2)
        assert_close(self, f"{tag}:dq", dq_k, dq_r, min_cos=0.99)
        self._mag(f"{tag}:dq", dq_k, dq_r, 6e-2)
        # dkv accumulates dK_score + dV over every head/token that picked a
        # block -> large summed magnitudes carry a bf16 quantization step; a
        # looser rel-L2 bound but the cosine floor stays tight.
        assert_close(self, f"{tag}:dkv", dkv_k, dkv_r, min_cos=0.99)
        self._mag(f"{tag}:dkv", dkv_k, dkv_r, 1.2e-1)
        if sink_np is not None:
            self.assertIsNotNone(ds_k)
            if sink_value is not None and sink_value <= -1e20:
                # Both gradients are exactly zero for a disabled sink; cosine is
                # undefined for two zero vectors, so compare values directly.
                self.assertTrue(bool(paddle.equal_all(ds_k, ds_r)))
            else:
                assert_close(self, f"{tag}:dsink", ds_k, ds_r, min_cos=0.99)
                self._mag(f"{tag}:dsink", ds_k, ds_r, 8e-2)
                # Attribute the residual: kernel dSink error must sit at the bf16
                # noise floor, proving it is rounding and not a formula defect.
                self._confirm_sink_rounding(
                    tag, ds_k, ds_r, q_np, k_np, idx, vr, sm, sink_np, g_np, dv
                )

    # ----- test cases -----------------------------------------------------
    def test_randcot_online_dk576_h64_sinkless(self):
        # csa online config: Dk=576, Dv=512 (kv_lora_rank=512), H=64.
        _skip_if_no_tl(self)
        self._run(
            "rc_dk576_h64", 1, 256, 64, 576, 512, 8, [96, 160], None, seed=11
        )

    def test_randcot_online_dk512_dv448_h64_sinkless(self):
        # ernielite_layer43 online config: kv_lora_rank=448 -> Dk=512, Dv=448.
        _skip_if_no_tl(self)
        self._run(
            "rc_dk512_dv448_h64",
            1,
            256,
            64,
            512,
            448,
            8,
            [88, 168],
            None,
            seed=12,
        )

    def test_randcot_online_dk512_dv448_h64_finite_sink(self):
        # Same online dims with a finite learnable sink under random cotangent.
        _skip_if_no_tl(self)
        self._run(
            "rc_dk512_dv448_sink",
            1,
            256,
            64,
            512,
            448,
            8,
            [128, 128],
            0.5,
            seed=13,
        )

    def test_randcot_online_dk512_dv448_h64_negative_sink(self):
        # A disabled virtual sink must match the sinkless reference at online dims.
        _skip_if_no_tl(self)
        self._run(
            "rc_dk512_dv448_neg_sink",
            1,
            256,
            64,
            512,
            448,
            8,
            [96, 160],
            None,
            seed=16,
            sink_value=-1e30,
        )

    def test_randcot_single_doc_dk576_h64_finite_sink(self):
        _skip_if_no_tl(self)
        self._run("rc_single_sink", 1, 256, 64, 576, 512, 8, None, 0.5, seed=14)

    def test_randcot_packed_ragged_h16_sinkless(self):
        # Packed ragged docs (all unaligned to BLOCK_B) + random cotangent.
        _skip_if_no_tl(self)
        self._run(
            "rc_ragged",
            1,
            288,
            16,
            576,
            512,
            6,
            [40, 88, 133, 27],
            None,
            seed=15,
        )

    def test_invalid_dk256_dv448_rejected(self):
        # Guard the documented constraint Dv <= Dk: the requested Dk=256/Dv=448
        # latent shape is invalid and the op must reject it (value is the
        # leading Dv slice of the Dk latent).
        _skip_if_no_tl(self)
        from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
            block_sparse_mqa_attention_tl,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.reference_sparse import (
            build_random_block_indices,
            make_causal_valid_range,
        )

        s, h, dk = 128, 8, 256
        q = paddle.randn([1, s, h, dk]).cast("bfloat16")
        kf = paddle.randn([1, s, dk]).cast("bfloat16")
        vr = make_causal_valid_range(s, batch=1)
        idx = build_random_block_indices(vr, 4, self.BLOCK_B, s, seed=1)
        with self.assertRaises((AssertionError, ValueError, RuntimeError)):
            block_sparse_mqa_attention_tl(
                q,
                kf,
                idx,
                vr,
                sm_scale=dk**-0.5,
                block_B=self.BLOCK_B,
                kv_lora_rank=448,
                attn_sink=None,
            )


if __name__ == "__main__":
    unittest.main()
