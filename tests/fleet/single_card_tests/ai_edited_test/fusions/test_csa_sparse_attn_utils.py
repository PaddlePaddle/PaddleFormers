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

"""Unit tests for CSA sparse-attn shared helpers and unfused/dispatch paths.

These cover the pure-Python / pure-Paddle code paths that do not depend on the
FlashMLA or cuDNN custom ops: input validation, local->global index
conversion, the unfused einsum reference attention, backend dispatch, the
FlashMLA wrapper's arch-alignment / unavailable-fallback helpers, and the
``tilelang_ops`` lazy re-export of ``csa_sparse_attn``.
"""

import unittest

import paddle

try:
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu:0")
    from paddleformers.fleet.fusions import csa_sparse_attn_utils

    _IMPORT_OK = paddle.is_compiled_with_cuda()
except Exception:  # pragma: no cover - import guard for non-GPU collection
    _IMPORT_OK = False


@unittest.skipUnless(_IMPORT_OK, "paddlefleet import requires a CUDA device")
class TestPrepareInputs(unittest.TestCase):
    def _inputs(self):
        q = paddle.randn([2, 3, 4, 8], dtype="float32")
        kv = paddle.randn([2, 6, 8], dtype="float32")
        sink = paddle.randn([4], dtype="float32")
        idx = paddle.randint(0, 6, [2, 3, 5]).cast("int32")
        return q, kv, sink, idx

    def test_rejects_bad_query_rank(self):
        q, kv, sink, idx = self._inputs()
        with self.assertRaisesRegex(ValueError, "q must have shape"):
            csa_sparse_attn_utils.prepare_inputs(
                q.reshape([2, 3, 4 * 8]), kv, sink, idx
            )

    def test_rejects_bad_kv_rank(self):
        q, kv, sink, idx = self._inputs()
        with self.assertRaisesRegex(ValueError, "kv must have shape"):
            csa_sparse_attn_utils.prepare_inputs(
                q, kv.reshape([2, 6 * 8]), sink, idx
            )

    def test_rejects_bad_topk_rank(self):
        q, kv, sink, idx = self._inputs()
        with self.assertRaisesRegex(ValueError, "topk_idxs must have shape"):
            csa_sparse_attn_utils.prepare_inputs(
                q, kv, sink, idx.reshape([2 * 3, 5])
            )

    def test_casts_dtypes(self):
        q, kv, sink, idx = self._inputs()
        _, _, sink_out, idx_out = csa_sparse_attn_utils.prepare_inputs(
            q, kv, sink.cast("bfloat16"), idx.cast("int64")
        )
        self.assertEqual(idx_out.dtype, paddle.int32)
        self.assertEqual(sink_out.dtype, paddle.float32)

    def test_passthrough_when_dtypes_already_match(self):
        q, kv, sink, idx = self._inputs()
        q_out, kv_out, sink_out, idx_out = csa_sparse_attn_utils.prepare_inputs(
            q, kv, sink, idx
        )
        self.assertIs(q_out, q)
        self.assertIs(kv_out, kv)
        self.assertIs(sink_out, sink)
        self.assertIs(idx_out, idx)


@unittest.skipUnless(_IMPORT_OK, "paddlefleet import requires a CUDA device")
class TestLocalToGlobalFlat(unittest.TestCase):
    def test_offsets_and_invalid_passthrough(self):
        b, sq, topk, skv = 2, 3, 4, 6
        local = paddle.to_tensor(
            [
                [[0, 1, 2, -1], [3, 4, 5, -1], [0, 5, -1, -1]],
                [[1, 2, 3, 4], [-1, -1, 0, 5], [2, 2, 2, 2]],
            ],
            dtype="int64",
        )
        out = csa_sparse_attn_utils._local_to_global_flat(local, skv)
        self.assertEqual(out.dtype, paddle.int32)
        self.assertEqual(out.shape, [b * sq, topk])

        flat = local.reshape([b * sq, topk]).numpy()
        got = out.numpy()
        for row in range(b * sq):
            batch = row // sq
            for c in range(topk):
                v = int(flat[row, c])
                expected = -1 if v < 0 else batch * skv + v
                self.assertEqual(int(got[row, c]), expected)


@unittest.skipUnless(_IMPORT_OK, "paddlefleet import requires a CUDA device")
class TestUnfusedCompressedSparseAttn(unittest.TestCase):
    def _inputs(self, b=1, sq=2, np_heads=3, hn=4, n_kv=5, topk=3):
        paddle.seed(2026)
        query = paddle.randn([b, sq, np_heads, hn], dtype="float32")
        kv_full = paddle.randn([b, n_kv, hn], dtype="float32")
        attn_sink = paddle.randn([np_heads], dtype="float32")
        topk_idxs = paddle.randint(0, n_kv, [b, sq, topk]).cast("int32")
        return query, kv_full, attn_sink, topk_idxs

    def test_output_shape(self):
        from paddleformers.fleet.fusions.csa_sparse_attn import (
            unfused_compressed_sparse_attn,
        )

        b, sq, np_heads, hn = 2, 3, 4, 8
        query, kv_full, attn_sink, topk_idxs = self._inputs(
            b, sq, np_heads, hn, n_kv=6, topk=4
        )
        out = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, softmax_scale=1.0 / hn**0.5
        )
        self.assertEqual(out.shape, [b, sq, np_heads * hn])
        self.assertEqual(out.dtype, query.dtype)

    def test_all_invalid_indices_yield_zero_output(self):
        from paddleformers.fleet.fusions.csa_sparse_attn import (
            unfused_compressed_sparse_attn,
        )

        query, kv_full, attn_sink, _ = self._inputs()
        # All topk positions invalid -> attention weights are all zero, so the
        # weighted sum (output) must be exactly zero everywhere.
        topk_idxs = paddle.full([1, 2, 3], -1, dtype="int32")
        out = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, softmax_scale=0.5
        )
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_matches_manual_softmax_reference(self):
        import math

        import numpy as np

        from paddleformers.fleet.fusions.csa_sparse_attn import (
            unfused_compressed_sparse_attn,
        )

        # Single head, single query, one invalid slot, zero sink -> compare to
        # a hand-computed stable softmax over the two valid KV rows.
        query = paddle.randn([1, 1, 1, 4], dtype="float32")
        kv_full = paddle.randn([1, 5, 4], dtype="float32")
        attn_sink = paddle.zeros([1], dtype="float32")
        topk_idxs = paddle.to_tensor([[[0, 2, -1]]], dtype="int32")
        out = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, softmax_scale=1.0
        )

        q = query.numpy()[0, 0, 0]
        kv = kv_full.numpy()[0]
        scores = [float((q * kv[0]).sum()), float((q * kv[2]).sum())]
        m = max(*scores, 0.0)
        exp = [math.exp(s - m) for s in scores]
        exp_sink = math.exp(0.0 - m)
        denom = sum(exp) + exp_sink
        ref = exp[0] / denom * kv[0] + exp[1] / denom * kv[2]
        self.assertLess(float(np.abs(out.numpy()[0, 0] - ref).max()), 1e-5)


@unittest.skipUnless(_IMPORT_OK, "paddlefleet import requires a CUDA device")
class TestCsaSparseAttnDispatch(unittest.TestCase):
    def _inputs(self):
        paddle.seed(7)
        query = paddle.randn([1, 2, 3, 4], dtype="float32")
        kv_full = paddle.randn([1, 5, 4], dtype="float32")
        attn_sink = paddle.randn([3], dtype="float32")
        topk_idxs = paddle.randint(0, 5, [1, 2, 3]).cast("int32")
        return query, kv_full, attn_sink, topk_idxs

    def test_unfused_dispatch_matches_direct_call(self):
        from paddleformers.fleet.fusions.csa_sparse_attn import (
            csa_sparse_attn,
            unfused_compressed_sparse_attn,
        )

        query, kv_full, attn_sink, topk_idxs = self._inputs()
        direct = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, softmax_scale=0.5
        )
        dispatched = csa_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, 0.5, backend="unfused"
        )
        self.assertTrue(bool((direct == dispatched).all()))

    def test_invalid_backend_raises(self):
        from paddleformers.fleet.fusions.csa_sparse_attn import csa_sparse_attn

        query, kv_full, attn_sink, topk_idxs = self._inputs()
        with self.assertRaisesRegex(
            ValueError, "csa_sparse_attn_backend='bogus' is invalid"
        ):
            csa_sparse_attn(
                query, kv_full, attn_sink, topk_idxs, 0.5, backend="bogus"
            )


@unittest.skipUnless(_IMPORT_OK, "paddlefleet import requires a CUDA device")
class TestTilelangOpsLazyExports(unittest.TestCase):
    def test_indexer_symbols_are_lazily_exported(self):
        import paddleformers.fleet.tilelang_ops as tl

        for name in (
            "csa_attn_target_reducesum",
            "csa_indexer_bwd",
            "csa_indexer_topk_fwd",
        ):
            self.assertTrue(callable(getattr(tl, name)), name)

    def test_csa_sparse_attn_is_lazily_exported(self):
        import paddleformers.fleet.tilelang_ops as tl
        from paddleformers.fleet.fusions.csa_sparse_attn import csa_sparse_attn

        self.assertIs(tl.csa_sparse_attn, csa_sparse_attn)

    def test_unknown_attribute_raises(self):
        import paddleformers.fleet.tilelang_ops as tl

        with self.assertRaises(AttributeError):
            _ = tl.definitely_not_a_real_symbol_xyz


@unittest.skipUnless(_IMPORT_OK, "paddlefleet import requires a CUDA device")
class TestFlashMlaWrapperHelpers(unittest.TestCase):
    def test_topk_alignment_matches_arch(self):
        from paddleformers.fleet.cudnn_ops.attn import (
            csa_sparse_attn_fwd_cudnn as m,
        )

        align = m._get_topk_alignment()
        sm = paddle.cuda.get_device_capability()
        self.assertEqual(align, 64 if sm[0] >= 10 else 128)

    def test_raises_when_flash_mla_unavailable(self):
        from paddleformers.fleet.cudnn_ops.attn import (
            csa_sparse_attn_fwd_cudnn as m,
        )

        saved = m._flash_mla_sparse_fwd
        m._flash_mla_sparse_fwd = None
        try:
            with self.assertRaisesRegex(
                RuntimeError, "flash_mla is not available"
            ):
                m.flash_mla_sparse_attn(
                    None,
                    None,
                    None,
                    paddle.zeros([1, 1, 1], dtype="int32"),
                )
        finally:
            m._flash_mla_sparse_fwd = saved

    def test_import_guard_sets_none_when_flash_mla_missing(self):
        import importlib
        import sys
        import types

        from paddleformers.fleet.cudnn_ops.attn import (
            csa_sparse_attn_fwd_cudnn as m,
        )

        # Reload the module with a flash_mla stub that lacks the symbol so the
        # `except (ImportError, RuntimeError)` guard runs and sets the fallback
        # to None, then restore the real module for the rest of the suite.
        saved_fm = sys.modules.get("paddlefleet_ops.flash_mla")
        sys.modules["paddlefleet_ops.flash_mla"] = types.ModuleType(
            "paddlefleet_ops.flash_mla"
        )
        try:
            reloaded = importlib.reload(m)
            self.assertIsNone(reloaded._flash_mla_sparse_fwd)
        finally:
            if saved_fm is not None:
                sys.modules["paddlefleet_ops.flash_mla"] = saved_fm
            else:
                sys.modules.pop("paddlefleet_ops.flash_mla", None)
            importlib.reload(m)


@unittest.skipUnless(_IMPORT_OK, "paddlefleet import requires a CUDA device")
class TestCudnnBackendDispatch(unittest.TestCase):
    """Cover the PyLayer fwd/bwd cudnn dispatch without real FlashMLA/cuDNN
    kernels by monkeypatching them with shape-faithful stubs."""

    def test_cudnn_forward_backward_dispatch(self):
        import paddleformers.fleet.cudnn_ops as cudnn_pkg
        from paddleformers.fleet.cudnn_ops.attn import (
            csa_sparse_attn_fwd_cudnn as fwd,
        )
        from paddleformers.fleet.fusions.csa_sparse_attn import csa_sparse_attn

        b, sq, h, hn, s_kv, topk = 1, 2, 3, 4, 6, 4

        def fake_fwd(
            q, kv, attn_sink, topk_idxs, sm_scale=None, indexer_topk=0
        ):
            bb, ss, hh, dd = q.shape
            out = paddle.ones([bb, ss, hh, dd], dtype=q.dtype)
            lse = paddle.zeros([bb, ss, hh], dtype="float32")
            return out, lse, None

        def fake_bwd(
            q_flat,
            kv_flat,
            o_flat,
            do_flat,
            lse_flat,
            attn_sink,
            topk_idxs_flat,
            softmax_scale=None,
        ):
            return (
                paddle.ones_like(q_flat),
                paddle.ones_like(kv_flat),
                paddle.ones_like(attn_sink),
            )

        orig_fwd = fwd.flash_mla_sparse_attn
        orig_bwd = getattr(cudnn_pkg, "csa_sparse_attn_bwd_cudnn", None)
        fwd.flash_mla_sparse_attn = fake_fwd
        cudnn_pkg.csa_sparse_attn_bwd_cudnn = fake_bwd
        try:
            q = paddle.randn([b, sq, h, hn], dtype="float32")
            q.stop_gradient = False
            kv = paddle.randn([b, s_kv, hn], dtype="float32")
            kv.stop_gradient = False
            sink = paddle.randn([h], dtype="float32")
            sink.stop_gradient = False
            idx = paddle.randint(0, s_kv, [b, sq, topk]).cast("int32")

            out = csa_sparse_attn(q, kv, sink, idx, 0.5, backend="cudnn")
            self.assertEqual(out.shape, [b, sq, h * hn])

            out.sum().backward()
            self.assertEqual(q.grad.shape, [b, sq, h, hn])
            self.assertEqual(kv.grad.shape, [b, s_kv, hn])
            self.assertEqual(sink.grad.shape, [h])
        finally:
            fwd.flash_mla_sparse_attn = orig_fwd
            if orig_bwd is not None:
                cudnn_pkg.csa_sparse_attn_bwd_cudnn = orig_bwd
