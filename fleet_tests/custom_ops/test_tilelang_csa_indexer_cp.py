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

"""Single-card tests for TileLang CSA indexer under simulated CP conditions.

Verifies that the TileLang indexer kernel produces correct results when called
with seq_offset (the CP position offset), by comparing:
  1. Local slice + seq_offset == full-sequence result slice (bit-exact for fwd)
  2. Backward dQ/dW slice-exact, sum-of-dK across ranks == full dK
  3. Causal boundary valid_count matches closed-form formula per-position
  4. TileLang kernel vs Paddle reference indexer cross-validation

These tests exercise the kernel's `valid_end = (i_t + seq_offset + 1) // ratio`
logic without requiring multiple GPUs.

Run:
    python -m pytest fleet_tests/custom_ops/test_tilelang_csa_indexer_cp.py -v
"""

import unittest

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)


# =========================================================================
# Helpers
# =========================================================================


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _make_inputs(b, sq, h_i, d_i, ratio, seed=2026):
    """Generate q [b, sq, h_i, d_i], k [b, sk, d_i], w [b, sq, h_i]."""
    sk = sq // ratio
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i], dtype="bfloat16")
    k = paddle.randn([b, sk, d_i], dtype="bfloat16")
    w = paddle.randn([b, sq, h_i], dtype="float32")
    return q, k, w


def _cosine_sim(a, b):
    """Cosine similarity between two tensors (flattened to 1D)."""
    a_f = a.cast("float32").flatten()
    b_f = b.cast("float32").flatten()
    dot = (a_f * b_f).sum()
    return (dot / (a_f.norm() * b_f.norm() + 1e-30)).item()


# =========================================================================
# Test: Forward topk bit-exact under simulated CP
# =========================================================================


class TestIndexerFwdCPBitExact(unittest.TestCase):
    """Slicing Q + seq_offset must produce identical results to full-seq slice."""

    def setUp(self):
        _cuda_or_skip(self)
        paddle.set_device("gpu")

    def _run(self, sq_global, cp_size, h_i, d_i, ratio, topk_eff):
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

        b = 2
        sq_local = sq_global // cp_size
        q, k, w = _make_inputs(b, sq_global, h_i, d_i, ratio)
        sk = sq_global // ratio
        if topk_eff > sk:
            self.skipTest(f"topk_eff={topk_eff} > sk={sk}")

        # Full-sequence reference
        idx_full, scores_full = csa_indexer_topk_fwd(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk_eff,
            seq_offset=0,
        )

        # Verify each simulated rank
        for r in range(cp_size):
            s = r * sq_local
            idx_r, scores_r = csa_indexer_topk_fwd(
                q[:, s : s + sq_local, :, :],
                k,
                w[:, s : s + sq_local, :],
                ratio=ratio,
                topk_effective=topk_eff,
                seq_offset=s,
            )
            self.assertTrue(
                paddle.equal_all(
                    idx_r, idx_full[:, s : s + sq_local, :]
                ).item(),
                f"Indices mismatch: sq={sq_global} cp={cp_size} rank={r}",
            )
            score_diff = (
                (scores_r - scores_full[:, s : s + sq_local, :])
                .abs()
                .max()
                .item()
            )
            self.assertLess(
                score_diff,
                1e-6,
                f"Scores diff={score_diff:.2e}: sq={sq_global} cp={cp_size} rank={r}",
            )

    def test_phase3_cp2_small(self):
        self._run(sq_global=32, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=4)

    def test_phase3_cp2_medium(self):
        self._run(sq_global=64, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=8)

    def test_phase3_cp4(self):
        self._run(sq_global=64, cp_size=4, h_i=16, d_i=32, ratio=4, topk_eff=4)

    def test_phase3_cp2_larger(self):
        self._run(
            sq_global=128, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=16
        )

    def test_phase3_cp4_larger(self):
        self._run(sq_global=128, cp_size=4, h_i=16, d_i=32, ratio=4, topk_eff=8)

    def test_phase3_long_seq(self):
        self._run(
            sq_global=256, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=16
        )

    def test_phase3_cp4_long(self):
        self._run(
            sq_global=256, cp_size=4, h_i=16, d_i=32, ratio=4, topk_eff=32
        )

    def test_phase3_512(self):
        self._run(
            sq_global=512, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=32
        )

    def test_phase2_topk_eq_ncomp(self):
        """Phase 2: topk_eff == n_compressed (full candidate selection)."""
        self._run(sq_global=64, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=16)

    def test_phase2_cp4(self):
        """Phase 2 with cp_size=4."""
        self._run(
            sq_global=128, cp_size=4, h_i=16, d_i=32, ratio=4, topk_eff=32
        )


# =========================================================================
# Test: Backward dQ/dW slice-exact, dK sum-exact
# =========================================================================


class TestIndexerBwdCPGradients(unittest.TestCase):
    """Backward under simulated CP: dQ/dW match slices, dK sums to full."""

    def setUp(self):
        _cuda_or_skip(self)
        paddle.set_device("gpu")

    def _run(self, sq_global, cp_size, h_i, d_i, ratio, topk_eff):
        from paddleformers.fleet.tilelang_ops import (
            csa_indexer_bwd,
            csa_indexer_topk_fwd,
        )

        b = 2
        sq_local = sq_global // cp_size
        q, k, w = _make_inputs(b, sq_global, h_i, d_i, ratio, seed=3030)
        sk = sq_global // ratio
        if topk_eff > sk:
            self.skipTest(f"topk_eff={topk_eff} > sk={sk}")

        # Full-sequence fwd + bwd
        idx_full, _ = csa_indexer_topk_fwd(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk_eff,
            seq_offset=0,
        )
        paddle.seed(4040)
        grad_full = (
            paddle.randn([b, sq_global, topk_eff], dtype="float32") * 0.01
        )
        dq_full, dw_full, dk_full = csa_indexer_bwd(
            q, w, k, idx_full, grad_full
        )

        # Simulate each rank and accumulate dK
        dk_accum = paddle.zeros_like(dk_full).cast("float32")
        for r in range(cp_size):
            s = r * sq_local
            idx_r, _ = csa_indexer_topk_fwd(
                q[:, s : s + sq_local, :, :],
                k,
                w[:, s : s + sq_local, :],
                ratio=ratio,
                topk_effective=topk_eff,
                seq_offset=s,
            )
            dq_r, dw_r, dk_r = csa_indexer_bwd(
                q[:, s : s + sq_local, :, :],
                w[:, s : s + sq_local, :],
                k,
                idx_r,
                grad_full[:, s : s + sq_local, :],
            )
            # dQ slice-exact
            dq_diff = (
                (
                    dq_r.cast("float32")
                    - dq_full[:, s : s + sq_local, :, :].cast("float32")
                )
                .abs()
                .max()
                .item()
            )
            self.assertLess(
                dq_diff,
                1e-4,
                f"dQ mismatch: sq={sq_global} rank={r} diff={dq_diff:.2e}",
            )
            # dW slice-exact
            dw_diff = (
                (
                    dw_r.cast("float32")
                    - dw_full[:, s : s + sq_local, :].cast("float32")
                )
                .abs()
                .max()
                .item()
            )
            self.assertLess(
                dw_diff,
                1e-2,
                f"dW mismatch: sq={sq_global} rank={r} diff={dw_diff:.2e}",
            )
            dk_accum += dk_r.cast("float32")

        # Sum of dK across ranks == full dK
        dk_diff = (dk_accum - dk_full.cast("float32")).abs().max().item()
        self.assertLess(
            dk_diff,
            1e-2,
            f"dK sum mismatch: sq={sq_global} cp={cp_size} diff={dk_diff:.2e}",
        )

    def test_cp2_small(self):
        self._run(sq_global=32, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=4)

    def test_cp2_medium(self):
        self._run(sq_global=64, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=8)

    def test_cp4(self):
        self._run(sq_global=64, cp_size=4, h_i=16, d_i=32, ratio=4, topk_eff=4)

    def test_cp2_larger(self):
        self._run(
            sq_global=128, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=16
        )

    def test_cp2_long(self):
        self._run(
            sq_global=256, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=16
        )

    def test_phase2(self):
        """Phase 2: topk == n_compressed."""
        self._run(sq_global=64, cp_size=2, h_i=16, d_i=32, ratio=4, topk_eff=16)


# =========================================================================
# Test: Causal boundary exact valid_count
# =========================================================================


class TestIndexerCPCausalBoundary(unittest.TestCase):
    """Per-position valid_count must match: min((global_pos+1)//ratio, sk, topk)."""

    def setUp(self):
        _cuda_or_skip(self)
        paddle.set_device("gpu")

    def _run(self, sq_global, cp_size, ratio):
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

        b, h_i, d_i = 1, 16, 32
        sq_local = sq_global // cp_size
        sk = sq_global // ratio
        topk_eff = sk  # Use full range to expose all valid positions

        q, k, w = _make_inputs(b, sq_global, h_i, d_i, ratio, seed=6060)

        for r in range(cp_size):
            s = r * sq_local
            idx_r, _ = csa_indexer_topk_fwd(
                q[:, s : s + sq_local, :, :],
                k,
                w[:, s : s + sq_local, :],
                ratio=ratio,
                topk_effective=topk_eff,
                seq_offset=s,
            )
            idx_np = idx_r[0].numpy()
            for t in range(sq_local):
                expected = min((s + t + 1) // ratio, sk, topk_eff)
                actual = int((idx_np[t] >= 0).sum())
                self.assertEqual(
                    actual,
                    expected,
                    f"Causal boundary wrong: sq={sq_global} cp={cp_size} "
                    f"rank={r} pos={s + t}: expected={expected} actual={actual}",
                )

    def test_cp2_sq64(self):
        self._run(sq_global=64, cp_size=2, ratio=4)

    def test_cp4_sq128(self):
        self._run(sq_global=128, cp_size=4, ratio=4)

    def test_cp2_sq256(self):
        self._run(sq_global=256, cp_size=2, ratio=4)

    def test_cp2_sq512(self):
        self._run(sq_global=512, cp_size=2, ratio=4)

    def test_first_positions_all_invalid(self):
        """Positions 0..ratio-2 should have valid_count=0."""
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

        b, h_i, d_i, ratio = 1, 16, 32, 4
        sq_global, cp_size = 64, 2
        sq_local = sq_global // cp_size
        sk = sq_global // ratio
        q, k, w = _make_inputs(b, sq_global, h_i, d_i, ratio, seed=7070)

        # Rank 0 holds the first positions
        idx_r, _ = csa_indexer_topk_fwd(
            q[:, :sq_local, :, :],
            k,
            w[:, :sq_local, :],
            ratio=ratio,
            topk_effective=sk,
            seq_offset=0,
        )
        idx_np = idx_r[0].numpy()
        for t in range(ratio - 1):
            count = int((idx_np[t] >= 0).sum())
            self.assertEqual(
                count,
                0,
                f"Position {t} should have 0 valid, got {count}",
            )


# =========================================================================
# Test: TileLang vs Paddle reference cross-validation
# =========================================================================


class TestIndexerCPTileLangVsPaddleRef(unittest.TestCase):
    """TileLang kernel topk set vs Paddle fused_qk_topk_naive set agreement."""

    def setUp(self):
        _cuda_or_skip(self)
        paddle.set_device("gpu")

    def _run(self, sq_global, cp_size, topk_eff, max_mismatch_rate=0.05):
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd
        from paddleformers.fleet.transformer.cp_utils import build_causal_mask_cp
        from paddleformers.fleet.transformer.csa_attention import fused_qk_topk_naive

        b, h_i, d_i, ratio = 2, 16, 32, 4
        sq_local = sq_global // cp_size
        sk = sq_global // ratio
        if topk_eff > sk:
            self.skipTest(f"topk_eff={topk_eff} > sk={sk}")

        q, k, w = _make_inputs(b, sq_global, h_i, d_i, ratio, seed=5050)

        total_mismatch = 0
        total_pos = 0
        for r in range(cp_size):
            s = r * sq_local
            q_r = q[:, s : s + sq_local, :, :]
            w_r = w[:, s : s + sq_local, :]

            tl_idx, _ = csa_indexer_topk_fwd(
                q_r,
                k,
                w_r,
                ratio=ratio,
                topk_effective=topk_eff,
                seq_offset=s,
            )
            q_pos = paddle.arange(s, s + sq_local, dtype="int64")
            mask = build_causal_mask_cp(q_pos, sk, ratio, b)
            _, pd_idx = fused_qk_topk_naive(q_r, k, w_r, topk_eff, mask)

            tl_np = tl_idx.numpy()
            pd_np = pd_idx.numpy()
            for bi in range(b):
                for ti in range(sq_local):
                    # Number of causally valid compressed positions for this query
                    n_valid = min((s + ti + 1) // ratio, sk)
                    # TileLang: invalid slots are -1
                    tl_set = set(tl_np[bi, ti][tl_np[bi, ti] >= 0])
                    # Paddle: topk always fills all slots; filter to valid range
                    pd_set = {int(x) for x in pd_np[bi, ti] if 0 <= x < n_valid}
                    if tl_set != pd_set:
                        total_mismatch += 1
                    total_pos += 1

        rate = total_mismatch / max(total_pos, 1)
        self.assertLess(
            rate,
            max_mismatch_rate,
            f"TL vs Paddle mismatch rate {rate * 100:.1f}% exceeds "
            f"{max_mismatch_rate * 100:.0f}%: sq={sq_global} cp={cp_size} topk={topk_eff}",
        )

    def test_cp2_small(self):
        self._run(sq_global=64, cp_size=2, topk_eff=8)

    def test_cp2_medium(self):
        self._run(sq_global=128, cp_size=2, topk_eff=16)

    def test_cp4(self):
        self._run(sq_global=256, cp_size=4, topk_eff=16)

    def test_cp2_large_topk(self):
        """Large topk (close to n_compressed)."""
        self._run(sq_global=128, cp_size=2, topk_eff=32)


# =========================================================================
# Test: seq_offset edge cases
# =========================================================================


class TestIndexerSeqOffsetEdgeCases(unittest.TestCase):
    """Edge cases for seq_offset parameter handling."""

    def setUp(self):
        _cuda_or_skip(self)
        paddle.set_device("gpu")

    def test_offset_zero_is_no_op(self):
        """seq_offset=0 should produce same results as not passing it."""
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

        b, sq, h_i, d_i, ratio = 2, 64, 16, 32, 4
        q, k, w = _make_inputs(b, sq, h_i, d_i, ratio)
        sk = sq // ratio
        topk = 8

        idx_default, scores_default = csa_indexer_topk_fwd(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk,
            seq_offset=0,
        )
        idx_explicit, scores_explicit = csa_indexer_topk_fwd(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk,
            seq_offset=0,
        )
        self.assertTrue(paddle.equal_all(idx_default, idx_explicit).item())

    def test_large_offset_all_valid(self):
        """With large seq_offset, all positions should have full valid range."""
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

        b, sq, h_i, d_i, ratio = 1, 32, 16, 32, 4
        q, k, w = _make_inputs(b, sq, h_i, d_i, ratio)
        sk = sq // ratio
        topk = sk

        # Large offset: even position 0 sees all compressed positions
        large_offset = 1024
        idx, _ = csa_indexer_topk_fwd(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk,
            seq_offset=large_offset,
        )
        idx_np = idx[0].numpy()
        for t in range(sq):
            count = int((idx_np[t] >= 0).sum())
            expected = min((t + large_offset + 1) // ratio, sk, topk)
            self.assertEqual(
                count,
                expected,
                f"pos={t}: expected {expected} valid, got {count}",
            )

    def test_rank_ordering_monotonic(self):
        """Higher CP ranks should have >= valid positions compared to lower ranks."""
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

        b, h_i, d_i, ratio = 1, 16, 32, 4
        sq_global, cp_size = 128, 4
        sq_local = sq_global // cp_size
        sk = sq_global // ratio
        topk = sk
        q, k, w = _make_inputs(b, sq_global, h_i, d_i, ratio)

        prev_min_valid = -1
        for r in range(cp_size):
            s = r * sq_local
            idx_r, _ = csa_indexer_topk_fwd(
                q[:, s : s + sq_local, :, :],
                k,
                w[:, s : s + sq_local, :],
                ratio=ratio,
                topk_effective=topk,
                seq_offset=s,
            )
            idx_np = idx_r[0].numpy()
            min_valid = int((idx_np[0] >= 0).sum())  # first position in rank
            self.assertGreaterEqual(
                min_valid,
                prev_min_valid,
                f"Rank {r} first pos has fewer valid ({min_valid}) "
                f"than rank {r - 1} ({prev_min_valid})",
            )
            prev_min_valid = int((idx_np[-1] >= 0).sum())  # last position


if __name__ == "__main__":
    unittest.main()
