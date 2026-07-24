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

"""Unit tests for cuDNN DSA indexer forward.

Covers:
- csa_indexer_fwd_cudnn.py (_validate_indexer_inputs, cudnn_indexer_forward,
  cudnn_indexer_topk, cudnn_indexer_topk_fwd)
- cudnn_ops/__init__.py (lazy __getattr__ for fwd symbols)
- cudnn_ops/indexer/__init__.py (lazy __getattr__ for fwd symbols)
"""

import unittest

import paddle

# =========================================================================
# Helpers
# =========================================================================

_SKIP_CONDITION = (
    not paddle.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10
)
_SKIP_REASON = "cuDNN DSA indexer requires Blackwell GPU (SM10x)"


def _require_sm100(cls):
    return unittest.skipIf(_SKIP_CONDITION, _SKIP_REASON)(cls)


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _make_indexer_inputs(b, sq, sk, h_i, d_i, dtype="bfloat16", seed=2026):
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i]).astype(dtype)
    k = paddle.randn([b, sk, d_i]).astype(dtype)
    w = paddle.randn([b, sq, h_i]).astype(dtype)
    return q, k, w


def _all_equal(tensor, value):
    return bool((tensor == value).all().item())


def _sorted_compare_indices(out_indices, ref_indices):
    out_sorted = paddle.sort(out_indices, axis=-1)
    ref_sorted = paddle.sort(ref_indices, axis=-1)
    return bool((out_sorted == ref_sorted).all().item())


def _build_causal_mask(batch, seq_len, seq_len_comp, ratio):
    comp_ids = paddle.arange(seq_len_comp, dtype="int64").reshape(
        [1, 1, seq_len_comp]
    )
    valid_end = paddle.arange(1, seq_len + 1, dtype="int64").reshape(
        [1, seq_len, 1]
    ) // int(ratio)
    valid = (comp_ids < valid_end).expand([batch, seq_len, seq_len_comp])
    neg_inf = paddle.full(
        [batch, seq_len, seq_len_comp], float("-inf"), dtype="float32"
    )
    return paddle.where(valid, paddle.zeros_like(neg_inf), neg_inf)


def _paddle_indexer_scores_and_topk(
    index_q, index_k_comp, weights, ratio, topk
):
    from paddleformers.fleet.transformer.dsa_attention import (
        fused_qk_topk_naive,
    )

    scores, indices = fused_qk_topk_naive(
        index_q,
        index_k_comp,
        weights.cast("float32"),
        index_topk=min(int(topk), int(index_k_comp.shape[1])),
        mask=_build_causal_mask(
            int(index_q.shape[0]),
            int(index_q.shape[1]),
            int(index_k_comp.shape[1]),
            ratio,
        ),
    )
    if indices.shape[-1] < int(topk):
        padding = paddle.full(
            [
                int(index_q.shape[0]),
                int(index_q.shape[1]),
                int(topk) - int(indices.shape[-1]),
            ],
            -1,
            dtype=indices.dtype,
        )
        indices = paddle.concat([indices, padding], axis=-1)
    return scores, indices.cast("int32")


# =========================================================================
# Test cases: input validation
# =========================================================================


@_require_sm100
class TestValidateIndexerInputs(unittest.TestCase):
    """Tests for _validate_indexer_inputs."""

    def setUp(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            _validate_indexer_inputs,
        )

        self._validate = _validate_indexer_inputs
        self.B, self.S, self.H, self.D = 1, 16, 32, 128
        self.Sk = self.S // 4
        self.index_q = paddle.zeros(
            [self.B, self.S, self.H, self.D], dtype="bfloat16"
        )
        self.index_k = paddle.zeros([self.B, self.Sk, self.D], dtype="bfloat16")
        self.weights = paddle.zeros([self.B, self.S, self.H], dtype="bfloat16")

    def test_valid_inputs_pass(self):
        self._validate(self.index_q, self.index_k, self.weights)

    def test_type_error_index_q(self):
        with self.assertRaises(TypeError):
            self._validate([1, 2], self.index_k, self.weights)

    def test_type_error_index_k(self):
        with self.assertRaises(TypeError):
            self._validate(self.index_q, "bad", self.weights)

    def test_type_error_weights(self):
        with self.assertRaises(TypeError):
            self._validate(self.index_q, self.index_k, 123)

    def test_wrong_ndim_index_q(self):
        with self.assertRaises(ValueError):
            self._validate(
                paddle.zeros([self.B, self.S, self.H], dtype="bfloat16"),
                self.index_k,
                self.weights,
            )

    def test_wrong_ndim_index_k(self):
        with self.assertRaises(ValueError):
            self._validate(
                self.index_q,
                paddle.zeros([self.B, self.Sk, self.D, 1], dtype="bfloat16"),
                self.weights,
            )

    def test_wrong_ndim_weights(self):
        with self.assertRaises(ValueError):
            self._validate(
                self.index_q,
                self.index_k,
                paddle.zeros([self.B, self.S], dtype="bfloat16"),
            )

    def test_batch_mismatch(self):
        with self.assertRaises(ValueError):
            self._validate(
                self.index_q,
                paddle.zeros([2, self.Sk, self.D], dtype="bfloat16"),
                self.weights,
            )

    def test_shape_mismatch_dim(self):
        with self.assertRaises(ValueError):
            self._validate(
                self.index_q,
                paddle.zeros([self.B, self.Sk, 64], dtype="bfloat16"),
                self.weights,
            )

    def test_invalid_heads(self):
        with self.assertRaises(ValueError):
            self._validate(
                paddle.zeros([self.B, self.S, 16, self.D], dtype="bfloat16"),
                paddle.zeros([self.B, self.Sk, self.D], dtype="bfloat16"),
                paddle.zeros([self.B, self.S, 16], dtype="bfloat16"),
            )

    def test_invalid_dim(self):
        with self.assertRaises(ValueError):
            self._validate(
                paddle.zeros([self.B, self.S, 32, 64], dtype="bfloat16"),
                paddle.zeros([self.B, self.Sk, 64], dtype="bfloat16"),
                paddle.zeros([self.B, self.S, 32], dtype="bfloat16"),
            )


@_require_sm100
class TestTopkEffectiveValidation(unittest.TestCase):
    """Tests for cudnn_indexer_topk_fwd topk_effective <= 0."""

    def test_topk_effective_zero_raises(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        B, S, H, D = 1, 16, 32, 128
        Sk = S // 4
        q = paddle.zeros([B, S, H, D], dtype="bfloat16")
        k = paddle.zeros([B, Sk, D], dtype="bfloat16")
        w = paddle.zeros([B, S, H], dtype="bfloat16")
        with self.assertRaises(ValueError):
            cudnn_indexer_topk_fwd(q, k, w, topk_effective=0)

    def test_topk_effective_negative_raises(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        B, S, H, D = 1, 16, 32, 128
        Sk = S // 4
        q = paddle.zeros([B, S, H, D], dtype="bfloat16")
        k = paddle.zeros([B, Sk, D], dtype="bfloat16")
        w = paddle.zeros([B, S, H], dtype="bfloat16")
        with self.assertRaises(ValueError):
            cudnn_indexer_topk_fwd(q, k, w, topk_effective=-1)


# =========================================================================
# Test cases: forward kernel
# =========================================================================


@_require_sm100
class TestCudnnIndexerForward(unittest.TestCase):
    """Tests for cudnn_indexer_forward (score computation)."""

    def setUp(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_forward,
        )

        self.cudnn_indexer_forward = cudnn_indexer_forward

    def test_output_shape_and_dtype(self):
        B, S_q, H_i, D_i, ratio = 2, 128, 64, 128, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        scores = self.cudnn_indexer_forward(q, k, w, ratio=ratio)
        self.assertEqual(list(scores.shape), [B, S_q, S_k])
        self.assertEqual(scores.dtype, paddle.float32)

    def test_masked_positions_are_neginf(self):
        B, S_q, H_i, D_i, ratio = 1, 32, 64, 128, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        scores = self.cudnn_indexer_forward(q, k, w, ratio=ratio)
        # Position 0: (0+1)//4 = 0 valid KV -> all scores should be -inf
        row0 = scores[0, 0, :].numpy()
        self.assertTrue(
            all(v == float("-inf") for v in row0),
            f"Position 0 should be all -inf, got {row0}",
        )

    def test_scores_match_paddle_reference(self):
        B, S_q, H_i, D_i, ratio = 1, 64, 64, 128, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i, seed=2028)
        scores = self.cudnn_indexer_forward(q, k, w, ratio=ratio, sm_scale=1.0)
        ref_scores, _ = _paddle_indexer_scores_and_topk(
            q, k, w, ratio=ratio, topk=S_k
        )
        valid = paddle.isfinite(ref_scores)
        max_abs_diff = (scores[valid] - ref_scores[valid]).abs().max().item()
        self.assertTrue(
            paddle.allclose(
                scores[valid], ref_scores[valid], rtol=1e-2, atol=1e-2
            ).item(),
            f"cuDNN indexer scores mismatch with Paddle reference, max_abs_diff={max_abs_diff}",
        )


@_require_sm100
class TestCudnnIndexerTopkFwd(unittest.TestCase):
    """Tests for cudnn_indexer_topk_fwd (combined score + top-K)."""

    def setUp(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        self.cudnn_indexer_topk_fwd = cudnn_indexer_topk_fwd

    def test_output_shape_and_dtype(self):
        B, S_q, H_i, D_i, ratio, topk = 2, 128, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, lengths = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        self.assertEqual(list(indices.shape), [B, S_q, topk])
        self.assertEqual(indices.dtype, paddle.int32)
        self.assertEqual(list(lengths.shape), [B, S_q])
        self.assertEqual(lengths.dtype, paddle.int32)

    def test_early_positions_all_invalid(self):
        """Positions 0..ratio-2 have (s+1)//ratio==0 -> all indices -1."""
        B, S_q, H_i, D_i, ratio, topk = 2, 64, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        for b in range(B):
            for s in range(ratio - 1):
                self.assertTrue(
                    _all_equal(indices[b, s, :], -1),
                    f"batch {b}, position {s} should be all -1",
                )

    def test_valid_indices_in_causal_range(self):
        """All non-negative indices satisfy idx < (s+1)//ratio."""
        B, S_q, H_i, D_i, ratio, topk = 2, 128, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        for b in range(B):
            for s in range(S_q):
                max_valid = (s + 1) // ratio
                row = indices[b, s, :].numpy()
                for idx in row:
                    if idx >= 0:
                        self.assertLess(
                            idx,
                            max_valid,
                            f"batch {b}, pos {s}: index {idx} >= max_valid {max_valid}",
                        )

    def test_topk_length_matches_valid_count(self):
        B, S_q, H_i, D_i, ratio, topk = 1, 64, 32, 128, 4, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, lengths = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        expected = (indices >= 0).sum(axis=-1).cast("int32")
        self.assertTrue(
            (lengths == expected).all().item(),
            "topk_length mismatch with actual valid count",
        )

    def test_h32_support(self):
        """Verify H_i=32 (qhead_per_kv_head=32) also works."""
        B, S_q, H_i, D_i, ratio, topk = 1, 64, 32, 128, 4, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        self.assertEqual(list(indices.shape), [B, S_q, topk])

    def test_index_sets_match_paddle_reference(self):
        B, S_q, H_i, D_i, ratio, topk = 2, 64, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i, seed=2029)
        indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        _, ref_indices = _paddle_indexer_scores_and_topk(
            q, k, w, ratio=ratio, topk=topk
        )
        self.assertTrue(
            _sorted_compare_indices(indices, ref_indices),
            "cuDNN and Paddle indexer top-k sets should match",
        )

    def test_seq_offset_matches_sliced_global_run(self):
        """CP causal-only mode: local chunk with seq_offset equals global slice."""
        B, S_global, H_i, D_i, ratio, topk = 1, 64, 64, 128, 4, 8
        S_k = S_global // ratio
        q, k, w = _make_indexer_inputs(B, S_global, S_k, H_i, D_i, seed=2031)
        offset = 32
        S_local = 32
        global_indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        local_indices, _ = self.cudnn_indexer_topk_fwd(
            q[:, offset : offset + S_local],
            k,
            w[:, offset : offset + S_local],
            ratio=ratio,
            topk_effective=topk,
            seq_offset=offset,
        )
        self.assertTrue(
            _sorted_compare_indices(
                local_indices, global_indices[:, offset : offset + S_local]
            ),
            "cuDNN seq_offset top-k sets should match sliced global run",
        )

    def test_seq_offset_with_valid_range_matches_sliced_global_run(self):
        """CP docmask fallback: local valid_range + seq_offset equals global slice."""
        from paddleformers.fleet.transformer.csa_attention import (
            get_valid_range,
        )

        B, S_global, H_i, D_i, ratio, topk = 1, 64, 64, 128, 4, 8
        S_k = S_global // ratio
        q, k, w = _make_indexer_inputs(B, S_global, S_k, H_i, D_i, seed=2032)
        startend = paddle.to_tensor(
            [32] * 32 + [64] * 32, dtype="int32"
        ).reshape([1, 1, S_global, 1])
        valid_range = get_valid_range(ratio, B, S_global, startend)
        offset = 32
        S_local = 32
        global_indices, _ = self.cudnn_indexer_topk_fwd(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk,
            valid_range=valid_range,
            startend_row_indices=startend,
        )
        local_indices, _ = self.cudnn_indexer_topk_fwd(
            q[:, offset : offset + S_local],
            k,
            w[:, offset : offset + S_local],
            ratio=ratio,
            topk_effective=topk,
            valid_range=valid_range[:, offset : offset + S_local],
            startend_row_indices=startend,
            seq_offset=offset,
        )
        self.assertTrue(
            _sorted_compare_indices(
                local_indices, global_indices[:, offset : offset + S_local]
            ),
            "cuDNN seq_offset docmask top-k sets should match sliced global run",
        )


@_require_sm100
class TestCudnnVsTileLangCrossValidation(unittest.TestCase):
    """Cross-validate cuDNN and TileLang indexer backends produce same sets."""

    def setUp(self):
        try:
            paddle.enable_compat(scope={"tilelang"}, silent=True)
            from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

            self.csa_indexer_topk_fwd = csa_indexer_topk_fwd
        except Exception:
            self.skipTest("TileLang CSA indexer not available")
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        self.cudnn_indexer_topk_fwd = cudnn_indexer_topk_fwd

    def test_index_sets_match(self):
        B, S_q, H_i, D_i, ratio, topk = 2, 64, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i, seed=42)
        cudnn_indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        tl_indices, _ = self.csa_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        self.assertTrue(
            _sorted_compare_indices(cudnn_indices, tl_indices),
            "cuDNN and TileLang indexer top-k sets should match",
        )


# =========================================================================
# Test cases: package imports
# =========================================================================


@_require_sm100
class TestPackageImports(unittest.TestCase):
    """Tests for cudnn_ops and cudnn_ops/indexer package exports."""

    def test_cudnn_ops_unknown_attr(self):
        import paddleformers.fleet.cudnn_ops as pkg

        with self.assertRaises(AttributeError):
            _ = pkg.nonexistent_symbol_xyz

    def test_cudnn_ops_indexer_unknown_attr(self):
        import paddleformers.fleet.cudnn_ops.indexer as pkg

        with self.assertRaises(AttributeError):
            _ = pkg.nonexistent_symbol_xyz

    def test_cudnn_ops_resolves_cudnn_indexer_forward(self):
        import paddleformers.fleet.cudnn_ops as pkg

        fn = pkg.cudnn_indexer_forward
        self.assertTrue(callable(fn))

    def test_cudnn_ops_indexer_resolves_cudnn_indexer_topk(self):
        import paddleformers.fleet.cudnn_ops.indexer as pkg

        fn = pkg.cudnn_indexer_topk
        self.assertTrue(callable(fn))


if __name__ == "__main__":
    unittest.main()
