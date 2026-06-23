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

import unittest

import paddle
import paddle.nn.functional as F

paddle.enable_compat(scope={"tilelang"}, silent=True)


# =========================================================================
# Helpers
# =========================================================================


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _make_indexer_inputs(b, sq, sk, h_i, d_i, dtype="bfloat16", seed=2026):
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i]).astype(dtype)
    k = paddle.randn([b, sk, d_i]).astype(dtype)
    w = paddle.randn([b, sq, h_i]).astype("float32")
    return q, k, w


def _make_loss_inputs(b, sq, sk, h_i, d_i, np_, hn, seed=2027):
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i]).astype("bfloat16")
    k = paddle.randn([b, sk, d_i]).astype("bfloat16")
    weights = paddle.randn([b, sq, h_i]).astype("float32")
    query_mla = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
    key_comp_mla = paddle.randn([b, sk, hn]).astype("bfloat16").detach()
    return q, k, weights, query_mla, key_comp_mla


def _build_csa_causal_mask(b, sq, sk, ratio):
    comp_ids = paddle.arange(sk, dtype="int64").reshape([1, 1, 1, sk])
    valid_end = (
        paddle.arange(1, sq + 1, dtype="int64").reshape([1, 1, sq, 1]) // ratio
    )
    valid = (comp_ids < valid_end).expand([b, 1, sq, sk])
    neg_inf = paddle.full([b, 1, sq, sk], float("-inf"), dtype="float32")
    return paddle.where(valid, paddle.zeros_like(neg_inf), neg_inf)


def _all_equal(tensor, value):
    return bool((tensor == value).all().item())


def _sorted_compare_indices(out_indices, ref_indices):
    out_sorted = paddle.sort(out_indices, axis=-1)
    ref_sorted = paddle.sort(ref_indices, axis=-1)
    return bool((out_sorted == ref_sorted).all().item())


def _assert_close(actual, expected, rtol, atol, msg):
    a = actual.cast("float32") if actual.dtype != paddle.float32 else actual
    e = (
        expected.cast("float32")
        if expected.dtype != paddle.float32
        else expected
    )
    if not paddle.allclose(a, e, rtol=rtol, atol=atol).item():
        diff = (a - e).abs()
        raise AssertionError(
            f"{msg}\n  max abs diff: {diff.max().item():.4e}\n"
            f"  max rel diff: {(diff / e.abs().clip(min=1e-12)).max().item():.4e}"
        )


# =========================================================================
# Reference implementations
# =========================================================================


def _ref_csa_indexer_topk(
    index_q, index_k_comp, weights, ratio, topk_effective
):
    scores = paddle.einsum(
        "bshd,btd->bsht", index_q.cast("float32"), index_k_comp.cast("float32")
    )
    scores = F.relu(scores)
    scores = (scores * weights.cast("float32").unsqueeze(-1)).sum(axis=2)
    scores = scores * (index_q.shape[-1] ** -0.5)
    batch, seq_len, seq_len_comp = scores.shape
    comp_ids = paddle.arange(seq_len_comp, dtype="int64").reshape(
        [1, 1, seq_len_comp]
    )
    positions = paddle.arange(1, seq_len + 1, dtype="int64").reshape(
        [1, seq_len, 1]
    )
    valid_end = positions // ratio
    valid_mask = comp_ids < valid_end
    scores = paddle.where(
        valid_mask, scores, paddle.full_like(scores, float("-inf"))
    )
    actual_topk = min(topk_effective, seq_len_comp)
    topk_scores_raw, topk_indices = paddle.topk(scores, k=actual_topk, axis=-1)
    valid_topk = paddle.take_along_axis(
        paddle.expand(valid_mask, [batch, seq_len, seq_len_comp]).cast("int32"),
        topk_indices,
        axis=-1,
    ).cast("bool")
    topk_indices = paddle.where(
        valid_topk, topk_indices, paddle.full_like(topk_indices, -1)
    )
    topk_scores_raw = paddle.where(
        valid_topk,
        topk_scores_raw,
        paddle.full_like(topk_scores_raw, float("-inf")),
    )
    topk_probs = F.softmax(topk_scores_raw, axis=-1)
    topk_probs = paddle.where(
        valid_topk, topk_probs, paddle.zeros_like(topk_probs)
    )
    if topk_effective > actual_topk:
        pad = topk_effective - actual_topk
        topk_indices = paddle.concat(
            [
                topk_indices,
                paddle.full(
                    [batch, seq_len, pad], -1, dtype=topk_indices.dtype
                ),
            ],
            axis=-1,
        )
        topk_probs = paddle.concat(
            [
                topk_probs,
                paddle.zeros([batch, seq_len, pad], dtype=topk_probs.dtype),
            ],
            axis=-1,
        )
    return topk_indices.cast("int32"), topk_probs.cast("float32")


def _paddle_ref_csa_indexer_topk(q, k, weights, ratio, topk_effective):
    from paddleformers.fleet.transformer.dsa_attention import (
        fused_qk_topk_naive,
    )

    b, sq, h_i, d_i = q.shape
    sk = k.shape[1]
    sm_scale = d_i**-0.5
    comp_ids = paddle.arange(sk, dtype="int64").reshape([1, 1, sk])
    valid_end = (
        paddle.arange(1, sq + 1, dtype="int64").reshape([1, sq, 1]) // ratio
    )
    valid_mask = (comp_ids < valid_end).expand([b, sq, sk])
    neg_inf = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    causal_mask = paddle.where(valid_mask, paddle.zeros_like(neg_inf), neg_inf)
    actual_topk = min(int(topk_effective), int(sk))
    index_scores, ref_topk_indices = fused_qk_topk_naive(
        q, k, weights, index_topk=actual_topk, mask=causal_mask
    )
    index_scores_scaled = index_scores * sm_scale
    masked_scaled = index_scores_scaled + causal_mask
    topk_scores_raw, topk_indices = paddle.topk(
        masked_scaled, k=actual_topk, axis=-1
    )
    topk_indices = paddle.clip(topk_indices, min=0, max=sk - 1)
    valid_topk = paddle.take_along_axis(
        valid_mask.cast("int32"), topk_indices, axis=-1
    ).cast("bool")
    topk_indices = paddle.where(
        valid_topk,
        topk_indices.cast("int32"),
        paddle.full_like(topk_indices, -1, dtype="int32"),
    )
    topk_scores_raw = paddle.where(
        valid_topk,
        topk_scores_raw,
        paddle.full_like(topk_scores_raw, float("-inf")),
    )
    row_has_valid = valid_topk.any(axis=-1, keepdim=True)
    safe_scores = paddle.where(
        row_has_valid, topk_scores_raw, paddle.zeros_like(topk_scores_raw)
    )
    topk_probs = F.softmax(safe_scores.cast("float32"), axis=-1)
    topk_probs = paddle.where(
        row_has_valid, topk_probs, paddle.zeros_like(topk_probs)
    )
    topk_probs = paddle.where(
        valid_topk, topk_probs, paddle.zeros_like(topk_probs)
    )
    if int(topk_effective) > actual_topk:
        pad = int(topk_effective) - actual_topk
        topk_indices = paddle.concat(
            [topk_indices, paddle.full([b, sq, pad], -1, dtype="int32")],
            axis=-1,
        )
        topk_probs = paddle.concat(
            [topk_probs, paddle.zeros([b, sq, pad], dtype="float32")], axis=-1
        )
    return topk_indices, topk_probs


# =========================================================================
# Kernel tests
# =========================================================================


class TestTileLangCSAIndexerInterfaceValidation(unittest.TestCase):
    """Raw kernel interfaces should fail before TileLang JIT on bad inputs."""

    def test_forward_rejects_shape_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            csa_indexer_topk_fwd_interface,
        )

        q = paddle.empty([1, 4, 8, 16], dtype="bfloat16")
        k = paddle.empty([2, 1, 16], dtype="bfloat16")
        w = paddle.empty([1, 4, 8], dtype="float32")

        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            csa_indexer_topk_fwd_interface(q, k, w, ratio=4, topk_effective=1)

    def test_backward_rejects_topk_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            csa_indexer_bwd_interface,
        )

        q = paddle.empty([1, 4, 8, 16], dtype="bfloat16")
        k = paddle.empty([1, 1, 16], dtype="bfloat16")
        w = paddle.empty([1, 4, 8], dtype="float32")
        topk_indices = paddle.empty([1, 4, 2], dtype="int32")
        grad_scores = paddle.empty([1, 4, 1], dtype="float32")

        with self.assertRaisesRegex(ValueError, "topk mismatch"):
            csa_indexer_bwd_interface(q, w, k, topk_indices, grad_scores)

    def test_attn_target_rejects_non_power_of_two_dim(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            csa_attn_target_reducesum_interface,
        )

        query = paddle.empty([1, 4, 8, 24], dtype="bfloat16")
        key = paddle.empty([1, 1, 24], dtype="bfloat16")
        topk_indices = paddle.empty([1, 4, 1], dtype="int32")

        with self.assertRaisesRegex(ValueError, "power of 2"):
            csa_attn_target_reducesum_interface(
                query, key, topk_indices, softmax_scale=1.0
            )


class TestTileLangCSAIndexerKernel(unittest.TestCase):
    """Correctness of raw TileLang kernel interfaces."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is required")
        paddle.set_device("gpu")

    def _run_kernel_fwd_case(self, topk_effective):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            csa_indexer_topk_fwd_interface,
        )

        batch, seq_len, seq_len_comp, heads, dim, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(
            batch, seq_len, seq_len_comp, heads, dim, seed=2026
        )
        out_idx, out_scores = csa_indexer_topk_fwd_interface(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk_effective,
            block_K=32,
            num_threads=128,
        )
        ref_idx, ref_scores = _ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertEqual(tuple(out_idx.shape), (batch, seq_len, topk_effective))
        self.assertTrue(paddle.all(out_idx.cpu() == ref_idx.cpu()).item())
        valid = ref_idx >= 0
        paddle.testing.assert_close(
            out_scores.cpu()[valid.cpu()],
            ref_scores.cpu()[valid.cpu()],
            rtol=6e-2,
            atol=2e-2,
        )
        self.assertTrue(
            paddle.all(
                out_scores.cpu()[~valid.cpu()] == ref_scores.cpu()[~valid.cpu()]
            ).item()
        )
        self.assertTrue(paddle.all(out_idx[:, :3, :] == -1).item())

    def test_kernel_fwd_selected_topk(self):
        self._run_kernel_fwd_case(topk_effective=2)

    def test_kernel_fwd_full_candidate(self):
        self._run_kernel_fwd_case(topk_effective=4)

    def test_kernel_fwd_output_padding(self):
        self._run_kernel_fwd_case(topk_effective=6)

    def _ref_csa_indexer_bwd(
        self, index_q, weights, index_k_comp, topk_indices, grad_scores
    ):
        q = index_q.detach().clone()
        q.stop_gradient = False
        w = weights.detach().clone()
        w.stop_gradient = False
        k = index_k_comp.detach().clone()
        k.stop_gradient = False
        scores = paddle.einsum(
            "bshd,btd->bsht", q.cast("float32"), k.cast("float32")
        )
        scores = F.relu(scores * (q.shape[-1] ** -0.5))
        scores = (scores * w.cast("float32").unsqueeze(-1)).sum(axis=2)
        valid = topk_indices >= 0
        safe_indices = paddle.clip(topk_indices, min=0).cast("int64")
        selected = paddle.take_along_axis(scores, safe_indices, axis=-1)
        selected = paddle.where(valid, selected, paddle.zeros_like(selected))
        (selected * grad_scores.cast("float32")).sum().backward()
        return q.grad, w.grad, k.grad

    def _run_kernel_bwd_case(self, topk_effective):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            csa_indexer_bwd_interface,
        )
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            csa_indexer_topk_fwd_interface,
        )

        batch, seq_len, seq_len_comp, heads, dim, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(
            batch, seq_len, seq_len_comp, heads, dim, seed=2027
        )
        topk_indices, _ = csa_indexer_topk_fwd_interface(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk_effective,
            block_K=32,
            num_threads=128,
        )
        grad_scores = paddle.randn(
            [batch, seq_len, topk_effective], dtype="float32"
        )
        grad_scores = paddle.where(
            topk_indices >= 0, grad_scores, paddle.zeros_like(grad_scores)
        ).contiguous()
        out_dq, out_dw, out_dk = csa_indexer_bwd_interface(
            q,
            w,
            k,
            topk_indices.contiguous(),
            grad_scores,
            block_I=32,
            num_threads=128,
        )
        ref_dq, ref_dw, ref_dk = self._ref_csa_indexer_bwd(
            q, w, k, topk_indices, grad_scores
        )
        self.assertEqual(tuple(out_dq.shape), tuple(q.shape))
        paddle.testing.assert_close(
            out_dq.cast("float32").cpu(),
            ref_dq.cast("float32").cpu(),
            rtol=6e-2,
            atol=2e-2,
        )
        paddle.testing.assert_close(
            out_dw.cpu(), ref_dw.cpu(), rtol=6e-2, atol=3e-2
        )
        paddle.testing.assert_close(
            out_dk.cpu(), ref_dk.cast("float32").cpu(), rtol=6e-2, atol=3e-2
        )

    def test_kernel_bwd_selected_topk(self):
        self._run_kernel_bwd_case(topk_effective=2)

    def test_kernel_bwd_full_candidate(self):
        self._run_kernel_bwd_case(topk_effective=4)

    def test_kernel_bwd_output_padding(self):
        self._run_kernel_bwd_case(topk_effective=6)


# =========================================================================
# Wrapper tests
# =========================================================================


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang CSA Indexer kernel requires CUDA",
)
class TestTileLangCSAIndexerWrapperForward(unittest.TestCase):
    def setUp(self):
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

        self._kernel = csa_indexer_topk_fwd

    def test_phase3_selected_topk_matches_reference(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 2
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertEqual(list(out_idx.shape), [b, sq, topk_effective])
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))
        valid = ref_idx >= 0
        self.assertTrue(
            paddle.allclose(
                paddle.masked_select(out_prob, valid),
                paddle.masked_select(ref_prob, valid),
                rtol=8e-2,
                atol=3e-2,
            ).item()
        )

    def test_phase2_full_candidate_matches_reference(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = sk
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))
        valid = ref_idx >= 0
        self.assertTrue(
            paddle.allclose(
                paddle.masked_select(out_prob, valid),
                paddle.masked_select(ref_prob, valid),
                rtol=8e-2,
                atol=3e-2,
            ).item()
        )

    def test_padded_n_compressed_matches_reference(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 6
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertEqual(list(out_idx.shape), [b, sq, topk_effective])
        self.assertTrue(_all_equal(out_idx[:, :, sk:], -1))
        self.assertTrue(_all_equal(out_prob[:, :, sk:], 0))
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))

    def test_causal_t0_t1_t2_have_no_compressed_block(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(q, k, w, ratio=ratio, topk_effective=4)
        self.assertTrue(_all_equal(out_idx[:, :3, :], -1))
        self.assertTrue(_all_equal(out_prob[:, :3, :], 0))

    def test_causal_t3_only_block_zero_visible(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, _ = self._kernel(q, k, w, ratio=ratio, topk_effective=4)
        row = out_idx[0, 3].numpy().tolist()
        self.assertIn(0, row)
        self.assertEqual(sum(int(x == -1) for x in row), 3)

    def test_causal_t7_blocks_zero_and_one_visible(self):
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, _ = self._kernel(q, k, w, ratio=ratio, topk_effective=4)
        row = sorted(out_idx[0, 7].numpy().tolist())
        valid = [x for x in row if x != -1]
        self.assertEqual(sorted(valid), [0, 1])

    def test_short_sequence_with_valid_end_less_than_topk(self):
        b, sq, sk, h_i, d_i, ratio = 1, 8, 4, 64, 128, 4
        topk_effective = 4
        q, k, w = _make_indexer_inputs(b, sq, sk, h_i, d_i)
        out_idx, _ = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, _ = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        for t in range(sq):
            valid_end = (t + 1) // ratio
            row = out_idx[0, t].numpy().tolist()
            n_valid = sum(int(x >= 0) for x in row)
            self.assertEqual(n_valid, min(valid_end, topk_effective))
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang CSA attn_target kernel requires CUDA",
)
class TestCSAAttnTargetReducesum(unittest.TestCase):
    """Isolated correctness tests for csa_attn_target_reducesum kernel."""

    def _run_and_compare(self, b, sq, sk, np_, hn, topk_eff, ratio, seed=2040):
        from paddleformers.fleet.tilelang_ops import (
            csa_attn_target_reducesum,
            csa_indexer_topk_fwd,
        )
        from paddleformers.fleet.transformer.csa_attention import (
            _compute_attn_target_on_selected_set,
        )

        paddle.seed(seed)
        h_i, d_i = 64, 128
        q = paddle.randn([b, sq, h_i, d_i]).astype("bfloat16")
        k = paddle.randn([b, sk, d_i]).astype("bfloat16")
        w = paddle.randn([b, sq, h_i]).astype("float32")
        query_mla = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
        # MLA invariant: compressed key is shared across all heads → [B, S_comp, D]
        key_comp_mla = (
            paddle.randn([b, sk, hn]).astype("bfloat16").contiguous().detach()
        )
        softmax_scale = hn**-0.5

        topk_indices, _ = csa_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk_eff
        )
        tl_target = csa_attn_target_reducesum(
            query_mla, key_comp_mla, topk_indices, softmax_scale
        )
        pd_target = _compute_attn_target_on_selected_set(
            query_mla, key_comp_mla, topk_indices, softmax_scale, None
        )
        _assert_close(
            tl_target,
            pd_target,
            rtol=6e-2,
            atol=2e-2,
            msg=f"target mismatch [b={b},sq={sq},sk={sk},np={np_},topk={topk_eff}]",
        )
        # L1 normalization: valid rows should sum to ~1
        valid = topk_indices >= 0
        row_valid = valid.any(axis=-1)
        if row_valid.any().item():
            row_sums = tl_target[row_valid].sum(axis=-1)
            _assert_close(
                row_sums,
                paddle.ones_like(row_sums),
                rtol=1e-4,
                atol=1e-4,
                msg="L1 normalization violated",
            )
        # Invalid rows should be all-zero
        if (~row_valid).any().item():
            zeros = tl_target[~row_valid]
            self.assertTrue(
                (zeros.cast("float32").abs() < 1e-6).all().item(),
                "invalid rows should be all-zero",
            )

    def test_multi_head_replicate(self):
        """heads=128 (>64) triggers REPLICATE_H=2 path in kernel.

        The kernel splits 128 MLA heads into two groups of 64, computes
        partial softmax per group, then sums and L1-normalizes. Keys are
        head-shared (MLA invariant) so both groups read the same K vector.
        A bug would show as broken partial-sum aggregation.
        """
        self._run_and_compare(
            b=1, sq=16, sk=4, np_=128, hn=128, topk_eff=2, ratio=4, seed=2041
        )

    def test_multi_block_topk_with_padding(self):
        """topk_eff=48 with block_I=32 → padded to 64 → 2 tile iterations.

        This tests: the online softmax correctly accumulates across
        multiple K-blocks rather than just one; (2) the interface layer
        correctly pads topk_indices with -1 to align to block_I and trims
        back to topk_eff=48 in the output; (3) the pad slots with index=-1
        are masked to -inf and contribute zero probability.
        """
        self._run_and_compare(
            b=2, sq=32, sk=16, np_=64, hn=128, topk_eff=48, ratio=4, seed=2042
        )

    def test_online_softmax_numerical_stability(self):
        """Validate online softmax under adversarial numeric conditions.

        Exercises three properties specific to 2-pass online softmax:
        1. Cross-block max shift: block 1 has logits ~10x larger than block 0,
           forcing the rescale exp(old_max - new_max) to a very small value.
           A bug in the online update would produce incorrect probabilities.
        2. All-invalid row: a query position with all topk_indices = -1 must
           produce all-zero output (tests NaN-safe path when row_max = -inf).
        3. Single valid entry: only 1 valid index per row; output must be 1.0
           for that entry (tests exact normalization in degenerate case).
        """
        from paddleformers.fleet.tilelang_ops import csa_attn_target_reducesum
        from paddleformers.fleet.transformer.csa_attention import (
            _compute_attn_target_on_selected_set,
        )

        _cuda_or_skip(self)
        paddle.set_device("gpu")

        b, np_, hn = 1, 64, 128
        sk = 8  # compressed sequence length
        topk = 64  # 2 blocks of block_I=32
        softmax_scale = hn**-0.5

        # Construct keys: first 4 keys are "small", last 4 are "large" (10x scale).
        # This ensures block 1 (indices 4-7) dominates, forcing cross-block rescale.
        paddle.seed(7777)
        key_small = paddle.randn([b, 4, hn]).astype("bfloat16") * 0.1
        key_large = paddle.randn([b, 4, hn]).astype("bfloat16") * 1.0
        key_comp_mla = (
            paddle.concat([key_small, key_large], axis=1).contiguous().detach()
        )

        # --- Case 1: cross-block max shift ---
        # Place small-key indices (0-3) in block 0 (positions 0-3) and large-key
        # indices (4-7) in block 1 (positions 32-35), so that the online softmax
        # must rescale row_sum when it encounters the larger block_max in block 1.
        sq = 4
        query_mla = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
        topk_indices_case1 = paddle.full([b, sq, topk], -1, dtype="int32")
        for t in range(sq):
            topk_indices_case1[0, t, 0:4] = paddle.to_tensor(
                [0, 1, 2, 3], dtype="int32"
            )
            topk_indices_case1[0, t, 32:36] = paddle.to_tensor(
                [4, 5, 6, 7], dtype="int32"
            )
        topk_indices_case1 = topk_indices_case1.contiguous()
        tl_out = csa_attn_target_reducesum(
            query_mla, key_comp_mla, topk_indices_case1, softmax_scale
        )
        pd_out = _compute_attn_target_on_selected_set(
            query_mla, key_comp_mla, topk_indices_case1, softmax_scale, None
        )
        _assert_close(
            tl_out,
            pd_out,
            rtol=5e-3,
            atol=2e-3,
            msg="cross-block max shift mismatch",
        )
        # L1 check
        row_sums = tl_out.sum(axis=-1)
        _assert_close(
            row_sums,
            paddle.ones_like(row_sums),
            rtol=1e-4,
            atol=1e-4,
            msg="L1 norm violated (case 1)",
        )

        # --- Case 2: all-invalid row ---
        sq = 2
        query_mla2 = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
        # Row 0: all invalid; Row 1: has valid entries
        topk_indices_case2 = paddle.full([b, sq, topk], -1, dtype="int32")
        topk_indices_case2[0, 1, 0] = 5
        topk_indices_case2[0, 1, 1] = 6
        topk_indices_case2 = topk_indices_case2.contiguous()
        tl_out2 = csa_attn_target_reducesum(
            query_mla2, key_comp_mla, topk_indices_case2, softmax_scale
        )
        # Row 0 must be all-zero
        self.assertTrue(
            (tl_out2[0, 0].cast("float32").abs() < 1e-6).all().item(),
            "all-invalid row should be zero",
        )
        # Row 1 must sum to 1
        row1_sum = tl_out2[0, 1].sum().item()
        self.assertAlmostEqual(
            row1_sum, 1.0, places=4, msg=f"single-row L1 violated: {row1_sum}"
        )

        # --- Case 3: single valid entry → must be exactly 1.0 ---
        sq = 2
        query_mla3 = paddle.randn([b, sq, np_, hn]).astype("bfloat16").detach()
        topk_indices_case3 = paddle.full([b, sq, topk], -1, dtype="int32")
        topk_indices_case3[0, 0, 0] = 3  # only 1 valid per row
        topk_indices_case3[0, 1, 0] = 7
        topk_indices_case3 = topk_indices_case3.contiguous()
        tl_out3 = csa_attn_target_reducesum(
            query_mla3, key_comp_mla, topk_indices_case3, softmax_scale
        )
        # The single valid slot must have probability = 1.0
        self.assertAlmostEqual(
            tl_out3[0, 0, 0].item(),
            1.0,
            places=4,
            msg="single-valid-entry prob should be 1.0",
        )
        self.assertAlmostEqual(
            tl_out3[0, 1, 0].item(),
            1.0,
            places=4,
            msg="single-valid-entry prob should be 1.0",
        )
        # All other slots must be 0
        self.assertTrue(
            (tl_out3[0, 0, 1:].cast("float32").abs() < 1e-6).all().item(),
            "non-valid slots should be zero",
        )


# =========================================================================
# Input validation tests (cover error-raising branches in wrapper/interface)
# =========================================================================


class TestCSAIndexerInputValidation(unittest.TestCase):
    """Cover TypeError/ValueError branches in csa_indexer.py validation."""

    def test_validate_indexer_inputs_non_tensor_q(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_indexer_inputs,
        )

        k = paddle.empty([1, 4, 16])
        w = paddle.empty([1, 8, 16])
        with self.assertRaises(TypeError):
            _validate_indexer_inputs("not_a_tensor", k, w)

    def test_validate_indexer_inputs_non_tensor_k(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_indexer_inputs,
        )

        q = paddle.empty([1, 8, 16, 32])
        w = paddle.empty([1, 8, 16])
        with self.assertRaises(TypeError):
            _validate_indexer_inputs(q, [1, 2, 3], w)

    def test_validate_indexer_inputs_non_tensor_w(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_indexer_inputs,
        )

        q = paddle.empty([1, 8, 16, 32])
        k = paddle.empty([1, 4, 32])
        with self.assertRaises(TypeError):
            _validate_indexer_inputs(q, k, None)

    def test_validate_indexer_inputs_wrong_q_ndim(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_indexer_inputs,
        )

        q = paddle.empty([1, 8, 16])  # 3D instead of 4D
        k = paddle.empty([1, 4, 16])
        w = paddle.empty([1, 8, 16])
        with self.assertRaises(ValueError):
            _validate_indexer_inputs(q, k, w)

    def test_validate_indexer_inputs_wrong_k_ndim(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_indexer_inputs,
        )

        q = paddle.empty([1, 8, 16, 32])
        k = paddle.empty([1, 4, 32, 1])  # 4D instead of 3D
        w = paddle.empty([1, 8, 16])
        with self.assertRaises(ValueError):
            _validate_indexer_inputs(q, k, w)

    def test_validate_indexer_inputs_wrong_w_ndim(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_indexer_inputs,
        )

        q = paddle.empty([1, 8, 16, 32])
        k = paddle.empty([1, 4, 32])
        w = paddle.empty([1, 8])  # 2D instead of 3D
        with self.assertRaises(ValueError):
            _validate_indexer_inputs(q, k, w)

    def test_validate_indexer_inputs_shape_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_indexer_inputs,
        )

        q = paddle.empty([1, 8, 16, 32])
        k = paddle.empty([1, 4, 32])
        w = paddle.empty([1, 8, 8])  # heads=8 != q heads=16
        with self.assertRaises(ValueError):
            _validate_indexer_inputs(q, k, w)

    def test_validate_topk_and_grad_non_tensor_topk(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_topk_and_grad,
        )

        q = paddle.empty([1, 8, 16, 32])
        grad = paddle.empty([1, 8, 4])
        with self.assertRaises(TypeError):
            _validate_topk_and_grad(q, "not_tensor", grad)

    def test_validate_topk_and_grad_non_tensor_grad(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_topk_and_grad,
        )

        q = paddle.empty([1, 8, 16, 32])
        topk = paddle.empty([1, 8, 4], dtype="int32")
        with self.assertRaises(TypeError):
            _validate_topk_and_grad(q, topk, 42)

    def test_validate_topk_and_grad_wrong_topk_ndim(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_topk_and_grad,
        )

        q = paddle.empty([1, 8, 16, 32])
        topk = paddle.empty([1, 8, 4, 1], dtype="int32")  # 4D
        grad = paddle.empty([1, 8, 4])
        with self.assertRaises(ValueError):
            _validate_topk_and_grad(q, topk, grad)

    def test_validate_topk_and_grad_wrong_grad_ndim(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_topk_and_grad,
        )

        q = paddle.empty([1, 8, 16, 32])
        topk = paddle.empty([1, 8, 4], dtype="int32")
        grad = paddle.empty([1, 8])  # 2D
        with self.assertRaises(ValueError):
            _validate_topk_and_grad(q, topk, grad)

    def test_validate_topk_and_grad_shape_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_topk_and_grad,
        )

        q = paddle.empty([1, 8, 16, 32])
        topk = paddle.empty([1, 8, 4], dtype="int32")
        grad = paddle.empty([1, 8, 3])  # topk_dim=3 != 4
        with self.assertRaises(ValueError):
            _validate_topk_and_grad(q, topk, grad)

    def test_validate_topk_and_grad_batch_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _validate_topk_and_grad,
        )

        q = paddle.empty([1, 8, 16, 32])
        topk = paddle.empty([2, 8, 4], dtype="int32")  # batch=2 != 1
        grad = paddle.empty([2, 8, 4])
        with self.assertRaises(ValueError):
            _validate_topk_and_grad(q, topk, grad)

    def test_prepare_forward_inputs_zero_topk(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _prepare_forward_inputs,
        )

        q = paddle.empty([1, 8, 16, 32])
        k = paddle.empty([1, 4, 32])
        w = paddle.empty([1, 8, 16])
        with self.assertRaises(ValueError):
            _prepare_forward_inputs(q, k, w, topk_effective=0)

    def test_prepare_forward_inputs_casts_weights(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _prepare_forward_inputs,
        )

        q = paddle.empty([1, 8, 16, 32])
        k = paddle.empty([1, 4, 32])
        w = paddle.empty([1, 8, 16], dtype="float16")  # not fp32
        _, _, w_out, _ = _prepare_forward_inputs(q, k, w, topk_effective=2)
        self.assertEqual(w_out.dtype, paddle.float32)

    def test_prepare_backward_inputs_casts_types(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            _prepare_backward_inputs,
        )

        q = paddle.empty([1, 8, 16, 32])
        k = paddle.empty([1, 4, 32])
        w = paddle.empty([1, 8, 16], dtype="float16")
        topk = paddle.empty([1, 8, 2], dtype="int64")  # not int32
        grad = paddle.empty([1, 8, 2], dtype="float16")  # not fp32
        _, w_out, _, topk_out, grad_out = _prepare_backward_inputs(
            q, w, k, topk, grad
        )
        self.assertEqual(w_out.dtype, paddle.float32)
        self.assertEqual(topk_out.dtype, paddle.int32)
        self.assertEqual(grad_out.dtype, paddle.float32)

    def test_csa_attn_target_wrapper_type_checks(self):
        """Cover TypeError branches in csa_attn_target_reducesum wrapper."""
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            csa_attn_target_reducesum,
        )

        q = paddle.empty([1, 4, 8, 32])
        k = paddle.empty([1, 2, 32])
        topk = paddle.empty([1, 4, 2], dtype="int32")

        with self.assertRaises(TypeError):
            csa_attn_target_reducesum("bad", k, topk, 1.0)
        with self.assertRaises(TypeError):
            csa_attn_target_reducesum(q, "bad", topk, 1.0)
        with self.assertRaises(TypeError):
            csa_attn_target_reducesum(q, k, "bad", 1.0)

    def test_csa_attn_target_wrapper_ndim_checks(self):
        """Cover ValueError branches for ndim in csa_attn_target_reducesum."""
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            csa_attn_target_reducesum,
        )

        with self.assertRaises(ValueError):
            csa_attn_target_reducesum(
                paddle.empty([1, 4, 32]),  # 3D query
                paddle.empty([1, 2, 32]),
                paddle.empty([1, 4, 2], dtype="int32"),
                1.0,
            )
        with self.assertRaises(ValueError):
            csa_attn_target_reducesum(
                paddle.empty([1, 4, 8, 32]),
                paddle.empty([1, 2, 32, 1]),  # 4D key
                paddle.empty([1, 4, 2], dtype="int32"),
                1.0,
            )
        with self.assertRaises(ValueError):
            csa_attn_target_reducesum(
                paddle.empty([1, 4, 8, 32]),
                paddle.empty([1, 2, 32]),
                paddle.empty([1, 4, 2, 1], dtype="int32"),  # 4D topk
                1.0,
            )

    def test_csa_attn_target_wrapper_batch_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            csa_attn_target_reducesum,
        )

        with self.assertRaises(ValueError):
            csa_attn_target_reducesum(
                paddle.empty([1, 4, 8, 32]),
                paddle.empty([2, 2, 32]),  # batch=2
                paddle.empty([1, 4, 2], dtype="int32"),
                1.0,
            )

    def test_csa_attn_target_wrapper_seq_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            csa_attn_target_reducesum,
        )

        with self.assertRaises(ValueError):
            csa_attn_target_reducesum(
                paddle.empty([1, 4, 8, 32]),
                paddle.empty([1, 2, 32]),
                paddle.empty([1, 3, 2], dtype="int32"),  # seq=3 != 4
                1.0,
            )

    def test_csa_attn_target_wrapper_dim_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer import (
            csa_attn_target_reducesum,
        )

        with self.assertRaises(ValueError):
            csa_attn_target_reducesum(
                paddle.empty([1, 4, 8, 32]),
                paddle.empty([1, 2, 16]),  # dim=16 != 32
                paddle.empty([1, 4, 2], dtype="int32"),
                1.0,
            )


class TestCSAIndexerFwdInterfaceValidation(unittest.TestCase):
    """Cover error branches in csa_indexer_fwd.py interface validation."""

    def test_rejects_non_contiguous(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([2, 4, 8, 16], dtype="bfloat16").transpose(
            [0, 2, 1, 3]
        )  # non-contiguous
        k = paddle.empty([2, 2, 16], dtype="bfloat16")
        w = paddle.empty([2, 4, 8], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, w, 2, 32, 0)

    def test_rejects_wrong_ndim(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([2, 4, 128], dtype="bfloat16")  # 3D
        k = paddle.empty([2, 2, 128], dtype="bfloat16")
        w = paddle.empty([2, 4, 64], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, w, 2, 32, 0)

    def test_rejects_heads_not_multiple_of_8(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 7, 16], dtype="bfloat16")  # heads=7
        k = paddle.empty([1, 2, 16], dtype="bfloat16")
        w = paddle.empty([1, 4, 7], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, w, 2, 32, 0)

    def test_rejects_zero_topk(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 16], dtype="bfloat16")
        k = paddle.empty([1, 2, 16], dtype="bfloat16")
        w = paddle.empty([1, 4, 8], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, w, 0, 32, 0)

    def test_rejects_nonzero_num_stages(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 16], dtype="bfloat16")
        k = paddle.empty([1, 2, 16], dtype="bfloat16")
        w = paddle.empty([1, 4, 8], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, w, 2, 32, 1)

    def test_tilelang_dtype_fp16(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            _tilelang_dtype,
        )

        t = paddle.empty([1], dtype="float16")
        self.assertEqual(_tilelang_dtype(t), "float16")

    def test_tilelang_dtype_rejects_fp32(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            _tilelang_dtype,
        )

        t = paddle.empty([1], dtype="float32")
        with self.assertRaises(TypeError):
            _tilelang_dtype(t)

    def test_next_power_of_2_edge_cases(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_fwd import (
            _next_power_of_2,
        )

        self.assertEqual(_next_power_of_2(0), 1)
        self.assertEqual(_next_power_of_2(1), 1)
        self.assertEqual(_next_power_of_2(3), 4)
        self.assertEqual(_next_power_of_2(32), 32)
        self.assertEqual(_next_power_of_2(33), 64)


class TestCSAIndexerBwdInterfaceValidation(unittest.TestCase):
    """Cover error branches in csa_indexer_bwd.py interface validation."""

    def test_rejects_non_contiguous(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 16], dtype="bfloat16").transpose(
            [0, 2, 1, 3]
        )
        w = paddle.empty([1, 4, 8], dtype="float32")
        k = paddle.empty([1, 2, 16], dtype="bfloat16")
        topk = paddle.empty([1, 4, 2], dtype="int32")
        grad = paddle.empty([1, 4, 2], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, w, k, topk, grad, 32, 0)

    def test_rejects_wrong_ndim(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 128], dtype="bfloat16")  # 3D
        w = paddle.empty([1, 4, 64], dtype="float32")
        k = paddle.empty([1, 2, 128], dtype="bfloat16")
        topk = paddle.empty([1, 4, 2], dtype="int32")
        grad = paddle.empty([1, 4, 2], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, w, k, topk, grad, 32, 0)

    def test_rejects_heads_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 16], dtype="bfloat16")
        w = paddle.empty([1, 4, 16], dtype="float32")  # heads=16 != 8
        k = paddle.empty([1, 2, 16], dtype="bfloat16")
        topk = paddle.empty([1, 4, 2], dtype="int32")
        grad = paddle.empty([1, 4, 2], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, w, k, topk, grad, 32, 0)

    def test_rejects_dim_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 16], dtype="bfloat16")
        w = paddle.empty([1, 4, 8], dtype="float32")
        k = paddle.empty([1, 2, 32], dtype="bfloat16")  # dim=32 != 16
        topk = paddle.empty([1, 4, 2], dtype="int32")
        grad = paddle.empty([1, 4, 2], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, w, k, topk, grad, 32, 0)

    def test_rejects_topk_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 16], dtype="bfloat16")
        w = paddle.empty([1, 4, 8], dtype="float32")
        k = paddle.empty([1, 2, 16], dtype="bfloat16")
        topk = paddle.empty([1, 4, 2], dtype="int32")
        grad = paddle.empty([1, 4, 3], dtype="float32")  # topk=3 != 2
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, w, k, topk, grad, 32, 0)

    def test_rejects_nonzero_num_stages(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 16], dtype="bfloat16")
        w = paddle.empty([1, 4, 8], dtype="float32")
        k = paddle.empty([1, 2, 16], dtype="bfloat16")
        topk = paddle.empty([1, 4, 2], dtype="int32")
        grad = paddle.empty([1, 4, 2], dtype="float32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, w, k, topk, grad, 32, 1)

    def test_tilelang_dtype_fp16(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            _tilelang_dtype,
        )

        t = paddle.empty([1], dtype="float16")
        self.assertEqual(_tilelang_dtype(t), "float16")

    def test_tilelang_dtype_rejects_fp32(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_indexer_bwd import (
            _tilelang_dtype,
        )

        t = paddle.empty([1], dtype="float32")
        with self.assertRaises(TypeError):
            _tilelang_dtype(t)


class TestCSAAttnTargetInterfaceValidation(unittest.TestCase):
    """Cover error branches in csa_attn_target.py interface validation."""

    def test_rejects_non_contiguous(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 32], dtype="bfloat16").transpose(
            [0, 2, 1, 3]
        )
        k = paddle.empty([1, 2, 32], dtype="bfloat16")
        topk = paddle.empty([1, 4, 2], dtype="int32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, topk, 32, 0)

    def test_rejects_wrong_ndim(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 32], dtype="bfloat16")  # 3D
        k = paddle.empty([1, 2, 32], dtype="bfloat16")
        topk = paddle.empty([1, 4, 2], dtype="int32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, topk, 32, 0)

    def test_rejects_dim_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 32], dtype="bfloat16")
        k = paddle.empty([1, 2, 16], dtype="bfloat16")  # dim=16 != 32
        topk = paddle.empty([1, 4, 2], dtype="int32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, topk, 32, 0)

    def test_rejects_batch_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 32], dtype="bfloat16")
        k = paddle.empty([2, 2, 32], dtype="bfloat16")  # batch=2
        topk = paddle.empty([1, 4, 2], dtype="int32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, topk, 32, 0)

    def test_rejects_seq_mismatch(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 32], dtype="bfloat16")
        k = paddle.empty([1, 2, 32], dtype="bfloat16")
        topk = paddle.empty([1, 3, 2], dtype="int32")  # seq=3 != 4
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, topk, 32, 0)

    def test_rejects_heads_not_multiple_of_64(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 65, 32], dtype="bfloat16")  # heads=65
        k = paddle.empty([1, 2, 32], dtype="bfloat16")
        topk = paddle.empty([1, 4, 2], dtype="int32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, topk, 32, 0)

    def test_rejects_nonzero_num_stages(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            _validate_interface_inputs,
        )

        q = paddle.empty([1, 4, 8, 32], dtype="bfloat16")
        k = paddle.empty([1, 2, 32], dtype="bfloat16")
        topk = paddle.empty([1, 4, 2], dtype="int32")
        with self.assertRaises(ValueError):
            _validate_interface_inputs(q, k, topk, 32, 1)

    def test_tilelang_dtype_fp16(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            _tilelang_dtype,
        )

        t = paddle.empty([1], dtype="float16")
        self.assertEqual(_tilelang_dtype(t), "float16")

    def test_tilelang_dtype_rejects_fp32(self):
        from paddleformers.fleet.tilelang_ops.indexer.csa_attn_target import (
            _tilelang_dtype,
        )

        t = paddle.empty([1], dtype="float32")
        with self.assertRaises(TypeError):
            _tilelang_dtype(t)


# =========================================================================
# TileLangCSAIndexerLossAutoScaler backward path test
# =========================================================================


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang CSA Indexer kernel requires CUDA",
)
class TestTileLangCSAIndexerLossAutoScaler(unittest.TestCase):
    """Test that TileLangCSAIndexerLossAutoScaler correctly backprops.

    This covers the backward() method of TileLangCSAIndexerLossAutoScaler
    (csa_attention.py lines 609-646) which hooks indexer loss gradients into
    the main output's backward graph.
    """

    def test_auto_scaler_backward_produces_correct_grads(self):
        from paddleformers.fleet.transformer.csa_attention import (
            TileLangCSAIndexerLossAutoScaler,
        )
        from paddleformers.fleet.transformer.dsa_attention import (
            DSAIndexerLossAutoScaler,
        )

        paddle.set_device("gpu")
        b, sq, h_i, d_i, sk = 1, 16, 64, 128, 4
        ratio = 4
        loss_coeff = 1.0

        paddle.seed(5050)
        index_q = paddle.randn([b, sq, h_i, d_i]).astype("bfloat16")
        index_k_comp = paddle.randn([b, sk, d_i]).astype("bfloat16")
        weights = paddle.randn([b, sq, h_i]).astype("float32")

        # Get topk and compute target to build the state tuple
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

        topk_indices, topk_probs = csa_indexer_topk_fwd(
            index_q,
            index_k_comp,
            weights,
            ratio=ratio,
            topk_effective=sk,
        )

        # Simulate target (random normalized distribution)
        paddle.seed(5051)
        target_raw = paddle.rand([b, sq, sk])
        target_raw = paddle.where(
            topk_indices >= 0, target_raw, paddle.zeros_like(target_raw)
        )
        target = target_raw / target_raw.sum(axis=-1, keepdim=True).clip(
            min=1e-10
        )

        # Make non-leaf tensors by passing through identity ops.
        # Paddle PyLayer doesn't allow inplace strategy on leaf vars.
        output_leaf = paddle.randn([b, sq, 128]).astype("bfloat16")
        output_leaf.stop_gradient = False
        output = output_leaf + 0  # non-leaf

        index_q_leaf = index_q.detach()
        index_q_leaf.stop_gradient = False
        index_q_d = index_q_leaf + 0  # non-leaf

        weights_leaf = weights.detach()
        weights_leaf.stop_gradient = False
        weights_d = weights_leaf + 0  # non-leaf

        index_k_leaf = index_k_comp.detach()
        index_k_leaf.stop_gradient = False
        index_k_d = index_k_leaf + 0  # non-leaf

        DSAIndexerLossAutoScaler._main_loss_backward_scale = None
        result = TileLangCSAIndexerLossAutoScaler.apply(
            output,
            index_q_d,
            weights_d,
            index_k_d,
            topk_indices.detach(),
            topk_probs.detach(),
            target.detach(),
            loss_coeff,
        )
        # Backward through the auto-scaler
        result.sum().backward()

        # Verify: output grad should be passed through (identity)
        self.assertTrue(
            paddle.allclose(
                output_leaf.grad.cast("float32"),
                paddle.ones_like(output_leaf).cast("float32"),
                atol=1e-5,
            ).item(),
            "output gradient should be ones (identity pass-through)",
        )
        # Verify: indexer param grads should be non-zero
        self.assertGreater(
            index_q_leaf.grad.cast("float32").abs().max().item(),
            0.0,
            "index_q grad should be non-zero",
        )
        self.assertGreater(
            weights_leaf.grad.cast("float32").abs().max().item(),
            0.0,
            "weights grad should be non-zero",
        )
        self.assertGreater(
            index_k_leaf.grad.cast("float32").abs().max().item(),
            0.0,
            "index_k grad should be non-zero",
        )
        # Verify: no NaN/Inf
        for name, g in [
            ("dQ", index_q_leaf.grad),
            ("dW", weights_leaf.grad),
            ("dK", index_k_leaf.grad),
        ]:
            self.assertTrue(
                paddle.isfinite(g.cast("float32")).all().item(),
                f"{name} contains NaN/Inf",
            )


# =========================================================================
# TileLang forward-only (no loss) path in CompressedSparseAttention.forward()
# =========================================================================


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang CSA kernel requires CUDA",
)
class TestCSAForwardTileLangFwdOnlyPath(unittest.TestCase):
    """Cover the TileLang fwd-only indexer path (csa_attention.py ~1209-1226).

    When csa_tilelang_enable_indexer=True but training=False (or loss_coeff=0),
    the code enters the fwd-only branch that calls csa_indexer_topk_fwd under
    no_grad and uses its indices for sparse attention.
    """

    def test_fwd_only_tilelang_matches_paddle_indexer(self):
        import types

        from paddle import nn
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )
        from paddleformers.fleet.transformer.csa_attention import (
            CompressedSparseAttention,
            CompressedSparseAttentionSublayersSpec,
            Compressor,
            CompressorSublayersSpec,
            CSAIndexer,
            CSAIndexerSublayersSpec,
        )

        paddle.set_device("gpu")

        class _Lin(nn.Layer):
            def __init__(self, input_size, output_size, **kwargs):
                super().__init__()
                self.weight = self.create_parameter(
                    shape=[output_size, input_size],
                    dtype="bfloat16",
                    default_initializer=nn.initializer.Normal(std=0.02),
                )

            def forward(self, x):
                return paddle.matmul(x, self.weight.T), None

        class _Norm(nn.Layer):
            def __init__(self, hidden_size=None, **kwargs):
                super().__init__()
                self.weight = self.create_parameter(
                    shape=[hidden_size],
                    dtype="float32",
                    default_initializer=nn.initializer.Constant(1.0),
                )

            def forward(self, x):
                return (
                    x
                    * paddle.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
                    * self.weight.cast(x.dtype)
                )

        head_dim, hidden_size, ratio = 64, 256, 4
        config = types.SimpleNamespace(
            num_attention_heads=8,
            v_head_dim=head_dim,
            hidden_size=hidden_size,
            q_lora_rank=64,
            qk_pos_emb_head_dim=32,
            csa_window_size=64,
            csa_compress_ratios=[ratio],
            csa_dense_mode=False,
            dsa_index_n_heads=16,
            dsa_index_head_dim=32,
            dsa_index_topk=16,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_use_sparse_loss=False,
            csa_tilelang_enable_indexer=True,
            csa_sparse_attn_backend="unfused",
            csa_indexer_backend="tilelang",
            init_method=None,
            init_method_std=0.02,
            layernorm_epsilon=1e-5,
            num_hidden_layers=1,
        )
        rope = RotaryEmbedding(32, rotary_percent=1.0, rotary_base=160000)

        comp_spec = CompressorSublayersSpec(
            linear_wkv=_Lin, linear_wgate=_Lin, norm=_Norm
        )
        idx_comp_spec = CompressorSublayersSpec(
            linear_wkv=_Lin, linear_wgate=_Lin, norm=_Norm
        )
        idx_spec = CSAIndexerSublayersSpec(
            linear_wq_b=_Lin,
            linear_weights_proj=_Lin,
            compressor=LayerSpec(
                layer=Compressor, sublayers_spec=idx_comp_spec
            ),
        )
        attn_spec = CompressedSparseAttentionSublayersSpec(
            compressor=LayerSpec(layer=Compressor, sublayers_spec=comp_spec),
            indexer=LayerSpec(layer=CSAIndexer, sublayers_spec=idx_spec),
        )

        # Build two instances with same seed: one with tilelang, one without
        paddle.seed(9999)
        csa_tl = CompressedSparseAttention(
            config=config,
            sublayers_spec=attn_spec,
            layer_number=1,
            attn_mask_type=None,
            attention_type="self",
            k_channels=head_dim,
            v_channels=head_dim,
            compress_ratio=ratio,
            rotary_pos_emb=rope,
        )
        csa_tl.eval()

        config_no_tl = types.SimpleNamespace(**vars(config))
        config_no_tl.csa_tilelang_enable_indexer = False

        paddle.seed(9999)
        csa_ref = CompressedSparseAttention(
            config=config_no_tl,
            sublayers_spec=attn_spec,
            layer_number=1,
            attn_mask_type=None,
            attention_type="self",
            k_channels=head_dim,
            v_channels=head_dim,
            compress_ratio=ratio,
            rotary_pos_emb=rope,
        )
        csa_ref.eval()

        # Run forward
        b, sq = 1, 64
        paddle.seed(8888)
        query = paddle.randn([b, sq, 8, head_dim], dtype="bfloat16")
        key = paddle.randn([b, sq, 1, head_dim], dtype="bfloat16")
        x = paddle.randn([b, sq, hidden_size], dtype="bfloat16")
        qr = paddle.randn([b, sq, 64], dtype="bfloat16")

        out_tl = csa_tl(query, key, key, None, x=x, qr=qr)
        out_ref = csa_ref(query, key, key, None, x=x, qr=qr)

        # They should match (same indexer weights → same topk → same attention)
        diff = (
            (out_tl.cast("float32") - out_ref.cast("float32"))
            .abs()
            .max()
            .item()
        )
        self.assertLess(
            diff, 0.1, f"TileLang fwd-only vs Paddle mismatch: {diff}"
        )


if __name__ == "__main__":
    unittest.main()
