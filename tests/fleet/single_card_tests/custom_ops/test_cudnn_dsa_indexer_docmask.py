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

"""Unit tests for the cuDNN CSA indexer document-mask helpers.

Covers the pure-Paddle helpers in
``paddleformers.fleet.cudnn_ops.indexer.docmask_utils``:

* topk_local_to_global / topk_global_to_local: per-document-local <-> global
  compressed-buffer id conversion, including the round-trip identity, the
  ``-1`` invalid-slot preservation, and the multi-document example from the
  task description.
* thd_to_bshd_b1 / bshd_b1_to_thd: packed-THD <-> BSHD(b==1, padded) layout
  conversion, including pad/de-pad round-trips and pad_value correctness.

These run on any device (CPU/GPU) -- no GPU kernel dependency.
"""

import unittest
from unittest.mock import patch

import paddle

from paddleformers.fleet.cudnn_ops.indexer.docmask_utils import (
    bshd_b1_to_thd,
    shift_scores_to_local_window,
    thd_to_bshd_b1,
    topk_global_to_local,
    topk_local_to_global,
    valid_range_to_counts,
)


class TestCudnnDocmaskMetadataReuse(unittest.TestCase):
    def test_thd_fast_path_uses_precomputed_doc_lens(self):
        from paddleformers.fleet.cudnn_ops.indexer import csa_indexer_fwd_cudnn as mod

        index_q = paddle.zeros([1, 8, 32, 128], dtype="bfloat16")
        index_k_comp = paddle.zeros([1, 2, 128], dtype="bfloat16")
        weights = paddle.zeros([1, 8, 32], dtype="bfloat16")
        valid_range = paddle.zeros([1, 8, 2], dtype="int32")
        startend_row_indices = paddle.to_tensor(
            [999] * 8, dtype="int32"
        ).reshape([1, 1, 8, 1])

        with patch.object(
            mod,
            "_doc_lens_from_startend",
            side_effect=AssertionError("should reuse doc_lens"),
        ):
            result = mod._cudnn_indexer_topk_fwd_docmask_thd(
                index_q,
                index_k_comp,
                weights,
                ratio=4,
                topk_effective=2,
                sm_scale=1.0,
                valid_range=valid_range,
                startend_row_indices=startend_row_indices,
                doc_lens=[4, 4],
            )

        self.assertIsNone(result)

    def test_thd_fast_path_rejects_mismatched_global_startend(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            _cudnn_indexer_topk_fwd_docmask_thd,
        )

        ratio, sq_local, sq_global, h, d = 4, 64, 128, 32, 128
        index_q = paddle.zeros([1, sq_local, h, d], dtype="bfloat16")
        index_k_comp = paddle.zeros(
            [1, sq_global // ratio, d], dtype="bfloat16"
        )
        weights = paddle.zeros([1, sq_local, h], dtype="bfloat16")
        valid_range = paddle.zeros([1, sq_local, 2], dtype="int32")
        startend_row_indices = paddle.full(
            [1, 1, sq_global, 1], sq_global, dtype="int32"
        )

        result = _cudnn_indexer_topk_fwd_docmask_thd(
            index_q,
            index_k_comp,
            weights,
            ratio=ratio,
            topk_effective=8,
            sm_scale=1.0,
            valid_range=valid_range,
            startend_row_indices=startend_row_indices,
            doc_lens=[sq_global],
        )
        self.assertIsNone(result)


class TestTopkLocalGlobal(unittest.TestCase):
    """topk_local_to_global / topk_global_to_local."""

    def test_multi_doc_example_local_to_global(self):
        """Task example: 3 docs, per-doc-local topk -> global flat ids.

        doc_col_start = [0, 3, 5] (doc0 has 3 compressed cols, doc1 has 2).
        local  [0,1,2 | 0,2,3 | 0,4,7]
        global [0,1,2 | 3,5,6 | 5,9,12]
        """
        # 3 queries, one per document, topk=3.
        topk_local = paddle.to_tensor(
            [[0, 1, 2], [0, 2, 3], [0, 4, 7]], dtype="int32"
        )
        valid_range = paddle.to_tensor([[0, 3], [3, 5], [5, 13]], dtype="int32")
        out = topk_local_to_global(topk_local, valid_range)
        expected = paddle.to_tensor(
            [[0, 1, 2], [3, 5, 6], [5, 9, 12]], dtype="int32"
        )
        self.assertTrue(paddle.equal_all(out, expected).item())

    def test_global_to_local_inverse(self):
        topk_global = paddle.to_tensor(
            [[0, 1, 2], [3, 5, 6], [5, 9, 12]], dtype="int32"
        )
        valid_range = paddle.to_tensor([[0, 3], [3, 5], [5, 13]], dtype="int32")
        out = topk_global_to_local(topk_global, valid_range)
        expected = paddle.to_tensor(
            [[0, 1, 2], [0, 2, 3], [0, 4, 7]], dtype="int32"
        )
        self.assertTrue(paddle.equal_all(out, expected).item())

    def test_round_trip_bshd(self):
        """local -> global -> local is identity on a [B, S, topk] tensor."""
        paddle.seed(0)
        b, s, topk = 2, 16, 8
        doc_start = paddle.randint(0, 100, [b, s, 1]).astype("int32")
        valid_range = paddle.concat([doc_start, doc_start + 50], axis=-1)
        local = paddle.randint(0, 50, [b, s, topk]).astype("int32")
        g = topk_local_to_global(local, valid_range)
        back = topk_global_to_local(g, valid_range)
        self.assertTrue(paddle.equal_all(back, local).item())

    def test_invalid_slots_preserved(self):
        """-1 slots stay -1 through both directions."""
        topk_local = paddle.to_tensor([[0, -1, 2], [-1, -1, 3]], dtype="int32")
        valid_range = paddle.to_tensor([[5, 10], [3, 9]], dtype="int32")
        g = topk_local_to_global(topk_local, valid_range)
        expected_g = paddle.to_tensor([[5, -1, 7], [-1, -1, 6]], dtype="int32")
        self.assertTrue(paddle.equal_all(g, expected_g).item())
        back = topk_global_to_local(g, valid_range)
        self.assertTrue(paddle.equal_all(back, topk_local).item())

    def test_zero_offset_is_identity(self):
        """doc_col_start == 0 (single-doc) leaves valid ids unchanged."""
        topk = paddle.to_tensor([[0, 5, 9], [1, -1, 7]], dtype="int32")
        valid_range = paddle.to_tensor([[0, 9], [0, 9]], dtype="int32")
        out = topk_local_to_global(topk, valid_range)
        self.assertTrue(paddle.equal_all(out, topk).item())

    def test_dtype_preserved(self):
        topk = paddle.to_tensor([[0, 1]], dtype="int32")
        valid_range = paddle.to_tensor([[4, 8]], dtype="int32")
        out = topk_local_to_global(topk, valid_range)
        self.assertEqual(out.dtype, paddle.int32)

    def test_shape_mismatch_raises(self):
        topk = paddle.to_tensor([[0, 1, 2]], dtype="int32")
        bad_valid = paddle.to_tensor([[0, 3], [1, 4]], dtype="int32")
        with self.assertRaises(ValueError):
            topk_local_to_global(topk, bad_valid)

    def test_valid_range_last_dim_must_be_2(self):
        topk = paddle.to_tensor([[0, 1, 2]], dtype="int32")
        bad_valid = paddle.to_tensor([[0, 3, 5]], dtype="int32")
        with self.assertRaises(ValueError):
            topk_local_to_global(topk, bad_valid)


class TestThdBshdConversion(unittest.TestCase):
    """thd_to_bshd_b1 / bshd_b1_to_thd."""

    def test_thd_to_bshd_pads_and_adds_batch(self):
        thd = paddle.arange(6).reshape([3, 2]).astype("float32")
        out = thd_to_bshd_b1(thd, pad_len=5, pad_value=0)
        self.assertEqual(out.shape, [1, 5, 2])
        # body preserved
        self.assertTrue(paddle.equal_all(out[0, :3], thd).item())
        # padding zeros
        self.assertTrue(
            paddle.equal_all(out[0, 3:], paddle.zeros([2, 2])).item()
        )

    def test_thd_to_bshd_pad_value_minus_one(self):
        thd = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype="int32")
        out = thd_to_bshd_b1(thd, pad_len=4, pad_value=-1)
        self.assertEqual(out.shape, [1, 4, 3])
        self.assertTrue(
            paddle.equal_all(
                out[0, 2:], paddle.full([2, 3], -1, dtype="int32")
            ).item()
        )

    def test_thd_to_bshd_no_pad_when_equal(self):
        thd = paddle.arange(4).reshape([2, 2]).astype("float32")
        out = thd_to_bshd_b1(thd, pad_len=2)
        self.assertEqual(out.shape, [1, 2, 2])
        self.assertTrue(paddle.equal_all(out[0], thd).item())

    def test_thd_to_bshd_pad_len_too_small_raises(self):
        thd = paddle.zeros([5, 2], dtype="float32")
        with self.assertRaises(ValueError):
            thd_to_bshd_b1(thd, pad_len=3)

    def test_bshd_to_thd_strips_padding_and_batch(self):
        bshd = paddle.arange(10).reshape([1, 5, 2]).astype("float32")
        out = bshd_b1_to_thd(bshd, total_len=3)
        self.assertEqual(out.shape, [3, 2])
        self.assertTrue(paddle.equal_all(out, bshd[0, :3]).item())

    def test_bshd_to_thd_requires_batch_one(self):
        bshd = paddle.zeros([2, 5, 2], dtype="float32")
        with self.assertRaises(ValueError):
            bshd_b1_to_thd(bshd, total_len=3)

    def test_bshd_to_thd_total_len_too_large_raises(self):
        bshd = paddle.zeros([1, 4, 2], dtype="float32")
        with self.assertRaises(ValueError):
            bshd_b1_to_thd(bshd, total_len=5)

    def test_round_trip_thd_bshd_thd(self):
        """THD -> BSHD(pad) -> THD recovers the original packed tensor."""
        paddle.seed(1)
        thd = paddle.randn([7, 4, 3]).astype("float32")  # [T, H, D]
        bshd = thd_to_bshd_b1(thd, pad_len=16, pad_value=0)
        self.assertEqual(bshd.shape, [1, 16, 4, 3])
        back = bshd_b1_to_thd(bshd, total_len=7)
        self.assertTrue(paddle.equal_all(back, thd).item())

    def test_round_trip_preserves_indices_dtype(self):
        thd = paddle.randint(0, 100, [5, 8]).astype("int32")
        bshd = thd_to_bshd_b1(thd, pad_len=8, pad_value=-1)
        back = bshd_b1_to_thd(bshd, total_len=5)
        self.assertEqual(back.dtype, paddle.int32)
        self.assertTrue(paddle.equal_all(back, thd).item())


class TestValidRangeToCounts(unittest.TestCase):
    """valid_range_to_counts."""

    def test_basic_counts(self):
        vr = paddle.to_tensor(
            [[[0, 3], [3, 5], [5, 13]]], dtype="int32"
        )  # [1, 3, 2]
        counts = valid_range_to_counts(vr)
        self.assertEqual(counts.shape, [1, 3])
        self.assertTrue(
            paddle.equal_all(
                counts, paddle.to_tensor([[3, 2, 8]], dtype="int32")
            ).item()
        )

    def test_empty_range_clamped_to_zero(self):
        # zeroed-out padding rows (start == end == 0) -> count 0.
        vr = paddle.to_tensor([[[0, 0], [4, 4]]], dtype="int32")
        counts = valid_range_to_counts(vr)
        self.assertTrue(
            paddle.equal_all(
                counts, paddle.to_tensor([[0, 0]], dtype="int32")
            ).item()
        )

    def test_dtype_is_int32(self):
        vr = paddle.to_tensor([[2, 7]], dtype="int64")
        counts = valid_range_to_counts(vr)
        self.assertEqual(counts.dtype, paddle.int32)

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            valid_range_to_counts(paddle.zeros([3, 3], dtype="int32"))


class TestShiftScoresToLocalWindow(unittest.TestCase):
    """shift_scores_to_local_window."""

    def test_left_aligns_window(self):
        # one query, valid window [3, 5) over 8 compressed cols.
        scores = paddle.to_tensor(
            [[[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]]],
            dtype="float32",
        )
        vr = paddle.to_tensor([[[3, 5]]], dtype="int32")
        local, counts = shift_scores_to_local_window(scores, vr)
        self.assertTrue(
            paddle.equal_all(
                counts, paddle.to_tensor([[2]], dtype="int32")
            ).item()
        )
        # cols 3,4 (=13,14) move to front; tail is -inf.
        row = local.numpy()[0, 0]
        self.assertEqual(row[0], 13.0)
        self.assertEqual(row[1], 14.0)
        self.assertTrue((row[2:] == float("-inf")).all())

    def test_topk_after_shift_then_remap_matches_global(self):
        """End-to-end (no kernel): argmax within local window -> global id.

        Picks the single largest valid score per query directly (a stand-in
        for the radix top-k), maps it back to global ids, and checks it equals
        the true global argmax restricted to the valid window.
        """
        paddle.seed(7)
        b, s, sk = 1, 4, 12
        scores = paddle.randn([b, s, sk]).astype("float32")
        # three docs: cols [0,4), [4,8), [8,12); one query per doc + one extra.
        vr = paddle.to_tensor(
            [[[0, 4], [4, 8], [8, 12], [4, 8]]], dtype="int32"
        )
        local, counts = shift_scores_to_local_window(scores, vr)
        # local argmax (top-1) over the left-aligned window
        local_top1 = local.argmax(axis=-1).cast("int32")  # [b, s]
        # remap to global via topk_local_to_global on a [b,s,1] tensor
        g = topk_local_to_global(local_top1.unsqueeze(-1), vr)[..., 0]
        # reference: global argmax restricted to each query's valid window
        col = paddle.arange(sk, dtype="int32").reshape([1, 1, sk])
        in_win = (col >= vr[..., 0:1]) & (col < vr[..., 1:2])
        masked = paddle.where(
            in_win, scores, paddle.full_like(scores, float("-inf"))
        )
        ref = masked.argmax(axis=-1).cast("int32")
        self.assertTrue(paddle.equal_all(g, ref).item())

    def test_bad_scores_rank_raises(self):
        with self.assertRaises(ValueError):
            shift_scores_to_local_window(
                paddle.zeros([4, 8], dtype="float32"),
                paddle.zeros([4, 2], dtype="int32"),
            )


def _spec_docmask_inputs(doc_lens, ratio, packed_sk=None):
    """Build expected docmask metadata directly from document lengths.

    This oracle intentionally does not call PaddleFleet docmask metadata or
    range helpers, keeping expected values independent of the implementation.
    """
    ratio = int(ratio)
    doc_lens = [int(length) for length in doc_lens]
    sq = sum(doc_lens)
    real_widths = [length // ratio for length in doc_lens]
    real_sk = sum(real_widths)
    if packed_sk is None:
        packed_sk = real_sk
    if packed_sk < real_sk:
        raise ValueError(
            f"packed_sk={packed_sk} is smaller than real_sk={real_sk}"
        )

    ends = []
    ranges = []
    token_start = 0
    col_start = 0
    for doc_len, width in zip(doc_lens, real_widths):
        doc_end = token_start + doc_len
        for pos_in_doc in range(doc_len):
            ends.append(doc_end)
            available = min((pos_in_doc + 1) // ratio, width)
            ranges.append([col_start, col_start + available])
        token_start = doc_end
        col_start += width

    startend = paddle.to_tensor(ends, dtype="int32").reshape([1, 1, sq, 1])
    valid_range = paddle.to_tensor(ranges, dtype="int32").reshape([1, sq, 2])
    return startend, valid_range, real_sk, int(packed_sk)


def _spec_indexer_topk(index_q, index_k, weights, doc_lens, ratio, topk):
    """Compute expected global top-k ids with an independent reference."""
    _, valid_range, real_sk, packed_sk = _spec_docmask_inputs(
        doc_lens, ratio, int(index_k.shape[1])
    )
    if packed_sk != int(index_k.shape[1]):
        raise AssertionError("oracle packed width must match index_k")

    q = index_q.cast("float32")
    k = index_k.cast("float32")
    w = weights.cast("float32")
    scale = float(index_q.shape[-1]) ** -0.5
    scores = paddle.einsum("bshd,btd->bsht", q, k) * scale
    scores = paddle.nn.functional.relu(scores)
    scores = (scores * w.unsqueeze(-1)).sum(axis=2)

    ranges = valid_range.numpy()[0]
    score_np = scores.numpy()[0]
    expected = []
    lengths = []
    for query, (start, end) in enumerate(ranges):
        start, end = int(start), min(int(end), real_sk)
        count = min(int(topk), max(end - start, 0))
        lengths.append(count)
        if count == 0:
            row = []
        else:
            local_order = score_np[query, start:end].argsort()[::-1][:count]
            row = [start + int(index) for index in local_order]
        expected.append(row + [-1] * (int(topk) - count))
    return (
        paddle.to_tensor(expected, dtype="int32").unsqueeze(0),
        paddle.to_tensor(lengths, dtype="int32").unsqueeze(0),
        valid_range,
    )


def _assert_topk_sets_equal(testcase, actual, expected, message):
    actual_np = actual.numpy()[0]
    expected_np = expected.numpy()[0]
    for query in range(actual_np.shape[0]):
        testcase.assertEqual(
            {int(index) for index in actual_np[query] if index >= 0},
            {int(index) for index in expected_np[query] if index >= 0},
            f"{message}, query={query}: actual={actual_np[query]}, expected={expected_np[query]}",
        )


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "cuDNN indexer top-k requires Blackwell GPU (SM100)",
)
class TestCudnnIndexerTopkDocmask(unittest.TestCase):
    """End-to-end forward docmask through cudnn_indexer_topk (radix kernel)."""

    def _ref_per_doc_topk(self, scores, valid_range, topk):
        """Per-query reference: top-k global ids restricted to the valid window.

        Mirrors the radix kernel semantics: within each query's
        ``[valid_start, valid_end)`` window pick the ``min(topk, count)``
        largest scores, returning their global compressed-buffer ids (sorted by
        descending score). Padding to ``topk`` is ``-1``.
        """
        b, s, sk = scores.shape
        col = paddle.arange(sk, dtype="int32").reshape([1, 1, sk])
        in_win = (col >= valid_range[..., 0:1]) & (col < valid_range[..., 1:2])
        masked = paddle.where(
            in_win, scores, paddle.full_like(scores, float("-inf"))
        )
        ref = paddle.full([b, s, topk], -1, dtype="int32")
        counts = (valid_range[..., 1] - valid_range[..., 0]).clip(min=0)
        masked_np = masked.numpy()
        counts_np = counts.numpy()
        import numpy as np

        ref_np = ref.numpy()
        for bi in range(b):
            for si in range(s):
                c = int(counts_np[bi, si])
                if c <= 0:
                    continue
                k = min(topk, c)
                order = np.argsort(-masked_np[bi, si])[:k]
                ref_np[bi, si, :k] = order.astype("int32")
        return paddle.to_tensor(ref_np)

    def test_docmask_topk_within_doc_windows(self):
        """User scenario: doc lens 23 + 9, ratio 4 -> 5 + 2 compressed cols.

        Compressed buffer (Sk=8): doc0 cols [0,5), doc1 cols [5,7), 1 pad col.
        Every selected id must lie inside its query's document window; no
        cross-document leakage; empty-range queries select nothing.
        """
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )
        from paddleformers.fleet.transformer.csa_attention import get_valid_range

        paddle.seed(0)
        ratio, sq, sk, topk = 4, 32, 8, 4
        startend = paddle.concat(
            [
                paddle.full([23], 23, dtype="int32"),
                paddle.full([9], 32, dtype="int32"),
            ]
        ).reshape([1, 1, sq, 1])
        valid_range = get_valid_range(ratio, 1, sq, startend)
        scores = paddle.randn([1, sq, sk]).astype("float32")

        topk_indices, topk_length = cudnn_indexer_topk(
            scores, sq, ratio, topk, valid_range=valid_range
        )
        ti = topk_indices.numpy()[0]
        tl = topk_length.numpy()[0]
        vr = valid_range.numpy()[0]
        for q in range(sq):
            start, end = int(vr[q, 0]), int(vr[q, 1])
            picks = [int(x) for x in ti[q] if x >= 0]
            for x in picks:
                self.assertTrue(
                    start <= x < end,
                    f"query {q}: id {x} outside window [{start},{end})",
                )
            # topk_length must equal the number of valid picks; empty-range
            # queries must report length 0 (locks the -1/length contract).
            self.assertEqual(
                int(tl[q]),
                len(picks),
                f"query {q}: topk_length {int(tl[q])} != valid picks {len(picks)}",
            )
            if end == start:
                self.assertEqual(
                    picks, [], f"empty-range query {q} picked {picks}"
                )
                self.assertEqual(
                    int(tl[q]), 0, f"empty-range query {q} length != 0"
                )

    def test_docmask_matches_per_doc_reference(self):
        """Selected id sets match a pure-numpy per-window top-k reference."""
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )

        paddle.seed(3)
        # 3 docs of compressed widths 4, 3, 5 packed into Sk=12; one query each.
        sk, topk = 12, 3
        valid_range = paddle.to_tensor(
            [[[0, 4], [4, 7], [7, 12]]], dtype="int32"
        )
        scores = paddle.randn([1, 3, sk]).astype("float32")
        topk_indices, _ = cudnn_indexer_topk(
            scores, 3, 4, topk, valid_range=valid_range
        )
        ref = self._ref_per_doc_topk(scores, valid_range, topk)
        # Compare as sets per query (radix tie order may differ from argsort).
        ti = topk_indices.numpy()[0]
        rf = ref.numpy()[0]
        for q in range(3):
            self.assertEqual(
                {int(x) for x in ti[q] if x >= 0},
                {int(x) for x in rf[q] if x >= 0},
                f"query {q} selected-set mismatch: got {ti[q]}, ref {rf[q]}",
            )

    def test_docmask_none_matches_causal_baseline(self):
        """valid_range=None reproduces the legacy causal-only path exactly."""
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )

        paddle.seed(5)
        ratio, sq, sk, topk = 4, 16, 8, 4
        scores = paddle.randn([1, sq, sk]).astype("float32")
        idx_a, _ = cudnn_indexer_topk(scores, sq, ratio, topk, valid_range=None)
        idx_b, _ = cudnn_indexer_topk(scores, sq, ratio, topk)
        self.assertTrue(paddle.equal_all(idx_a, idx_b).item())

    def _run_spec_case(self, doc_lens, ratio, topk, seed, packed_sk=None):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        paddle.seed(seed)
        h, d = 64, 128
        startend, valid_range, real_sk, sk = _spec_docmask_inputs(
            doc_lens, ratio, packed_sk
        )
        sq = sum(doc_lens)
        index_q = paddle.randn([1, sq, h, d]).astype("bfloat16")
        index_k = paddle.randn([1, sk, d]).astype("bfloat16")
        weights = paddle.randn([1, sq, h]).astype("bfloat16")
        expected_idx, expected_len, expected_range = _spec_indexer_topk(
            index_q, index_k, weights, doc_lens, ratio, topk
        )
        self.assertTrue(paddle.equal_all(valid_range, expected_range).item())

        actual_idx, actual_len, _ = cudnn_indexer_topk_fwd(
            index_q,
            index_k,
            weights,
            ratio=ratio,
            topk_effective=topk,
            valid_range=valid_range,
            startend_row_indices=startend,
            doc_lens=doc_lens,
            return_topk_scores=True,
        )

        self.assertTrue(
            paddle.equal_all(actual_len, expected_len).item(),
            f"topk lengths mismatch: actual={actual_len.numpy()}, expected={expected_len.numpy()}",
        )
        _assert_topk_sets_equal(
            self,
            actual_idx,
            expected_idx,
            f"doc_lens={doc_lens}, real_sk={real_sk}",
        )
        return actual_idx, actual_len, expected_range, real_sk

    def test_aligned_docs_match_independent_spec(self):
        """Aligned documents exercise THD while expectations come from the spec."""
        self._run_spec_case([16, 24, 8], ratio=4, topk=8, seed=17)

    def test_non_ratio_aligned_docs_match_independent_spec(self):
        """Tail queries remain valid and cannot cross document boundaries."""
        _, lengths, valid_range, _ = self._run_spec_case(
            [23, 9], ratio=4, topk=8, seed=19, packed_sk=8
        )
        ranges = valid_range.numpy()[0]
        self.assertEqual(ranges[22].tolist(), [0, 5])
        self.assertEqual(ranges[31].tolist(), [5, 7])
        self.assertEqual(int(lengths.numpy()[0, 22]), 5)
        self.assertEqual(int(lengths.numpy()[0, 31]), 2)

    def test_short_document_matches_independent_spec(self):
        """A one-column document remains isolated when execution falls back."""
        indices, lengths, valid_range, _ = self._run_spec_case(
            [4, 12], ratio=4, topk=8, seed=23
        )
        ranges = valid_range.numpy()[0]
        self.assertEqual(ranges[3].tolist(), [0, 1])
        self.assertEqual(ranges[7].tolist(), [1, 2])
        self.assertEqual(int(lengths.numpy()[0, 3]), 1)
        self.assertEqual({int(x) for x in indices.numpy()[0, 3] if x >= 0}, {0})

    def test_packed_padding_is_never_selected(self):
        """Extra physical KV columns are padding, not another document."""
        indices, _, _, real_sk = self._run_spec_case(
            [16, 8], ratio=4, topk=8, seed=29, packed_sk=9
        )
        selected = [int(x) for x in indices.numpy().reshape([-1]) if x >= 0]
        self.assertTrue(selected)
        self.assertTrue(all(index < real_sk for index in selected))

    def test_cp_local_query_slice_matches_global_spec(self):
        """A CP-style local Q slice preserves global docmask semantics."""
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        paddle.seed(31)
        doc_lens, ratio, topk, h, d = [16, 24], 4, 8, 64, 128
        sq = sum(doc_lens)
        startend, valid_range, _, sk = _spec_docmask_inputs(doc_lens, ratio)
        index_q = paddle.randn([1, sq, h, d]).astype("bfloat16")
        index_k = paddle.randn([1, sk, d]).astype("bfloat16")
        weights = paddle.randn([1, sq, h]).astype("bfloat16")
        expected_idx, expected_len, _ = _spec_indexer_topk(
            index_q, index_k, weights, doc_lens, ratio, topk
        )

        offset, local_sq = 20, 12
        actual_idx, actual_len = cudnn_indexer_topk_fwd(
            index_q[:, offset : offset + local_sq],
            index_k,
            weights[:, offset : offset + local_sq],
            ratio=ratio,
            topk_effective=topk,
            valid_range=valid_range[:, offset : offset + local_sq],
            startend_row_indices=startend,
            doc_lens=doc_lens,
            seq_offset=offset,
        )
        expected_slice = expected_idx[:, offset : offset + local_sq]
        self.assertTrue(
            paddle.equal_all(
                actual_len, expected_len[:, offset : offset + local_sq]
            ).item()
        )
        _assert_topk_sets_equal(
            self, actual_idx, expected_slice, "CP local query slice"
        )


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "cuDNN indexer top-k requires Blackwell GPU (SM100)",
)
class TestMapCompressedTopkToKvFullDocmask(unittest.TestCase):
    """End-to-end: cudnn docmask topk -> _map_compressed_topk_to_kv_full.

    Verifies the single-document causal cutoff inside
    ``_map_compressed_topk_to_kv_full`` (``id < (t+1)//ratio``, computed from
    the GLOBAL query position) does not falsely reject any in-window global
    compressed id produced by the docmask-aware forward, and that every kept id
    decodes back into its query's document window. Uses document lengths that
    are NOT multiples of ratio to stress the floor-accumulation of doc column
    starts.
    """

    def test_no_false_rejection_and_in_window(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )
        from paddleformers.fleet.transformer.csa_attention import (
            _map_compressed_topk_to_kv_full,
            get_valid_range,
        )

        paddle.seed(0)
        ratio = 4
        # doc lengths NOT divisible by ratio: 6, 7, 19 (sum=32).
        #   doc0 len6  -> cutoff4  -> 1 compressed col  [0,1)
        #   doc1 len7  -> cutoff4  -> 1 compressed col  [1,2)
        #   doc2 len19 -> cutoff16 -> 4 compressed cols [2,6)
        # compressed buffer Sk = sq//ratio = 8 (cols [6,8) are pad).
        doc_lens = [6, 7, 19]
        sq = sum(doc_lens)
        sk = sq // ratio  # 8
        topk = 4
        offset = sq  # compressed entries follow raw KV in kv_full

        ends = []
        acc = 0
        for dl in doc_lens:
            acc += dl
            ends.extend([acc] * dl)
        startend = paddle.to_tensor(ends, dtype="int32").reshape([1, 1, sq, 1])
        valid_range = get_valid_range(ratio, 1, sq, startend)  # [1, sq, 2]

        scores = paddle.randn([1, sq, sk]).astype("float32")
        topk_global, _ = cudnn_indexer_topk(
            scores, sq, ratio, topk, valid_range=valid_range
        )

        mapped = _map_compressed_topk_to_kv_full(topk_global, sq, ratio, offset)

        ti = topk_global.numpy()[0]
        mp = mapped.numpy()[0]
        vr = valid_range.numpy()[0]
        for q in range(sq):
            start, end = int(vr[q, 0]), int(vr[q, 1])
            in_topk = [int(x) for x in ti[q] if x >= 0]
            kept = [int(x) for x in mp[q] if x >= 0]
            # (1) No false rejection: count of valid slots preserved.
            self.assertEqual(
                len(kept),
                len(in_topk),
                f"query {q}: map dropped valid ids "
                f"(in={in_topk}, kept_mapped={mp[q].tolist()}, "
                f"window=[{start},{end}))",
            )
            # (2) Every kept id decodes (minus offset) into the doc window.
            for m in kept:
                comp = m - offset
                self.assertTrue(
                    start <= comp < end,
                    f"query {q}: mapped id {m} -> compressed {comp} "
                    f"outside window [{start},{end})",
                )

    def test_matches_manual_compressed_mapping(self):
        """Mapped ids equal (in-window global id + offset), elementwise."""
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )
        from paddleformers.fleet.transformer.csa_attention import (
            _map_compressed_topk_to_kv_full,
            get_valid_range,
        )

        paddle.seed(1)
        ratio = 4
        doc_lens = [10, 22]  # not multiples of ratio at the boundary sum
        sq = sum(doc_lens)  # 32
        sk = sq // ratio  # 8
        topk = 4
        offset = sq

        ends = []
        acc = 0
        for dl in doc_lens:
            acc += dl
            ends.extend([acc] * dl)
        startend = paddle.to_tensor(ends, dtype="int32").reshape([1, 1, sq, 1])
        valid_range = get_valid_range(ratio, 1, sq, startend)

        scores = paddle.randn([1, sq, sk]).astype("float32")
        topk_global, _ = cudnn_indexer_topk(
            scores, sq, ratio, topk, valid_range=valid_range
        )
        mapped = _map_compressed_topk_to_kv_full(topk_global, sq, ratio, offset)

        # Since the forward already constrains topk_global to in-window ids,
        # the map is exactly: valid -> id+offset, invalid(-1) -> -1.
        valid = topk_global >= 0
        expected = paddle.where(
            valid,
            topk_global + offset,
            paddle.full_like(topk_global, -1),
        )
        self.assertTrue(paddle.equal_all(mapped, expected).item())


# ---------------------------------------------------------------------------
# End-to-end: docmask topk -> window concat -> sparse fwd (flash_mla / cudnn)
# ---------------------------------------------------------------------------


def _docmask_topk_idxs_for_attn(doc_lens, ratio, window_size, topk, seed):
    """Build kv_full topk_idxs for sparse attention under docmask (b==1).

    Returns ``(topk_idxs [1, sq, W+topk], sq, n_compressed, valid_range,
    doc_row_valid)`` where ``doc_row_valid[q]`` marks queries that belong to a
    real (non-padding) document position. Window + compressed indices are built
    with the same docmask-aware helpers the production forward uses, then
    concatenated exactly like ``CompressedSparseAttention.forward``.
    """
    from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
        cudnn_indexer_topk,
    )
    from paddleformers.fleet.transformer.csa_attention import (
        _map_compressed_topk_to_kv_full,
        get_valid_range,
        get_window_topk_idxs,
    )

    paddle.seed(seed)
    sq = sum(doc_lens)
    n_compressed = sq // ratio
    offset = sq  # compressed entries follow raw KV inside kv_full

    ends = []
    acc = 0
    for dl in doc_lens:
        acc += dl
        ends.extend([acc] * dl)
    startend = paddle.to_tensor(ends, dtype="int32").reshape([1, 1, sq, 1])

    valid_range = get_valid_range(ratio, 1, sq, startend)  # [1, sq, 2]
    scores = paddle.randn([1, sq, n_compressed]).astype("float32")
    topk_compressed, _ = cudnn_indexer_topk(
        scores, sq, ratio, topk, valid_range=valid_range
    )
    compress_idxs = _map_compressed_topk_to_kv_full(
        topk_compressed, sq, ratio, offset
    )
    window_idxs = get_window_topk_idxs(window_size, 1, sq, startend)

    if compress_idxs.dtype != window_idxs.dtype:
        compress_idxs = compress_idxs.cast(window_idxs.dtype)
    topk_idxs = paddle.concat([window_idxs, compress_idxs], axis=-1).cast(
        "int32"
    )
    # A query row is "real" when its window has at least one valid slot.
    doc_row_valid = (window_idxs >= 0).any(axis=-1)[0]  # [sq]
    return topk_idxs, sq, n_compressed, valid_range, doc_row_valid


def _flash_mla_available():
    try:
        import paddlefleet_ops

        from paddleformers.fleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn

        return (
            paddlefleet_ops.is_flash_mla_available()
            and csa_sparse_attn_fwd_cudnn._flash_mla_sparse_fwd is not None
        )
    except (ImportError, RuntimeError, AttributeError):
        return False


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "docmask sparse-fwd compatibility requires Blackwell GPU (SM100)",
)
class TestDocmaskSparseFwdCompat(unittest.TestCase):
    """docmask topk -> window concat -> sparse attention forward.

    The ``backend="cudnn"`` path combines FlashMLA forward with cuDNN
    backward, while the TileLang backend uses its own forward and backward.
    These tests exercise the production FlashMLA wrapper directly and through
    the ``csa_sparse_attn`` PyLayer entry point, checking
    directly and through the ``csa_sparse_attn`` PyLayer entry point, checking
    that docmask-produced topk_idxs (window + compressed, kv_full-local, -1
    padding) feed cleanly in, that real (non-padding) query rows produce finite
    output, and that fully-masked padding rows (a docmask-only situation that
    single-document inputs never hit) stay benign.
    """

    # DSv4 sparse-MQA shape: head_dim must be 512, num_heads 64.
    HEAD_DIM = 512
    NUM_HEADS = 64

    def _build(self, doc_lens, ratio=4, window_size=8, topk=128, seed=0):
        topk_idxs, sq, n_comp, valid_range, doc_row_valid = (
            _docmask_topk_idxs_for_attn(
                doc_lens, ratio, window_size, topk, seed
            )
        )
        s_kvfull = sq + n_comp
        paddle.seed(seed + 1000)
        query = paddle.randn(
            [1, sq, self.NUM_HEADS, self.HEAD_DIM], dtype=paddle.bfloat16
        )
        kv_full = paddle.randn(
            [1, s_kvfull, self.HEAD_DIM], dtype=paddle.bfloat16
        )
        attn_sink = paddle.randn([self.NUM_HEADS], dtype=paddle.float32)
        sm_scale = self.HEAD_DIM**-0.5
        return query, kv_full, attn_sink, topk_idxs, sm_scale, doc_row_valid

    def _check_real_rows_finite(self, out, doc_row_valid, name):
        # out: [1, sq, H*D] or [1, sq, H, D]; reduce to per-row finiteness.
        out_f = out.reshape([out.shape[1], -1]).cast("float32")
        finite_per_row = paddle.isfinite(out_f).all(axis=-1)  # [sq]
        real = doc_row_valid  # [sq] bool
        real_finite = paddle.where(
            real, finite_per_row, paddle.ones_like(finite_per_row)
        )
        self.assertTrue(
            bool(real_finite.all().item()),
            f"{name}: some real (non-padding) query row is non-finite",
        )

    @unittest.skipUnless(_flash_mla_available(), "flash_mla not available")
    def test_flash_mla_fwd_docmask(self):
        from paddleformers.fleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
            flash_mla_sparse_attn,
        )

        # doc lens not multiples of ratio to exercise padding query rows.
        query, kv_full, attn_sink, topk_idxs, sm_scale, doc_row_valid = (
            self._build([23, 9], seed=1)
        )
        out, lse, _ = flash_mla_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, sm_scale=sm_scale
        )
        self.assertEqual(out.shape[1], query.shape[1])
        self._check_real_rows_finite(out, doc_row_valid, "flash_mla")

    @unittest.skipUnless(_flash_mla_available(), "flash_mla not available")
    def test_csa_sparse_attn_pylayer_docmask(self):
        """Through csa_sparse_attn PyLayer (backend selects bwd; fwd=FlashMLA)."""
        from paddleformers.fleet.fusions.csa_sparse_attn import csa_sparse_attn

        query, kv_full, attn_sink, topk_idxs, sm_scale, doc_row_valid = (
            self._build([23, 9], seed=2)
        )
        # backend="cudnn" runs FlashMLA forward and cuDNN backward.
        try:
            out = csa_sparse_attn(
                query, kv_full, attn_sink, topk_idxs, sm_scale, backend="cudnn"
            )
        except (RuntimeError, ImportError) as e:
            self.skipTest(f"csa_sparse_attn cudnn-bwd path unavailable: {e}")
        self.assertEqual(out.shape[1], query.shape[1])
        self._check_real_rows_finite(out, doc_row_valid, "csa_sparse_attn")

    @unittest.skipUnless(_flash_mla_available(), "flash_mla not available")
    def test_real_rows_match_single_doc_equivalent(self):
        """doc0's packed rows == doc0 run alone (no cross-doc contamination).

        Build a packed [doc0 | doc1] topk via the docmask helpers, then build
        doc0 alone (same doc0 length, its own startend). doc0's query rows,
        window ids and compressed ids all live in doc0's id range, so slicing
        doc0 out of the packed run must equal the standalone doc0 run when fed
        the SAME query / kv_full slice. Any leak of doc1 KV into doc0 rows would
        break the equality. Uses doc lengths that are multiples of ratio so
        doc0 occupies a clean prefix [0, d0) of raw KV and [sq, sq+d0//ratio)
        of compressed KV, making the standalone slice exact.
        """
        from paddleformers.fleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
            flash_mla_sparse_attn,
        )

        ratio, window, topk = 4, 8, 128
        d0, d1 = 64, 64
        sq = d0 + d1
        n0 = d0 // ratio  # doc0 compressed cols
        topk_idxs, _sq, n_comp, _vr, _drv = _docmask_topk_idxs_for_attn(
            [d0, d1], ratio, window, topk, seed=3
        )
        self.assertEqual(_sq, sq)
        s_kvfull = sq + n_comp

        paddle.seed(1234)
        query = paddle.randn(
            [1, sq, self.NUM_HEADS, self.HEAD_DIM], dtype=paddle.bfloat16
        )
        kv_full = paddle.randn(
            [1, s_kvfull, self.HEAD_DIM], dtype=paddle.bfloat16
        )
        attn_sink = paddle.randn([self.NUM_HEADS], dtype=paddle.float32)
        sm_scale = self.HEAD_DIM**-0.5

        out_packed, _, _ = flash_mla_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, sm_scale=sm_scale
        )

        # Standalone doc0: query rows [0, d0); its kv_full is doc0's raw KV
        # [0, d0) followed by doc0's compressed cols [sq, sq+n0). Remap doc0's
        # packed topk ids into this compact [0, d0+n0) buffer.
        q0 = query[:, :d0].contiguous()
        kv0 = paddle.concat(
            [kv_full[:, :d0], kv_full[:, sq : sq + n0]], axis=1
        ).contiguous()
        idx0 = topk_idxs[:, :d0].clone().numpy()
        # raw-KV ids [0,d0) stay; compressed ids [sq, sq+n0) -> [d0, d0+n0); -1 stays.
        remap = idx0.copy()
        comp_mask = idx0 >= sq
        remap[comp_mask] = idx0[comp_mask] - sq + d0
        idx0_remapped = paddle.to_tensor(remap, dtype="int32")

        out_doc0, _, _ = flash_mla_sparse_attn(
            q0, kv0, attn_sink, idx0_remapped, sm_scale=sm_scale
        )

        packed_doc0 = out_packed[:, :d0].reshape([d0, -1]).cast("float32")
        alone_doc0 = out_doc0.reshape([d0, -1]).cast("float32")
        # bf16 gather + fp32 accumulate: identical inputs -> identical output.
        self.assertTrue(
            paddle.allclose(
                packed_doc0, alone_doc0, rtol=1e-2, atol=1e-2
            ).item(),
            "doc0 packed output diverges from standalone doc0 "
            "(cross-document contamination through sparse fwd gather)",
        )

    @unittest.skipUnless(_flash_mla_available(), "flash_mla not available")
    def test_fully_masked_rows_are_benign(self):
        """Fully ``-1`` query rows (docmask-only) must not produce NaN/Inf.

        Natural docmask packing never yields an all-masked row (every query
        attends to its own sliding window), but we construct one explicitly to
        prove the attention-sink denominator floors softmax: a row with zero
        valid KV slots yields a finite (≈0) output instead of NaN. This is the
        property that lets padding rows pass through the sparse fwd without
        contaminating downstream tensors.
        """
        from paddleformers.fleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
            flash_mla_sparse_attn,
        )

        h, d, sq, skv, topk = self.NUM_HEADS, self.HEAD_DIM, 8, 16, 128
        paddle.seed(7)
        q = paddle.randn([1, sq, h, d], dtype=paddle.bfloat16)
        kv = paddle.randn([1, skv, d], dtype=paddle.bfloat16)
        sink = paddle.randn([h], dtype=paddle.float32)

        idx = paddle.full([1, sq, topk], -1, dtype="int32").numpy()
        # rows 0..5 valid (a few real slots); rows 6,7 fully masked.
        for r in range(6):
            for j in range(4):
                idx[0, r, j] = j
        topk_idxs = paddle.to_tensor(idx)

        out, _, _ = flash_mla_sparse_attn(
            q, kv, sink, topk_idxs, sm_scale=d**-0.5
        )
        out_f = out.reshape([sq, -1]).cast("float32")
        self.assertTrue(
            bool(paddle.isfinite(out_f).all().item()),
            "fully-masked rows produced NaN/Inf (attention sink should floor "
            "the softmax denominator)",
        )


def _tilelang_indexer_bwd_available():
    try:
        from paddleformers.fleet.tilelang_ops import csa_indexer_bwd  # noqa: F401

        return True
    except (ImportError, RuntimeError, AttributeError):
        return False


def _cos_rms(actual, ref):
    a = actual.cast("float32").flatten()
    r = ref.cast("float32").flatten()
    na, nr = a.norm(), r.norm()
    denom = (na * nr).item()
    cos = 1.0 if denom == 0.0 else (a @ r / (na * nr)).item()
    rms_err = (a - r).pow(2).mean().sqrt().item()
    rms_ref = r.pow(2).mean().sqrt().item()
    return cos, rms_err / max(rms_ref, 1e-12)


def _ref_indexer_bwd_autograd(
    index_q,
    weights,
    index_k,
    topk_indices,
    target,
    topk_probs,
    loss_coeff,
    grad_loss,
    ratio,
):
    """Pure-paddle autograd reference matching the cuDNN backward semantics.

    Forward: scores = relu(q@k^T * sm_scale) weighted-summed over heads, gather
    at topk_indices, masked. Loss = sum( ((topk_probs - target) * scale *
    grad_loss).detach() * topk_score ), i.e. the linear KL-logit grad the cuDNN
    kernel reduces to when the clipped-log path is not triggered (the same
    reference used by test_cudnn_dsa_indexer_bwd.test_parity_against_reference).
    """
    import paddle.nn.functional as F

    b, sq, h, d = index_q.shape
    sk = index_k.shape[1]
    topk = topk_indices.shape[-1]
    sm = d**-0.5

    q = index_q.cast("float32").detach()
    q.stop_gradient = False
    k = index_k.cast("float32").detach()
    k.stop_gradient = False
    w = weights.cast("float32").detach()
    w.stop_gradient = False

    scores = paddle.einsum("bshd,btd->bsht", q, k) * sm
    scores = F.relu(scores)
    scores = (scores * w.unsqueeze(-1)).sum(axis=2)  # [b, sq, sk]

    idx = topk_indices.cast("int64")
    valid = topk_indices >= 0
    safe = paddle.where(valid, idx, paddle.zeros_like(idx))
    topk_score = paddle.take_along_axis(scores, safe, axis=-1)
    topk_score = paddle.where(valid, topk_score, paddle.zeros_like(topk_score))

    scale = loss_coeff / float(b * sq)
    og = (topk_probs - target) * scale
    if grad_loss is not None:
        og = og * grad_loss
    loss = (og.detach() * topk_score).sum()
    loss.backward()
    return q.grad, w.grad, k.grad


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "cuDNN indexer backward requires Blackwell GPU (SM100)",
)
class TestCudnnVsTilelangIndexerBwdDocmask(unittest.TestCase):
    """P2-3: cuDNN docmask backward gradient correctness on packed multi-doc.

    cuDNN and TileLang indexer backwards take **different OGrad contracts**:
    TileLang consumes an externally computed ``grad_scores`` (grad w.r.t. topk
    logits) and only back-props logit->(q,k,w); cuDNN consumes (target,
    topk_probs, loss_coeff) and computes the clipped-log KL grad_signal
    internally. They are therefore not directly comparable by feeding identical
    tensors. Instead we validate both against a shared pure-paddle autograd
    reference (the same reference test_cudnn_dsa_indexer_bwd uses), under a
    realistic packed multi-document docmask input produced by
    get_valid_range(startend) + the cuDNN docmask forward top-k.

    This locks: (a) cuDNN docmask backward matches the autograd ground truth;
    (b) TileLang docmask backward matches the same ground truth; hence the two
    backends are mutually consistent on packed multi-doc gradients.
    """

    def setUp(self):
        from paddleformers.fleet.cudnn_ops import csa_indexer_bwd as cudnn_bwd

        self.cudnn_bwd = cudnn_bwd
        self.has_tl = _tilelang_indexer_bwd_available()
        if self.has_tl:
            from paddleformers.fleet.tilelang_ops import csa_indexer_bwd as tl_bwd

            self.tl_bwd = tl_bwd

    def _make_packed(self, doc_lens, ratio=4, h=64, d=128, topk=128, seed=0):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )
        from paddleformers.fleet.transformer.csa_attention import get_valid_range

        paddle.seed(seed)
        sq = sum(doc_lens)
        n_comp = sq // ratio
        ends, acc = [], 0
        for dl in doc_lens:
            acc += dl
            ends.extend([acc] * dl)
        startend = paddle.to_tensor(ends, dtype="int32").reshape([1, 1, sq, 1])
        valid_range = get_valid_range(ratio, 1, sq, startend)

        index_q = paddle.randn([1, sq, h, d]).astype("bfloat16")
        index_k = paddle.randn([1, n_comp, d]).astype("bfloat16")
        weights = paddle.randn([1, sq, h]).astype("bfloat16")
        topk_idx, _ = cudnn_indexer_topk_fwd(
            index_q,
            index_k,
            weights,
            ratio=ratio,
            topk_effective=topk,
            valid_range=valid_range,
        )
        # target / topk_probs as masked-softmax over the valid slots, with
        # fully-invalid (padding) rows zeroed — exactly the production contract
        # from _compute_tilelang_csa_indexer_loss_forward. The per-row sum over
        # valid slots is 1, which is what makes cuDNN's clipped-log KL
        # grad_signal reduce to the linear scale*(predict - target) the
        # autograd reference uses.
        valid = topk_idx >= 0
        row_valid = valid.any(axis=-1, keepdim=True)
        neg_inf = paddle.full([1], float("-inf"), dtype="float32")

        def _masked_softmax(seed_shift):
            paddle.seed(seed + seed_shift)
            logits = paddle.randn([1, sq, topk]).astype("float32")
            logits = paddle.where(
                valid, logits, neg_inf.broadcast_to(logits.shape)
            )
            logits = paddle.where(row_valid, logits, paddle.zeros_like(logits))
            p = paddle.nn.functional.softmax(logits, axis=-1)
            return p * row_valid.cast("float32")

        target = _masked_softmax(101)
        topk_probs = _masked_softmax(202)
        return index_q, index_k, weights, topk_idx, target, topk_probs, n_comp

    def _check(self, doc_lens, seed, ratio=4, topk=128):
        loss_coeff = 0.01
        gl = paddle.to_tensor(1.0, dtype="float32")
        iq, ik, w, ti, tgt, tp, _ = self._make_packed(
            doc_lens, ratio=ratio, topk=topk, seed=seed
        )

        gq_ref, gw_ref, gk_ref = _ref_indexer_bwd_autograd(
            iq, w, ik, ti, tgt, tp, loss_coeff, gl, ratio
        )

        gq_c, gw_c, gk_c = self.cudnn_bwd(
            iq.clone(),
            w.clone(),
            ik.clone(),
            tgt.clone(),
            tp.clone(),
            ti.clone(),
            loss_coeff=loss_coeff,
            grad_loss=gl,
        )
        for name, a, r in (
            ("cudnn d_index_q", gq_c, gq_ref),
            ("cudnn d_weights", gw_c, gw_ref),
            ("cudnn d_index_k", gk_c, gk_ref),
        ):
            cos, rms = _cos_rms(a, r)
            self.assertGreaterEqual(cos, 0.97, f"{name}: cos {cos:.4f}")
            self.assertLessEqual(rms, 0.55, f"{name}: rms_rel {rms:.4f}")

        if self.has_tl:
            grad_scores = (tp - tgt) * (loss_coeff / float(sum(doc_lens)))
            gq_t, gw_t, gk_t = self.tl_bwd(
                iq.clone(),
                w.clone(),
                ik.clone(),
                ti.clone(),
                grad_scores.clone(),
            )
            for name, a, r in (
                ("tilelang d_index_q", gq_t, gq_ref),
                ("tilelang d_weights", gw_t, gw_ref),
                ("tilelang d_index_k", gk_t, gk_ref),
            ):
                cos, rms = _cos_rms(a, r)
                self.assertGreaterEqual(cos, 0.97, f"{name}: cos {cos:.4f}")
                self.assertLessEqual(rms, 0.55, f"{name}: rms_rel {rms:.4f}")

    def test_two_docs_ratio4(self):
        self._check([23, 9], seed=7)

    def test_three_docs_uneven(self):
        self._check([40, 28, 60], seed=11)

    def test_deterministic_path(self):
        old = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
            "FLAGS_cudnn_deterministic"
        ]
        paddle.set_flags({"FLAGS_cudnn_deterministic": 1})
        try:
            self._check([23, 9], seed=7)
        finally:
            paddle.set_flags({"FLAGS_cudnn_deterministic": old})


class TestCudnnIndexerQueryTiling(unittest.TestCase):
    """Query-dim tiling of the packed-global fallback.

    Runs on CPU by stubbing the two cuDNN kernels; verifies the tiling
    heuristic and that tiled forward+top-k reproduces the single-shot result
    with correct per-tile ``seq_offset`` and concatenation order.
    """

    def test_resolve_query_tile_heuristic(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            _resolve_indexer_query_tile,
        )

        # tile*S_k bounded by 64Mi score elems: S_k=32768 -> tile=2048.
        self.assertEqual(_resolve_indexer_query_tile(131072, 32768), 2048)
        # Heuristic never exceeds S_q.
        self.assertEqual(_resolve_indexer_query_tile(1024, 32768), 1024)

    def _run_impl(self, mod, sq, sk, topk, query_tile, return_topk_scores):
        seen_offsets = []

        def fake_forward(
            index_q, index_k_comp, weights, ratio, sm_scale, seq_offset
        ):
            m = int(index_q.shape[1])
            seen_offsets.append(int(seq_offset))
            rows = (
                paddle.arange(seq_offset, seq_offset + m, dtype="float32")
                .reshape([1, m, 1])
                .tile([1, 1, sk])
            )
            cols = paddle.arange(sk, dtype="float32").reshape([1, 1, sk])
            return rows * sk + cols  # [1, m, sk], encodes global row & col

        def fake_topk(scores, sq_local, ratio, topk_, valid_range, seq_offset):
            m = int(scores.shape[1])
            gid = paddle.arange(
                seq_offset, seq_offset + m, dtype="int32"
            ).reshape([1, m, 1])
            pad = paddle.full([1, m, topk_ - 1], -1, dtype="int32")
            indices = paddle.concat([gid, pad], axis=-1)
            length = paddle.ones([1, m], dtype="int32")
            return indices, length

        index_q = paddle.zeros([1, sq, 32, 128], dtype="bfloat16")
        index_k_comp = paddle.zeros([1, sk, 128], dtype="bfloat16")
        weights = paddle.zeros([1, sq, 32], dtype="bfloat16")

        with (
            patch.object(
                mod, "cudnn_indexer_forward", side_effect=fake_forward
            ),
            patch.object(mod, "cudnn_indexer_topk", side_effect=fake_topk),
            patch.object(mod, "_DEFAULT_QUERY_TILE_ELEMS", query_tile * sk),
        ):
            out = mod._cudnn_indexer_topk_fwd_impl(
                index_q,
                index_k_comp,
                weights,
                ratio=4,
                topk_effective=topk,
                valid_range=None,
                startend_row_indices=None,
                return_topk_scores=return_topk_scores,
            )
        return out, seen_offsets

    def test_tiled_matches_single_shot(self):
        from paddleformers.fleet.cudnn_ops.indexer import csa_indexer_fwd_cudnn as mod

        sq, sk, topk = 10, 16, 2

        (single_idx, single_len), single_offsets = self._run_impl(
            mod, sq, sk, topk, query_tile=sq, return_topk_scores=False
        )
        (tiled_idx, tiled_len), tiled_offsets = self._run_impl(
            mod, sq, sk, topk, query_tile=3, return_topk_scores=False
        )

        # Slot 0 carries the global query id; tiling must preserve row order.
        expected_ids = paddle.arange(sq, dtype="int32")
        self.assertEqual(single_idx[0, :, 0].tolist(), expected_ids.tolist())
        self.assertEqual(tiled_idx[0, :, 0].tolist(), expected_ids.tolist())
        self.assertEqual(tiled_idx.tolist(), single_idx.tolist())
        self.assertEqual(tiled_len.tolist(), single_len.tolist())
        # Single-shot issues one forward at offset 0; tile=3 issues 4 chunks.
        self.assertEqual(single_offsets, [0])
        self.assertEqual(tiled_offsets, [0, 3, 6, 9])

    def test_tiled_matches_single_shot_with_scores(self):
        from paddleformers.fleet.cudnn_ops.indexer import csa_indexer_fwd_cudnn as mod

        sq, sk, topk = 10, 16, 2

        (single_idx, _, single_scores), _ = self._run_impl(
            mod, sq, sk, topk, query_tile=sq, return_topk_scores=True
        )
        (tiled_idx, _, tiled_scores), _ = self._run_impl(
            mod, sq, sk, topk, query_tile=4, return_topk_scores=True
        )

        self.assertEqual(tiled_idx.tolist(), single_idx.tolist())
        self.assertEqual(tiled_scores.tolist(), single_scores.tolist())
        # Slot 0 gathers score[row, id=row] = row*sk + row; slot 1 is masked.
        for i in range(sq):
            self.assertEqual(single_scores[0, i, 0].item(), float(i * sk + i))
            self.assertEqual(single_scores[0, i, 1].item(), float("-inf"))


if __name__ == "__main__":
    unittest.main()
