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

"""Unit tests for the HySparse block-TopK selection host pipeline
(:mod:`paddleformers.fleet.tilelang_ops.hysparse.pipeline`).

These are pure-tensor tests (no TileLang / FA4 kernels): they pin down the
eq.(3) block-score recovery, the group-wise head aggregation, the
document-relative valid-block masking, and the ``[B, S, topk]`` shape contract
(including ``topk > num_blocks`` padding and invalid-block ``-1`` slots).
"""

import unittest

import paddle

from paddleformers.fleet.tilelang_ops.hysparse.pipeline import (
    block_scores_from_logit,
    select_topk_blocks,
)

_NEG_INF = float("-inf")


def _skip_if_no_cuda(tc):
    if not paddle.device.is_compiled_with_cuda():
        tc.skipTest("CUDA build of Paddle required")
    if paddle.device.cuda.device_count() == 0:
        tc.skipTest("no CUDA device available")


def _valid_range(bos, eos, b=1, s=1):
    vr = paddle.to_tensor([[bos, eos]], dtype="int32").reshape([1, 1, 2])
    return vr.expand([b, s, 2]).contiguous()


class TestBlockScoresFromLogit(unittest.TestCase):
    def test_matches_manual_exp(self):
        _skip_if_no_cuda(self)
        # block_logit [B,H,S,nb], lse [B,S,H]. scores = exp(block_logit - lse).
        block_logit = paddle.to_tensor(
            [1.0, 3.0, 2.0, 0.0], dtype="float32"
        ).reshape([1, 1, 1, 4])
        lse = paddle.to_tensor([0.5], dtype="float32").reshape([1, 1, 1])
        scores = block_scores_from_logit(block_logit, lse)
        expect = paddle.exp(block_logit.reshape([4]) - 0.5)
        self.assertEqual(list(scores.shape), [1, 1, 1, 4])
        self.assertLess(float((scores.reshape([4]) - expect).abs().max()), 1e-5)

    def test_masked_and_nan_guard_to_zero(self):
        _skip_if_no_cuda(self)
        # -inf block_logit with -inf lse -> (-inf)-(-inf)=nan -> exp=nan; the
        # guard must turn any non-finite score into 0. Finite entries survive.
        block_logit = paddle.to_tensor(
            [_NEG_INF, 2.0], dtype="float32"
        ).reshape([1, 1, 1, 2])
        lse = paddle.to_tensor([_NEG_INF], dtype="float32").reshape([1, 1, 1])
        scores = block_scores_from_logit(block_logit, lse).reshape([2])
        self.assertTrue(bool(paddle.isfinite(scores).all()))
        self.assertEqual(float(scores[0]), 0.0)  # masked -> 0


class TestSelectTopkBlocks(unittest.TestCase):
    def _logit(self, values):
        # [1, H, 1, nb] from a python list-of-lists (per head).
        t = paddle.to_tensor(values, dtype="float32")
        h, nb = t.shape
        return t.reshape([1, h, 1, nb])

    def _zero_lse(self, h):
        return paddle.zeros([1, 1, h], dtype="float32")

    def test_head_agg_max_selects_largest(self):
        _skip_if_no_cuda(self)
        # single head, lse=0 -> scores monotonic in logit; top2 of [1,3,2,0].
        block_logit = self._logit([[1.0, 3.0, 2.0, 0.0]])
        idx = select_topk_blocks(
            block_logit,
            self._zero_lse(1),
            _valid_range(0, 4),
            2,
            1,
            head_agg="max",
        )
        self.assertEqual(list(idx.shape), [1, 1, 2])
        got = sorted(idx.reshape([2]).numpy().tolist())
        self.assertEqual(got, [1, 2])

    def test_head_agg_sum_across_heads(self):
        _skip_if_no_cuda(self)
        # A case where sum and max disagree. Per-head scores (lse=0 -> exp(logit)):
        #   head0: blk0=.6 blk1=.5 blk2=0   head1: blk0=0 blk1=.5 blk2=.6
        # max-agg:  [.6, .5, .6] -> top2 = {blk0, blk2}  (blk1 excluded)
        # sum-agg:  [.6, 1., .6] -> top2 must include blk1.
        import math

        l6, l5 = math.log(0.6), math.log(0.5)
        block_logit = self._logit([[l6, l5, _NEG_INF], [_NEG_INF, l5, l6]])
        idx_sum = select_topk_blocks(
            block_logit,
            self._zero_lse(2),
            _valid_range(0, 3),
            2,
            1,
            head_agg="sum",
        )
        self.assertIn(1, set(idx_sum.reshape([2]).numpy().tolist()))
        idx_max = select_topk_blocks(
            block_logit,
            self._zero_lse(2),
            _valid_range(0, 3),
            2,
            1,
            head_agg="max",
        )
        self.assertNotIn(1, set(idx_max.reshape([2]).numpy().tolist()))

    def test_invalid_blocks_become_minus_one(self):
        _skip_if_no_cuda(self)
        # eos=2, block_B=1 -> blocks 0,1 valid; 2,3 invalid. topk=3 -> the third
        # slot lands on an invalid block and must be -1.
        block_logit = self._logit([[1.0, 3.0, 9.0, 9.0]])
        idx = select_topk_blocks(
            block_logit,
            self._zero_lse(1),
            _valid_range(0, 2),
            3,
            1,
        )
        got = sorted(idx.reshape([3]).numpy().tolist())
        self.assertEqual(got, [-1, 0, 1])

    def test_topk_exceeds_num_blocks_pads_minus_one(self):
        _skip_if_no_cuda(self)
        # nb=2, topk=4, block_B=1, both blocks valid -> width stays 4 and the two
        # surplus slots are -1 padding.
        block_logit = self._logit([[1.0, 2.0]])
        idx = select_topk_blocks(
            block_logit,
            self._zero_lse(1),
            _valid_range(0, 2),
            4,
            1,
        )
        self.assertEqual(list(idx.shape), [1, 1, 4])
        vals = idx.reshape([4]).numpy().tolist()
        self.assertEqual(sorted(v for v in vals if v >= 0), [0, 1])
        self.assertEqual(sum(1 for v in vals if v == -1), 2)

    def test_topk_non_positive_raises(self):
        _skip_if_no_cuda(self)
        block_logit = self._logit([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            select_topk_blocks(
                block_logit, self._zero_lse(1), _valid_range(0, 2), 0, 1
            )

    def test_unknown_head_agg_raises(self):
        _skip_if_no_cuda(self)
        block_logit = self._logit([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            select_topk_blocks(
                block_logit,
                self._zero_lse(1),
                _valid_range(0, 2),
                1,
                1,
                head_agg="mean",
            )


if __name__ == "__main__":
    unittest.main()
