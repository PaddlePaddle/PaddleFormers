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

"""HySparse: FA4-fused block-score TopK == scatter (eq.3) reference TopK.

This test guards the *量纲 (scale) contract* fixed in flash-attention PR #164:
the FA4 sm100 forward now emits the **scaled** per-block max logit
(``softmax_scale * q.k`` -- the exact value fed into softmax), and the host
:func:`select_topk_blocks` therefore recovers the eq.(3) block score as
``exp(block_logit - lse)`` (no host-side ``* sm_scale``). This test verifies the
whole path is self-consistent, from *computing the score* to *deriving TopK
from the score*, against an independent "scatter" reference operator that
implements HySparse paper (arXiv 2602.03560) eq.(3) with plain matmul + softmax:

* the FA4 fused ``block_logit`` matches the reference scaled per-block max logit
  on every finite (unmasked) entry, and the masked pattern matches exactly;
* the eq.(3) block scores ``exp(block_logit - lse)`` match; and
* the per-query TopK block indices selected from the two score sources are
  identical.

The scatter reference is the model: for query ``i`` and key block ``b`` it takes
the max SCALED logit over the block's causally-valid columns, converts it to the
block's max attention weight ``exp(max_logit - lse)`` (eq.3 block importance),
aggregates across the query-group heads by a group-wise max, and TopK-selects.

FA4 block-score fusion runs only on SM 10.x (Blackwell); the test skips
otherwise.
"""

import math
import unittest

import paddle

_NEG_INF = float("-inf")


def _sm100_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        testcase.skipTest(
            f"FA4 block-score fusion requires SM 10.x (Blackwell); got SM {major}.x"
        )


def _num_blocks(seqlen_k, block_B):
    return (seqlen_k + block_B - 1) // block_B


def _causal_valid_range(b, s):
    """Single-document causal ``valid_range`` [B, S, 2]: bos=0, eos=t+1."""
    eos = (
        paddle.arange(1, s + 1, dtype="int32")
        .reshape([1, s, 1])
        .expand([b, s, 1])
    )
    bos = paddle.zeros([b, s, 1], dtype="int32")
    return paddle.concat([bos, eos], axis=-1).contiguous()


def _scatter_reference(q, k, sm_scale, block_B):
    """Scatter (eq.3) reference: scaled per-block max logit + natural-log LSE.

    q, k: [B, S, H, D] bf16 (single-document causal, S == S_kv).
    Returns (block_logit_ref [B,H,S,nb] fp32, lse_ref [B,S,H] fp32) computed with
    a full-precision matmul + softmax -- the independent model for the FA4 path.
    """
    b, s, h, d = q.shape
    sk = k.shape[1]
    nb = _num_blocks(sk, block_B)
    qf = q.astype("float32").transpose([0, 2, 1, 3])  # [B,H,S,D]
    kf = k.astype("float32").transpose([0, 2, 1, 3])  # [B,H,Sk,D]
    logits = paddle.matmul(qf, kf, transpose_y=True) * sm_scale  # [B,H,S,Sk]
    # bottom-right aligned causal (== standard causal when s == sk).
    row = paddle.arange(s).reshape([s, 1])
    col = paddle.arange(sk).reshape([1, sk])
    masked = (col > row + (sk - s)).reshape([1, 1, s, sk])
    logits = paddle.where(masked, paddle.full_like(logits, _NEG_INF), logits)
    # natural-log LSE over keys -> [B,H,S] -> [B,S,H] (pipeline layout).
    lse_bhs = paddle.logsumexp(logits, axis=-1)  # [B,H,S]
    lse_ref = lse_bhs.transpose([0, 2, 1]).contiguous()  # [B,S,H]
    # per-block max of the scaled logit -> [B,H,S,nb].
    pad = nb * block_B - sk
    if pad > 0:
        logits = paddle.concat(
            [logits, paddle.full([b, h, s, pad], _NEG_INF, dtype="float32")],
            axis=-1,
        )
    logits = logits.reshape([b, h, s, nb, block_B])
    block_logit_ref = logits.max(axis=-1)  # [B,H,S,nb]
    return block_logit_ref, lse_ref


def _maxerr_finite(got, ref):
    """Masked-pattern match + max abs diff on finite (unmasked) entries."""
    import numpy as np

    got_np = got.astype("float32").numpy()
    ref_np = ref.astype("float32").numpy()
    # -inf blocks may surface as true -inf or as -FLT_MAX from a reduce over an
    # all -inf slice; treat anything below -1e30 as "masked".
    thr = -1e30
    got_masked = got_np <= thr
    ref_masked = ref_np <= thr
    pattern_mismatch = int((got_masked != ref_masked).sum())
    finite = ~ref_masked
    diff = (
        np.abs(got_np[finite] - ref_np[finite])
        if finite.any()
        else np.array([0.0])
    )
    return pattern_mismatch, float(diff.max())


class TestHySparseFA4TopKConsistency(unittest.TestCase):
    # HySparse decompressed-MHA full-attn config knobs (kept small so the test
    # is fast; head dim 256 matches the real d_n(192)+d_r(64) MLA head).
    BLOCK_B = 64
    TOPK = 4

    def _run(self, b, s, h, d, dv, seed=2026):
        _sm100_or_skip(self)
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
            select_topk_blocks,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.pipeline import (
            block_scores_from_logit,
        )

        paddle.seed(seed)
        q = paddle.randn([b, s, h, d], dtype="bfloat16")
        k = paddle.randn([b, s, h, d], dtype="bfloat16")
        v = paddle.randn([b, s, h, dv], dtype="bfloat16")
        sm_scale = 1.0 / math.sqrt(d)
        valid_range = _causal_valid_range(b, s)

        # ---- FA4-fused path: compute score, then TopK from score. ----
        out, lse, block_logit = block_score_fa4_attn_fwd(
            q,
            k,
            v,
            valid_range=valid_range,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            causal=True,
        )
        fa4_idx = select_topk_blocks(
            block_logit, lse, valid_range, self.TOPK, self.BLOCK_B
        )

        # ---- Scatter (eq.3) reference: same two stages, plain matmul. ----
        block_logit_ref, lse_ref = _scatter_reference(
            q, k, sm_scale, self.BLOCK_B
        )
        ref_idx = select_topk_blocks(
            block_logit_ref, lse_ref, valid_range, self.TOPK, self.BLOCK_B
        )

        # (1) scaled per-block max logit agrees (masked pattern + finite values).
        pattern_mismatch, logit_err = _maxerr_finite(
            block_logit, block_logit_ref
        )
        self.assertEqual(
            pattern_mismatch,
            0,
            "block_logit masked pattern mismatch vs scatter ref",
        )
        # bf16 tensor-core MMA error, scaled by sm_scale (raw ~0.06*sqrt(d)).
        logit_tol = 0.06 * math.sqrt(d) * sm_scale
        self.assertLessEqual(
            logit_err,
            logit_tol,
            f"block_logit finite max|diff|={logit_err:.4e} > tol={logit_tol:.4e}",
        )

        # (2) eq.(3) block scores exp(block_logit - lse) agree.
        scores_fa4 = block_scores_from_logit(block_logit, lse)
        scores_ref = block_scores_from_logit(block_logit_ref, lse_ref)
        _, score_err = _maxerr_finite(scores_fa4, scores_ref)
        self.assertLessEqual(
            score_err,
            5e-2,
            f"eq.3 block scores max|diff|={score_err:.4e} too large",
        )

        # (3) end-to-end: TopK block indices from the two score sources match.
        # Compare as sets per query row (TopK order is irrelevant for the
        # downstream gather; -1 padding slots must line up too).
        fa4_sorted = paddle.sort(fa4_idx.astype("int64"), axis=-1)
        ref_sorted = paddle.sort(ref_idx.astype("int64"), axis=-1)
        mismatch = int((fa4_sorted != ref_sorted).astype("int32").sum().item())
        total = fa4_sorted.numel().item()
        # Allow a tiny tie-driven slack: when two blocks have near-equal scores
        # bf16 noise can swap which lands in the last TopK slot.
        self.assertLessEqual(
            mismatch,
            max(1, int(0.005 * total)),
            f"TopK index mismatch {mismatch}/{total} between FA4 and scatter ref",
        )

    def test_topk_consistency_small(self):
        # small heads / head-dim: exercises the score->TopK path end to end.
        self._run(b=1, s=256, h=8, d=64, dv=64)

    def test_topk_consistency_h64(self):
        # split-D head_dim=256 (real HySparse decompressed-MHA full-attn head),
        # H=64 production MLA head count. S=384 (6 blocks) keeps TopK=4 selective
        # while trimming the O(H*S^2) fp32 scatter reference vs a larger S.
        self._run(b=1, s=384, h=64, d=256, dv=256)


if __name__ == "__main__":
    unittest.main()
