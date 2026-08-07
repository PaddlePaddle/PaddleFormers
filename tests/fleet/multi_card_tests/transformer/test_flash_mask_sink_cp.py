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

"""Multi-card tests for FA4 attention-sink under context parallelism.

Verifies that flashmask_attention_cp with a learnable_sink produces forward
output and input/sink gradients matching a single-rank full-sequence reference.
Also covers the refined-recompute CP path
(RefinedRcomputeFlashMaskCpAttention) with a learnable_sink, driven through
``recompute`` to force its two forward passes, vs the non-rr *_cp path.

Key detail: CP uses the DualChunkSwap load-balancing layout (scatter_balance),
so the per-rank input slice AND the reference-output slice must both be taken
with scatter_balance -- NOT contiguous slicing.

Constraints exercised (per FlashMaskContextParallel):
  - causal=False only
  - startend_row_indices.shape[-1] == 2
  - seq divisible by (cp_size * 2)
"""

import math
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

paddle.set_flags({"FLAGS_flash_attn_version": 4})

from paddleformers.fleet.context_parallel_utils import (
    flashmask_attention_cp,
    scatter_balance,
)

DTYPE = paddle.bfloat16
SEED = 2026
COS_SIM_THRESHOLD = 0.95
# A non-trivial sink bias so the sink measurably perturbs the reference output,
# giving the discriminative "closer to with-sink than no-sink" check signal.
SINK_BIAS = 4.0

CP_SIZE = None
CP_RANK = None
CP_GROUP = None


def setUpModule():
    # FA4 attention-sink lives only in the cute backend, built solely for
    # compute capability >= 10 (sm100/Blackwell). Skip before fleet.init so
    # non-sm100 ranks exit cleanly without initializing distributed.
    from paddlefleet_ops import is_flash_mask_available

    if not is_flash_mask_available():
        raise unittest.SkipTest(
            "FA4 attention-sink CP requires the cute backend "
            "(sm100, capability >= 10)"
        )
    global CP_SIZE, CP_RANK, CP_GROUP
    strategy = fleet.DistributedStrategy()
    world = dist.get_world_size()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": world,
        "sep_degree": 1,
        "cp_degree": world,
        "ep_degree": world,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)
    CP_GROUP = fleet.get_hybrid_communicate_group().get_context_parallel_group()
    CP_RANK = CP_GROUP.rank
    CP_SIZE = CP_GROUP.nranks


def _cosine_sim(actual, expected):
    a = actual.cast("float32").flatten()
    b = expected.cast("float32").flatten()
    dot = (a * b).sum()
    return (dot / (a.norm() * b.norm() + 1e-30)).item()


def _noncausal_startend_row_indices(batch_size, seq_len):
    """Full non-causal [b, 1, seq_len, 2] startend_row_indices.

    For a 2-col non-causal mask the columns are (down_start, up_end):
      col0 (down_start): rows >= down_start are masked (-inf) for that key
      col1 (up_end):     rows <  up_end    are masked (-inf) for that key
    So "mask nothing" (full attention) is down_start=seq_len, up_end=0 --
    matching the repo's own generate_non_causal_mask (down_left=seqlen_k,
    up_right=0). The previous (0, seq_len) masked EVERY row -> all-zero output.
    """
    down_start = np.full((batch_size, 1, seq_len, 1), seq_len, dtype=np.int32)
    up_end = np.zeros((batch_size, 1, seq_len, 1), dtype=np.int32)
    indices = np.concatenate([down_start, up_end], axis=-1)
    return paddle.to_tensor(indices, dtype=paddle.int32)


def _full_attention_ref(q, k, v, learnable_sink):
    """fp32 full non-causal attention with optional sink, no masking.

    q,k,v: [b, s, h, d]. learnable_sink: [h] or None. Returns [b, s, h, d].
    """
    qf = q.astype("float32").transpose([0, 2, 1, 3])
    kf = k.astype("float32").transpose([0, 2, 1, 3])
    vf = v.astype("float32").transpose([0, 2, 1, 3])

    h = qf.shape[1]
    d = qf.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    scores = paddle.matmul(qf * softmax_scale, kf, transpose_y=True)

    row_max = scores.max(axis=-1, keepdim=True)
    if learnable_sink is not None:
        sink = learnable_sink.astype("float32").reshape([1, h, 1, 1])
        row_max = paddle.maximum(row_max, sink)
        exp_sink = paddle.exp(sink - row_max)
    else:
        exp_sink = 0.0

    exp_scores = paddle.exp(scores - row_max)
    denom = exp_scores.sum(axis=-1, keepdim=True) + exp_sink
    attention = exp_scores / denom
    out = paddle.matmul(attention.astype(vf.dtype), vf)
    return out.transpose([0, 2, 1, 3])


def _scatter_leaf(full_tensor):
    """Per-rank DualChunkSwap slice of a full tensor as a fresh CP leaf.

    scatter_balance performs the same balanced (both-ends) split that
    flashmask_attention_cp expects its already-sharded inputs to follow.
    We detach first so autograd treats the per-rank slice as a leaf.
    """
    local = scatter_balance(full_tensor.detach(), group=CP_GROUP, axis=1)
    local = local.detach()
    local.stop_gradient = False
    return local


class TestFlashMaskSinkCP(unittest.TestCase):
    """flashmask_attention_cp + learnable_sink vs a full-sequence reference.

    Each rank holds the scatter_balance slice of identical full q/k/v; the
    reference output/grads are sliced the same way for comparison.
    """

    def _full_inputs(self, batch_size, seq_len, nheads, d):
        paddle.seed(SEED)
        np.random.seed(SEED)
        q = paddle.randn([batch_size, seq_len, nheads, d], dtype=DTYPE)
        k = paddle.randn([batch_size, seq_len, nheads, d], dtype=DTYPE)
        v = paddle.randn([batch_size, seq_len, nheads, d], dtype=DTYPE)
        return q, k, v

    def _run(
        self, batch_size, seq_len, nheads, d, use_sink, sink_trainable=True
    ):
        assert seq_len % (CP_SIZE * 2) == 0, (
            f"seq_len {seq_len} must be divisible by cp_size*2 {CP_SIZE * 2}"
        )

        q_full, k_full, v_full = self._full_inputs(
            batch_size, seq_len, nheads, d
        )
        # Full-sequence reference leaves (single-rank golden).
        q_ref, k_ref, v_ref = [
            x.detach().clone() for x in (q_full, k_full, v_full)
        ]
        for t in (q_ref, k_ref, v_ref):
            t.stop_gradient = False

        if use_sink:
            paddle.seed(SEED + 1)
            # Use a large-magnitude sink so its effect on the softmax denom is
            # unmistakable: a path that silently drops the sink will diverge
            # well below COS_SIM_THRESHOLD instead of staying borderline.
            sink_full = paddle.randn([nheads], dtype=DTYPE) + SINK_BIAS
            sink_ref = sink_full.detach().clone()
            sink_ref.stop_gradient = not sink_trainable
            sink_local = sink_full.detach().clone()
            sink_local.stop_gradient = not sink_trainable
        else:
            sink_full = sink_ref = sink_local = None

        # Per-rank scatter_balance slices (the layout CP expects).
        q_local = _scatter_leaf(q_full)
        k_local = _scatter_leaf(k_full)
        v_local = _scatter_leaf(v_full)

        startend_row_indices = _noncausal_startend_row_indices(
            batch_size, seq_len
        )

        out_local = flashmask_attention_cp(
            q_local,
            k_local,
            v_local,
            startend_row_indices,
            causal=False,
            learnable_sink=sink_local,
        )

        # Reference forward over the full sequence, then slice the same way.
        out_ref_full = _full_attention_ref(q_ref, k_ref, v_ref, sink_ref)
        out_ref_local = scatter_balance(
            out_ref_full.detach(), group=CP_GROUP, axis=1
        )

        cos = _cosine_sim(out_local, out_ref_local)
        self.assertGreaterEqual(
            cos,
            COS_SIM_THRESHOLD,
            f"[rank {CP_RANK}] fwd cosine {cos} < {COS_SIM_THRESHOLD}",
        )

        if use_sink:
            # Discriminative check: the sink must materially change the output,
            # and out_local must track the WITH-sink reference rather than the
            # no-sink one. A path that silently drops the sink would instead sit
            # closer to the no-sink reference.
            #
            # Note: a global cosine vs the no-sink output is a poor probe here.
            # With full non-causal attention over `seq_len` keys, the single
            # exp(sink - row_max) term is diluted by the seq_len-way denominator,
            # so the sink only perturbs the output slightly (cosine stays ~0.99)
            # even though it is correctly applied. We therefore compare absolute
            # distances to the two references instead.
            out_nosink_full = _full_attention_ref(q_ref, k_ref, v_ref, None)
            out_nosink_local = scatter_balance(
                out_nosink_full.detach(), group=CP_GROUP, axis=1
            )

            def _max_abs_diff(a, b):
                return (
                    (a.cast("float32") - b.cast("float32")).abs().max().item()
                )

            # Sanity: the sink must actually change the reference output,
            # otherwise this test cannot distinguish "applied" from "ignored".
            ref_sink_effect = _max_abs_diff(out_ref_local, out_nosink_local)
            self.assertGreater(
                ref_sink_effect,
                1e-3,
                f"[rank {CP_RANK}] reference sink effect {ref_sink_effect} is "
                f"too small to be a meaningful probe; increase SINK_BIAS",
            )

            # out_local must be closer to the WITH-sink reference than to the
            # no-sink one; if the sink were silently ignored these would flip.
            dist_to_sink = _max_abs_diff(out_local, out_ref_local)
            dist_to_nosink = _max_abs_diff(out_local, out_nosink_local)
            self.assertLess(
                dist_to_sink,
                dist_to_nosink,
                f"[rank {CP_RANK}] output is closer to the no-sink reference "
                f"(d={dist_to_nosink}) than to the with-sink reference "
                f"(d={dist_to_sink}); learnable_sink is likely being ignored",
            )

        # Backward: identical upstream grad on both sides (sliced for local).
        paddle.seed(SEED + 2)
        g_full = paddle.randn(out_ref_full.shape, dtype=out_ref_full.dtype)
        g_local = scatter_balance(g_full.detach(), group=CP_GROUP, axis=1)

        out_local.backward(g_local.clone())
        out_ref_full.backward(g_full.clone())

        for name, a_full in (
            ("dq", q_ref.grad),
            ("dk", k_ref.grad),
            ("dv", v_ref.grad),
        ):
            ref_local = scatter_balance(a_full.detach(), group=CP_GROUP, axis=1)
            got = {"dq": q_local.grad, "dk": k_local.grad, "dv": v_local.grad}[
                name
            ]
            c = _cosine_sim(got, ref_local)
            self.assertGreaterEqual(
                c,
                COS_SIM_THRESHOLD,
                f"[rank {CP_RANK}] {name} cosine {c} < {COS_SIM_THRESHOLD}",
            )

        if use_sink and sink_trainable:
            # The in-kernel all_reduce is removed: each CP rank's dsink is only
            # the partial sum over the query rows it owns. The global dsink is
            # produced trainer-side (grad scale + allreduce). Here we SUM across
            # the CP group explicitly before it should match the full-sequence
            # reference.
            self.assertIsNotNone(sink_local.grad)
            self.assertEqual(list(sink_local.grad.shape), [nheads])
            self.assertIsNotNone(sink_ref.grad)

            summed = sink_local.grad.detach().clone()
            if CP_SIZE > 1:
                # A single rank's partial sum should NOT already equal the
                # global reference (proves nothing reduces across CP ranks
                # inside the kernel).
                local_gap = (
                    (
                        sink_local.grad.cast("float32")
                        - sink_ref.grad.cast("float32")
                    )
                    .abs()
                    .max()
                    .item()
                )
                self.assertGreater(
                    local_gap,
                    1e-3,
                    f"[rank {CP_RANK}] per-rank dsink already matches the "
                    f"global reference (gap={local_gap}); an in-kernel reduce "
                    f"may still be summing across CP ranks",
                )
                dist.all_reduce(summed, group=CP_GROUP)

            sc = _cosine_sim(summed, sink_ref.grad)
            self.assertGreaterEqual(
                sc,
                COS_SIM_THRESHOLD,
                f"[rank {CP_RANK}] summed dsink cosine {sc} < {COS_SIM_THRESHOLD}",
            )
        elif use_sink and not sink_trainable:
            # Fixed off-by-one sink: stop_gradient -> CP returns the 3-tuple,
            # so the sink leaf must receive no gradient.
            self.assertIsNone(sink_local.grad)

    def test_cp_trainable_sink(self):
        self._run(2, 256, 4, 128, use_sink=True)

    def test_cp_sink_none(self):
        self._run(2, 256, 4, 128, use_sink=False)

    def test_cp_offbyone_fixed_sink(self):
        # Fixed (stop_gradient) sink -> backward returns 3-tuple, grad is None.
        self._run(2, 256, 4, 128, use_sink=True, sink_trainable=False)

    def test_cp_small_head_dim_sink(self):
        self._run(2, 128, 4, 64, use_sink=True)


class TestRefinedRecomputeFlashMaskSinkCP(unittest.TestCase):
    """rr-CP FlashMask attention + learnable_sink.

    The rr-CP path keys off ``framework._dygraph_tracer()._has_grad`` to pick
    between two forward passes: the first (``_has_grad`` False) runs the real CP
    kernel under no_grad and stashes tensors; the second (``_has_grad`` True)
    rebuilds the graph via ``FlashMaskAttnCpFunctor``, whose custom backward
    computes the grads. We drive both passes manually so the test exercises the
    rr-CP functor's fwd/bwd directly, and compare against the non-rr
    ``flashmask_attention_cp`` sink path on the same scatter_balance layout.
    """

    def _full_inputs(self, batch_size, seq_len, nheads, d):
        paddle.seed(SEED)
        np.random.seed(SEED)
        q = paddle.randn([batch_size, seq_len, nheads, d], dtype=DTYPE)
        k = paddle.randn([batch_size, seq_len, nheads, d], dtype=DTYPE)
        v = paddle.randn([batch_size, seq_len, nheads, d], dtype=DTYPE)
        return q, k, v

    def _run(self, batch_size, seq_len, nheads, d, sink_trainable=True):
        from paddle import framework

        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        assert seq_len % (CP_SIZE * 2) == 0

        q_full, k_full, v_full = self._full_inputs(
            batch_size, seq_len, nheads, d
        )

        paddle.seed(SEED + 1)
        sink_full = paddle.randn([nheads], dtype=DTYPE) + SINK_BIAS

        startend_row_indices = _noncausal_startend_row_indices(
            batch_size, seq_len
        )

        # Non-rr reference on the same per-rank scatter_balance layout.
        q_ref = _scatter_leaf(q_full)
        k_ref = _scatter_leaf(k_full)
        v_ref = _scatter_leaf(v_full)
        sink_ref = sink_full.detach().clone()
        sink_ref.stop_gradient = not sink_trainable
        out_ref = flashmask_attention_cp(
            q_ref,
            k_ref,
            v_ref,
            startend_row_indices,
            causal=False,
            learnable_sink=sink_ref,
        )

        # rr-CP path: drive the two-pass mechanism manually.
        q_local = _scatter_leaf(q_full)
        k_local = _scatter_leaf(k_full)
        v_local = _scatter_leaf(v_full)
        sink_local = sink_full.detach().clone()
        sink_local.stop_gradient = not sink_trainable

        rr_attn = RefinedRcomputeFlashMaskCpAttention()
        tracer = framework._dygraph_tracer()
        prev_has_grad = tracer._has_grad

        tracer._has_grad = False
        try:
            rr_attn(
                q_local,
                k_local,
                v_local,
                startend_row_indices,
                causal=False,
                learnable_sink=sink_local,
            )
        finally:
            tracer._has_grad = prev_has_grad

        tracer._has_grad = True
        try:
            out_local = rr_attn(
                q_local,
                k_local,
                v_local,
                startend_row_indices,
                causal=False,
                learnable_sink=sink_local,
            )
        finally:
            tracer._has_grad = prev_has_grad

        cos = _cosine_sim(out_local, out_ref)
        self.assertGreaterEqual(
            cos,
            COS_SIM_THRESHOLD,
            f"[rank {CP_RANK}] rr-CP fwd cosine {cos} < {COS_SIM_THRESHOLD}",
        )

        paddle.seed(SEED + 2)
        g = paddle.randn(out_ref.shape, dtype=out_ref.dtype)
        out_local.backward(g.clone())
        out_ref.backward(g.clone())

        for name, got, ref in (
            ("dq", q_local.grad, q_ref.grad),
            ("dk", k_local.grad, k_ref.grad),
            ("dv", v_local.grad, v_ref.grad),
        ):
            self.assertIsNotNone(got, f"[rank {CP_RANK}] rr-CP {name} is None")
            c = _cosine_sim(got, ref)
            self.assertGreaterEqual(
                c,
                COS_SIM_THRESHOLD,
                f"[rank {CP_RANK}] rr-CP {name} cosine {c} < {COS_SIM_THRESHOLD}",
            )

        if sink_trainable:
            self.assertIsNotNone(sink_local.grad)
            self.assertEqual(list(sink_local.grad.shape), [nheads])
            self.assertIsNotNone(sink_ref.grad)
            sc = _cosine_sim(sink_local.grad, sink_ref.grad)
            self.assertGreaterEqual(
                sc,
                COS_SIM_THRESHOLD,
                f"[rank {CP_RANK}] rr-CP dsink cosine {sc} < {COS_SIM_THRESHOLD}",
            )
        else:
            # Fixed sink is stop_gradient -> rr-CP returns no sink grad.
            self.assertIsNone(sink_local.grad)

    def test_rr_cp_trainable_sink(self):
        self._run(2, 256, 4, 128)

    def test_rr_cp_fixed_sink_no_grad(self):
        self._run(2, 256, 4, 128, sink_trainable=False)


if __name__ == "__main__":
    unittest.main()
