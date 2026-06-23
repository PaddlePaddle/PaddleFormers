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

"""CPU-runnable routing tests for the multimax fused-CE branch of
``LanguageLoss.forward_impl``.

These tests stub ``LigerFusedLinearCrossEntropyFunction.apply`` (the GPU/Triton
fused kernel) to capture the forwarded arguments, so the routing logic can be
exercised on CPU. They protect against regressions in:

- 3-tuple emission (no multimax) still works and does NOT append extra args.
- 5-tuple emission (multimax enabled) appends ``multimax_ranges`` /
  ``multimax_ts`` in that order at positions 8 and 9.
- The fused branch only fires when ``logits`` is a tuple; tensor inputs are
  routed to the regular CE path (covered indirectly by other tests).
"""

import types
import unittest
from unittest import mock

import paddle

import paddlefleet_ops

# sonicmoe ecosystem op is not always loadable in CI envs; the multimax
# feature does not depend on it, so neutralize the gating before importing
# anything that pulls paddlefleet.models.
paddlefleet_ops.is_sonic_moe_available = lambda: False


def _capturing_apply():
    """Return a mock for LigerFusedLinearCrossEntropyFunction.apply that:
    - records the forwarded *args in ``calls`` for inspection,
    - returns a deterministic [BT] float32 tensor of zeros so the caller's
      reshape/lossmask logic still runs end-to-end.
    """
    calls = []

    def _apply(*args):
        calls.append(args)
        # args[0] is _input shape [BT, H]; return [BT] zeros.
        bt = args[0].shape[0]
        return paddle.zeros([bt], dtype=paddle.float32)

    return calls, _apply


class _StubLanguageLoss:
    """Minimal stand-in exposing only the attributes ``forward_impl`` reads.

    Constructing the real ``LanguageLoss`` requires fleet init + parallel
    state; we don't need any of that to test the multimax routing branch,
    which only depends on a few config flags. We bind ``forward_impl`` as
    an unbound method to keep the production code path under test.
    """

    def __init__(self, fused_chunk=1, parallel_ce=False, ignored_index=-100):
        self.config = types.SimpleNamespace(
            fused_linear_ce_loss_chunk=fused_chunk,
            gpt_model_use_experimental_version=False,
            cp_balance_mode="default",
        )
        self.enable_parallel_cross_entropy = parallel_ce
        self.ignored_index = ignored_index


class TestLanguageLossMultimaxRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import here so the sonicmoe gate above is in effect first.
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        cls.LanguageLoss = LanguageLoss

    def _make_inputs(self, B=2, S=4, H=8):
        hidden = paddle.randn([B, S, H], dtype="float32")
        # weight is [V, H]; vocab size doesn't matter for routing.
        V = 16
        weight = paddle.randn([V, H], dtype="float32")
        bias = None
        labels = paddle.randint(0, V, [B, S], dtype="int64")
        return hidden, weight, bias, labels, B, S

    def _patch_target(self):
        # Patch the symbol where it is *imported* (inside forward_impl, the
        # function is imported lazily from paddleformers.fleet.triton_ops.fused_...).
        return mock.patch(
            "paddleformers.fleet.triton_ops.fused_linear_cross_entropy."
            "LigerFusedLinearCrossEntropyFunction.apply",
            autospec=False,
        )

    def _bypass_cp_gather(self):
        # On a single-card CPU run there is no context-parallel group; the
        # call to ``get_context_parallel_world_size`` returns 1 anyway, so
        # the gather branch is naturally skipped. We patch it to be safe in
        # case the import path differs at module load time.
        return mock.patch(
            "paddleformers.fleet.models.common.language_loss.language_loss."
            "get_context_parallel_world_size",
            return_value=1,
        )

    def test_3tuple_routes_without_multimax_args(self):
        """3-tuple input: apply() must be called with 8 positional args; no
        multimax tensors appended."""
        hidden, weight, bias, labels, B, S = self._make_inputs()
        stub = _StubLanguageLoss(fused_chunk=1)
        calls, capture = _capturing_apply()

        with self._patch_target() as p, self._bypass_cp_gather():
            p.side_effect = capture
            loss = self.LanguageLoss.forward_impl(
                stub, (hidden, weight, bias), labels
            )

        self.assertEqual(len(calls), 1, "apply must be called exactly once")
        forwarded = calls[0]
        # Expected positional layout (see fused_linear_cross_entropy.py):
        # 0:_input 1:weight 2:target 3:bias 4:ignore_index 5:reduction
        # 6:num_chunks 7:ec_align  (8/9 only when multimax)
        self.assertEqual(
            len(forwarded),
            8,
            f"expected 8 args without multimax, got {len(forwarded)}",
        )
        self.assertEqual(forwarded[3], None, "bias must be forwarded as None")
        self.assertEqual(
            forwarded[4], stub.ignored_index, "ignore_index mismatch"
        )
        self.assertEqual(forwarded[5], "none", "reduction must be 'none'")
        self.assertEqual(
            forwarded[6],
            stub.config.fused_linear_ce_loss_chunk,
            "num_chunks mismatch",
        )
        self.assertIsInstance(loss, paddle.Tensor)

    def test_5tuple_routes_with_multimax_args(self):
        """5-tuple input: apply() must receive 10 positional args, with
        multimax_ranges at index 8 and multimax_ts at index 9."""
        hidden, weight, bias, labels, B, S = self._make_inputs()
        ranges = paddle.zeros([4], dtype="float32")
        ts = paddle.zeros([4], dtype="float32")

        stub = _StubLanguageLoss(fused_chunk=2)
        calls, capture = _capturing_apply()

        with self._patch_target() as p, self._bypass_cp_gather():
            p.side_effect = capture
            self.LanguageLoss.forward_impl(
                stub, (hidden, weight, bias, ranges, ts), labels
            )

        self.assertEqual(len(calls), 1)
        forwarded = calls[0]
        self.assertEqual(
            len(forwarded),
            10,
            f"expected 10 args with multimax, got {len(forwarded)}",
        )
        # Identity by `is` to confirm exactly the same tensors are forwarded
        # (no implicit copy/cast in the routing layer).
        self.assertIs(forwarded[8], ranges, "multimax_ranges position wrong")
        self.assertIs(forwarded[9], ts, "multimax_ts position wrong")
        self.assertEqual(
            forwarded[6],
            stub.config.fused_linear_ce_loss_chunk,
            "num_chunks mismatch",
        )

    def test_parallel_ce_with_tuple_logits_asserts(self):
        """Tensor-parallel parallel_output=True is incompatible with the
        fused tuple path; LanguageLoss must assert rather than silently
        producing wrong gradients."""
        hidden, weight, bias, labels, _B, _S = self._make_inputs()
        stub = _StubLanguageLoss(fused_chunk=1, parallel_ce=True)
        with self.assertRaises(AssertionError):
            self.LanguageLoss.forward_impl(stub, (hidden, weight, bias), labels)


if __name__ == "__main__":
    unittest.main()
