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

"""CPU-runnable tests for the PyLayer wrapper around the fused-linear-CE
Triton kernel.

These tests stub the GPU-only inner functions
(``fused_linear_cross_entropy_forward`` and
``fused_linear_cross_entropy_backward``) so the PyLayer's autograd plumbing —
specifically the per-parameter ``stop_gradient`` caching in ``forward`` and
the freeze-respecting branches in ``backward`` — can be exercised on CPU.

Regression coverage for:
- multimax_ranges/multimax_ts caching in ``forward``.
- backward returning ``None`` for frozen multimax params and not touching
  their ``main_grad`` / backward hooks, even when the kernel emitted a
  gradient tensor (multimax_requires_grad is the OR of the two params).
- backward continuing to populate the trainable param when only one of the
  pair is frozen.
"""

import unittest
from unittest import mock

import paddle

from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
    fused_linear_cross_entropy as fused_module,
)


def _stub_forward(
    _input,
    weight,
    target,
    bias,
    ignore_index,
    reduction,
    num_chunks,
    ec_align,
    multimax_ranges=None,
    multimax_ts=None,
):
    """Drop-in replacement that returns shape-correct fake grads.

    Mirrors the real function's return-tuple contract:
    - 4-tuple when multimax is disabled,
    - 6-tuple when multimax is enabled.

    The ``multimax_requires_grad`` flag in the real kernel is the OR of the
    two params' grad-requirements, so the kernel ALWAYS emits both grads
    when multimax is enabled. We mimic that here by always returning
    non-None grads for both ranges and ts when the params are present —
    that's exactly the pathological case the backward must handle.
    """
    BT = _input.shape[0]
    loss = paddle.zeros([BT], dtype=paddle.float32)
    grad_input = paddle.ones_like(_input) if not _input.stop_gradient else None
    grad_weight = paddle.ones_like(weight) if not weight.stop_gradient else None
    grad_bias = (
        paddle.ones_like(bias)
        if (bias is not None and not bias.stop_gradient)
        else None
    )
    if multimax_ranges is None:
        return loss, grad_input, grad_weight, grad_bias
    grad_mm_ranges = paddle.full(multimax_ranges.shape, 7.0, dtype="float32")
    grad_mm_ts = paddle.full(multimax_ts.shape, 11.0, dtype="float32")
    return loss, grad_input, grad_weight, grad_bias, grad_mm_ranges, grad_mm_ts


def _stub_backward(grad_output, grad_input, grad_weight, grad_bias):
    """Identity pass-through; the PyLayer's logic post-backward is what
    we're testing, not the scaling math."""
    return grad_input, grad_weight, grad_bias


class TestPyLayerMultimaxFreeze(unittest.TestCase):
    def setUp(self):
        self._fwd_patch = mock.patch.object(
            fused_module,
            "fused_linear_cross_entropy_forward",
            side_effect=_stub_forward,
        )
        self._bwd_patch = mock.patch.object(
            fused_module,
            "fused_linear_cross_entropy_backward",
            side_effect=_stub_backward,
        )
        self._fwd_patch.start()
        self._bwd_patch.start()
        self.addCleanup(self._fwd_patch.stop)
        self.addCleanup(self._bwd_patch.stop)

    def _make_inputs(self, BT=4, H=8, V=16):
        x = paddle.randn([BT, H], dtype="float32")
        w = paddle.randn([V, H], dtype="float32")
        target = paddle.randint(0, V, [BT])
        ranges = paddle.to_tensor([0.0, 1.0, 0.0, 1.0])
        ts = paddle.to_tensor([0.1, 0.1, 0.1, 0.1])
        return x, w, target, ranges, ts

    def test_both_multimax_trainable(self):
        x, w, target, ranges, ts = self._make_inputs()
        x.stop_gradient = False
        w.stop_gradient = False
        ranges.stop_gradient = False
        ts.stop_gradient = False

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x, w, target, None, -100, "none", 1, False, ranges, ts
        ).sum()
        loss.backward()

        self.assertIsNotNone(ranges.grad)
        self.assertIsNotNone(ts.grad)

    def test_only_ranges_frozen(self):
        """multimax_ranges frozen, multimax_ts trainable: kernel emits both
        grads (OR semantics), but backward must skip the frozen one."""
        x, w, target, ranges, ts = self._make_inputs()
        x.stop_gradient = False
        w.stop_gradient = False
        ranges.stop_gradient = True  # frozen
        ts.stop_gradient = False

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x, w, target, None, -100, "none", 1, False, ranges, ts
        ).sum()
        loss.backward()

        # Frozen param must not receive a grad.
        self.assertIsNone(ranges.grad)
        # Trainable param still gets its grad.
        self.assertIsNotNone(ts.grad)

    def test_only_ts_frozen(self):
        x, w, target, ranges, ts = self._make_inputs()
        x.stop_gradient = False
        w.stop_gradient = False
        ranges.stop_gradient = False
        ts.stop_gradient = True  # frozen

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x, w, target, None, -100, "none", 1, False, ranges, ts
        ).sum()
        loss.backward()

        self.assertIsNone(ts.grad)
        self.assertIsNotNone(ranges.grad)

    def test_both_frozen_no_main_grad_writes(self):
        """Both multimax params frozen + ``main_grad`` attribute present:
        backward must NOT touch ``main_grad`` for either param even though
        the kernel produced grads."""
        x, w, target, ranges, ts = self._make_inputs()
        x.stop_gradient = False
        w.stop_gradient = False
        ranges.stop_gradient = True
        ts.stop_gradient = True
        # Pre-set main_grad to a sentinel; assert it stays untouched.
        ranges.main_grad = paddle.full(ranges.shape, 99.0, dtype="float32")
        ts.main_grad = paddle.full(ts.shape, 99.0, dtype="float32")
        hook_fired = {"ranges": False, "ts": False}
        ranges._apply_backward_hook = lambda: hook_fired.__setitem__(
            "ranges", True
        )
        ts._apply_backward_hook = lambda: hook_fired.__setitem__("ts", True)

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x, w, target, None, -100, "none", 1, False, ranges, ts
        ).sum()
        loss.backward()

        self.assertTrue(
            paddle.all(ranges.main_grad == 99.0).item(),
            "frozen multimax_ranges.main_grad was modified",
        )
        self.assertTrue(
            paddle.all(ts.main_grad == 99.0).item(),
            "frozen multimax_ts.main_grad was modified",
        )
        self.assertFalse(hook_fired["ranges"])
        self.assertFalse(hook_fired["ts"])

    def test_partial_freeze_with_main_grad(self):
        """Only ts frozen, both have ``main_grad``. ranges' main_grad must
        get accumulated; ts' main_grad must stay untouched."""
        x, w, target, ranges, ts = self._make_inputs()
        x.stop_gradient = False
        w.stop_gradient = False
        ranges.stop_gradient = False
        ts.stop_gradient = True  # frozen
        ranges.main_grad = paddle.zeros(ranges.shape, dtype="float32")
        ts.main_grad = paddle.full(ts.shape, 99.0, dtype="float32")

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x, w, target, None, -100, "none", 1, False, ranges, ts
        ).sum()
        loss.backward()

        # ranges' main_grad got the kernel-emitted 7.0 accumulated.
        self.assertTrue(
            paddle.all(ranges.main_grad == 7.0).item(),
            f"ranges.main_grad expected 7.0, got "
            f"{ranges.main_grad.numpy().tolist()}",
        )
        # ts' main_grad untouched (frozen).
        self.assertTrue(
            paddle.all(ts.main_grad == 99.0).item(),
            "frozen ts.main_grad was modified",
        )


if __name__ == "__main__":
    unittest.main()
