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

"""Negative-control test: FusedDSAIndexerLoss is NOT the source of the
"backward should return None" PyLayer contract crash.

Why this test exists
--------------------
``FusedDSAIndexerLoss.backward`` returns ``(grad_q, grad_weights, grad_k,
None, None, None)`` for its 6 Tensor inputs ``(q, weights, k, query, key, mask)``.
A natural worry is: if all 6 inputs are passed in with ``stop_gradient=True``,
positions 0/1/2 would return non-None and violate Paddle's PyLayer contract
(which requires ``None`` at any position whose forward input had
``stop_gradient=True``).

This test verifies that worry is unfounded. When every Tensor input is
stop_gradient=True, the output ``loss`` itself ends up stop_gradient=True too,
so ``loss.backward()`` is a no-op (no backward node fires) and no contract
check is ever invoked. The test PASSES, demonstrating that this PyLayer is
safe to call with all-detached inputs even with the current backward
implementation.

(The real source of the production "backward should return None at 0
position" crash for the CSA indexer training path is the TP linear PyLayer
``LinearWithGradAccumulationAndAsyncCommunication``; see the dedicated
reproducer in fleet_tests/tensor_parallel/.)

Keep this test as a regression guard: if someone later changes
FusedDSAIndexerLoss in a way that makes its output stop_gradient=False under
all-detached inputs, the contract violation will surface and this test will
fail loudly.
"""

import unittest

import paddle

from paddleformers.fleet.transformer.dsa_attention import FusedDSAIndexerLoss


class TestFusedDSAIndexerLossDetachedInputs(unittest.TestCase):
    """Verify FusedDSAIndexerLoss is safe under all-detached Tensor inputs.

    Setup mimics a hypothetical caller that detaches every Tensor input
    before invoking ``FusedDSAIndexerLoss.apply``. Expected behavior:

      * the returned ``loss`` is stop_gradient=True
      * ``loss.backward()`` is a no-op (no backward node fires)
      * none of the inputs receive ``.grad``

    See module docstring for full rationale.
    """

    def setUp(self):
        paddle.seed(0)
        # Indexer dims
        self.sq, self.sk = 8, 8
        self.b = 2
        self.h, self.d = 4, 32
        # MLA dims
        self.np, self.hn = 4, 64
        self.topk = 4
        self.softmax_scale = self.hn**-0.5

    def _make_detached_inputs(self):
        """Build inputs where every Tensor is stop_gradient=True.

        Returns 6 tensors (q, weights, k, query, key, mask) so PyLayer sees
        the same 6-tensor signature as the 6 returns from ``backward``.
        """
        q = paddle.randn(
            [self.b, self.sq, self.h, self.d], dtype="float32"
        ).detach()
        weights = paddle.randn(
            [self.b, self.sq, self.h], dtype="float32"
        ).detach()
        k = paddle.randn([self.b, self.sk, self.d], dtype="float32").detach()
        query = paddle.randn(
            [self.b, self.sq, self.np, self.hn], dtype="float32"
        ).detach()
        key = paddle.randn(
            [self.b, self.sk, self.np, self.hn], dtype="float32"
        ).detach()

        # Sanity: every tensor input must be stop_gradient=True for this test
        # to exercise the all-detached topology.
        for name, t in [
            ("q", q),
            ("weights", weights),
            ("k", k),
            ("query", query),
            ("key", key),
        ]:
            assert t.stop_gradient is True, (
                f"{name} should be stop_gradient=True after detach()"
            )

        causal = paddle.triu(
            paddle.full([self.sq, self.sk], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, sq, sk]

        return q, weights, k, query, key, mask

    def test_backward_with_all_detached_inputs_does_not_violate_pylayer_contract(
        self,
    ):
        q, weights, k, query, key, mask = self._make_detached_inputs()

        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            mask,
            False,
            None,
        )

        # When every Tensor input is stop_gradient=True, Paddle propagates
        # stop_gradient=True onto the output, so backward() is a no-op and
        # the PyLayer's backward implementation is never called -- hence no
        # "should return None at X position" check is triggered.
        try:
            loss.backward()
        except Exception as e:
            self.fail(
                "FusedDSAIndexerLoss.backward() with all-detached inputs raised "
                f"{type(e).__name__}: {e}\n"
                "Expected backward() to be a silent no-op (output should inherit "
                "stop_gradient=True from all-detached inputs)."
            )

        # No .grad should have been written to any input.
        self.assertIsNone(
            q.grad, "q.grad must be None for stop_gradient=True input"
        )
        self.assertIsNone(
            weights.grad,
            "weights.grad must be None for stop_gradient=True input",
        )
        self.assertIsNone(
            k.grad, "k.grad must be None for stop_gradient=True input"
        )


if __name__ == "__main__":
    unittest.main()
