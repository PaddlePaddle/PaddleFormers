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

"""Regression test: LinearWithGradAccumulationAndAsyncCommunication.backward
honors Paddle's PyLayer contract when ``input.stop_gradient=True``.

Background
----------
``LinearWithGradAccumulationAndAsyncCommunication`` (a ``paddle.autograd.Function``
used internally by ``ColumnParallelLinear`` / ``RowParallelLinear``) is invoked
through ``apply`` with three Tensor inputs ``(input, weight, bias)``.

Paddle's PyLayer C++ contract (py_layer_node.cc) is strict:

    For any Tensor input whose stop_gradient was True in forward, the
    corresponding position in backward's return tuple MUST be None.

Historically this Function computed ``grad_input = grad_output @ weight.t()``
unconditionally and returned it at position 0, which violated the contract
whenever a caller passed a detached tensor as ``input`` (e.g. for gradient
isolation -- the CSA attention indexer training path: ``x_det = x.detach()``
fed into the indexer's ``ColumnParallelLinear``). That triggered:

    InvalidArgumentError: GradNodePyLayer_LinearWithGradAccumulationAndAsyncCommunication's
    backward function should return None at 0 position, because it's forward
    Tensor's stopgradient is true.

(Note: PyTorch's ``torch.autograd.Function`` has a more lenient contract --
returning a tensor for a non-grad input is silently discarded -- so the same
code shape works in Megatron-LM upstream but crashed here until fixed.)

What this test does
-------------------
Calls the Function directly with a detached input on CPU, no TP, no SP, no
grad-fusion, and asserts that:

  * ``backward()`` completes cleanly (no PyLayer contract error)
  * the detached ``input`` does NOT receive a ``.grad``
  * the trainable ``weight`` DOES receive a finite ``.grad``

Failure modes guarded against:

  * Future regression of the unconditional ``grad_input`` computation.
  * Accidentally suppressing ``grad_weight`` when input is detached.
"""

import unittest

import paddle

from paddleformers.fleet.tensor_parallel.layers import (
    LinearWithGradAccumulationAndAsyncCommunication,
)


class TestLinearGradAccumDetachedInput(unittest.TestCase):
    """Detached-input backward must skip grad_input but still produce grad_weight."""

    def setUp(self):
        paddle.seed(0)
        self.s, self.b, self.h_in, self.h_out = 4, 2, 16, 8

    def _make_inputs(self):
        # Mimic a typical "gradient isolation" pattern: an upstream tensor is
        # detached before being fed to a trainable Linear (so gradients should
        # NOT flow back to the upstream graph through this linear).
        x = paddle.randn([self.s, self.b, self.h_in], dtype="float32")
        input_det = x.detach()
        assert input_det.stop_gradient is True

        # Trainable weight (matches a ColumnParallelLinear weight).
        weight = paddle.randn([self.h_in, self.h_out], dtype="float32")
        weight.stop_gradient = False
        return input_det, weight

    def test_backward_with_detached_input_completes_cleanly(self):
        input_det, weight = self._make_inputs()

        output = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_det,
            weight,
            None,  # bias
            False,  # gradient_accumulation_fusion
            False,  # allreduce_dgrad
            False,  # sequence_parallel
            None,  # grad_output_buffer
            0,  # wgrad_deferral_limit
            None,  # tp_group
        )

        # Must NOT raise: backward must return None at position 0 since
        # input_det.stop_gradient=True.
        try:
            output.sum().backward()
        except Exception as e:
            self.fail(
                f"backward() raised {type(e).__name__}: {e}\n"
                "Expected clean backward when input.stop_gradient=True; this "
                "is a regression of the PyLayer contract fix."
            )

        # Detached input must not receive a gradient.
        self.assertIsNone(
            input_det.grad,
            "input_det.grad must be None (input.stop_gradient was True)",
        )
        # Trainable weight must still receive a finite gradient.
        self.assertIsNotNone(
            weight.grad,
            "weight.grad must be computed even when input is detached",
        )
        self.assertEqual(list(weight.grad.shape), [self.h_in, self.h_out])
        self.assertTrue(paddle.isfinite(weight.grad).all().item())


if __name__ == "__main__":
    unittest.main()
