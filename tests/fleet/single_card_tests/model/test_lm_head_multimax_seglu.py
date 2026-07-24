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

"""Targeted unit tests for the multimax SegLU function used by GPTLMHead.

Covers the pure-math layer of the multimax feature:
- SegLU(x, ranges=0, ts=0) is the identity (resume safety / step-0 invariant).
- SegLU is element-wise.
- SegLU does NOT mutate the input tensor (must clone for autograd correctness).
- SegLU has the expected value at simple analytic points.
"""

import unittest

import paddle

import paddlefleet_ops

# sonicmoe ecosystem op is not available in some CI envs; the SegLU function
# itself does not depend on it, so neutralize the gating before importing
# anything from paddleformers.fleet.models.gpt.
paddlefleet_ops.is_sonic_moe_available = lambda: False


class TestSegLU(unittest.TestCase):
    """Math-level tests for the multimax SegLU activation."""

    @classmethod
    def setUpClass(cls):
        from paddleformers.fleet.models.gpt.lm_head import SegLU

        cls.SegLU = staticmethod(SegLU)

    def test_zero_params_is_identity(self):
        """ranges=ts=0 -> SegLU should be the identity (init-time invariant)."""
        x = paddle.randn([2, 3, 5], dtype="float32")
        ranges = paddle.zeros([4], dtype="float32")
        ts = paddle.zeros([4], dtype="float32")
        y = self.SegLU(x, ranges, ts)
        self.assertEqual(list(y.shape), list(x.shape))
        # Identity at init guarantees resume from checkpoints lacking these
        # params produces bit-identical logits at step 0.
        self.assertTrue(
            paddle.allclose(y, x, atol=1e-6).item(),
            "SegLU(x, 0, 0) must equal x exactly",
        )

    def test_input_not_mutated(self):
        """SegLU must clone before in-place +=; mutating x would corrupt autograd."""
        x = paddle.randn([4, 7], dtype="float32")
        x_orig = x.clone()
        ranges = paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype="float32")
        ts = paddle.to_tensor([0.5, 0.5, 0.1, 0.1], dtype="float32")
        _ = self.SegLU(x, ranges, ts)
        # x must be unchanged after the call
        self.assertTrue(
            paddle.allclose(x, x_orig, atol=0.0).item(),
            "SegLU must not mutate its input tensor",
        )

    def test_elementwise_shape_preserved(self):
        """Output shape == input shape across various rank tensors."""
        ranges = paddle.zeros([4], dtype="float32")
        ts = paddle.to_tensor([0.1, 0.2, 0.3, 0.4], dtype="float32")
        for shape in [[8], [3, 16], [2, 4, 32], [1, 2, 3, 64]]:
            x = paddle.randn(shape, dtype="float32")
            y = self.SegLU(x, ranges, ts)
            self.assertEqual(list(y.shape), shape)

    def test_inside_window_no_modulation(self):
        """For a value strictly inside [ranges[1], ranges[0]] = (-1, 1) and
        away from the squared cutoffs at ranges[2..3]=0, SegLU should equal x.

        Linear part: ts[0]*relu(ranges[0]-x) is 0 when x>ranges[0],
                     ts[1]*relu(x-ranges[1]) is 0 when x<ranges[1].
        Quadratic part: ts[2]*relu(ranges[2]-x)**2 is 0 when x>ranges[2],
                        ts[3]*relu(x-ranges[3])**2 is 0 when x<ranges[3].
        Choose ranges such that all four relu(...) args are negative -> SegLU
        is identity for any x.
        """
        x = paddle.to_tensor([-2.0, 0.0, 2.0], dtype="float32")
        ts = paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype="float32")
        # relu(ranges[0] - x) = 0 iff ranges[0] <= x  -> use ranges[0] = -1e30
        # relu(x - ranges[1]) = 0 iff x <= ranges[1]  -> use ranges[1] = +1e30
        # relu(ranges[2] - x) = 0 iff ranges[2] <= x  -> use ranges[2] = -1e30
        # relu(x - ranges[3]) = 0 iff x <= ranges[3]  -> use ranges[3] = +1e30
        ranges = paddle.to_tensor([-1e30, 1e30, -1e30, 1e30], dtype="float32")
        y = self.SegLU(x, ranges, ts)
        self.assertTrue(paddle.allclose(y, x, atol=0.0).item())

    def test_known_linear_modulation(self):
        """With ts=[1,0,0,0] and ranges=[1,0,0,0], SegLU(x) = x + relu(1 - x).

        For x = 0:   1 + relu(1) = 0 + 1 = 1
        For x = 0.5: 0.5 + relu(0.5) = 1.0
        For x = 2:   2 + relu(-1) = 2 + 0 = 2
        """
        x = paddle.to_tensor([0.0, 0.5, 2.0], dtype="float32")
        ranges = paddle.to_tensor([1.0, 0.0, 0.0, 0.0], dtype="float32")
        ts = paddle.to_tensor([1.0, 0.0, 0.0, 0.0], dtype="float32")
        y = self.SegLU(x, ranges, ts)
        expected = paddle.to_tensor([1.0, 1.0, 2.0], dtype="float32")
        self.assertTrue(
            paddle.allclose(y, expected, atol=1e-6).item(),
            f"got {y.numpy().tolist()}, expected {expected.numpy().tolist()}",
        )

    def test_known_quadratic_modulation(self):
        """With ts=[0,0,1,0] and ranges=[0,0,1,0], SegLU(x) = x + relu(1 - x)**2.

        For x = -1: -1 + relu(2)**2 = -1 + 4 = 3
        For x = 0:   0 + relu(1)**2 = 0 + 1 = 1
        For x = 2:   2 + relu(-1)**2 = 2 + 0 = 2
        """
        x = paddle.to_tensor([-1.0, 0.0, 2.0], dtype="float32")
        ranges = paddle.to_tensor([0.0, 0.0, 1.0, 0.0], dtype="float32")
        ts = paddle.to_tensor([0.0, 0.0, 1.0, 0.0], dtype="float32")
        y = self.SegLU(x, ranges, ts)
        expected = paddle.to_tensor([3.0, 1.0, 2.0], dtype="float32")
        self.assertTrue(
            paddle.allclose(y, expected, atol=1e-6).item(),
            f"got {y.numpy().tolist()}, expected {expected.numpy().tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
