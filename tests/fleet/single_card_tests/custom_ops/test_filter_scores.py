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

import numpy as np
import paddle

from paddlefleet_ops import filter_scores, filter_scores_grad


class _FilterScores(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, probs, indices):
        topk_scores = filter_scores(probs, indices)
        ctx.save_for_backward(indices)
        return topk_scores

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensor()
        probs_grad = filter_scores_grad(indices, grad_output)
        return probs_grad, None


class TestFilterScoresOp(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        self.filter_scores_op = _FilterScores.apply

    def _get_numpy_reference(self, probs_np, indices_np):
        flat_probs = probs_np.flatten()
        flat_indices = indices_np.flatten()
        valid_mask = flat_indices != -1
        expected_scores = flat_probs[valid_mask]
        return expected_scores

    def _run_forward_test(self, shape, dtype):
        probs_np = np.random.rand(*shape).astype(dtype)
        indices_np = np.random.randint(low=0, high=100, size=shape).astype(
            np.int64
        )
        invalid_mask = np.random.choice([True, False], size=shape, p=[0.5, 0.5])
        indices_np[invalid_mask] = -1
        expected_topk_scores = self._get_numpy_reference(probs_np, indices_np)

        probs = paddle.to_tensor(probs_np, place=paddle.CUDAPlace(0))
        indices = paddle.to_tensor(indices_np, place=paddle.CUDAPlace(0))
        probs.stop_gradient = False

        topk_scores = self.filter_scores_op(probs, indices)

        np.testing.assert_allclose(
            topk_scores.numpy(),
            expected_topk_scores,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Forward pass failed for shape={shape}, dtype={dtype}",
        )
        print(f"Forward test passed for shape={shape}, dtype={dtype}")

        return probs, indices, topk_scores

    def test_forward_float32(self):
        self._run_forward_test(shape=(8, 128), dtype="float32")

    def test_gradient(self):
        shape = (16, 64)
        dtype = "float32"

        probs, indices, topk_scores = self._run_forward_test(
            shape=shape, dtype=dtype
        )

        grads = paddle.grad(
            outputs=[topk_scores.sum()],
            inputs=[probs],
        )
        probs_grad = grads[0]

        indices_np = indices.numpy()
        expected_grad_np = np.zeros(shape, dtype=dtype)
        expected_grad_np[indices_np != -1] = 1.0

        np.testing.assert_allclose(
            probs_grad.numpy(),
            expected_grad_np,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Backward pass (gradient) check failed.",
        )
        print("Backward (gradient) test passed.")

    def test_edge_cases(self):
        probs = paddle.randn([10, 10])
        indices = paddle.full([10, 10], -1, dtype="int64")
        topk_scores = self.filter_scores_op(probs, indices)
        self.assertEqual(topk_scores.numel(), 0, "Failed for all-invalid case")
        print("Edge case passed: all invalid indices")

        probs = paddle.randn([10, 10])
        indices = paddle.full([10, 10], 1, dtype="int64")
        topk_scores = self.filter_scores_op(probs, indices)
        self.assertEqual(
            topk_scores.numel(), probs.numel(), "Failed for all-valid case"
        )
        np.testing.assert_allclose(topk_scores.numpy(), probs.numpy().flatten())
        print("Edge case passed: all valid indices")

        probs = paddle.to_tensor([], dtype="float32")
        indices = paddle.to_tensor([], dtype="int64")
        topk_scores = self.filter_scores_op(probs, indices)
        self.assertEqual(topk_scores.numel(), 0, "Failed for empty input case")
        print("Edge case passed: empty input")


if __name__ == "__main__":
    unittest.main()
