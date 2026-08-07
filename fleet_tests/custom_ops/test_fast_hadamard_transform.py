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

try:
    import paddlefleet_ops

    from paddleformers.fleet.transformer import dsa_attention

    _HAS_FAST_HADAMARD = (
        paddlefleet_ops.is_fast_hadamard_transform_available()
        and dsa_attention._fast_hadamard_transform is not None
    )
except (ImportError, RuntimeError, AttributeError):
    _HAS_FAST_HADAMARD = False


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and _HAS_FAST_HADAMARD,
    "Fast Hadamard transform requires CUDA and fast_hadamard_transform",
)
class TestFastHadamardTransform(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)

    def test_forward_backward(self):
        x = paddle.randn([1, 512, 128], "bfloat16")

        x_ref = x.detach().requires_grad_()
        y_ref = dsa_attention.rotate_activation(x_ref)

        x_tgt = x.detach().requires_grad_()
        y_tgt = dsa_attention.rotate_activation(x_tgt, use_fast_hadamard=True)

        grad = paddle.randn_like(y_ref)
        y_ref.backward(grad)
        y_tgt.backward(grad)

        # forward is binary-equal
        np.testing.assert_allclose(y_ref.float(), y_tgt.float(), atol=0, rtol=0)

        # Note: the backward of the Hadamard transform is identical to its forward.
        # Tridao reuses the same kernel for both forward and backward passes, whereas
        # Paddle relies on autograd, so Tridao is expected to be more numerically stable.
        np.testing.assert_allclose(
            x_ref.grad.float(), x_tgt.grad.float(), atol=1e-3, rtol=1e-3
        )

    def test_unavailable_assert(self):
        x = paddle.randn([1, 1, 32], "bfloat16")
        func = dsa_attention._fast_hadamard_transform
        dsa_attention._fast_hadamard_transform = None

        try:
            with self.assertRaisesRegex(
                RuntimeError, "fast_hadamard_transform is not available"
            ):
                dsa_attention.rotate_activation(x, use_fast_hadamard=True)
        finally:
            dsa_attention._fast_hadamard_transform = func


if __name__ == "__main__":
    unittest.main()
