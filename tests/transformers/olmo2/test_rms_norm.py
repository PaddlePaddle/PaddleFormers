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

from paddleformers.transformers.olmo2.modeling import Olmo2RMSNorm


class Olmo2RMSNormTest(unittest.TestCase):
    def test_static_dynamic_batch_and_sequence(self):
        previous_device = paddle.get_device()
        try:
            paddle.set_device("cpu")
            layer = Olmo2RMSNorm(32)
            compiled = paddle.jit.to_static(
                layer,
                full_graph=True,
                backend=None,
                input_spec=[paddle.static.InputSpec([None, None, 32], "float32")],
            )
            for shape in [(1, 7, 32), (2, 9, 32)]:
                x = paddle.randn(shape)
                np.testing.assert_array_equal(compiled(x).numpy(), layer(x).numpy())
        finally:
            paddle.set_device(previous_device)

    def test_row_layout_preserves_forward_and_gradients(self):
        paddle.seed(123)
        for shape in [(32,), (7, 32), (2, 7, 32), (2, 3, 7, 32)]:
            with self.subTest(shape=shape):
                layer = Olmo2RMSNorm(shape[-1], eps=1e-6)
                x = paddle.randn(shape, dtype="float32")
                x.stop_gradient = False
                reference_x = x.detach().clone()
                reference_x.stop_gradient = False
                reference_weight = layer.weight.detach().clone()
                reference_weight.stop_gradient = False
                reference = reference_weight * (
                    reference_x * paddle.rsqrt(reference_x.pow(2).mean(-1, keepdim=True) + 1e-6)
                )
                actual = layer(x)
                reference_grads = paddle.grad(reference.square().mean(), [reference_x, reference_weight])
                actual_grads = paddle.grad(actual.square().mean(), [x, layer.weight])
                self.assertEqual(list(actual.shape), list(shape))
                np.testing.assert_array_equal(actual.numpy(), reference.numpy())
                for actual_grad, reference_grad in zip(actual_grads, reference_grads):
                    np.testing.assert_allclose(actual_grad.numpy(), reference_grad.numpy(), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
