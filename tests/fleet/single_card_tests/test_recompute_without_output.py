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
from paddle import nn
from paddle.distributed.fleet.recompute.recompute import custom_state_manager

from paddleformers.fleet.tensor_parallel import RecomputeWithoutOutput


class TestRecomputeWithoutOutput(unittest.TestCase):
    def test_pure_tensors_with_grad(self):
        fc1 = nn.Linear(32, 64)
        fc2 = nn.Linear(64, 32)

        a = paddle.randn([8, 32])
        a.stop_gradient = False

        ctx = RecomputeWithoutOutput()
        b = ctx.recompute(fc1, a, preserve_rng_state=False)

        c = fc2(b)
        mem_before_discard = paddle.device.memory_allocated()
        ctx.discard_output_and_register_recompute(c)
        mem_after_discard = paddle.device.memory_allocated()
        c.backward()

        self.assertIsNotNone(a.grad)
        self.assertEqual(mem_before_discard - mem_after_discard, b.size * b.itemsize)

    def test_non_tensor_input(self):
        a = paddle.randn([8, 16, 32])
        a.stop_gradient = False

        ctx = RecomputeWithoutOutput()
        b = ctx.recompute(paddle.nn.functional.layer_norm, a, a.shape[1:])

        c = paddle.sum(b)
        ctx.discard_output_and_register_recompute(c)
        c.backward()

        self.assertIsNotNone(a.grad)

    def test_no_grad_input(self):
        a = paddle.randn([16, 32])
        a.stop_gradient = False
        b = paddle.randn([32, 48])
        b.stop_gradient = False
        c = paddle.randn([16, 1])

        def scale_matmul(x, y, z):
            x = x * z["scale"]
            return paddle.matmul(x, y)

        ctx = RecomputeWithoutOutput()
        d = ctx.recompute(scale_matmul, a, b, {"scale": c})

        e = d.sum()
        ctx.discard_output_and_register_recompute(e)
        e.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

    def test_rng_consistency(self):
        a = paddle.randn([32])
        a.stop_gradient = False

        custom_state_manager.set_custom_get_state_func(lambda x=None: None)
        custom_state_manager.set_custom_set_state_func(lambda x=None: None)

        def random_mul(x):
            return x * paddle.randn_like(x)

        paddle.seed(2026)
        b = random_mul(a)
        c = b.sum()
        c.backward()
        a_grad_ref = a.grad.clone()
        a.clear_grad()

        paddle.seed(2026)
        ctx = RecomputeWithoutOutput()
        b = ctx.recompute(random_mul, a)
        c = paddle.sum(b)
        ctx.discard_output_and_register_recompute(c)
        c.backward()

        np.testing.assert_allclose(a.grad, a_grad_ref, atol=0, rtol=0)

    def test_amp(self):
        fc = nn.Linear(16, 32)
        fc = paddle.amp.decorate(fc, level="O2", dtype="float16")

        a = paddle.randn([8, 16])
        a.stop_gradient = False

        with paddle.amp.auto_cast(level="O2", dtype="float16"):
            ctx = RecomputeWithoutOutput()
            b = ctx.recompute(fc, a)
            c = paddle.sum(b)
            ctx.discard_output_and_register_recompute(c)
            c.backward()

        self.assertEqual(c.dtype, paddle.float16)
        self.assertIsNotNone(a.grad)

    def test_multiple_outputs(self):
        a = paddle.randn([8, 32])
        a.stop_gradient = False

        def split_mul(x):
            p, q = x.split(2, axis=-1)
            return p.sin() * q, p * q.cos()

        b, c = split_mul(a)
        d = b + c
        d.backward()
        a_grad_ref = a.grad.clone()
        a.clear_grad()

        ctx = RecomputeWithoutOutput()
        b, c = ctx.recompute(split_mul, a, share_grad_holder=True)
        d = b + c
        ctx.discard_output_and_register_recompute(d)
        d.backward()

        np.testing.assert_allclose(a.grad, a_grad_ref, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
