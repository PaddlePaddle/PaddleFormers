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
import paddle.nn.functional as F

from paddleformers.fleet.triton_ops import SigmoidGateFusionTriton


class TestSigmoidGateFusionTriton(unittest.TestCase):
    def setUp(self):
        self.shape_cases = ([1024], [33, 127], [4, 8, 64])
        self.dtype_cases = ["float32", "float16"]
        if paddle.device.cuda.get_device_capability()[0] >= 8:
            self.dtype_cases.append("bfloat16")

    def _make_inputs(self, shape, dtype):
        paddle.seed(2026)
        attn_out = paddle.randn(shape, dtype=dtype)
        gate = paddle.randn(shape, dtype=dtype)
        attn_out.stop_gradient = False
        gate.stop_gradient = False
        return attn_out, gate

    def _run_triton(self, shape, dtype):
        attn_out, gate = self._make_inputs(shape, dtype)
        out = SigmoidGateFusionTriton.apply(attn_out, gate)
        loss = out.sum()
        loss.backward()
        return out, attn_out.grad, gate.grad

    def _run_reference(self, shape, dtype):
        attn_out, gate = self._make_inputs(shape, dtype)
        out = attn_out * F.sigmoid(gate)
        loss = out.sum()
        loss.backward()
        return out, attn_out.grad, gate.grad

    def test_forward_backward_matches_reference(self):
        for shape in self.shape_cases:
            for dtype in self.dtype_cases:
                tri_out, tri_attn_grad, tri_gate_grad = self._run_triton(shape, dtype)
                ref_out, ref_attn_grad, ref_gate_grad = self._run_reference(shape, dtype)

                np.testing.assert_allclose(tri_out.float(), ref_out.float(), atol=0, rtol=0)
                np.testing.assert_allclose(tri_attn_grad.float(), ref_attn_grad.float(), atol=0, rtol=0)
                np.testing.assert_allclose(tri_gate_grad.float(), ref_gate_grad.float(), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
