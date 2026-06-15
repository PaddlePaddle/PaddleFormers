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

from paddleformers.fleet.transformer.paddle_norm import RMSNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class TestRMSNormFusionTriton(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        self.config = TransformerConfig(
            hidden_size=128,
            num_attention_heads=4,
            normalization="RMSNorm",
            rms_norm_eps=1e-6,
        )
        self.ref_norm = RMSNorm(self.config)
        self.ref_norm = paddle.amp.decorate(self.ref_norm, level="O2", dtype="bfloat16")

    def test_forward_backward(self):
        x = paddle.randn([1024, 128], dtype="bfloat16")
        x.stop_gradient = False
        dy = paddle.randn_like(x) * 0.01

        # Reference
        y0 = self.ref_norm(x)
        y0.backward(dy)
        dx0, dw0 = x.grad.clone(), self.ref_norm.weight.grad.clone()

        # Reset grads
        x.clear_gradient()
        self.ref_norm.weight.clear_gradient()

        # Triton
        paddle.enable_compat(scope={"triton"}, silent=True)
        from paddleformers.fleet.triton_ops import RMSNormFusionTriton

        y1 = RMSNormFusionTriton.apply(x, self.ref_norm.weight, self.config.rms_norm_eps)
        y1.backward(dy)
        dx1, dw1 = x.grad, self.ref_norm.weight.grad
        paddle.disable_compat()

        np.testing.assert_allclose(y0.float(), y1.float(), rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(dx0.float(), dx1.float(), rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(dw0.float(), dw1.float(), rtol=1e-4, atol=5e-3)


if __name__ == "__main__":
    unittest.main()
