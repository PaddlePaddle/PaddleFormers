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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest

import numpy as np
import paddle


class TestBiasDropoutAddFunc(unittest.TestCase):
    """Tests for _bias_dropout_add_func."""

    def setUp(self):
        paddle.seed(42)

    def test_with_bias_training(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True
        bias.stop_gradient = True

        out = _bias_dropout_add_func((x, bias), residual, 0.0, True)
        self.assertEqual(out.shape, [4, 8])
        # With p=0, dropout is identity, so out = (x + bias) + residual
        expected = x + bias + residual
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_without_bias_training(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True

        out = _bias_dropout_add_func((x, None), residual, 0.0, True)
        self.assertEqual(out.shape, [4, 8])
        expected = x + residual
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_with_bias_eval_inplace(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True
        bias.stop_gradient = True
        residual.stop_gradient = False

        out = _bias_dropout_add_func((x, bias), residual, 0.0, False)
        self.assertEqual(out.shape, [4, 8])

    def test_without_bias_eval_inplace(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True
        residual.stop_gradient = False

        out = _bias_dropout_add_func((x, None), residual, 0.0, False)
        self.assertEqual(out.shape, [4, 8])

    def test_with_bias_eval_not_inplace(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        # Not inplace because x requires gradient
        x.stop_gradient = False
        bias.stop_gradient = True
        residual.stop_gradient = False

        out = _bias_dropout_add_func((x, bias), residual, 0.0, False)
        self.assertEqual(out.shape, [4, 8])

    def test_without_bias_eval_not_inplace(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        # Not inplace because x requires gradient
        x.stop_gradient = False
        residual.stop_gradient = False

        out = _bias_dropout_add_func((x, None), residual, 0.0, False)
        self.assertEqual(out.shape, [4, 8])

    def test_with_dropout_nonzero_prob(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True
        bias.stop_gradient = True

        # With non-zero dropout, output should differ from no-dropout
        out = _bias_dropout_add_func((x, bias), residual, 0.5, True)
        self.assertEqual(out.shape, [4, 8])


class TestBiasDropoutAddUnfused(unittest.TestCase):
    """Tests for bias_dropout_add_unfused."""

    def setUp(self):
        paddle.seed(42)

    def test_training_mode(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            bias_dropout_add_unfused,
        )

        fn = bias_dropout_add_unfused(True)
        x = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True
        bias.stop_gradient = True

        out = fn((x, bias), residual, 0.0)
        self.assertEqual(out.shape, [4, 8])

    def test_eval_mode(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            bias_dropout_add_unfused,
        )

        fn = bias_dropout_add_unfused(False)
        x = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True
        bias.stop_gradient = True

        out = fn((x, bias), residual, 0.0)
        self.assertEqual(out.shape, [4, 8])

    def test_without_bias(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            bias_dropout_add_unfused,
        )

        fn = bias_dropout_add_unfused(True)
        x = paddle.randn([4, 8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True

        out = fn((x, None), residual, 0.0)
        self.assertEqual(out.shape, [4, 8])


class TestGetBiasDropoutAdd(unittest.TestCase):
    """Tests for get_bias_dropout_add."""

    def test_returns_callable(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add

        fn = get_bias_dropout_add(True, fused=True)
        self.assertTrue(callable(fn))

    def test_training_mode_usage(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add

        fn = get_bias_dropout_add(True, fused=True)
        x = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True
        bias.stop_gradient = True

        out = fn((x, bias), residual, 0.0)
        self.assertEqual(out.shape, [4, 8])

    def test_eval_mode_usage(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add

        fn = get_bias_dropout_add(False, fused=True)
        x = paddle.randn([4, 8], dtype=paddle.float32)
        bias = paddle.randn([8], dtype=paddle.float32)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True
        bias.stop_gradient = True

        out = fn((x, bias), residual, 0.0)
        self.assertEqual(out.shape, [4, 8])


if __name__ == "__main__":
    unittest.main()
