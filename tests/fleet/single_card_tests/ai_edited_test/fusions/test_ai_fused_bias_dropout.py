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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest

import paddle


class TestBiasDropoutAddFunc(unittest.TestCase):
    """Tests for _bias_dropout_add_func."""

    def test_with_bias_training(self):
        """Test with bias during training."""
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8])
        bias = paddle.randn([8])
        residual = paddle.randn([4, 8])
        out = _bias_dropout_add_func((x, bias), residual, 0.0, True)
        self.assertEqual(out.shape, [4, 8])

    def test_without_bias_training(self):
        """Test without bias during training."""
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8])
        residual = paddle.randn([4, 8])
        out = _bias_dropout_add_func((x, None), residual, 0.0, True)
        self.assertEqual(out.shape, [4, 8])

    def test_with_bias_eval(self):
        """Test with bias in eval mode."""
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8])
        bias = paddle.randn([8])
        residual = paddle.randn([4, 8])
        x.stop_gradient = True
        residual.stop_gradient = False
        bias.stop_gradient = True
        out = _bias_dropout_add_func((x, bias), residual, 0.0, False)
        self.assertEqual(out.shape, [4, 8])

    def test_residual_dtype_conversion(self):
        """Test residual is cast to match x dtype."""
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            _bias_dropout_add_func,
        )

        x = paddle.randn([4, 8], dtype=paddle.float16)
        residual = paddle.randn([4, 8], dtype=paddle.float32)
        out = _bias_dropout_add_func((x, None), residual, 0.0, True)
        self.assertEqual(out.dtype, paddle.float16)

    def test_bias_dropout_add_unfused(self):
        """Test bias_dropout_add_unfused returns callable."""
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            bias_dropout_add_unfused,
        )

        func = bias_dropout_add_unfused(True)
        self.assertTrue(callable(func))

    def test_get_bias_dropout_add(self):
        """Test get_bias_dropout_add returns callable."""
        from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add

        func = get_bias_dropout_add(True, fused=True)
        self.assertTrue(callable(func))


if __name__ == "__main__":
    unittest.main()
