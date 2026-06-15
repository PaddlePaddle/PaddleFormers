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

from paddleformers.fleet.fp8.act_quant import act_quant
from paddleformers.fleet.fp8.qat import fp8_simulate, fp8_simulate_qat


def print_io(label, x, y):
    """Print input/output comparison."""
    diff = (y - x.astype(y.dtype)).abs()
    print(f"\n----- [{label}] -----")
    print(f"  Input  (first 8): {x.astype('float32').numpy().flatten()[:8]}")
    print(f"  Output (first 8): {y.astype('float32').numpy().flatten()[:8]}")
    print(f"  Diff   (first 8): {diff.astype('float32').numpy().flatten()[:8]}")
    print(
        f"  Max abs error: {diff.astype('float32').max().item():.6f}, Mean abs error: {diff.astype('float32').mean().item():.6f}"
    )


class TestFP8SimulateQAT(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)
        self.block_size = 128

    def test_output_shape(self):
        """Output shape should match input shape."""
        x = paddle.randn([2, 4, 256])
        x.stop_gradient = False
        y = fp8_simulate_qat(x, self.block_size)
        print_io("test_output_shape", x, y)
        self.assertEqual(y.shape, x.shape)

    def test_output_dtype(self):
        """Output dtype should match input dtype."""
        for dtype in ["float32", "float16", "bfloat16"]:
            x = paddle.randn([2, 256]).astype(dtype)
            x.stop_gradient = False
            y = fp8_simulate_qat(x, self.block_size)
            print_io(f"test_output_dtype [{dtype}]", x, y)
            self.assertEqual(y.dtype, x.dtype)

    def test_forward_matches_fp8_simulate(self):
        """Forward should produce same result as fp8_simulate."""
        x = paddle.randn([2, 256])
        x.stop_gradient = False
        y_qat = fp8_simulate_qat(x, self.block_size)
        y_ref = fp8_simulate(x, self.block_size)
        print_io("test_forward_matches (qat)", x, y_qat)
        print_io("test_forward_matches (ref)", x, y_ref)
        np.testing.assert_allclose(y_qat.numpy(), y_ref.numpy(), rtol=1e-5, atol=1e-5)

    def test_quantization_error_bounded(self):
        """Quantization error should be bounded (not too large)."""
        x = paddle.randn([4, 256])
        x.stop_gradient = False
        y = fp8_simulate_qat(x, self.block_size)
        print_io("test_quantization_error_bounded", x, y)
        err = (y - x).abs().max().item()
        self.assertLess(err, x.abs().max().item() * 0.2)

    def test_ste_backward(self):
        """Backward should pass gradient through unchanged (STE)."""
        x = paddle.randn([2, 256])
        x.stop_gradient = False
        y = fp8_simulate_qat(x, self.block_size)
        print_io("test_ste_backward", x, y)
        loss = y.sum()
        loss.backward()
        print(f"  Grad (first 8): {x.grad.numpy().flatten()[:8]}")
        np.testing.assert_allclose(x.grad.numpy(), np.ones_like(x.grad.numpy()), rtol=1e-5, atol=1e-5)

    def test_different_block_sizes(self):
        """Should work with different block sizes."""
        x = paddle.randn([2, 512])
        x.stop_gradient = False
        for bs in [64, 128, 256]:
            y = fp8_simulate_qat(x, bs)
            print_io(f"test_different_block_sizes [bs={bs}]", x, y)
            self.assertEqual(y.shape, x.shape)

    def test_act_quant_no_scale_fmt(self):
        """act_quant without scale_fmt (non-power-of-2 path)."""
        x = paddle.randn([2, 256])
        y_out, s_out = act_quant(x, block_size=128, scale_fmt=None)
        self.assertEqual(y_out.shape, x.shape)
        self.assertEqual(s_out.shape, [2, 2])

    def test_act_quant_inplace(self):
        """act_quant with inplace=True."""
        x = paddle.randn([2, 256]).astype("bfloat16")
        x_clone = x.clone()
        result = act_quant(x_clone, block_size=128, scale_fmt="ue8m0", inplace=True)
        # inplace returns the input tensor modified
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)

    def test_compressor_no_rotate_qat(self):
        """Compressor with rotate=False and use_fp8_qat=True covers lines 1295-1296."""
        import types

        from paddle import nn

        from paddleformers.fleet.transformer.csa_attention import (
            Compressor,
            CompressorSublayersSpec,
        )

        class _Linear(nn.Layer):
            def __init__(self, in_size, out_size, **kwargs):
                super().__init__()
                self.weight = self.create_parameter(
                    shape=[out_size, in_size],
                    dtype="float32",
                    default_initializer=nn.initializer.Normal(std=0.02),
                )

            def forward(self, x):
                return paddle.matmul(x, self.weight.T), None

        class _Norm(nn.Layer):
            def __init__(self, hidden_size=None, **kwargs):
                super().__init__()
                size = hidden_size or 1
                self.weight = self.create_parameter(
                    shape=[size],
                    default_initializer=nn.initializer.Constant(1.0),
                )
                self.eps = 1e-5

            def forward(self, x):
                return x * paddle.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight

        head_dim = 128
        pos_dim = 64
        hidden_size = 256
        compress_ratio = 2

        config = types.SimpleNamespace(
            hidden_size=hidden_size,
            qk_pos_emb_head_dim=pos_dim,
            init_method=None,
            init_method_std=0.02,
            rms_norm_eps=1e-5,
            num_hidden_layers=1,
            use_fp8_qat=True,
        )

        spec = CompressorSublayersSpec(
            linear_wkv=_Linear,
            linear_wgate=_Linear,
            norm=_Norm,
        )

        compressor = Compressor(
            config=config,
            sublayers_spec=spec,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=None,
        )

        x = paddle.randn([2, 64, hidden_size])
        kv = compressor(x)
        # nope_dim = head_dim - pos_dim = 64, must be divisible by block_size=64
        self.assertEqual(kv.shape, [2, 32, head_dim])
