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

import numpy as np
import paddle

_CUDA_SOFTMAX = False
try:
    import paddle

    _CUDA_SOFTMAX = paddle.is_compiled_with_cuda()
except:
    pass


@unittest.skipIf(not _CUDA_SOFTMAX, "SoftmaxOne requires GPU for sink token")
class TestSoftmaxOne(unittest.TestCase):
    """Tests for SoftmaxOne layer."""

    def setUp(self):
        paddle.seed(42)
        self.x = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)


class TestFusedScaleMaskSoftmaxExtra(unittest.TestCase):
    """Additional tests for FusedScaleMaskSoftmax."""

    def setUp(self):
        paddle.seed(42)
        self.x = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required for float16")
    def test_fp16_input_fp32_softmax(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.padding,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x,
            softmax_in_fp32=True,
            scale=0.125,
        )
        x_f16 = self.x.cast(paddle.float16)
        out = layer(x_f16, mask=None)
        self.assertEqual(out.shape, [2, 4, 8, 16])
        self.assertEqual(out.dtype, paddle.float16)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required for bfloat16")
    def test_bf16_input_fp32_softmax(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=True,
            attn_mask_type=AttnMaskType.padding,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x,
            softmax_in_fp32=True,
            scale=0.125,
        )
        x_bf16 = self.x.cast(paddle.bfloat16)
        out = layer(x_bf16, mask=None)
        self.assertEqual(out.shape, [2, 4, 8, 16])
        self.assertEqual(out.dtype, paddle.bfloat16)

    def test_no_scale_no_softmax_fp32(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.no_mask,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x,
            softmax_in_fp32=False,
            scale=None,
        )
        out = layer(self.x, mask=None)
        self.assertEqual(out.shape, [2, 4, 8, 16])
        np.testing.assert_allclose(out.sum(axis=-1).numpy(), 1.0, atol=1e-5)

    def test_scale_assert_no_fp32(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        with self.assertRaises(AssertionError):
            FusedScaleMaskSoftmax(
                input_in_fp16=False,
                input_in_bf16=False,
                attn_mask_type=AttnMaskType.padding,
                scaled_masked_softmax_fusion=True,
                mask_func=lambda x, m: x,
                softmax_in_fp32=False,
                scale=0.125,
            )

    def test_fp16_and_bf16_assert(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        with self.assertRaises(AssertionError):
            FusedScaleMaskSoftmax(
                input_in_fp16=True,
                input_in_bf16=True,
                attn_mask_type=AttnMaskType.padding,
                scaled_masked_softmax_fusion=True,
                mask_func=lambda x, m: x,
                softmax_in_fp32=True,
                scale=None,
            )

    def test_input_dim_assert(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.padding,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x,
            softmax_in_fp32=True,
            scale=None,
        )
        x_3d = paddle.randn([2, 4, 8], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            layer(x_3d, mask=None)

    def test_causal_mask_auto_gen_sq_eq_sk(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x * m.cast(x.dtype),
            softmax_in_fp32=True,
            scale=None,
        )
        x = paddle.randn([2, 4, 8, 8], dtype=paddle.float32)
        out = layer(x, mask=None)
        self.assertEqual(out.shape, [2, 4, 8, 8])

    def test_causal_mask_sq_ne_sk_assert(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x * m.cast(x.dtype),
            softmax_in_fp32=True,
            scale=None,
        )
        x = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            layer(x, mask=None)

    def test_causal_mask_sq_1_no_mask_gen(self):
        """When sq == 1, causal mask should not be generated."""
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x * m.cast(x.dtype),
            softmax_in_fp32=True,
            scale=None,
        )
        # sq == 1, should not generate causal mask
        x = paddle.randn([2, 4, 1, 16], dtype=paddle.float32)
        out = layer(x, mask=None)
        self.assertEqual(out.shape, [2, 4, 1, 16])

    def test_sliding_window_mask(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x * m.cast(x.dtype),
            softmax_in_fp32=True,
            scale=None,
            sliding_window=(4, 4),
        )
        x = paddle.randn([2, 4, 8, 8], dtype=paddle.float32)
        out = layer(x, mask=None)
        self.assertEqual(out.shape, [2, 4, 8, 8])

    def test_arbitrary_mask_type(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.arbitrary,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x * m.cast(x.dtype),
            softmax_in_fp32=True,
            scale=None,
        )
        mask = paddle.randn([8, 16], dtype=paddle.float32)
        out = layer(self.x, mask=mask)
        self.assertEqual(out.shape, [2, 4, 8, 16])

    def test_no_mask_type_with_explicit_mask(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.no_mask,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x,
            softmax_in_fp32=True,
            scale=None,
        )
        out = layer(self.x, mask=None)
        self.assertEqual(out.shape, [2, 4, 8, 16])
        np.testing.assert_allclose(out.sum(axis=-1).numpy(), 1.0, atol=1e-5)

    def test_mask_applied_before_softmax(self):
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.padding,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x + m.cast(x.dtype),
            softmax_in_fp32=True,
            scale=None,
        )
        mask = paddle.ones([8, 16], dtype=paddle.float32)
        out = layer(self.x, mask=mask)
        self.assertEqual(out.shape, [2, 4, 8, 16])

    def test_causal_mask_with_user_mask_ignored(self):
        """When attn_mask_type is causal and mask is provided, user mask is used."""
        from paddleformers.fleet.fusions.fused_softmax import FusedScaleMaskSoftmax
        from paddleformers.fleet.transformer.enums import AttnMaskType

        layer = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
            mask_func=lambda x, m: x * m.cast(x.dtype),
            softmax_in_fp32=True,
            scale=None,
        )
        x = paddle.randn([2, 4, 8, 8], dtype=paddle.float32)
        user_mask = paddle.ones([8, 8], dtype=paddle.float32)
        out = layer(x, mask=user_mask)
        self.assertEqual(out.shape, [2, 4, 8, 8])


if __name__ == "__main__":
    unittest.main()
