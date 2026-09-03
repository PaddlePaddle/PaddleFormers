# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest

import numpy as np
import paddle


class TestRepeatKV(unittest.TestCase):
    """Tests for paddleformers.nn.attention.utils.repeat_kv."""

    def test_repeat_kv_n_rep_1(self):
        """When n_rep is 1, the input tensor should be returned unchanged."""
        from paddleformers.nn.attention.utils import repeat_kv

        x = paddle.randn([2, 8, 4, 16])
        result = repeat_kv(x, 1)
        np.testing.assert_allclose(result.numpy(), x.numpy())

    def test_repeat_kv_n_rep_2(self):
        """When n_rep is 2, the key/value heads should be repeated twice."""
        from paddleformers.nn.attention.utils import repeat_kv

        x = paddle.randn([2, 8, 4, 16])
        result = repeat_kv(x, 2)
        self.assertEqual(result.shape, [2, 8, 8, 16])

    def test_repeat_kv_n_rep_4(self):
        """When n_rep is 4, heads should be quadrupled."""
        from paddleformers.nn.attention.utils import repeat_kv

        x = paddle.randn([1, 4, 2, 32])
        result = repeat_kv(x, 4)
        self.assertEqual(result.shape, [1, 4, 8, 32])

    def test_repeat_kv_values_match(self):
        """Repeated values should exactly match the original values."""
        from paddleformers.nn.attention.utils import repeat_kv

        x = paddle.randn([1, 2, 2, 4])
        result = repeat_kv(x, 3)
        # Shape: [1, 2, 6, 4], heads 0,1 = original 0; heads 2,3 = original 1; heads 4,5 = original 1
        # After unsqueeze(-2).tile([1,1,1,3,1]).reshape, the pattern is:
        # result[:, :, 0] = result[:, :, 1] = result[:, :, 2] = x[:, :, 0]
        # result[:, :, 3] = result[:, :, 4] = result[:, :, 5] = x[:, :, 1]
        np.testing.assert_allclose(result[:, :, 0, :].numpy(), x[:, :, 0, :].numpy())
        np.testing.assert_allclose(result[:, :, 1, :].numpy(), x[:, :, 0, :].numpy())
        np.testing.assert_allclose(result[:, :, 2, :].numpy(), x[:, :, 0, :].numpy())
        np.testing.assert_allclose(result[:, :, 3, :].numpy(), x[:, :, 1, :].numpy())
        np.testing.assert_allclose(result[:, :, 4, :].numpy(), x[:, :, 1, :].numpy())
        np.testing.assert_allclose(result[:, :, 5, :].numpy(), x[:, :, 1, :].numpy())

    def test_repeat_kv_single_kv_head(self):
        """Edge case: single key-value head repeated to multiple attention heads."""
        from paddleformers.nn.attention.utils import repeat_kv

        x = paddle.randn([2, 8, 1, 64])
        result = repeat_kv(x, 8)
        self.assertEqual(result.shape, [2, 8, 8, 64])

    def test_repeat_kv_batch_size_1(self):
        """Test with batch_size=1 for simple verification."""
        from paddleformers.nn.attention.utils import repeat_kv

        x = paddle.randn([1, 1, 3, 8])
        result = repeat_kv(x, 2)
        self.assertEqual(result.shape, [1, 1, 6, 8])
