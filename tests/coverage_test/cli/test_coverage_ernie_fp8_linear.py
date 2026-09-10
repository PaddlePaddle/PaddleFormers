# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import MagicMock

import numpy as np
import paddle

from paddleformers.cli.train.ernie_pretrain.models.fp8_linear import (
    Fp8FusedMlp,
    padding,
)


class TestPadding(unittest.TestCase):
    """Tests for the padding function."""

    def test_no_padding_needed_axis0(self):
        """Test that no padding is applied when already divisible by 512 on axis 0."""
        x = paddle.randn([512, 256])
        result = padding(x, 0)
        self.assertEqual(result.shape, [512, 256])

    def test_padding_needed_axis0(self):
        """Test padding is applied on axis 0."""
        x = paddle.randn([100, 256])
        result = padding(x, 0)
        # Should be padded to 512 or 128 boundary
        self.assertEqual(result.shape[0] % 128, 0)
        self.assertGreaterEqual(result.shape[0], 100)

    def test_no_padding_needed_axis1(self):
        """Test that no padding is applied when already divisible by 512 on axis 1."""
        x = paddle.randn([256, 512])
        result = padding(x, 1)
        self.assertEqual(result.shape, [256, 512])

    def test_padding_needed_axis1(self):
        """Test padding is applied on axis 1."""
        x = paddle.randn([256, 100])
        result = padding(x, 1)
        self.assertEqual(result.shape[1] % 128, 0)
        self.assertGreaterEqual(result.shape[1], 100)

    def test_padding_preserves_original_data_axis0(self):
        """Test that original data is preserved after padding on axis 0."""
        x = paddle.randn([100, 64])
        result = padding(x, 0)
        # Original data should be unchanged
        np.testing.assert_allclose(result[:100].numpy(), x.numpy(), rtol=1e-5)

    def test_padding_preserves_original_data_axis1(self):
        """Test that original data is preserved after padding on axis 1."""
        x = paddle.randn([64, 100])
        result = padding(x, 1)
        np.testing.assert_allclose(result[:, :100].numpy(), x.numpy(), rtol=1e-5)

    def test_padding_128_multiple(self):
        """Test that 128-multiple size is padded correctly."""
        x = paddle.randn([128, 64])
        result = padding(x, 0)
        # 128 is already divisible by 128, and 128 % 512 != 0
        # So it should be padded to 512
        self.assertEqual(result.shape[0] % 128, 0)

    def test_padding_already_512(self):
        """Test tensor already at 512 boundary."""
        x = paddle.randn([512, 512])
        result = padding(x, 0)
        self.assertEqual(result.shape, [512, 512])
        result = padding(x, 1)
        self.assertEqual(result.shape, [512, 512])


class TestFp8FusedMlp(unittest.TestCase):
    """Tests for Fp8FusedMlp layer (init only, no GPU forward)."""

    def test_init(self):
        """Test Fp8FusedMlp initialization."""
        config = MagicMock()
        config.hidden_size = 256
        config.intermediate_size = 512

        layer = Fp8FusedMlp(config)
        self.assertEqual(layer.hidden_size, 256)
        self.assertEqual(layer.intermediate_size, 512)
        self.assertEqual(layer.w1.shape, [256, 1024])  # intermediate_size * 2
        self.assertEqual(layer.w2.shape, [512, 256])
        self.assertEqual(layer.w1.dtype, paddle.bfloat16)
        self.assertEqual(layer.w2.dtype, paddle.bfloat16)


if __name__ == "__main__":
    unittest.main()
