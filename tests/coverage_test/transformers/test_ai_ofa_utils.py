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
import paddle.nn as nn

from paddleformers.transformers.ofa_utils import (
    compute_neuron_head_importance,
    encoder_layer_ofa_forward,
    encoder_ofa_forward,
    mha_ofa_forward,
    prepare_qkv_ofa,
    reorder_head,
    reorder_neuron,
    reorder_neuron_head,
)


class TestReorderNeuron(unittest.TestCase):
    """Tests for reorder_neuron function."""

    def test_reorder_neuron_dim0(self):
        # Linear(in_features=4, out_features=8) -> weight shape [4, 8]
        layer = nn.Linear(4, 8)
        original_weight = layer.weight.numpy().copy()
        # Reorder along dim 0 (in_features), index must have len=4
        index = paddle.to_tensor([3, 2, 1, 0], dtype="int64")
        reorder_neuron(layer, index, dim=0)
        new_weight = layer.weight.numpy()
        expected = original_weight[[3, 2, 1, 0]]
        np.testing.assert_allclose(new_weight, expected, atol=1e-6)

    def test_reorder_neuron_dim1(self):
        # Linear(in_features=4, out_features=8) -> weight shape [4, 8]
        layer = nn.Linear(4, 8)
        original_weight = layer.weight.numpy().copy()
        # Reorder along dim 1 (out_features), index must have len=8
        index = paddle.to_tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype="int64")
        reorder_neuron(layer, index, dim=1)
        new_weight = layer.weight.numpy()
        expected = original_weight[:, [7, 6, 5, 4, 3, 2, 1, 0]]
        np.testing.assert_allclose(new_weight, expected, atol=1e-6)

    def test_reorder_neuron_with_bias_dim0(self):
        # dim=0 does not reorder bias
        # Linear(in_features=4, out_features=8) -> weight [4,8], bias [8]
        layer = nn.Linear(4, 8)
        original_bias = layer.bias.numpy().copy()
        index = paddle.to_tensor([3, 2, 1, 0], dtype="int64")
        reorder_neuron(layer, index, dim=0)
        # When dim=0, bias is just assigned as-is (not reordered)
        new_bias = layer.bias.numpy()
        np.testing.assert_allclose(new_bias, original_bias, atol=1e-6)

    def test_reorder_neuron_with_bias_dim1(self):
        # Linear(in_features=4, out_features=8) -> weight [4,8], bias [8]
        layer = nn.Linear(4, 8)
        original_bias = layer.bias.numpy().copy()
        # Reorder along dim 1 (out_features), index must have len=8
        index = paddle.to_tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype="int64")
        reorder_neuron(layer, index, dim=1)
        new_bias = layer.bias.numpy()
        expected = original_bias[[7, 6, 5, 4, 3, 2, 1, 0]]
        np.testing.assert_allclose(new_bias, expected, atol=1e-6)


class TestOFAFunctionsExist(unittest.TestCase):
    """Tests that OFA utility functions are importable and callable."""

    def test_prepare_qkv_ofa_importable(self):
        self.assertTrue(callable(prepare_qkv_ofa))

    def test_mha_ofa_forward_importable(self):
        self.assertTrue(callable(mha_ofa_forward))

    def test_encoder_ofa_forward_importable(self):
        self.assertTrue(callable(encoder_ofa_forward))

    def test_encoder_layer_ofa_forward_importable(self):
        self.assertTrue(callable(encoder_layer_ofa_forward))

    def test_compute_neuron_head_importance_importable(self):
        self.assertTrue(callable(compute_neuron_head_importance))

    def test_reorder_neuron_head_importable(self):
        self.assertTrue(callable(reorder_neuron_head))

    def test_reorder_head_importable(self):
        self.assertTrue(callable(reorder_head))

    def test_reorder_neuron_importable(self):
        self.assertTrue(callable(reorder_neuron))


class TestEncoderOFAForward(unittest.TestCase):
    """Tests for encoder_ofa_forward function."""

    def test_src_mask_none_head_mask(self):
        class MockLayer:
            def __call__(self, x, src_mask=None):
                return x

        class MockEncoder:
            num_layers = 2
            layers = [MockLayer(), MockLayer()]
            norm = None

        encoder = MockEncoder()
        src = paddle.randn([2, 4, 8])
        result = encoder_ofa_forward(encoder, src, src_mask=[None, None])
        self.assertTrue(paddle.allclose(result, src))

    def test_src_mask_with_1d_head_mask(self):
        class MockLayer:
            def __call__(self, x, src_mask=None):
                return x

        class MockEncoder:
            num_layers = 2
            layers = [MockLayer(), MockLayer()]
            norm = None

        encoder = MockEncoder()
        src = paddle.randn([2, 4, 8])
        head_mask = paddle.ones([4], dtype="float32")
        result = encoder_ofa_forward(encoder, src, src_mask=[None, head_mask])
        self.assertIsNotNone(result)

    def test_src_mask_with_2d_head_mask(self):
        class MockLayer:
            def __call__(self, x, src_mask=None):
                return x

        class MockEncoder:
            num_layers = 2
            layers = [MockLayer(), MockLayer()]
            norm = None

        encoder = MockEncoder()
        src = paddle.randn([2, 4, 8])
        head_mask = paddle.ones([2, 4], dtype="float32")
        result = encoder_ofa_forward(encoder, src, src_mask=[None, head_mask])
        self.assertIsNotNone(result)

    def test_with_norm(self):
        class MockLayer:
            def __call__(self, x, src_mask=None):
                return x

        class MockEncoder:
            num_layers = 1
            layers = [MockLayer()]
            norm = nn.LayerNorm(8)

        encoder = MockEncoder()
        src = paddle.randn([2, 4, 8])
        result = encoder_ofa_forward(encoder, src, src_mask=[None, None])
        self.assertEqual(result.shape, [2, 4, 8])


if __name__ == "__main__":
    unittest.main()
