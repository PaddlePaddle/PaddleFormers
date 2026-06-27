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
from unittest.mock import patch

import paddle

from paddleformers.fleet.tensor_parallel.layers import Linear
from paddleformers.fleet.transformer.dsa_attention import (
    DSAIndexerLossAutoScaler,
    DSAIndexerSublayersSpec,
    FusedDSAIndexerLoss,
    Indexer,
    _unfused_dsa_attention,
    hadamard_transform,
    rotate_activation,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "dsa_index_n_heads": 1,
        "dsa_index_head_dim": 16,
        "dsa_index_topk": 4,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 16,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestHadamardTransform(unittest.TestCase):
    """Test hadamard_transform function."""

    def test_dim_1(self):
        x = paddle.randn([2, 1])
        out = hadamard_transform(x, scale=1.0)
        self.assertEqual(out.shape, [2, 1])

    def test_dim_2(self):
        x = paddle.randn([2, 2])
        out = hadamard_transform(x, scale=1.0)
        self.assertEqual(out.shape, [2, 2])

    def test_dim_4(self):
        x = paddle.randn([2, 4])
        out = hadamard_transform(x, scale=1.0)
        self.assertEqual(out.shape, [2, 4])

    def test_dim_8(self):
        x = paddle.randn([3, 8])
        out = hadamard_transform(x, scale=1.0)
        self.assertEqual(out.shape, [3, 8])

    def test_dim_16(self):
        x = paddle.randn([2, 16])
        out = hadamard_transform(x, scale=1.0)
        self.assertEqual(out.shape, [2, 16])

    def test_non_power_of_two_raises(self):
        x = paddle.randn([2, 3])
        with self.assertRaises(AssertionError):
            hadamard_transform(x)

    def test_zero_dim_raises(self):
        x = paddle.randn([2, 0])
        with self.assertRaises(AssertionError):
            hadamard_transform(x)

    def test_scale_factor(self):
        x = paddle.randn([2, 4])
        out = hadamard_transform(x, scale=2.0)
        # Check scale is applied
        out_noscale = hadamard_transform(x, scale=1.0)
        diff = (out - out_noscale * 2.0).abs().max()
        self.assertAlmostEqual(float(diff), 0.0, places=5)

    def test_3d_input(self):
        x = paddle.randn([2, 3, 8])
        out = hadamard_transform(x)
        self.assertEqual(out.shape, [2, 3, 8])

    def test_preserves_shape(self):
        shape = [2, 3, 4, 16]
        x = paddle.randn(shape)
        out = hadamard_transform(x)
        self.assertEqual(out.shape, shape)

    def test_inverse_property(self):
        # H * H = N * I for Hadamard matrix
        x = paddle.randn([2, 8])
        h1 = hadamard_transform(x, scale=1.0)
        h2 = hadamard_transform(h1, scale=1.0)
        # H2 * x should equal N * x for N=8
        diff = (h2 - x * 8.0).abs().max()
        self.assertAlmostEqual(float(diff), 0.0, places=4)


class TestRotateActivation(unittest.TestCase):
    """Test rotate_activation function."""

    def test_valid_bf16(self):
        x = paddle.randn([2, 4, 16]).cast("bfloat16")
        out = rotate_activation(x)
        self.assertEqual(out.shape, [2, 4, 16])

    def test_invalid_dtype_raises(self):
        x = paddle.randn([2, 4, 16]).cast("float32")
        with self.assertRaises(AssertionError):
            rotate_activation(x)

    def test_non_power_of_two_raises(self):
        x = paddle.randn([2, 3]).cast("bfloat16")
        with self.assertRaises(AssertionError):
            rotate_activation(x)

    def test_output_scale(self):
        x = paddle.randn([2, 16]).cast("bfloat16")
        out = rotate_activation(x)
        expected_scale = 16.0**-0.5
        out_manual = hadamard_transform(x, scale=expected_scale)
        diff = (out - out_manual).abs().max()
        self.assertAlmostEqual(float(diff), 0.0, places=4)


class TestUnfusedDSAAttention(unittest.TestCase):
    """Test _unfused_dsa_attention function."""

    def test_output_shape(self):
        b, s, nhpp, qk_hd, v_hd = 2, 4, 2, 8, 16
        query = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        key = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        value = paddle.randn([b, s, nhpp, v_hd], dtype="float32")
        out = _unfused_dsa_attention(query, key, value, None, 0.5)
        self.assertEqual(out.shape, [b, s, nhpp * v_hd])

    def test_with_mask(self):
        b, s, nhpp, qk_hd, v_hd = 1, 4, 2, 8, 16
        query = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        key = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        value = paddle.randn([b, s, nhpp, v_hd], dtype="float32")
        mask = paddle.zeros([b, 1, s, s], dtype="float32")
        out = _unfused_dsa_attention(query, key, value, mask, 0.5)
        self.assertEqual(out.shape, [b, s, nhpp * v_hd])

    def test_asymmetric_head_dims(self):
        b, s, nhpp = 1, 4, 1
        qk_hd, v_hd = 8, 16
        query = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        key = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        value = paddle.randn([b, s, nhpp, v_hd], dtype="float32")
        out = _unfused_dsa_attention(query, key, value, None, 0.5)
        self.assertEqual(out.shape, [b, s, nhpp * v_hd])

    def test_single_head(self):
        b, s, nhpp = 1, 4, 1
        qk_hd = v_hd = 8
        query = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        key = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        value = paddle.randn([b, s, nhpp, v_hd], dtype="float32")
        out = _unfused_dsa_attention(query, key, value, None, 0.5)
        self.assertEqual(out.shape, [b, s, nhpp * v_hd])

    def test_different_softmax_scale(self):
        b, s, nhpp, qk_hd, v_hd = 1, 4, 2, 8, 16
        query = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        key = paddle.randn([b, s, nhpp, qk_hd], dtype="float32")
        value = paddle.randn([b, s, nhpp, v_hd], dtype="float32")
        out1 = _unfused_dsa_attention(query, key, value, None, 0.1)
        out2 = _unfused_dsa_attention(query, key, value, None, 1.0)
        # Different scales should produce different outputs
        diff = (out1 - out2).abs().max()
        self.assertGreater(float(diff), 0.0)


def _make_indexer_sublayers_spec():
    return DSAIndexerSublayersSpec(
        linear_wq_b=Linear,
        linear_wk=Linear,
        k_norm=paddle.nn.LayerNorm,
        linear_weights_proj=Linear,
    )


class TestIndexer(unittest.TestCase):
    """Test Indexer class."""

    def test_construction(self):
        config = _make_config()
        spec = _make_indexer_sublayers_spec()
        indexer = Indexer(config, sublayers_spec=spec, layer_number=1)
        self.assertEqual(indexer.n_heads, 1)
        self.assertEqual(indexer.head_dim, 16)
        self.assertEqual(indexer.index_topk, 4)
        self.assertIsNotNone(indexer.wq_b)
        self.assertIsNotNone(indexer.wk)
        self.assertIsNotNone(indexer.k_norm)
        self.assertIsNotNone(indexer.weights_proj)

    def test_nope_head_dim(self):
        config = _make_config(dsa_index_head_dim=16, qk_rope_head_dim=8)
        spec = _make_indexer_sublayers_spec()
        indexer = Indexer(config, sublayers_spec=spec, layer_number=1)
        self.assertEqual(indexer.nope_head_dim, 8)

    @patch("paddleformers.fleet.transformer.dsa_attention._apply_rotary_pos_emb_bshd")
    def test_apply_rope(self, mock_rope):
        mock_rope.return_value = paddle.randn([2, 4, 8])
        config = _make_config()
        spec = _make_indexer_sublayers_spec()
        indexer = Indexer(config, sublayers_spec=spec, layer_number=1)
        x = paddle.randn([2, 4, 16])
        freqs = paddle.randn([1, 4, 1, 8])
        result = indexer._apply_rope(x, freqs, 1.0)
        self.assertEqual(result.shape, [2, 4, 16])
        mock_rope.assert_called()

    @patch("paddleformers.fleet.transformer.dsa_attention.rotate_activation")
    def test_forward_before_topk_shape(self, mock_rotate):
        mock_rotate.side_effect = lambda x: x
        config = _make_config()
        spec = _make_indexer_sublayers_spec()
        indexer = Indexer(config, sublayers_spec=spec, layer_number=1)
        hidden = paddle.randn([2, 4, 64], dtype="float32")
        q_latent = paddle.randn([2, 4, 16], dtype="float32")
        q, k, weights = indexer.forward_before_topk(hidden, q_latent)
        self.assertEqual(q.shape, [2, 4, 1, 16])
        self.assertEqual(k.shape, [2, 4, 16])
        self.assertEqual(weights.shape, [2, 4, 1])


class TestFusedDSAIndexerLoss(unittest.TestCase):
    """Test FusedDSAIndexerLoss."""

    def test_construction(self):
        loss_fn = FusedDSAIndexerLoss()
        self.assertIsNotNone(loss_fn)

    def test_forward_output_shape(self):
        loss_fn = FusedDSAIndexerLoss
        # FusedDSAIndexerLoss is a PyLayer, use .apply() to call
        # Required args: q, weights, k, query, key
        sq, b, h, d = 2, 1, 2, 4
        q = paddle.randn([b, sq, h, d], dtype="float32")
        weights = paddle.randn([b, sq, h], dtype="float32")
        k = paddle.randn([b, sq, d], dtype="float32")
        query = paddle.randn([b, sq, h, d], dtype="float32")
        key = paddle.randn([b, sq, h, d], dtype="float32")
        loss = loss_fn.apply(q, weights, k, query, key)
        self.assertEqual(loss.shape, [])


class TestDSAIndexerLossAutoScaler(unittest.TestCase):
    """Test DSAIndexerLossAutoScaler."""

    def test_set_scale(self):
        DSAIndexerLossAutoScaler.set_loss_scale(paddle.to_tensor(0.5))
        self.assertIsNotNone(DSAIndexerLossAutoScaler._main_loss_backward_scale)

    def test_forward_returns_output(self):
        output = paddle.randn([2, 4])
        indexer_loss = paddle.randn([2, 4])
        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        self.assertEqual(result.shape, output.shape)


if __name__ == "__main__":
    unittest.main()
