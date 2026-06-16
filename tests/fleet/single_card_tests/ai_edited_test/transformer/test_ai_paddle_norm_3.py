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

import paddle

from paddleformers.fleet.transformer.paddle_norm import (
    L2Norm,
    LayerNorm,
    RMSNorm,
    WrappedPaddleNorm,
    WrappedPaddleNormPipe,
    get_norm_extra_args,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "normalization": "RMSNorm",
        "rms_norm_eps": 1e-5,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestWrappedPaddleNormSelection(unittest.TestCase):
    """Tests for WrappedPaddleNorm normalization type selection."""

    def test_rmsnorm_selection(self):
        """Test WrappedPaddleNorm selects RMSNorm."""
        config = _make_config(normalization="RMSNorm")
        norm = WrappedPaddleNorm(config=config, hidden_size=64)
        self.assertIsInstance(norm, RMSNorm)

    def test_layernorm_selection(self):
        """Test WrappedPaddleNorm selects LayerNorm."""
        config = _make_config(normalization="LayerNorm")
        norm = WrappedPaddleNorm(config=config, hidden_size=64)
        self.assertIsInstance(norm, LayerNorm)

    def test_unsupported_normalization_raises(self):
        """Test WrappedPaddleNorm with unsupported normalization raises Exception."""
        config = _make_config(normalization="InvalidNorm")
        with self.assertRaises(Exception):
            WrappedPaddleNorm(config=config, hidden_size=64)

    def test_input_is_parallel_default(self):
        """Test default input_is_parallel value."""
        config = _make_config(
            sequence_parallel=False, tensor_model_parallel_size=1
        )
        norm = WrappedPaddleNorm(config=config, hidden_size=64)
        self.assertIsInstance(norm, RMSNorm)

    def test_input_is_parallel_explicit(self):
        """Test explicit input_is_parallel value."""
        config = _make_config()
        norm = WrappedPaddleNorm(
            config=config, hidden_size=64, input_is_parallel=True
        )
        self.assertIsInstance(norm, RMSNorm)


class TestWrappedPaddleNormPipeMTP(unittest.TestCase):
    """Tests for WrappedPaddleNormPipe with MTP configurations."""

    def test_mtp_disabled(self):
        """Test WrappedPaddleNormPipe with MTP disabled."""
        config = _make_config()
        norm_pipe = WrappedPaddleNormPipe(config=config, hidden_size=128)
        dict_args = {"hidden_states": paddle.randn([2, 4, 128])}
        result = norm_pipe(dict_args)
        self.assertEqual(result["hidden_states"].shape, [2, 4, 128])

    def test_mtp_load_weight_only(self):
        """Test WrappedPaddleNormPipe with mtp_load_weight_only=True."""
        config = _make_config(
            num_nextn_predict_layers=2, mtp_load_weight_only=True
        )
        norm_pipe = WrappedPaddleNormPipe(config=config, hidden_size=128)
        dict_args = {"hidden_states": paddle.randn([3, 4, 128])}
        result = norm_pipe(dict_args)
        self.assertEqual(result["hidden_states"].shape, [3, 4, 128])

    def test_mtp_preserves_other_dict_keys(self):
        """Test WrappedPaddleNormPipe preserves other dict keys."""
        config = _make_config()
        norm_pipe = WrappedPaddleNormPipe(config=config, hidden_size=128)
        dict_args = {
            "hidden_states": paddle.randn([2, 4, 128]),
            "attention_mask": paddle.ones([1, 1, 4, 4]),
        }
        result = norm_pipe(dict_args)
        self.assertIn("attention_mask", result)


class TestL2NormEdgeCases(unittest.TestCase):
    """Edge case tests for L2Norm."""

    def test_l2norm_with_zeros(self):
        """Test L2Norm with zero input."""
        norm = L2Norm(hidden_size=64)
        x = paddle.zeros([2, 4, 64])
        out = norm(x)
        self.assertFalse(paddle.isnan(out).any().item())

    def test_l2norm_with_large_values(self):
        """Test L2Norm with large input values."""
        norm = L2Norm(hidden_size=64)
        x = paddle.randn([2, 4, 64]) * 1000
        out = norm(x)
        # L2Norm normalizes so mean of squared values along last dim is ~1
        mean_sq = out.float().pow(2).mean(-1)
        self.assertTrue(
            paddle.allclose(mean_sq, paddle.ones_like(mean_sq), atol=0.1).item()
        )

    def test_l2norm_preserves_shape(self):
        """Test L2Norm preserves input shape."""
        norm = L2Norm(hidden_size=64)
        x = paddle.randn([2, 4, 64])
        out = norm(x)
        self.assertEqual(out.shape, x.shape)


class TestGetNormExtraArgsEdgeCases(unittest.TestCase):
    """Edge case tests for get_norm_extra_args."""

    def test_with_layer_spec_wrapped_paddle_norm(self):
        """Test with LayerSpec containing WrappedPaddleNorm."""
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        config = _make_config()
        layer_spec = LayerSpec(WrappedPaddleNorm)
        extra_args = get_norm_extra_args(layer_spec, config, 128, 1e-5, False)
        self.assertIn("hidden_size", extra_args)
        self.assertEqual(extra_args["hidden_size"], 128)
        self.assertEqual(extra_args["eps"], 1e-5)

    def test_with_rmsnorm_class(self):
        """Test with RMSNorm class directly."""
        config = _make_config()
        extra_args = get_norm_extra_args(RMSNorm, config, 64, 1e-6, True)
        self.assertEqual(extra_args["normalized_shape"], 64)
        self.assertEqual(extra_args["norm_eps"], 1e-6)
        self.assertTrue(extra_args["input_is_parallel"])


if __name__ == "__main__":
    unittest.main()
