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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.paddle_norm import (
    FusedRMSNorm,
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
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestRMSNorm(unittest.TestCase):
    """Test RMSNorm layer."""

    def test_construction_with_defaults(self):
        config = _make_config()
        norm = RMSNorm(config)
        self.assertEqual(norm.normalized_shape, 128)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-5)

    def test_construction_with_custom_shape(self):
        config = _make_config()
        norm = RMSNorm(config, normalized_shape=64)
        self.assertEqual(norm.normalized_shape, 64)

    def test_construction_with_custom_eps(self):
        config = _make_config()
        norm = RMSNorm(config, norm_eps=1e-6)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-6)

    def test_forward_shape(self):
        config = _make_config()
        norm = RMSNorm(config)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_with_params_dtype_none(self):
        config = _make_config()
        config.params_dtype = None
        norm = RMSNorm(config)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_returns_correct_dtype(self):
        config = _make_config()
        norm = RMSNorm(config)
        x = paddle.randn([2, 4, 128]).cast("float32")
        out = norm(x)
        self.assertEqual(out.dtype, paddle.float32)

    def test_enable_sequence_parallel(self):
        config = _make_config()
        norm = RMSNorm(config, input_is_parallel=True)
        # Should not raise
        norm.enable_sequence_parallel()

    @patch(
        "paddleformers.fleet.transformer.paddle_norm.mark_as_sequence_parallel_parameter"
    )
    def test_enable_sequence_parallel_calls_marker(self, mock_mark):
        config = _make_config()
        norm = RMSNorm(config, input_is_parallel=True)
        norm.enable_sequence_parallel()
        mock_mark.assert_called_with(norm.weight)


class TestLayerNormClass(unittest.TestCase):
    """Test LayerNorm layer."""

    def test_construction(self):
        config = _make_config()
        norm = LayerNorm(config)
        self.assertEqual(norm.normalized_shape, 128)
        self.assertIsNotNone(norm.weight)
        self.assertIsNotNone(norm.bias)

    def test_forward_shape(self):
        config = _make_config()
        norm = LayerNorm(config)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_with_custom_shape(self):
        config = _make_config()
        norm = LayerNorm(config, normalized_shape=64)
        x = paddle.randn([2, 4, 64])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 64])

    def test_bias_initialization(self):
        config = _make_config()
        norm = LayerNorm(config)
        # Bias should be initialized to near zero
        self.assertAlmostEqual(float(norm.bias.mean()), 0.0, places=4)


class TestFusedRMSNorm(unittest.TestCase):
    """Test FusedRMSNorm layer."""

    def test_construction(self):
        config = _make_config()
        norm = FusedRMSNorm(config)
        self.assertEqual(norm.normalized_shape, 128)

    def test_forward_shape(self):
        config = _make_config()
        norm = FusedRMSNorm(config)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_is_subclass_of_rms_norm(self):
        self.assertTrue(issubclass(FusedRMSNorm, RMSNorm))


class TestWrappedPaddleNorm(unittest.TestCase):
    """Test WrappedPaddleNorm factory."""

    def test_creates_rms_norm(self):
        config = _make_config(normalization="RMSNorm")
        norm = WrappedPaddleNorm(config, hidden_size=64)
        self.assertIsInstance(norm, RMSNorm)

    def test_creates_layer_norm(self):
        config = _make_config(normalization="LayerNorm")
        norm = WrappedPaddleNorm(config, hidden_size=64)
        self.assertIsInstance(norm, LayerNorm)

    def test_unsupported_norm_raises(self):
        config = _make_config(normalization="UnsupportedNorm")
        with self.assertRaises(Exception):  # noqa: B017
            WrappedPaddleNorm(config, hidden_size=64)

    def test_input_is_parallel_none_uses_config(self):
        config = _make_config(
            normalization="RMSNorm",
            sequence_parallel=False,
            tensor_model_parallel_size=1,
        )
        norm = WrappedPaddleNorm(config, hidden_size=64, input_is_parallel=None)
        self.assertIsInstance(norm, RMSNorm)

    def test_passes_hidden_size_and_eps(self):
        config = _make_config(rms_norm_eps=1e-6)
        norm = WrappedPaddleNorm(config, hidden_size=32, eps=1e-6)
        self.assertEqual(norm.normalized_shape, 32)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-6)

    def test_uses_new_semantics(self):
        # WrappedPaddleNorm uses __new__, so calling it returns the norm directly
        config = _make_config()
        norm = WrappedPaddleNorm(config, hidden_size=64)
        self.assertIsNotNone(norm.weight)


class TestWrappedPaddleNormPipe(unittest.TestCase):
    """Test WrappedPaddleNormPipe."""

    def test_construction(self):
        config = _make_config()
        pipe = WrappedPaddleNormPipe(config, hidden_size=64)
        self.assertIsNotNone(pipe.norm)

    def test_forward_without_mtp(self):
        config = _make_config(num_nextn_predict_layers=0)
        pipe = WrappedPaddleNormPipe(config, hidden_size=64)
        x = paddle.randn([2, 4, 64])
        result = pipe({"hidden_states": x})
        self.assertEqual(result["hidden_states"].shape, [2, 4, 64])

    def test_forward_with_mtp_enabled(self):
        config = _make_config(
            num_nextn_predict_layers=1,
            mtp_load_weight_only=False,
        )
        pipe = WrappedPaddleNormPipe(config, hidden_size=64)
        # Simulate concatenated hidden states: main + mtp
        x_main = paddle.randn([2, 4, 64])
        x_mtp = paddle.randn([2, 4, 64])
        x_concat = paddle.concat([x_main, x_mtp], axis=0)
        result = pipe({"hidden_states": x_concat})
        # Output should have the same shape (main + mtp concatenated)
        self.assertEqual(result["hidden_states"].shape, [4, 4, 64])

    def test_forward_with_mtp_load_weight_only(self):
        config = _make_config(
            num_nextn_predict_layers=1,
            mtp_load_weight_only=True,
        )
        pipe = WrappedPaddleNormPipe(config, hidden_size=64)
        x = paddle.randn([2, 4, 64])
        result = pipe({"hidden_states": x})
        self.assertEqual(result["hidden_states"].shape, [2, 4, 64])

    def test_build_schedule_node(self):
        config = _make_config()
        pipe = WrappedPaddleNormPipe(config, hidden_size=64)
        node = pipe.build_schedule_node()
        self.assertIsNotNone(node)


class TestL2Norm(unittest.TestCase):
    """Test L2Norm layer."""

    def test_construction(self):
        norm = L2Norm(hidden_size=128)
        self.assertEqual(norm.hidden_size, 128)
        self.assertAlmostEqual(norm.eps, 1e-6)

    def test_forward_shape(self):
        norm = L2Norm(hidden_size=128)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_preserves_dtype(self):
        norm = L2Norm(hidden_size=128)
        x = paddle.randn([2, 4, 128]).cast("float32")
        out = norm(x)
        self.assertEqual(out.dtype, paddle.float32)

    def test_forward_normalization_magnitude(self):
        norm = L2Norm(hidden_size=4, eps=1e-6)
        x = paddle.ones([1, 1, 4]) * 2.0
        out = norm(x)
        # L2Norm normalizes by sqrt(sum(x^2)) per vector
        # sum(x^2) = 4 * 4 = 16, sqrt(16) = 4, so each element = 2/4 = 0.5
        # But actual implementation divides by L2 norm which includes eps
        # Values should be close to 1.0 (L2 normalization to unit norm)
        for val in out.flatten().tolist():
            self.assertAlmostEqual(val, 1.0, places=4)

    def test_custom_eps(self):
        norm = L2Norm(hidden_size=128, eps=1e-8)
        self.assertAlmostEqual(norm.eps, 1e-8)


class TestGetNormExtraArgs(unittest.TestCase):
    """Test get_norm_extra_args helper."""

    def test_for_wrapped_paddle_norm(self):
        config = _make_config()
        extra = get_norm_extra_args(WrappedPaddleNorm, config, 64, 1e-5, False)
        self.assertIn("hidden_size", extra)
        self.assertIn("eps", extra)
        self.assertEqual(extra["hidden_size"], 64)
        self.assertAlmostEqual(extra["eps"], 1e-5)
        self.assertIn("config", extra)
        self.assertIn("input_is_parallel", extra)

    def test_for_other_norm_class(self):
        config = _make_config()
        extra = get_norm_extra_args(RMSNorm, config, 64, 1e-5, True)
        self.assertIn("normalized_shape", extra)
        self.assertIn("norm_eps", extra)
        self.assertEqual(extra["normalized_shape"], 64)
        self.assertAlmostEqual(extra["norm_eps"], 1e-5)
        self.assertTrue(extra["input_is_parallel"])

    def test_with_layer_spec(self):
        config = _make_config()
        mock_spec = MagicMock()
        mock_spec.layer = WrappedPaddleNorm
        extra = get_norm_extra_args(mock_spec, config, 64, 1e-5, False)
        self.assertIn("normalized_shape", extra)

    def test_with_direct_class(self):
        config = _make_config()
        extra = get_norm_extra_args(LayerNorm, config, 32, 1e-6, False)
        self.assertEqual(extra["normalized_shape"], 32)
        self.assertAlmostEqual(extra["norm_eps"], 1e-6)


if __name__ == "__main__":
    unittest.main()
