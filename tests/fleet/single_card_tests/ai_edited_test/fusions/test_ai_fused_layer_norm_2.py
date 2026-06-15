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
from unittest.mock import MagicMock, patch

import paddle


def _make_config(**kwargs):
    """Create a mock TransformerConfig."""
    config = MagicMock()
    config.layernorm_zero_centered_gamma = kwargs.get("layernorm_zero_centered_gamma", False)
    config.normalization = kwargs.get("normalization", "LayerNorm")
    config.persist_layer_norm = kwargs.get("persist_layer_norm", False)
    config.sequence_parallel = kwargs.get("sequence_parallel", False)
    return config


class TestFusedLayerNormInit(unittest.TestCase):
    """Tests for FusedLayerNorm initialization."""

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_init_basic(self):
        """Test basic initialization."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=1024)
        self.assertEqual(layer.eps, 1e-5)
        self.assertFalse(layer.zero_centered_gamma)
        self.assertFalse(layer.persist_layer_norm)

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_init_custom_eps(self):
        """Test initialization with custom epsilon."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=1024, eps=1e-6)
        self.assertEqual(layer.eps, 1e-6)

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_init_zero_centered_gamma(self):
        """Test initialization with zero_centered_gamma."""
        config = _make_config(layernorm_zero_centered_gamma=True)
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=1024)
        self.assertTrue(layer.zero_centered_gamma)

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_init_wrong_normalization_raises(self):
        """Test wrong normalization type raises assertion."""
        config = _make_config(normalization="RMSNorm")
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        with self.assertRaises(AssertionError):
            FusedLayerNorm(config, hidden_size=1024)

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        False,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_init_no_apex_raises(self):
        """Test that missing apex raises ValueError."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        with self.assertRaises(ValueError):
            FusedLayerNorm(config, hidden_size=9999)


class TestFusedLayerNormResetParameters(unittest.TestCase):
    """Tests for FusedLayerNorm.reset_parameters."""

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_reset_without_zero_centered(self):
        """Test reset with zero_centered_gamma=False."""
        config = _make_config(layernorm_zero_centered_gamma=False)
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=1024)
        # Weight should be ones, bias should be zeros
        self.assertTrue(paddle.allclose(layer.weight, paddle.ones_like(layer.weight)))
        self.assertTrue(paddle.allclose(layer.bias, paddle.zeros_like(layer.bias)))

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_reset_with_zero_centered(self):
        """Test reset with zero_centered_gamma=True."""
        config = _make_config(layernorm_zero_centered_gamma=True)
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=1024)
        # Both weight and bias should be zeros
        self.assertTrue(paddle.allclose(layer.weight, paddle.zeros_like(layer.weight)))
        self.assertTrue(paddle.allclose(layer.bias, paddle.zeros_like(layer.bias)))


class TestFusedLayerNormForward(unittest.TestCase):
    """Tests for FusedLayerNorm forward pass."""

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    @patch("paddleformers.fleet.fusions.fused_layer_norm.fused_layer_norm")
    def test_forward_calls_fused_ln(self, mock_fused_ln):
        """Test forward calls fused_layer_norm."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        mock_fused_ln.return_value = (
            paddle.randn([2, 4, 1024]),
            paddle.randn([2, 4, 1024]),
            paddle.randn([2, 4, 1024]),
        )
        layer = FusedLayerNorm(config, hidden_size=1024)
        x = paddle.randn([2, 4, 1024])
        result = layer(x)
        self.assertIsNotNone(result)

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_forward_shape_mismatch_raises(self):
        """Test forward raises on shape mismatch."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=512)
        x = paddle.randn([2, 4, 1024])
        with self.assertRaises(ValueError):
            layer(x)


class TestFusedLayerNormHiddenSize(unittest.TestCase):
    """Tests for FusedLayerNorm hidden_size handling."""

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_integral_hidden_size(self):
        """Test integral hidden_size converted to tuple."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=1024)
        # hidden_size should be converted to tuple
        self.assertIsInstance(layer.hidden_size, tuple)

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_tuple_hidden_size(self):
        """Test tuple hidden_size accepted."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=(1024,))
        self.assertEqual(layer.hidden_size, (1024,))


class TestFusedLayerNormSequenceParallel(unittest.TestCase):
    """Tests for FusedLayerNorm sequence parallel flag."""

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_sequence_parallel_set(self):
        """Test sequence_parallel flag is set from config."""
        config = _make_config(sequence_parallel=True)
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=1024)
        self.assertTrue(layer.sequence_parallel)

    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_FUSED_LAYER_NORM",
        True,
    )
    @patch(
        "paddleformers.fleet.fusions.fused_layer_norm.HAVE_PERSIST_LAYER_NORM",
        False,
    )
    def test_weight_sequence_parallel_flag(self):
        """Test weight has sequence_parallel attribute."""
        config = _make_config(sequence_parallel=True)
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm

        layer = FusedLayerNorm(config, hidden_size=1024)
        self.assertTrue(layer.weight.sequence_parallel)
        self.assertTrue(layer.bias.sequence_parallel)


if __name__ == "__main__":
    unittest.main()
