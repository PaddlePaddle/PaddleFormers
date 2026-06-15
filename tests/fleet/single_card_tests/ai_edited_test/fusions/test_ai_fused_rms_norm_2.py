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
    config.normalization = kwargs.get("normalization", "RMSNorm")
    config.sequence_parallel = kwargs.get("sequence_parallel", False)
    return config


class TestFusedRmsNormInit(unittest.TestCase):
    """Tests for FusedRmsNorm initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=1024)
        self.assertEqual(layer.eps, 1e-6)
        self.assertFalse(layer.zero_centered_gamma)

    def test_init_custom_eps(self):
        """Test initialization with custom epsilon."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=1024, eps=1e-8)
        self.assertEqual(layer.eps, 1e-8)

    def test_init_zero_centered_gamma(self):
        """Test initialization with zero_centered_gamma."""
        config = _make_config(layernorm_zero_centered_gamma=True)
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=1024)
        self.assertTrue(layer.zero_centered_gamma)

    def test_init_wrong_normalization_raises(self):
        """Test wrong normalization type raises assertion."""
        config = _make_config(normalization="LayerNorm")
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        with self.assertRaises(AssertionError):
            FusedRmsNorm(config, hidden_size=1024)


class TestFusedRmsNormResetParameters(unittest.TestCase):
    """Tests for FusedRmsNorm.reset_parameters."""

    def test_reset_without_zero_centered(self):
        """Test reset with zero_centered_gamma=False."""
        config = _make_config(layernorm_zero_centered_gamma=False)
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=1024)
        self.assertTrue(paddle.allclose(layer.weight, paddle.ones_like(layer.weight)))
        self.assertTrue(paddle.allclose(layer.bias, paddle.zeros_like(layer.bias)))

    def test_reset_with_zero_centered(self):
        """Test reset with zero_centered_gamma=True."""
        config = _make_config(layernorm_zero_centered_gamma=True)
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=1024)
        self.assertTrue(paddle.allclose(layer.weight, paddle.zeros_like(layer.weight)))
        self.assertTrue(paddle.allclose(layer.bias, paddle.zeros_like(layer.bias)))


class TestFusedRmsNormForward(unittest.TestCase):
    """Tests for FusedRmsNorm forward pass."""

    @patch("paddleformers.fleet.fusions.fused_rms_norm.fused_rms_norm")
    def test_forward_calls_fused_rms_norm(self, mock_fused_rms):
        """Test forward calls fused_rms_norm."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        mock_fused_rms.return_value = paddle.randn([2, 4, 1024])
        layer = FusedRmsNorm(config, hidden_size=1024)
        x = paddle.randn([2, 4, 1024])
        result = layer(x)
        mock_fused_rms.assert_called_once()

    @patch("paddleformers.fleet.fusions.fused_rms_norm.fused_rms_norm")
    def test_forward_tuple_output(self, mock_fused_rms):
        """Test forward handles tuple output from fused_rms_norm."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        mock_fused_rms.return_value = (
            paddle.randn([2, 4, 1024]),
            paddle.randn([2, 4, 1024]),
            paddle.randn([2, 4, 1024]),
        )
        layer = FusedRmsNorm(config, hidden_size=1024)
        x = paddle.randn([2, 4, 1024])
        result = layer(x)
        self.assertEqual(result.shape, [2, 4, 1024])

    def test_forward_shape_mismatch_raises(self):
        """Test forward raises on shape mismatch."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=512)
        x = paddle.randn([2, 4, 1024])
        with self.assertRaises(ValueError):
            layer(x)


class TestFusedRmsNormZeroCenteredGamma(unittest.TestCase):
    """Tests for FusedRmsNorm with zero_centered_gamma in forward."""

    @patch("paddleformers.fleet.fusions.fused_rms_norm.fused_rms_norm")
    def test_weight_adjusted_when_zero_centered(self, mock_fused_rms):
        """Test weight+1 when zero_centered_gamma=True."""
        config = _make_config(layernorm_zero_centered_gamma=True)
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        mock_fused_rms.return_value = paddle.randn([2, 4, 1024])
        layer = FusedRmsNorm(config, hidden_size=1024)
        x = paddle.randn([2, 4, 1024])
        result = layer(x)
        # The weight passed should be weight + 1
        call_args = mock_fused_rms.call_args
        weight_arg = call_args[0][1]
        # Since weight was initialized to zeros, weight + 1 = ones
        self.assertTrue(
            paddle.allclose(
                weight_arg.cast(paddle.float32),
                paddle.ones([1024], dtype=paddle.float32),
                atol=1e-5,
            )
        )


class TestFusedRmsNormHiddenSize(unittest.TestCase):
    """Tests for FusedRmsNorm hidden_size handling."""

    def test_integral_hidden_size(self):
        """Test integral hidden_size converted to tuple."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=1024)
        self.assertIsInstance(layer.hidden_size, tuple)

    def test_tuple_hidden_size(self):
        """Test tuple hidden_size accepted."""
        config = _make_config()
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=(1024,))
        self.assertEqual(layer.hidden_size, (1024,))


class TestFusedRmsNormSequenceParallel(unittest.TestCase):
    """Tests for FusedRmsNorm sequence parallel flag."""

    def test_sequence_parallel_set(self):
        """Test sequence_parallel flag is set from config."""
        config = _make_config(sequence_parallel=True)
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=1024)
        self.assertTrue(layer.sequence_parallel)

    def test_weight_sequence_parallel_flag(self):
        """Test weight has sequence_parallel attribute."""
        config = _make_config(sequence_parallel=True)
        from paddleformers.fleet.fusions.fused_rms_norm import FusedRmsNorm

        layer = FusedRmsNorm(config, hidden_size=1024)
        self.assertTrue(layer.weight.sequence_parallel)
        self.assertTrue(layer.bias.sequence_parallel)


if __name__ == "__main__":
    unittest.main()
