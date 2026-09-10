# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestLayerNorm(unittest.TestCase):
    """Tests for paddleformers.nn.norm.LayerNorm."""

    def _make_config(self, **overrides):
        config = MagicMock()
        config.hidden_size = overrides.get("hidden_size", 64)
        config.norm_eps = overrides.get("norm_eps", 1e-5)
        config.rms_norm_eps = 1e-6
        config.get = lambda key, default=None: overrides.get(key, default)
        return config

    def test_layer_norm_forward_shape(self):
        """LayerNorm should produce output with the same shape as input."""
        from paddleformers.nn.norm import LayerNorm

        config = self._make_config(hidden_size=32)
        norm = LayerNorm(config)
        x = paddle.randn([2, 8, 32])
        out = norm(x)
        self.assertEqual(out.shape, [2, 8, 32])

    def test_layer_norm_custom_hidden_size(self):
        """LayerNorm with custom hidden_size override."""
        from paddleformers.nn.norm import LayerNorm

        config = self._make_config(hidden_size=64)
        norm = LayerNorm(config, hidden_size=128)
        self.assertEqual(norm.hidden_size, 128)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_layer_norm_custom_eps(self):
        """LayerNorm with custom norm_eps override."""
        from paddleformers.nn.norm import LayerNorm

        config = self._make_config(norm_eps=1e-5)
        norm = LayerNorm(config, norm_eps=1e-8)
        self.assertAlmostEqual(norm.norm_eps, 1e-8)

    def test_layer_norm_config_stored(self):
        """LayerNorm should store config reference."""
        from paddleformers.nn.norm import LayerNorm

        config = self._make_config()
        norm = LayerNorm(config)
        self.assertIs(norm.config, config)


class TestRMSNorm(unittest.TestCase):
    """Tests for paddleformers.nn.norm.RMSNorm."""

    def _make_config(self, **overrides):
        config = MagicMock()
        config.hidden_size = overrides.get("hidden_size", 64)
        config.rms_norm_eps = overrides.get("rms_norm_eps", 1e-6)
        config.norm_eps = 1e-5
        config.get = lambda key, default=None: overrides.get(key, default)
        return config

    @patch("paddleformers.nn.norm.detect_device", return_value="gpu")
    @patch("paddleformers.nn.norm.fused_rms_norm_ext")
    def test_rms_norm_forward_shape_with_fused(self, mock_fused, mock_device):
        """RMSNorm with fuse_rms_norm=True should use fused implementation."""
        from paddleformers.nn.norm import RMSNorm

        mock_fused.return_value = (paddle.randn([2, 8, 64]),)
        config = self._make_config(fuse_rms_norm=True)
        norm = RMSNorm(config)
        x = paddle.randn([2, 8, 64], dtype="float32")
        out = norm(x)
        self.assertEqual(out.shape, [2, 8, 64])
        mock_fused.assert_called_once()

    @patch("paddleformers.nn.norm.detect_device", return_value="gpu")
    @patch("paddleformers.nn.norm.fused_rms_norm_ext")
    def test_rms_norm_iluvatar_skip_fused(self, mock_fused, mock_device):
        """RMSNorm on iluvatar_gpu should skip fused implementation."""
        from paddleformers.nn.norm import RMSNorm

        mock_device.return_value = "iluvatar_gpu"
        config = self._make_config(fuse_rms_norm=True)
        norm = RMSNorm(config)
        x = paddle.randn([2, 4, 64], dtype="float32")
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 64])
        mock_fused.assert_not_called()

    @patch("paddleformers.nn.norm.detect_device", return_value="gpu")
    @patch("paddleformers.nn.norm.fused_rms_norm_ext")
    def test_rms_norm_no_fuse(self, mock_fused, mock_device):
        """RMSNorm with fuse_rms_norm=False should use manual implementation."""
        from paddleformers.nn.norm import RMSNorm

        config = self._make_config(fuse_rms_norm=False)
        norm = RMSNorm(config)
        x = paddle.randn([2, 4, 64], dtype="float32")
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 64])
        mock_fused.assert_not_called()

    def test_rms_norm_custom_hidden_size(self):
        """RMSNorm with custom hidden_size override."""
        from paddleformers.nn.norm import RMSNorm

        config = self._make_config(hidden_size=64)
        norm = RMSNorm(config, hidden_size=128)
        self.assertEqual(norm.hidden_size, 128)

    def test_rms_norm_custom_eps(self):
        """RMSNorm with custom norm_eps override."""
        from paddleformers.nn.norm import RMSNorm

        config = self._make_config(rms_norm_eps=1e-6)
        norm = RMSNorm(config, norm_eps=1e-8)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-8)

    @patch("paddleformers.nn.norm.detect_device", return_value="gpu")
    @patch("paddleformers.nn.norm.fused_rms_norm_ext")
    def test_rms_norm_float16_dtype(self, mock_fused, mock_device):
        """RMSNorm should handle float16 weight dtype correctly in manual path."""
        from paddleformers.nn.norm import RMSNorm

        config = self._make_config(fuse_rms_norm=False)
        norm = RMSNorm(config)
        # Manually set weight to float16
        norm.weight = paddle.create_parameter(
            shape=[64], dtype="float16", default_initializer=paddle.nn.initializer.Constant(1.0)
        )
        x = paddle.randn([2, 4, 64], dtype="float32")
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 64])
        mock_fused.assert_not_called()


class TestNorm(unittest.TestCase):
    """Tests for paddleformers.nn.norm.Norm factory."""

    def _make_config(self, **overrides):
        config = MagicMock()
        config.hidden_size = 64
        config.rms_norm_eps = 1e-6
        config.norm_eps = 1e-5
        config.get = lambda key, default=None: overrides.get(key, default)
        return config

    def test_norm_create_default_rms(self):
        """Norm.create with no norm_type should default to rms_norm."""
        from paddleformers.nn.norm import Norm, RMSNorm

        config = self._make_config()
        norm = Norm.create(config)
        self.assertIsInstance(norm, RMSNorm)

    def test_norm_create_layer_norm(self):
        """Norm.create with norm_type='layer_norm' should return LayerNorm."""
        from paddleformers.nn.norm import LayerNorm, Norm

        config = self._make_config()
        norm = Norm.create(config, norm_type="layer_norm")
        self.assertIsInstance(norm, LayerNorm)

    def test_norm_create_custom_hidden_size(self):
        """Norm.create should pass through custom hidden_size."""
        from paddleformers.nn.norm import Norm

        config = self._make_config()
        norm = Norm.create(config, hidden_size=128)
        self.assertEqual(norm.hidden_size, 128)

    def test_norm_create_custom_eps(self):
        """Norm.create should pass through custom norm_eps."""
        from paddleformers.nn.norm import Norm

        config = self._make_config()
        norm = Norm.create(config, norm_eps=1e-8)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-8)

    def test_norm_create_has_bias_defaults_to_config(self):
        """Norm.create should read use_bias from config when has_bias is None."""
        from paddleformers.nn.norm import Norm

        config = self._make_config(use_bias=True)
        norm = Norm.create(config, norm_type="layer_norm")
        # LayerNorm always has bias, but has_bias should be passed through
        self.assertIsNotNone(norm.bias)
