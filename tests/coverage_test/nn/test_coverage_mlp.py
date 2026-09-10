# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest
from unittest.mock import MagicMock

import paddle


class TestMLP(unittest.TestCase):
    """Tests for paddleformers.nn.mlp.MLP."""

    def _make_config(self, **overrides):
        config = MagicMock()
        config.hidden_size = overrides.get("hidden_size", 64)
        config.intermediate_size = overrides.get("intermediate_size", 128)
        config.tensor_model_parallel_size = overrides.get("tensor_model_parallel_size", 1)
        config.sequence_parallel = False
        # Use a real dict for get() to avoid MagicMock comparison issues
        config_data = {
            "mlp_bias": overrides.get("mlp_bias", False),
            "fuse_swiglu": overrides.get("fuse_swiglu", False),
            "hidden_act": overrides.get("hidden_act", "silu"),
        }
        config.get = lambda key, default=None: config_data.get(key, default)
        return config

    def test_mlp_init_default(self):
        """MLP with default config should create gate_proj, up_proj, down_proj."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config()
        mlp = MLP(config)
        self.assertTrue(hasattr(mlp, "gate_proj"))
        self.assertTrue(hasattr(mlp, "up_proj"))
        self.assertTrue(hasattr(mlp, "down_proj"))
        self.assertFalse(mlp.fuse_up_gate)

    def test_mlp_init_fuse_up_gate(self):
        """MLP with fuse_up_gate=True should create a single fused projection."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config()
        mlp = MLP(config, fuse_up_gate=True)
        self.assertTrue(hasattr(mlp, "up_gate_proj"))
        self.assertTrue(mlp.fuse_up_gate)

    def test_mlp_init_custom_sizes(self):
        """MLP with custom hidden_size and intermediate_size."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config(hidden_size=32, intermediate_size=64)
        mlp = MLP(config)
        self.assertEqual(mlp.hidden_size, 32)
        self.assertEqual(mlp.intermediate_size, 64)

    def test_mlp_init_with_bias(self):
        """MLP with mlp_bias=True should pass has_bias to linear layers."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config(mlp_bias=True)
        mlp = MLP(config)
        self.assertTrue(mlp.has_bias)

    def test_mlp_init_custom_proj_names(self):
        """MLP with custom projection names should set attributes correctly."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config()
        mlp = MLP(
            config,
            gate_proj_name="my_gate",
            up_proj_name="my_up",
            down_proj_name="my_down",
        )
        self.assertTrue(hasattr(mlp, "my_gate"))
        self.assertTrue(hasattr(mlp, "my_up"))
        self.assertTrue(hasattr(mlp, "my_down"))

    def test_mlp_init_fuse_swiglu(self):
        """MLP with fuse_swiglu=True should set fuse_swiglu flag."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config(fuse_swiglu=True)
        mlp = MLP(config)
        self.assertTrue(mlp.fuse_swiglu)

    def test_mlp_act_fn(self):
        """MLP should set act_fn based on hidden_act config."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config(hidden_act="gelu")
        mlp = MLP(config)
        self.assertEqual(mlp.act_type, "gelu")

    def test_mlp_tensor_parallel_flag(self):
        """MLP should set tensor_parallel flag based on config."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config(tensor_model_parallel_size=1)
        mlp = MLP(config)
        self.assertFalse(mlp.tensor_parallel)

        # For tp_size > 1, mock the parallel linear classes to avoid distributed init
        from paddleformers.nn.linear import Linear

        mock_cls = MagicMock(return_value=paddle.nn.Linear(64, 128))
        original_mapping = Linear._global_mapping.copy()
        try:
            Linear._global_mapping["colwise"] = mock_cls
            Linear._global_mapping["rowwise"] = mock_cls
            config2 = self._make_config(tensor_model_parallel_size=2)
            mlp2 = MLP(config2)
            self.assertTrue(mlp2.tensor_parallel)
        finally:
            Linear._global_mapping = original_mapping

    def test_mlp_has_bias_default_false(self):
        """MLP should default has_bias to False from config."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config()
        mlp = MLP(config)
        self.assertFalse(mlp.has_bias)

    def test_mlp_gate_up_proj_name(self):
        """MLP should store gate_up_proj_name for fused gate/up projection."""
        from paddleformers.nn.mlp import MLP

        config = self._make_config()
        mlp = MLP(config)
        self.assertEqual(mlp.gate_up_proj_name, "up_gate_proj")
