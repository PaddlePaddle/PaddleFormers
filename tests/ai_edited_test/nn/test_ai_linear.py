# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest
from unittest.mock import MagicMock

import paddle.nn as nn


class TestLinear(unittest.TestCase):
    """Tests for paddleformers.nn.linear.Linear factory."""

    def _make_config(self, **overrides):
        config = MagicMock()
        config.tensor_model_parallel_size = overrides.get("tensor_model_parallel_size", 1)
        config.sequence_parallel = overrides.get("sequence_parallel", False)
        return config

    def test_linear_create_default(self):
        """With tp_size=1, linear_type should default to 'default' nn.Linear."""
        from paddleformers.nn.linear import Linear

        config = self._make_config()
        linear = Linear.create(64, 128, config=config)
        self.assertIsInstance(linear, nn.Linear)

    def test_linear_create_explicit_type(self):
        """Explicit linear_type should be used directly."""
        from paddleformers.nn.linear import Linear

        config = self._make_config()
        linear = Linear.create(64, 128, config=config, linear_type="default")
        self.assertIsInstance(linear, nn.Linear)

    def test_linear_create_no_type_no_config_raises(self):
        """Should raise ValueError when neither linear_type nor config is provided."""
        from paddleformers.nn.linear import Linear

        with self.assertRaises(ValueError):
            Linear.create(64, 128)

    def test_linear_create_with_config_infers_type(self):
        """When only config is given, linear_type is inferred from config."""
        from paddleformers.nn.linear import Linear

        config = self._make_config(tensor_model_parallel_size=1)
        linear = Linear.create(64, 128, config=config)
        self.assertIsInstance(linear, nn.Linear)

    def test_get_linear_type_single_gpu(self):
        """get_linear_type should return 'default' for tp_size=1."""
        from paddleformers.nn.linear import Linear

        config = self._make_config(tensor_model_parallel_size=1)
        self.assertEqual(Linear.get_linear_type(config), "default")

    def test_get_linear_type_colwise(self):
        """get_linear_type should return 'colwise' for multi-GPUs with default tp_plan."""
        from paddleformers.nn.linear import Linear

        config = self._make_config(tensor_model_parallel_size=2)
        self.assertEqual(Linear.get_linear_type(config, tp_plan="colwise"), "colwise")

    def test_get_linear_type_rowwise(self):
        """get_linear_type with rowwise tp_plan."""
        from paddleformers.nn.linear import Linear

        config = self._make_config(tensor_model_parallel_size=2)
        self.assertEqual(Linear.get_linear_type(config, tp_plan="rowwise"), "rowwise")

    def test_get_linear_type_sequence_parallel(self):
        """get_linear_type should prepend 'sequence_' when sequence_parallel is True."""
        from paddleformers.nn.linear import Linear

        config = self._make_config(tensor_model_parallel_size=2, sequence_parallel=True)
        self.assertEqual(Linear.get_linear_type(config, tp_plan="colwise"), "sequence_colwise")

    def test_get_linear_type_sequence_parallel_rowwise(self):
        """get_linear_type with rowwise tp_plan and sequence_parallel."""
        from paddleformers.nn.linear import Linear

        config = self._make_config(tensor_model_parallel_size=2, sequence_parallel=True)
        self.assertEqual(Linear.get_linear_type(config, tp_plan="rowwise"), "sequence_rowwise")

    def test_get_linear_kwargs_default(self):
        """get_linear_kwargs for 'default' should return bias_attr."""
        from paddleformers.nn.linear import Linear

        kwargs = Linear.get_linear_kwargs("default", has_bias=True)
        self.assertEqual(kwargs, {"bias_attr": True})

    def test_get_linear_kwargs_default_no_bias(self):
        """get_linear_kwargs for 'default' with has_bias=False."""
        from paddleformers.nn.linear import Linear

        kwargs = Linear.get_linear_kwargs("default", has_bias=False)
        self.assertEqual(kwargs, {"bias_attr": False})

    def test_get_linear_kwargs_colwise(self):
        """get_linear_kwargs for 'colwise' should include gather_output."""
        from paddleformers.nn.linear import Linear

        kwargs = Linear.get_linear_kwargs("colwise", has_bias=True, gather_output=True)
        self.assertEqual(kwargs, {"has_bias": True, "gather_output": True})

    def test_get_linear_kwargs_rowwise(self):
        """get_linear_kwargs for 'rowwise' should include input_is_parallel."""
        from paddleformers.nn.linear import Linear

        kwargs = Linear.get_linear_kwargs("rowwise", has_bias=False, input_is_parallel=False)
        self.assertEqual(kwargs, {"has_bias": False, "input_is_parallel": False})

    def test_get_linear_kwargs_sequence_colwise(self):
        """get_linear_kwargs for 'sequence_colwise'."""
        from paddleformers.nn.linear import Linear

        kwargs = Linear.get_linear_kwargs("sequence_colwise", has_bias=True, gather_output=False)
        self.assertEqual(kwargs, {"has_bias": True, "gather_output": False})

    def test_get_linear_kwargs_sequence_rowwise(self):
        """get_linear_kwargs for 'sequence_rowwise'."""
        from paddleformers.nn.linear import Linear

        kwargs = Linear.get_linear_kwargs("sequence_rowwise", has_bias=True, input_is_parallel=True)
        self.assertEqual(kwargs, {"has_bias": True, "input_is_parallel": True})

    def test_linear_create_colwise_with_parallel(self):
        """With tp_size>1 and colwise, Linear.create should use ColumnParallelLinear."""
        from paddleformers.nn.linear import Linear

        config = self._make_config(tensor_model_parallel_size=2)
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)
        original_mapping = Linear._global_mapping.copy()
        try:
            Linear._global_mapping["colwise"] = mock_cls
            Linear.create(64, 128, config=config, tp_plan="colwise")
            mock_cls.assert_called_once()
        finally:
            Linear._global_mapping = original_mapping

    def test_linear_create_sequence_parallel(self):
        """With sequence_parallel and tp_size>1, should use sequence_colwise type."""
        from paddleformers.nn.linear import Linear

        config = self._make_config(tensor_model_parallel_size=2, sequence_parallel=True)
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)
        original_mapping = Linear._global_mapping.copy()
        try:
            Linear._global_mapping["sequence_colwise"] = mock_cls
            Linear.create(64, 128, config=config, tp_plan="colwise")
            mock_cls.assert_called_once()
        finally:
            Linear._global_mapping = original_mapping

    def test_linear_create_rowwise(self):
        """Explicit rowwise linear_type should work."""
        from paddleformers.nn.linear import Linear

        config = self._make_config(tensor_model_parallel_size=2)
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)
        original_mapping = Linear._global_mapping.copy()
        try:
            Linear._global_mapping["rowwise"] = mock_cls
            Linear.create(64, 128, config=config, tp_plan="rowwise")
            mock_cls.assert_called_once()
        finally:
            Linear._global_mapping = original_mapping

    def test_linear_create_with_weight_attr(self):
        """weight_attr should be passed through to the linear constructor."""
        from paddleformers.nn.linear import Linear

        config = self._make_config()
        linear = Linear.create(64, 128, config=config)
        self.assertIsInstance(linear, nn.Linear)
        # PaddlePaddle Linear stores weight as [in_features, out_features]
        self.assertEqual(linear.weight.shape, [64, 128])
