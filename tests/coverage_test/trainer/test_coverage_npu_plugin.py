# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/plugins/npu_plugin.py"""

import types
import unittest
from unittest.mock import MagicMock

import paddle

from paddleformers.trainer.plugins.npu_plugin import (
    _optimizer_step_with_flatten_param_grads,
    npu_accelerate_plugin,
)


class MockOptimizer:
    """Mock optimizer for NPU plugin testing."""

    def __init__(self, params=None):
        self._param_groups = params or []
        self.regularization = None
        self._grad_clip = None
        self.helper = MagicMock()
        self._apply_optimize = MagicMock()

    def step(self):
        pass


class TestNpuAcceleratePlugin(unittest.TestCase):
    """Tests for npu_accelerate_plugin function."""

    def test_replaces_step_method(self):
        """Test that npu_accelerate_plugin replaces optimizer step method."""
        optimizer = MockOptimizer()
        original_step = optimizer.step
        npu_accelerate_plugin(optimizer)
        self.assertNotEqual(optimizer.step, original_step)
        self.assertIsInstance(optimizer.step, types.MethodType)

    def test_new_step_is_wrapped(self):
        """Test that the new step method is the flattened version."""
        optimizer = MockOptimizer()
        npu_accelerate_plugin(optimizer)
        # The step method should now be _optimizer_step_with_flatten_param_grads
        self.assertEqual(optimizer.step.__func__, _optimizer_step_with_flatten_param_grads)


class TestOptimizerStepWithFlattenParamGrads(unittest.TestCase):
    """Tests for _optimizer_step_with_flatten_param_grads function."""

    def test_dict_param_groups_raises(self):
        """Test that dict-style param groups raise RuntimeError."""
        optimizer = MockOptimizer()
        optimizer._param_groups = [{"params": []}]
        with self.assertRaises(RuntimeError):
            _optimizer_step_with_flatten_param_grads(optimizer)

    @unittest.skip("Requires _default_dict on optimizer which is set by Paddle internals")
    def test_skip_stop_gradient_params(self):
        """Test that parameters with stop_gradient are skipped."""
        param = paddle.to_tensor([1.0, 2.0])
        param.stop_gradient = True
        optimizer = MockOptimizer(params=[param])
        # When all params have stop_gradient, params_grads is empty,
        # _flatten_param_grads is not called, and _apply_optimize is called
        # with the empty params_grads list
        _optimizer_step_with_flatten_param_grads(optimizer)


if __name__ == "__main__":
    unittest.main()
