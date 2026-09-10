# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for peft/lora/loraga_utils.py"""

import unittest
from unittest.mock import MagicMock

import paddle
import paddle.nn as nn

from paddleformers.peft.lora.loraga_utils import (
    GradientOffloadHookContext,
    get_hook_enable,
    loraga_svd_module,
    set_hook_enable,
)


class TestHookEnable(unittest.TestCase):
    """Tests for set_hook_enable and get_hook_enable."""

    def test_set_hook_enable_true(self):
        """Test setting hook enable to True."""
        set_hook_enable(True)
        self.assertTrue(get_hook_enable())

    def test_set_hook_enable_false(self):
        """Test setting hook enable to False."""
        set_hook_enable(False)
        self.assertFalse(get_hook_enable())

    def test_set_hook_enable_toggle(self):
        """Test toggling hook enable state."""
        set_hook_enable(True)
        self.assertTrue(get_hook_enable())
        set_hook_enable(False)
        self.assertFalse(get_hook_enable())
        # Restore
        set_hook_enable(False)

    def tearDown(self):
        set_hook_enable(False)


class TestGradientOffloadHookContext(unittest.TestCase):
    """Tests for GradientOffloadHookContext context manager."""

    def setUp(self):
        set_hook_enable(False)
        self.model = nn.Linear(4, 4)
        self.gradient_dict = {}

    def tearDown(self):
        set_hook_enable(False)

    def test_enter_sets_hook_enable(self):
        """Test that entering the context enables the hook."""
        ctx = GradientOffloadHookContext(
            model=self.model, gradient_dict=self.gradient_dict, local_rank=0, loraga_init_iters=4
        )
        self.assertFalse(get_hook_enable())
        ctx.__enter__()
        self.assertTrue(get_hook_enable())
        ctx.__exit__(None, None, None)

    def test_exit_disables_hook_enable(self):
        """Test that exiting the context disables the hook."""
        ctx = GradientOffloadHookContext(
            model=self.model, gradient_dict=self.gradient_dict, local_rank=0, loraga_init_iters=4
        )
        ctx.__enter__()
        self.assertTrue(get_hook_enable())
        ctx.__exit__(None, None, None)
        self.assertFalse(get_hook_enable())

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        ctx = GradientOffloadHookContext(model=self.model, gradient_dict=self.gradient_dict)
        self.assertEqual(ctx.local_rank, 0)
        self.assertEqual(ctx.loraga_init_iters, 4)
        self.assertFalse(ctx.gradient_offload)

    def test_gradient_offload_cpu(self):
        """Test that gradient_offload moves gradients to CPU."""
        gradient_dict = {}
        ctx = GradientOffloadHookContext(
            model=self.model,
            gradient_dict=gradient_dict,
            local_rank=0,
            loraga_init_iters=1,
            gradient_offload=True,
        )
        with ctx:
            x = paddle.randn([2, 4])
            y = self.model(x)
            loss = y.sum()
            loss.backward()
        set_hook_enable(False)


class TestLoragaSvdModule(unittest.TestCase):
    """Tests for loraga_svd_module function."""

    def test_loraga_svd_module_stable_gamma(self):
        """Test loraga_svd_module with stable_gamma != -1."""
        # Create a mock LoRA module
        mock_module = MagicMock()
        mock_module.r = 2
        mock_module.scaling = 1.0

        # lora_A and lora_B need .dtype, .set_value(), and __matmul__ support
        mock_lora_A = MagicMock()
        mock_lora_A.dtype = paddle.float32
        mock_lora_A.__matmul__ = MagicMock(return_value=paddle.randn([16, 16]))
        mock_lora_B = MagicMock()
        mock_lora_B.dtype = paddle.float32
        mock_module.lora_A = mock_lora_A
        mock_module.lora_B = mock_lora_B
        mock_module.weight = MagicMock()
        mock_module.weight.data = paddle.randn([16, 16])

        grads = paddle.randn([16, 16])
        loraga_init_dict = {}

        loraga_svd_module(
            name="layers.0.lora",
            module=mock_module,
            grads=grads,
            stable_gamma=8,
            loraga_init_dict=loraga_init_dict,
        )

        # Check that lora_A and lora_B keys are in the init dict
        self.assertIn("0.lora.lora_A", loraga_init_dict)
        self.assertIn("0.lora.lora_B", loraga_init_dict)

    def test_loraga_svd_module_no_stable_gamma(self):
        """Test loraga_svd_module with stable_gamma == -1 (uses scaling)."""
        mock_module = MagicMock()
        mock_module.r = 2
        mock_module.scaling = 2.0

        # lora_A and lora_B need .dtype, .set_value(), and __matmul__ support
        mock_lora_A = MagicMock()
        mock_lora_A.dtype = paddle.float32
        mock_lora_A.__matmul__ = MagicMock(return_value=paddle.randn([16, 16]))
        mock_lora_B = MagicMock()
        mock_lora_B.dtype = paddle.float32
        mock_module.lora_A = mock_lora_A
        mock_module.lora_B = mock_lora_B
        mock_module.weight = MagicMock()
        mock_module.weight.data = paddle.randn([16, 16])

        grads = paddle.randn([16, 16])
        loraga_init_dict = {}

        loraga_svd_module(
            name="layers.0.lora",
            module=mock_module,
            grads=grads,
            stable_gamma=-1,
            loraga_init_dict=loraga_init_dict,
        )

        self.assertIn("0.lora.lora_A", loraga_init_dict)
        self.assertIn("0.lora.lora_B", loraga_init_dict)


if __name__ == "__main__":
    unittest.main()
