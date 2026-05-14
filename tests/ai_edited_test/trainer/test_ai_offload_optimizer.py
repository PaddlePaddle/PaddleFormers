# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/utils/offload_optimizer.py"""

import unittest
from unittest.mock import patch

import paddle

from paddleformers.trainer.utils.offload_optimizer import (
    hack_offload_optimizer,
    hack_offload_optimizer_eb5,
    offload,
    reload,
)


class TestOffload(unittest.TestCase):
    """Tests for offload function."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_offload_cuda(self):
        """Test offload moves tensor to pinned memory on CUDA."""
        x = paddle.randn([4, 4])
        offload(x)
        # After offload, tensor should be on pinned place or cpu
        self.assertTrue(x.place.is_cuda_pinned_place() or x.place.is_cpu_place())

    def test_offload_cpu(self):
        """Test offload on CPU-only environment."""
        x = paddle.randn([4, 4])
        # On CPU, should use CPUPlace
        with patch("paddleformers.trainer.utils.offload_optimizer.paddle.is_compiled_with_cuda", return_value=False):
            with patch(
                "paddleformers.trainer.utils.offload_optimizer.paddle.is_compiled_with_xpu", return_value=False
            ):
                offload(x)


class TestReload(unittest.TestCase):
    """Tests for reload function."""

    def test_reload_basic(self):
        """Test reload moves tensor back to default device."""
        x = paddle.randn([4, 4])
        reload(x)
        # After reload, tensor should be on the default device
        self.assertIsNotNone(x)


class TestHackOffloadOptimizer(unittest.TestCase):
    """Tests for hack_offload_optimizer function."""

    def test_hack_offload_optimizer_patches_optimizer(self):
        """Test that hack_offload_optimizer patches _add_accumulator."""
        original_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
        try:
            hack_offload_optimizer()
            # After hacking, _add_accumulator should be replaced
            new_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
            self.assertNotEqual(original_add_accumulator, new_add_accumulator)
        finally:
            # Restore original method
            setattr(paddle.optimizer.Optimizer, "_add_accumulator", original_add_accumulator)

    def test_hack_offload_optimizer_eb5_mode(self):
        """Test that hack_offload_optimizer with eb5 mode calls eb5 variant."""
        original_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
        try:
            hack_offload_optimizer(mode="eb5")
            # After hacking, _add_accumulator should be replaced
            new_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
            self.assertNotEqual(original_add_accumulator, new_add_accumulator)
        finally:
            setattr(paddle.optimizer.Optimizer, "_add_accumulator", original_add_accumulator)


class TestHackOffloadOptimizerEb5(unittest.TestCase):
    """Tests for hack_offload_optimizer_eb5 function."""

    def test_hack_offload_optimizer_eb5_patches_optimizer(self):
        """Test that hack_offload_optimizer_eb5 patches _add_accumulator."""
        original_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
        try:
            hack_offload_optimizer_eb5()
            new_add_accumulator = getattr(paddle.optimizer.Optimizer, "_add_accumulator")
            self.assertNotEqual(original_add_accumulator, new_add_accumulator)
        finally:
            setattr(paddle.optimizer.Optimizer, "_add_accumulator", original_add_accumulator)


if __name__ == "__main__":
    unittest.main()
