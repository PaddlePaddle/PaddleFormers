# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/unified_checkpoint/load_dynamic.py"""

import unittest

from paddleformers.trainer.unified_checkpoint.load_dynamic import (
    create_dispatch_table,
    create_optimizer_dispatch_table,
    distributed_send_recv,
    load_unified_checkpoint_dynamically,
    load_unified_optimizer_dynamically,
)


class TestCreateDispatchTable(unittest.TestCase):
    """Tests for create_dispatch_table function."""

    def test_function_exists(self):
        """Test that create_dispatch_table is callable."""
        self.assertTrue(callable(create_dispatch_table))


class TestCreateOptimizerDispatchTable(unittest.TestCase):
    """Tests for create_optimizer_dispatch_table function."""

    def test_function_exists(self):
        """Test that create_optimizer_dispatch_table is callable."""
        self.assertTrue(callable(create_optimizer_dispatch_table))


class TestDistributedSendRecv(unittest.TestCase):
    """Tests for distributed_send_recv function."""

    def test_function_exists(self):
        """Test that distributed_send_recv is callable."""
        self.assertTrue(callable(distributed_send_recv))


class TestLoadUnifiedCheckpointDynamically(unittest.TestCase):
    """Tests for load_unified_checkpoint_dynamically function."""

    def test_function_exists(self):
        """Test that load_unified_checkpoint_dynamically is callable."""
        self.assertTrue(callable(load_unified_checkpoint_dynamically))


class TestLoadUnifiedOptimizerDynamically(unittest.TestCase):
    """Tests for load_unified_optimizer_dynamically function."""

    def test_function_exists(self):
        """Test that load_unified_optimizer_dynamically is callable."""
        self.assertTrue(callable(load_unified_optimizer_dynamically))


if __name__ == "__main__":
    unittest.main()
