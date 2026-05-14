# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/unified_checkpoint/check_completion.py"""

import unittest
from unittest.mock import MagicMock, patch

from paddleformers.trainer.unified_checkpoint.check_completion import (
    check_unified_checkpoint,
    check_unified_optimizer,
)


class TestCheckUnifiedCheckpoint(unittest.TestCase):
    """Tests for check_unified_checkpoint function."""

    def test_missing_index_file_raises(self):
        """Test that missing index file raises exception."""
        args = MagicMock()
        args.dataset_rank = 0
        args.use_expert_parallel = False
        model = MagicMock()

        with patch("paddleformers.trainer.unified_checkpoint.check_completion.distributed_isfile", return_value=False):
            with self.assertRaises(Exception):
                check_unified_checkpoint(args, model, "/nonexistent/path")


class TestCheckUnifiedOptimizer(unittest.TestCase):
    """Tests for check_unified_optimizer function."""

    def test_missing_index_file_raises(self):
        """Test that missing optimizer index file raises exception."""
        args = MagicMock()
        model = MagicMock()
        optimizer = MagicMock()

        with patch("paddleformers.trainer.unified_checkpoint.check_completion.distributed_isfile", return_value=False):
            with self.assertRaises(Exception):
                check_unified_optimizer(args, model, optimizer, "/nonexistent/path")


if __name__ == "__main__":
    unittest.main()
