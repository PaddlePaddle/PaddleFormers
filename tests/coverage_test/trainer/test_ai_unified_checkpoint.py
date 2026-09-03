# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/unified_checkpoint/unified_checkpoint.py"""

import unittest
from unittest.mock import MagicMock

from paddleformers.trainer.unified_checkpoint.unified_checkpoint import (
    UnifiedCheckpointHandler,
)


class TestUnifiedCheckpointHandler(unittest.TestCase):
    """Tests for UnifiedCheckpointHandler class."""

    def _create_mock_args(self, async_save=False):
        """Create mock args for UnifiedCheckpointHandler."""
        args = MagicMock()
        args.unified_checkpoint_config = ["async_save"] if async_save else []
        return args

    def test_init(self):
        """Test UnifiedCheckpointHandler initialization."""
        args = self._create_mock_args()
        handler = UnifiedCheckpointHandler(args)
        self.assertEqual(handler.args, args)
        self.assertIsNotNone(handler.async_handler)

    def test_save_unified_checkpoint_callable(self):
        """Test that save_unified_checkpoint method exists."""
        args = self._create_mock_args()
        handler = UnifiedCheckpointHandler(args)
        self.assertTrue(callable(handler.save_unified_checkpoint))

    def test_load_unified_checkpoint_callable(self):
        """Test that load_unified_checkpoint method exists."""
        args = self._create_mock_args()
        handler = UnifiedCheckpointHandler(args)
        self.assertTrue(callable(handler.load_unified_checkpoint))


if __name__ == "__main__":
    unittest.main()
