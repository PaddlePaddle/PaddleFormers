# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/unified_checkpoint/load_local.py"""

import unittest
from unittest.mock import MagicMock, patch

from paddleformers.trainer.unified_checkpoint.load_local import (
    load_unified_checkpoint_locally,
    load_unified_optimizer_locally,
)


class TestLoadUnifiedCheckpointLocally(unittest.TestCase):
    """Tests for load_unified_checkpoint_locally function."""

    def test_missing_index_raises(self):
        """Test that missing index file raises ValueError."""
        args = MagicMock()
        args.use_expert_parallel = False
        args.data_parallel_rank = 0
        model = MagicMock()
        model.config = MagicMock()
        model.config.tensor_model_parallel_size = 1

        with patch(
            "paddleformers.trainer.unified_checkpoint.load_local.select_model_weight_index",
            side_effect=ValueError("Can't find index"),
        ):
            with self.assertRaises(ValueError):
                load_unified_checkpoint_locally(args, model, "/nonexistent/path")


class TestLoadUnifiedOptimizerLocally(unittest.TestCase):
    """Tests for load_unified_optimizer_locally function."""

    def test_function_exists(self):
        """Test that load_unified_optimizer_locally function exists and is callable."""
        self.assertTrue(callable(load_unified_optimizer_locally))


if __name__ == "__main__":
    unittest.main()
