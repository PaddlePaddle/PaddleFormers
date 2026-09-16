# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/utils/ckpt_converter.py"""

import unittest

from paddleformers.trainer.utils.ckpt_converter import (
    MODEL_STATE_FILE_MIN_SIZE,
    MODEL_WEIGHT_SUFFIX,
    OPTIMIZER_STATE_NAME_SUFFIX,
    OPTIMIZER_WEIGHT_SUFFIX,
    CheckpointConverter,
)


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_model_weight_suffix(self):
        """Test MODEL_WEIGHT_SUFFIX constant."""
        self.assertEqual(MODEL_WEIGHT_SUFFIX, ".pdparams")

    def test_optimizer_weight_suffix(self):
        """Test OPTIMIZER_WEIGHT_SUFFIX constant."""
        self.assertEqual(OPTIMIZER_WEIGHT_SUFFIX, ".pdopt")

    def test_model_state_file_min_size(self):
        """Test MODEL_STATE_FILE_MIN_SIZE constant."""
        self.assertIsInstance(MODEL_STATE_FILE_MIN_SIZE, int)

    def test_optimizer_state_name_suffix(self):
        """Test OPTIMIZER_STATE_NAME_SUFFIX list."""
        self.assertIsInstance(OPTIMIZER_STATE_NAME_SUFFIX, list)
        self.assertIn(".moment1", OPTIMIZER_STATE_NAME_SUFFIX)
        self.assertIn(".moment2", OPTIMIZER_STATE_NAME_SUFFIX)
        self.assertIn(".master_weight", OPTIMIZER_STATE_NAME_SUFFIX)


class TestCheckpointConverter(unittest.TestCase):
    """Tests for CheckpointConverter class."""

    def test_class_exists(self):
        """Test that CheckpointConverter class is importable."""
        self.assertTrue(callable(CheckpointConverter))


if __name__ == "__main__":
    unittest.main()
