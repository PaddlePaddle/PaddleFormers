# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import unittest
from unittest.mock import patch


class TestLauncher(unittest.TestCase):
    """Tests for paddleformers.cli.launcher module."""

    @patch("paddleformers.cli.launcher.run_export")
    @patch("paddleformers.cli.launcher.run_tuner")
    def test_launch_train(self, mock_run_tuner, mock_run_export):
        """Test launch() with 'train' command calls run_tuner."""
        with patch("sys.argv", ["launcher", "train"]):
            from paddleformers.cli.launcher import launch

            launch()
            mock_run_tuner.assert_called_once()
            mock_run_export.assert_not_called()

    @patch("paddleformers.cli.launcher.run_export")
    @patch("paddleformers.cli.launcher.run_tuner")
    def test_launch_export(self, mock_run_tuner, mock_run_export):
        """Test launch() with 'export' command calls run_export."""
        with patch("sys.argv", ["launcher", "export"]):
            from paddleformers.cli.launcher import launch

            launch()
            mock_run_export.assert_called_once()
            mock_run_tuner.assert_not_called()

    def test_launch_no_args_raises(self):
        """Test launch() with no arguments raises ValueError."""
        with patch("sys.argv", ["launcher"]):
            from paddleformers.cli.launcher import launch

            with self.assertRaises(ValueError) as ctx:
                launch()
            self.assertIn("larger than 1", str(ctx.exception))

    def test_launch_unknown_command_raises(self):
        """Test launch() with unknown command raises ValueError."""
        with patch("sys.argv", ["launcher", "unknown"]):
            from paddleformers.cli.launcher import launch

            with self.assertRaises(ValueError) as ctx:
                launch()
            self.assertIn("Unknown command", str(ctx.exception))
            self.assertIn("unknown", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
