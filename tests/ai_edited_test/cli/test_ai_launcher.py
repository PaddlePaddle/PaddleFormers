# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest
from unittest.mock import MagicMock, patch


class TestLaunch(unittest.TestCase):
    """Tests for paddleformers.cli.launcher module."""

    def setUp(self):
        # Pre-populate the lazy module caches and sys.modules to prevent
        # the heavy import chain from being triggered.
        self._saved_modules = {}
        self._mock_modules = {}

        # Create mock modules for the heavy import chain
        modules_to_mock = [
            "paddleformers.cli.train.auto_parallel",
            "paddleformers.cli.train.auto_parallel.run_auto_parallel",
            "paddleformers.cli.train.dpo",
            "paddleformers.cli.train.sft",
            "paddleformers.cli.train.deepseek_v3_pretrain",
            "paddleformers.cli.train.ernie_pretrain",
        ]

        for mod_name in modules_to_mock:
            if mod_name in sys.modules:
                self._saved_modules[mod_name] = sys.modules[mod_name]
            mock_mod = MagicMock()
            sys.modules[mod_name] = mock_mod
            self._mock_modules[mod_name] = mock_mod

        # Also mock the tuner and launcher modules' dependencies
        if "paddleformers.cli.train.tuner" in sys.modules:
            del sys.modules["paddleformers.cli.train.tuner"]
        if "paddleformers.cli.launcher" in sys.modules:
            del sys.modules["paddleformers.cli.launcher"]

    def tearDown(self):
        # Restore original modules
        for mod_name in list(self._mock_modules.keys()):
            if mod_name in self._saved_modules:
                sys.modules[mod_name] = self._saved_modules[mod_name]
            elif mod_name in sys.modules:
                del sys.modules[mod_name]

        # Clean up cached module references
        for mod_name in ["paddleformers.cli.train.tuner", "paddleformers.cli.launcher"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

    def test_launch_train(self):
        """Test launch() with 'train' command calls run_tuner."""
        from paddleformers.cli.launcher import launch

        with patch("paddleformers.cli.launcher.run_tuner") as mock_run_tuner:
            with patch("paddleformers.cli.launcher.run_export") as mock_run_export:
                with patch.object(sys, "argv", ["launcher", "train"]):
                    launch()
                    mock_run_tuner.assert_called_once()
                    mock_run_export.assert_not_called()

    def test_launch_export(self):
        """Test launch() with 'export' command calls run_export."""
        from paddleformers.cli.launcher import launch

        with patch("paddleformers.cli.launcher.run_tuner") as mock_run_tuner:
            with patch("paddleformers.cli.launcher.run_export") as mock_run_export:
                with patch.object(sys, "argv", ["launcher", "export"]):
                    launch()
                    mock_run_export.assert_called_once()
                    mock_run_tuner.assert_not_called()

    def test_launch_no_args_raises(self):
        """Test launch() with no arguments raises ValueError."""
        from paddleformers.cli.launcher import launch

        with patch("paddleformers.cli.launcher.run_tuner"):
            with patch("paddleformers.cli.launcher.run_export"):
                with patch.object(sys, "argv", ["launcher"]):
                    with self.assertRaises(ValueError) as ctx:
                        launch()
                    self.assertIn("larger than 1", str(ctx.exception))

    def test_launch_unknown_command_raises(self):
        """Test launch() with unknown command raises ValueError."""
        from paddleformers.cli.launcher import launch

        with patch("paddleformers.cli.launcher.run_tuner"):
            with patch("paddleformers.cli.launcher.run_export"):
                with patch.object(sys, "argv", ["launcher", "unknown"]):
                    with self.assertRaises(ValueError) as ctx:
                        launch()
                    self.assertIn("Unknown command", str(ctx.exception))
                    self.assertIn("unknown", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
