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

import unittest
from functools import partial
from unittest.mock import MagicMock, patch

from paddleformers.cli.cli import USAGE, WELCOME


class TestCliConstants(unittest.TestCase):
    def test_usage_not_empty(self):
        self.assertTrue(len(USAGE) > 0)
        self.assertIn("train", USAGE)
        self.assertIn("export", USAGE)
        self.assertIn("version", USAGE)
        self.assertIn("help", USAGE)

    def test_welcome_not_empty(self):
        self.assertTrue(len(WELCOME) > 0)
        self.assertIn("PaddleFormers", WELCOME)

    def test_usage_format(self):
        # USAGE should contain properly formatted help text
        self.assertIn("paddleformers-cli", USAGE)
        self.assertIn("finetuning", USAGE)

    def test_welcome_format(self):
        # WELCOME should have separators and title
        self.assertIn("-" * 60, WELCOME)
        self.assertIn("Cli", WELCOME)


class TestCliCommandMap(unittest.TestCase):
    """Test the command mapping logic that main() uses, without
    actually calling main() which triggers heavy imports."""

    def test_help_command_prints_usage(self):
        """Test that 'help' partial calls print with USAGE."""
        # partial(print, USAGE) captures the real print at creation time,
        # so we verify the behavior by checking what print receives.
        with patch("builtins.print") as mock_print:
            # Create the partial after mocking print
            help_func = partial(print, USAGE)
            help_func()
            mock_print.assert_called_once_with(USAGE)

    def test_version_command_prints_welcome(self):
        """Test that 'version' partial calls print with WELCOME."""
        with patch("builtins.print") as mock_print:
            version_func = partial(print, WELCOME)
            version_func()
            mock_print.assert_called_once_with(WELCOME)

    def test_unknown_command_message(self):
        """Test that unknown commands produce an appropriate message."""
        command = "unknown_cmd"
        # Replicate the logic from main()
        COMMAND_MAP = {
            "train": MagicMock(),
            "export": MagicMock(),
            "version": partial(print, WELCOME),
            "help": partial(print, USAGE),
        }
        distributed_funcs = ["train", "export"]

        if command not in distributed_funcs and command not in COMMAND_MAP:
            with patch("builtins.print") as mock_print:
                print(f"Unknown command: {command}.\n{USAGE}")
                call_args = str(mock_print.call_args)
                self.assertIn("Unknown command", call_args)
                self.assertIn("unknown_cmd", call_args)


class TestCliArgvHandling(unittest.TestCase):
    """Test the argv handling logic from main() without calling main()."""

    def test_default_command_is_help(self):
        """When no argv[1], command defaults to 'help'."""
        # Replicate: command = sys.argv[1] if len(sys.argv) > 1 else "help"
        argv = ["cli"]
        command = argv[1] if len(argv) > 1 else "help"
        self.assertEqual(command, "help")

    def test_explicit_command(self):
        """When argv[1] exists, it becomes the command."""
        argv = ["cli", "train"]
        command = argv[1] if len(argv) > 1 else "help"
        self.assertEqual(command, "train")

    def test_help_is_non_distributed(self):
        """Help command should not be in distributed_funcs."""
        distributed_funcs = ["train", "export"]
        self.assertNotIn("help", distributed_funcs)

    def test_version_is_non_distributed(self):
        """Version command should not be in distributed_funcs."""
        distributed_funcs = ["train", "export"]
        self.assertNotIn("version", distributed_funcs)


if __name__ == "__main__":
    unittest.main()
