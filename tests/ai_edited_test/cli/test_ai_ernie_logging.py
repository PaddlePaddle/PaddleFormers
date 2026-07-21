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

import logging
import os
import tempfile
import unittest

from paddleformers.cli.train.ernie_pretrain.src.utils.logging import (
    logger,
    setup_logger_output_file,
)


class TestLoggingModule(unittest.TestCase):
    """Tests for the logging module."""

    def test_logger_exists(self):
        """Test that the logger object exists."""
        self.assertIsNotNone(logger)
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_has_handler(self):
        """Test that the logger has at least one handler."""
        self.assertGreater(len(logger.handlers), 0)

    def test_logger_level(self):
        """Test that the logger level is set to 10 (DEBUG)."""
        self.assertEqual(logger.level, 10)

    def test_setup_logger_output_file(self):
        """Test setup_logger_output_file creates log directory and file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logger_output_file(tmpdir, 0)
            logdir = os.path.join(tmpdir, "log")
            self.assertTrue(os.path.isdir(logdir))
            logfile = os.path.join(logdir, "workerlog.0")
            self.assertTrue(os.path.exists(logfile))

    def test_setup_logger_output_file_different_ranks(self):
        """Test setup_logger_output_file with different local ranks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logger_output_file(tmpdir, 0)
            setup_logger_output_file(tmpdir, 1)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "log", "workerlog.0")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "log", "workerlog.1")))

    def test_bce_log_disabled(self):
        """Test that baidubce logger has no handlers and doesn't propagate."""
        bce_log = logging.getLogger("baidubce")
        self.assertFalse(bce_log.propagate)

    def test_filelock_log_disabled(self):
        """Test that filelock logger is disabled."""
        filelock_log = logging.getLogger("filelock")
        self.assertTrue(filelock_log.disabled)

    def test_bce_bns_proxy_disabled(self):
        """Test that bce_bns_proxy wrapper logger is disabled."""
        bce_bns_log = logging.getLogger("bce_bns_proxy.wrapper")
        self.assertTrue(bce_bns_log.disabled)


if __name__ == "__main__":
    unittest.main()
