# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import unittest

from paddleformers.cli.hparams.export_args import ExportArguments


class TestExportArguments(unittest.TestCase):
    """Tests for ExportArguments dataclass."""

    def test_default_copy_tokenizer_true(self):
        """Test default copy_tokenizer is True."""
        args = ExportArguments()
        self.assertTrue(args.copy_tokenizer)

    def test_copy_tokenizer_false(self):
        """Test setting copy_tokenizer to False."""
        args = ExportArguments(copy_tokenizer=False)
        self.assertFalse(args.copy_tokenizer)

    def test_copy_tokenizer_explicit_true(self):
        """Test explicitly setting copy_tokenizer to True."""
        args = ExportArguments(copy_tokenizer=True)
        self.assertTrue(args.copy_tokenizer)


if __name__ == "__main__":
    unittest.main()
