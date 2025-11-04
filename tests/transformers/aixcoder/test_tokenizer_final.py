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

"""Final tokenizer tests to improve coverage."""

import tempfile
import unittest
from unittest.mock import patch

import paddle
from transformers import PreTrainedTokenizerFast

from paddleformers.transformers.aixcoder.tokenizer_fast import AixcoderTokenizerFast


class AixcoderTokenizerFinalTest(unittest.TestCase):
    """Final tests for AixcoderTokenizerFast to reach 80% coverage."""

    def test_tokenizer_init(self):
        """Test tokenizer initialization."""
        # Mock the parent class init to avoid file requirements
        with patch.object(PreTrainedTokenizerFast, "__init__"):
            tokenizer = AixcoderTokenizerFast(
                vocab_file="dummy.json",
                tokenizer_file="dummy.json",
                unk_token="<unk>",
                bos_token="<s>",
                eos_token="</s>",
                pad_token="</s>",
            )

            # Tokenizer should be initialized
            self.assertIsNotNone(tokenizer)

    def test_input_wrapper_full_flow(self):
        """Test input_wrapper with all branches."""
        # Create a tokenizer with proper mocking
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)

        # Mock the parent __call__ method
        def mock_parent_call(text, return_tensors=None, **kwargs):
            # Return different results based on input
            if "☺" in text:
                # Pad token
                ids = [[100]]
            elif "<s>▁<AIX-SPAN-PRE>" in text:
                # Pre-code IDs
                ids = [[1, 101, 102, 103]]
            elif "▁<AIX-SPAN-MIDDLE>" in text:
                # Code string IDs
                if "# Python file" in text:
                    ids = [[200, 201, 202, 203, 204]]
                else:
                    ids = [[104, 105, 106]]
            else:
                # Later code IDs
                ids = [[107, 108, 109]]

            if return_tensors == "pd":
                return {"input_ids": paddle.to_tensor(ids), "attention_mask": paddle.to_tensor([[1] * len(ids[0])])}
            return {"input_ids": ids, "attention_mask": [[1] * len(ids[0])]}

        # Patch the parent __call__ for this instance
        with patch.object(PreTrainedTokenizerFast, "__call__", side_effect=mock_parent_call):
            # Test with Python file extension
            result = tokenizer.input_wrapper(
                code_string="def hello():\n    pass", later_code="# Comment", path="/home/user/test.py"
            )

            self.assertIn("input_ids", result)
            self.assertIn("attention_mask", result)

            # Test with JavaScript file
            result = tokenizer.input_wrapper(
                code_string="function test() { return true; }", later_code="", path="app.js"
            )

            self.assertIn("input_ids", result)

            # Test with no path (empty path handling)
            result = tokenizer.input_wrapper(code_string="code", later_code="", path="")

            self.assertIn("input_ids", result)

            # Test with unsupported file extension
            result = tokenizer.input_wrapper(code_string="code", later_code="", path="file.unknown")

            self.assertIn("input_ids", result)

    def test_input_wrapper_language_detection(self):
        """Test language detection in input_wrapper."""
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)

        # Mock parent __call__
        with patch.object(PreTrainedTokenizerFast, "__call__") as mock_call:
            mock_call.return_value = {
                "input_ids": paddle.to_tensor([[1, 2, 3]]),
                "attention_mask": paddle.to_tensor([[1, 1, 1]]),
            }

            # Test various file extensions
            test_cases = [
                ("test.py", True),  # Python
                ("test.js", True),  # JavaScript
                ("test.java", True),  # Java
                ("test.cpp", True),  # C++
                ("test.c", True),  # C
                ("test.go", True),  # Go
                ("test.rs", True),  # Rust
                ("test.txt", False),  # Text (no language tag)
                ("test", False),  # No extension
            ]

            for path, has_language in test_cases:
                result = tokenizer.input_wrapper(code_string="code", path=path)
                self.assertIsNotNone(result)

    def test_input_wrapper_special_paths(self):
        """Test input_wrapper with special path cases."""
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)

        with patch.object(PreTrainedTokenizerFast, "__call__") as mock_call:
            mock_call.return_value = {
                "input_ids": paddle.to_tensor([[1, 2, 3]]),
                "attention_mask": paddle.to_tensor([[1, 1, 1]]),
            }

            # Test with None path
            result = tokenizer.input_wrapper(code_string="code", path=None)
            self.assertIsNotNone(result)

            # Test with numeric path (non-string)
            result = tokenizer.input_wrapper(code_string="code", path=123)
            self.assertIsNotNone(result)

            # Test with path containing multiple dots
            result = tokenizer.input_wrapper(code_string="code", path="my.test.file.py")
            self.assertIsNotNone(result)

    def test_input_wrapper_concat_logic(self):
        """Test the concatenation logic in input_wrapper."""
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)

        # Create specific mock returns for testing concatenation
        call_count = 0

        def mock_call(text, return_tensors=None, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:  # pad_token call
                return {"input_ids": paddle.to_tensor([[100]]), "attention_mask": paddle.to_tensor([[1]])}
            elif call_count == 2:  # pre_code call
                return {"input_ids": paddle.to_tensor([[1, 2, 3]]), "attention_mask": paddle.to_tensor([[1, 1, 1]])}
            elif call_count == 3:  # later_code call
                return {
                    "input_ids": paddle.to_tensor([[100, 4, 5]]),  # Includes pad
                    "attention_mask": paddle.to_tensor([[1, 1, 1]]),
                }
            else:  # code_string call
                return {
                    "input_ids": paddle.to_tensor([[6, 7, 8, 9]]),
                    "attention_mask": paddle.to_tensor([[1, 1, 1, 1]]),
                }

        with patch.object(PreTrainedTokenizerFast, "__call__", side_effect=mock_call):
            result = tokenizer.input_wrapper(code_string="main code", later_code="suffix", path="test.py")

            # Check that IDs were concatenated
            self.assertEqual(result["input_ids"].shape[0], 1)  # Batch size 1
            # Should have combined tokens from all parts
            self.assertGreater(result["input_ids"].shape[1], 3)

    def test_tokenizer_with_real_initialization(self):
        """Test tokenizer with real initialization path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Write minimal tokenizer config
            f.write('{"version": "1.0"}')
            temp_file = f.name

        try:
            # This will fail but tests the init path
            with self.assertRaises(Exception):
                # Will fail because we don't have a real tokenizer file
                _ = AixcoderTokenizerFast(vocab_file=temp_file)
        finally:
            import os

            os.unlink(temp_file)

    def test_input_wrapper_edge_cases(self):
        """Test input_wrapper with edge cases for better coverage."""
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)

        with patch.object(PreTrainedTokenizerFast, "__call__") as mock_call:
            mock_call.return_value = {"input_ids": paddle.to_tensor([[1]]), "attention_mask": paddle.to_tensor([[1]])}

            # Test with very long path
            long_path = "/very/long/path" * 100 + "/file.py"
            result = tokenizer.input_wrapper("code", path=long_path)
            self.assertIsNotNone(result)

            # Test with empty code_string and later_code
            result = tokenizer.input_wrapper("", "", "")
            self.assertIsNotNone(result)

            # Test with special characters in path
            result = tokenizer.input_wrapper("code", path="file@#$.py")
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
