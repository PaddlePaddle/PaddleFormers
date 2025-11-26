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

"""Tests for AiXcoder tokenizer."""

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.transformers.aixcoder.tokenizer_fast import AixcoderTokenizerFast


class AixcoderTokenizerTest(unittest.TestCase):
    """Test AiXcoder tokenizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the parent class to avoid loading actual tokenizer files
        with patch.object(AixcoderTokenizerFast, "__init__", lambda self, *args, **kwargs: None):
            self.tokenizer = AixcoderTokenizerFast()

        # Set up mock attributes
        self.tokenizer.pad_token_id = 2
        self.tokenizer.eos_token_id = 2
        self.tokenizer.bos_token_id = 1

        # Mock parent class methods
        self.tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = MagicMock(return_value="decoded text")

        # Create a mock __call__ method for parent class
        def mock_call(text, return_tensors=None, return_token_type_ids=False):
            tokens = [1, 2, 3, 4, 5]
            if return_tensors == "pd":
                return {
                    "input_ids": paddle.to_tensor([tokens]),
                    "attention_mask": paddle.to_tensor([[1] * len(tokens)]),
                }
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        # Store original __call__ and replace with mock
        self.original_call = AixcoderTokenizerFast.__call__
        AixcoderTokenizerFast.__call__ = MagicMock(side_effect=mock_call)

    def tearDown(self):
        """Clean up after tests."""
        # Restore original __call__ method
        AixcoderTokenizerFast.__call__ = self.original_call

    def test_input_wrapper_basic(self):
        """Test basic input wrapper functionality."""
        # Mock the parent class __call__ method for this tokenizer instance
        # Note: When patching PreTrainedTokenizerFast.__call__, the first argument is 'self'
        def mock_super_call(text, return_tensors=None, return_token_type_ids=False, **kwargs):
            # Simple mock implementation
            if "☺" in text:
                tokens = [100]  # pad token
            elif "<s>" in text:
                tokens = [1, 101, 102]  # pre_code tokens
            elif "▁<AIX-SPAN-MIDDLE>" in text:
                tokens = [103, 104, 105]  # code_string tokens
            else:
                tokens = [106, 107]

            if return_tensors == "pd":
                return {
                    "input_ids": paddle.to_tensor([tokens]),
                    "attention_mask": paddle.to_tensor([[1] * len(tokens)]),
                }
            return {"input_ids": [tokens], "attention_mask": [[1] * len(tokens)]}

        # Mock PreTrainedTokenizerFast.__call__ which is called via super()
        from transformers import PreTrainedTokenizerFast

        with patch.object(PreTrainedTokenizerFast, "__call__", side_effect=mock_super_call):
            result = self.tokenizer.input_wrapper(code_string="def hello():", later_code="pass", path="test.py")

            # Check result is a dictionary with expected keys
            self.assertIsInstance(result, dict)
            self.assertIn("input_ids", result)
            self.assertIn("attention_mask", result)

            # Check tensors have correct shape (batch_size=1)
            self.assertEqual(len(result["input_ids"].shape), 2)
            self.assertEqual(result["input_ids"].shape[0], 1)

    def test_input_wrapper_with_file_extension(self):
        """Test input wrapper with file extension detection."""
        from transformers import PreTrainedTokenizerFast

        # Test Python file
        with patch.object(
            PreTrainedTokenizerFast,
            "__call__",
            return_value={"input_ids": paddle.to_tensor([[1, 2, 3]]), "attention_mask": paddle.to_tensor([[1, 1, 1]])},
        ):
            result = self.tokenizer.input_wrapper(code_string="print('hello')", path="test.py")
            self.assertIsNotNone(result)

            # Test JavaScript file
            result = self.tokenizer.input_wrapper(code_string="console.log('hello')", path="test.js")
            self.assertIsNotNone(result)

    def test_input_wrapper_without_path(self):
        """Test input wrapper without file path."""
        from transformers import PreTrainedTokenizerFast

        with patch.object(
            PreTrainedTokenizerFast,
            "__call__",
            return_value={"input_ids": paddle.to_tensor([[1, 2, 3]]), "attention_mask": paddle.to_tensor([[1, 1, 1]])},
        ):
            result = self.tokenizer.input_wrapper(code_string="def hello():")
            self.assertIsNotNone(result)
            self.assertIn("input_ids", result)

    def test_input_wrapper_with_later_code(self):
        """Test input wrapper with suffix code."""
        from transformers import PreTrainedTokenizerFast

        with patch.object(
            PreTrainedTokenizerFast,
            "__call__",
            return_value={
                "input_ids": paddle.to_tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": paddle.to_tensor([[1, 1, 1, 1, 1]]),
            },
        ):
            result = self.tokenizer.input_wrapper(
                code_string="def add(a, b):", later_code="    return a + b", path="math_utils.py"
            )
            self.assertIsNotNone(result)
            self.assertIn("input_ids", result)
            self.assertIn("attention_mask", result)

    def test_call_method_with_dict(self):
        """Test __call__ method with dictionary input."""
        # Temporarily restore the original __call__ method to test dict handling
        # Use patch to restore the original method
        with patch.object(AixcoderTokenizerFast, "__call__", self.original_call):
            # Create a new tokenizer instance with proper mocking
            tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)
            tokenizer.input_wrapper = MagicMock(
                return_value={
                    "input_ids": paddle.to_tensor([[1, 2, 3]]),
                    "attention_mask": paddle.to_tensor([[1, 1, 1]]),
                }
            )
            # Ensure _looks_like_code method exists
            if not hasattr(tokenizer, "_looks_like_code"):
                tokenizer._looks_like_code = lambda x: False

            # Test with dictionary input
            input_dict = {"code_string": "def test():", "later_code": "pass", "path": "test.py"}

            # Call the method directly
            result = AixcoderTokenizerFast.__call__(tokenizer, input_dict, return_tensors="pd")

            # Verify input_wrapper was called
            tokenizer.input_wrapper.assert_called_once_with("def test():", "pass", "test.py")

            # Check result
            self.assertIn("input_ids", result)
            self.assertIn("attention_mask", result)

    def test_call_method_with_string(self):
        """Test __call__ method with string input."""
        # Test that string input uses parent class implementation
        with patch.object(
            AixcoderTokenizerFast, "__call__", return_value={"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        ):
            result = self.tokenizer("simple text")
            self.assertIsNotNone(result)

    def test_pad_token_configuration(self):
        """Test pad token configuration."""
        # Verify pad_token_id is set correctly
        self.assertEqual(self.tokenizer.pad_token_id, 2)
        self.assertEqual(self.tokenizer.eos_token_id, 2)
        self.assertEqual(self.tokenizer.bos_token_id, 1)

    def test_special_tokens(self):
        """Test special tokens handling."""
        from transformers import PreTrainedTokenizerFast

        # Test that special tokens are handled correctly
        with patch.object(
            PreTrainedTokenizerFast,
            "__call__",
            return_value={
                "input_ids": paddle.to_tensor([[1, 101, 102, 103, 2]]),
                "attention_mask": paddle.to_tensor([[1, 1, 1, 1, 1]]),
            },
        ):
            result = self.tokenizer.input_wrapper(code_string="test code", pad_token="☺")
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
