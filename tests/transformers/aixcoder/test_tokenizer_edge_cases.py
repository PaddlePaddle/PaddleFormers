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

"""Edge case tests for AiXcoder tokenizer."""

import unittest
from unittest.mock import MagicMock, patch

import paddle
from transformers import PreTrainedTokenizerFast

from paddleformers.transformers.aixcoder.tokenizer_fast import AixcoderTokenizerFast


class AixcoderTokenizerEdgeCaseTest(unittest.TestCase):
    """Test edge cases for AiXcoder tokenizer."""

    def test_looks_like_code_detection(self):
        """Test _looks_like_code heuristic method."""
        # Create a mock tokenizer instance
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)

        # Test cases that should be detected as code
        code_samples = [
            "def hello():\n    print('world')",
            "class MyClass:\n    pass",
            "import numpy as np",
            "function getData() { return data; }",
            "const x = 10; let y = 20;",
            "public void main(String[] args) {}",
            "if (x == 5) { return true; }",
            "array[0] = value;",
            "result = a -> b => c",
        ]

        for sample in code_samples:
            self.assertTrue(tokenizer._looks_like_code(sample), f"Failed to detect as code: {sample}")

        # Test cases that should NOT be detected as code
        non_code_samples = [
            "Hello world",
            "This is a normal sentence.",
            "The weather is nice today",
            "I like pizza",
            "2 + 2 = 4",  # Simple math, not enough indicators
        ]

        for sample in non_code_samples:
            self.assertFalse(tokenizer._looks_like_code(sample), f"Incorrectly detected as code: {sample}")

    def test_looks_like_code_edge_cases(self):
        """Test edge cases for code detection."""
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)

        # Test with minimal code indicators (exactly 2)
        minimal_code = "{ }"
        self.assertTrue(tokenizer._looks_like_code(minimal_code))

        # Test with only 1 indicator (should not be detected as code)
        single_indicator = "This has a { bracket"
        self.assertFalse(tokenizer._looks_like_code(single_indicator))

        # Test empty string
        self.assertFalse(tokenizer._looks_like_code(""))

        # Test with many indicators
        complex_code = "def func(x): return x == 5 if x != 0 else None"
        self.assertTrue(tokenizer._looks_like_code(complex_code))

    @patch.object(PreTrainedTokenizerFast, "__call__")
    def test_call_with_code_detection(self, mock_super_call):
        """Test __call__ method with code detection logic."""
        mock_super_call.return_value = {
            "input_ids": paddle.to_tensor([[1, 2, 3]]),
            "attention_mask": paddle.to_tensor([[1, 1, 1]]),
        }

        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)
        tokenizer.input_wrapper = MagicMock(
            return_value={"input_ids": paddle.to_tensor([[4, 5, 6]]), "attention_mask": paddle.to_tensor([[1, 1, 1]])}
        )

        # Test with code-like text (should use input_wrapper)
        _ = tokenizer("def test(): pass")
        tokenizer.input_wrapper.assert_called_once()

        # Reset mock
        tokenizer.input_wrapper.reset_mock()

        # Test with later_code parameter (should use input_wrapper)
        _ = tokenizer("some text", later_code="return x")
        tokenizer.input_wrapper.assert_called_once()

        # Reset mock
        tokenizer.input_wrapper.reset_mock()

        # Test with path parameter (should use input_wrapper)
        _ = tokenizer("text", path="main.py")
        tokenizer.input_wrapper.assert_called_once()

    @patch.object(PreTrainedTokenizerFast, "__call__")
    def test_call_with_non_code_text(self, mock_super_call):
        """Test __call__ with non-code text falls back to parent."""
        mock_super_call.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)
        tokenizer.input_wrapper = MagicMock()

        # Test with normal text (should NOT use input_wrapper)
        _ = tokenizer("This is normal text")
        tokenizer.input_wrapper.assert_not_called()
        mock_super_call.assert_called()

    @patch.object(PreTrainedTokenizerFast, "__call__")
    def test_call_with_text_pair(self, mock_super_call):
        """Test __call__ with text_pair falls back to parent."""
        mock_super_call.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)
        tokenizer.input_wrapper = MagicMock()

        # Test with text_pair (should NOT use input_wrapper)
        _ = tokenizer("text1", text_pair="text2")
        tokenizer.input_wrapper.assert_not_called()
        mock_super_call.assert_called()

    @patch.object(PreTrainedTokenizerFast, "__call__")
    def test_call_with_all_parameters(self, mock_super_call):
        """Test __call__ with all possible parameters."""
        mock_super_call.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)
        tokenizer.input_wrapper = MagicMock()

        # Test with many parameters (non-code text)
        _ = tokenizer(
            "Normal text",
            text_pair=None,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            stride=0,
            is_split_into_words=False,
            pad_to_multiple_of=8,
            padding_side="right",
            return_tensors="pd",
            return_token_type_ids=False,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
            return_length=False,
            verbose=True,
        )

        # Should use parent implementation
        tokenizer.input_wrapper.assert_not_called()
        mock_super_call.assert_called_once()

    def test_call_with_custom_kwargs(self):
        """Test __call__ with custom kwargs (later_code, path)."""
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)
        tokenizer.input_wrapper = MagicMock(
            return_value={"input_ids": paddle.to_tensor([[1, 2, 3]]), "attention_mask": paddle.to_tensor([[1, 1, 1]])}
        )

        # Test with custom kwargs
        _ = tokenizer("def test():", later_code="    return True", path="/home/user/test.py")

        # Check input_wrapper was called with correct arguments
        tokenizer.input_wrapper.assert_called_with("def test():", "    return True", "/home/user/test.py")

    def test_call_default_path(self):
        """Test that default path is test.py when not specified."""
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)
        tokenizer.input_wrapper = MagicMock(
            return_value={"input_ids": paddle.to_tensor([[1, 2, 3]]), "attention_mask": paddle.to_tensor([[1, 1, 1]])}
        )

        # Test without path (should default to test.py)
        _ = tokenizer("def main():", later_code="pass")

        # Check input_wrapper was called with default path
        tokenizer.input_wrapper.assert_called_with("def main():", "pass", "test.py")

    @patch.object(PreTrainedTokenizerFast, "__call__")
    def test_call_with_empty_strings(self, mock_super_call):
        """Test __call__ with empty strings."""
        mock_super_call.return_value = {"input_ids": [[0]], "attention_mask": [[1]]}

        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)
        tokenizer.input_wrapper = MagicMock()

        # Test with empty string
        _ = tokenizer("")

        # Empty string doesn't look like code
        tokenizer.input_wrapper.assert_not_called()
        mock_super_call.assert_called()

    @patch.object(PreTrainedTokenizerFast, "__call__")
    def test_call_with_none_text(self, mock_super_call):
        """Test __call__ with None text (should handle gracefully)."""
        mock_super_call.return_value = {"input_ids": [[0]], "attention_mask": [[1]]}

        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)
        tokenizer.input_wrapper = MagicMock()

        # Test with None (should fall back to parent which will handle it)
        _ = tokenizer(None)

        # Should use parent implementation to handle None
        mock_super_call.assert_called()

    def test_code_detection_with_mixed_content(self):
        """Test code detection with mixed code and text."""
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)

        # Mixed content with enough code indicators
        mixed_content = """
        Here is my function:
        def calculate(x, y):
            return x + y
        """
        self.assertTrue(tokenizer._looks_like_code(mixed_content))

        # Mixed content with not enough indicators
        mixed_minimal = "I wrote a function today"
        self.assertFalse(tokenizer._looks_like_code(mixed_minimal))

    def test_code_detection_language_variety(self):
        """Test code detection across different programming languages."""
        tokenizer = AixcoderTokenizerFast.__new__(AixcoderTokenizerFast)

        # Python
        self.assertTrue(tokenizer._looks_like_code("from collections import defaultdict"))

        # JavaScript
        self.assertTrue(tokenizer._looks_like_code("const arr = [1, 2, 3];"))

        # Java
        self.assertTrue(tokenizer._looks_like_code("public static void main(String[] args) {}"))

        # C++
        self.assertTrue(tokenizer._looks_like_code("int main() { return 0; }"))

        # Go
        self.assertTrue(tokenizer._looks_like_code("func main() { fmt.Println() }"))

        # Rust
        self.assertTrue(tokenizer._looks_like_code("fn main() -> Result<(), Error> {}"))


if __name__ == "__main__":
    unittest.main()
