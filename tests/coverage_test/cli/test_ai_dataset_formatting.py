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
import types
import unittest
from unittest.mock import MagicMock

from datasets import Dataset, Value

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod

from paddleformers.cli.train.sft.dataset_formatting import (
    FORMAT_MAPPING,
    conversations_formatting_function,
    get_formatting_func_from_dataset,
    instructions_formatting_function,
    paddleformers_instructions_formatting_function,
)


class TestFormatMapping(unittest.TestCase):
    """Tests for FORMAT_MAPPING constant"""

    def test_chatml_format(self):
        self.assertIn("chatml", FORMAT_MAPPING)
        chatml = FORMAT_MAPPING["chatml"]
        self.assertIsInstance(chatml, list)
        self.assertEqual(len(chatml), 1)
        self.assertIn("content", chatml[0])
        self.assertIn("role", chatml[0])

    def test_instruction_format(self):
        self.assertIn("instruction", FORMAT_MAPPING)
        instruction = FORMAT_MAPPING["instruction"]
        self.assertIn("completion", instruction)
        self.assertIn("prompt", instruction)

    def test_paddleformers_format(self):
        self.assertIn("paddleformers", FORMAT_MAPPING)
        pf = FORMAT_MAPPING["paddleformers"]
        self.assertIn("src", pf)
        self.assertIn("tgt", pf)

    def test_three_formats(self):
        self.assertEqual(len(FORMAT_MAPPING), 3)


class TestConversationsFormattingFunction(unittest.TestCase):
    """Tests for conversations_formatting_function"""

    def test_returns_callable(self):
        mock_tokenizer = MagicMock()
        result = conversations_formatting_function(mock_tokenizer, "messages")
        self.assertTrue(callable(result))

    def test_format_batch(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_text"

        format_fn = conversations_formatting_function(mock_tokenizer, "messages")

        examples = {
            "messages": [
                [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
                [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}],
            ]
        }
        result = format_fn(examples)
        self.assertEqual(len(result), 2)

    def test_format_single(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_text"

        format_fn = conversations_formatting_function(mock_tokenizer, "messages")

        examples = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}
        result = format_fn(examples)
        self.assertEqual(result, "formatted_text")


class TestInstructionsFormattingFunction(unittest.TestCase):
    """Tests for instructions_formatting_function"""

    def test_returns_callable(self):
        mock_tokenizer = MagicMock()
        result = instructions_formatting_function(mock_tokenizer)
        self.assertTrue(callable(result))

    def test_format_batch(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_text"

        format_fn = instructions_formatting_function(mock_tokenizer)

        examples = {
            "prompt": ["What is AI?", "What is ML?"],
            "completion": ["AI is...", "ML is..."],
        }
        result = format_fn(examples)
        self.assertEqual(len(result), 2)
        # Verify apply_chat_template was called with proper structure
        self.assertEqual(mock_tokenizer.apply_chat_template.call_count, 2)

    def test_format_single(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_text"

        format_fn = instructions_formatting_function(mock_tokenizer)

        examples = {
            "prompt": "What is AI?",
            "completion": "AI is...",
        }
        result = format_fn(examples)
        self.assertEqual(result, "formatted_text")


class TestPaddleFormersFormattingFunction(unittest.TestCase):
    """Tests for paddleformers_instructions_formatting_function"""

    def test_returns_callable(self):
        mock_tokenizer = MagicMock()
        result = paddleformers_instructions_formatting_function(mock_tokenizer)
        self.assertTrue(callable(result))

    def test_format_batch(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_text"

        format_fn = paddleformers_instructions_formatting_function(mock_tokenizer)

        examples = {
            "src": ["source1", "source2"],
            "tgt": ["target1", "target2"],
        }
        result = format_fn(examples)
        self.assertEqual(len(result), 2)

    def test_format_single(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_text"

        format_fn = paddleformers_instructions_formatting_function(mock_tokenizer)

        examples = {
            "src": "source_text",
            "tgt": "target_text",
        }
        result = format_fn(examples)
        self.assertEqual(result, "formatted_text")


class TestGetFormattingFuncFromDataset(unittest.TestCase):
    """Tests for get_formatting_func_from_dataset"""

    def test_non_dataset_returns_none(self):
        mock_tokenizer = MagicMock()
        result = get_formatting_func_from_dataset("not_a_dataset", mock_tokenizer)
        self.assertIsNone(result)

    def test_unsupported_dataset_returns_none(self):
        mock_tokenizer = MagicMock()
        # Create a dataset with unsupported features
        ds = MagicMock(spec=Dataset)
        ds.features = {"unsupported_col": Value(dtype="string", id=None)}
        result = get_formatting_func_from_dataset(ds, mock_tokenizer)
        self.assertIsNone(result)

    def test_instruction_format_detected(self):
        mock_tokenizer = MagicMock()
        ds = MagicMock(spec=Dataset)
        ds.features = FORMAT_MAPPING["instruction"]
        result = get_formatting_func_from_dataset(ds, mock_tokenizer)
        self.assertIsNotNone(result)

    def test_paddleformers_format_detected(self):
        mock_tokenizer = MagicMock()
        ds = MagicMock(spec=Dataset)
        ds.features = FORMAT_MAPPING["paddleformers"]
        result = get_formatting_func_from_dataset(ds, mock_tokenizer)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
