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

from paddleformers.datasets.template.formatter import (
    EmptyFormatter,
    FunctionFormatter,
    StringFormatter,
    ThinkingFormatter,
    ToolFormatter,
)


class TestEmptyFormatter(unittest.TestCase):
    """Tests for EmptyFormatter."""

    def test_apply_returns_slots(self):
        formatter = EmptyFormatter(slots=["hello", "world"])
        result = formatter.apply()
        self.assertEqual(result, ["hello", "world"])

    def test_apply_empty_slots(self):
        formatter = EmptyFormatter(slots=[])
        result = formatter.apply()
        self.assertEqual(result, [])

    def test_no_placeholder_allowed(self):
        """EmptyFormatter should not contain placeholders."""
        with self.assertRaises(ValueError):
            EmptyFormatter(slots=["{{content}}"])

    def test_no_placeholder_allowed_mixed(self):
        """EmptyFormatter with any placeholder raises error."""
        with self.assertRaises(ValueError):
            EmptyFormatter(slots=["prefix_{{name}}"])

    def test_set_slots_allowed(self):
        """EmptyFormatter can have set slots without placeholder check."""
        formatter = EmptyFormatter(slots=[{"bos_token"}])
        result = formatter.apply()
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], set)


class TestStringFormatter(unittest.TestCase):
    """Tests for StringFormatter."""

    def test_apply_with_placeholder(self):
        formatter = StringFormatter(slots=["Hello {{content}} world"])
        result = formatter.apply(content="beautiful")
        self.assertEqual(result, ["Hello beautiful world"])

    def test_apply_multiple_placeholders(self):
        formatter = StringFormatter(slots=["{{greeting}} {{name}}"])
        result = formatter.apply(greeting="Hi", name="Alice")
        self.assertEqual(result, ["Hi Alice"])

    def test_requires_placeholder(self):
        """StringFormatter must contain a placeholder."""
        with self.assertRaises(ValueError):
            StringFormatter(slots=["no placeholder here"])

    def test_apply_with_set_slot(self):
        formatter = StringFormatter(slots=["{{content}}", {"eos_token"}])
        result = formatter.apply(content="test")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "test")
        self.assertIsInstance(result[1], set)

    def test_apply_invalid_slot_type_raises(self):
        """StringFormatter with int slot raises ValueError during __post_init__
        because int has no placeholder."""
        with self.assertRaises(ValueError):
            StringFormatter(slots=[123])

    def test_apply_non_string_value_with_placeholder(self):
        """StringFormatter raises RuntimeError when applying non-string value to placeholder."""
        formatter = StringFormatter(slots=["{{content}}"])
        with self.assertRaises(RuntimeError):
            formatter.apply(content=123)


class TestFunctionFormatter(unittest.TestCase):
    """Tests for FunctionFormatter."""

    def test_apply_single_function(self):
        formatter = FunctionFormatter(slots=["{{content}}"], tool_format="default")
        result = formatter.apply(content='[{"name": "test_func", "arguments": {"a": 1}}]')
        self.assertEqual(len(result), 1)
        self.assertIn("test_func", result[0])

    def test_apply_with_type_key(self):
        formatter = FunctionFormatter(slots=["{{content}}"], tool_format="default")
        result = formatter.apply(
            content='[{"type": "function", "function": {"name": "my_func", "arguments": {"x": 1}}}]'
        )
        self.assertIn("my_func", result[0])

    def test_apply_with_thought_words_no_match(self):
        """Test with thought_words that do not match content."""
        formatter = FunctionFormatter(slots=["{{content}}"], tool_format="default")
        content = '[{"name": "test_func", "arguments": {"a": 1}}]'
        result = formatter.apply(content=content, thought_words=("<think\n", "\n</think\n"))
        self.assertIsInstance(result, list)

    def test_apply_with_thought_words_match(self):
        """Test with thought_words that match content."""
        formatter = FunctionFormatter(slots=["{{content}}"], tool_format="default")
        content = '<think\nthinking\n</think\n[{"name": "test_func", "arguments": {"a": 1}}]'
        result = formatter.apply(content=content, thought_words=("<think\n", "\n</think\n"))
        self.assertIsInstance(result, list)

    def test_apply_invalid_json_raises(self):
        formatter = FunctionFormatter(slots=["{{content}}"], tool_format="default")
        with self.assertRaises(RuntimeError):
            formatter.apply(content="not valid json")


class TestToolFormatter(unittest.TestCase):
    """Tests for ToolFormatter."""

    def test_apply_with_tools(self):
        formatter = ToolFormatter(tool_format="default")
        tools_json = '[{"type": "function", "function": {"name": "test_func", "description": "A test", "parameters": {"type": "object", "properties": {}}}}]'
        result = formatter.apply(content=tools_json)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("test_func", result[0])

    def test_apply_empty_tools(self):
        formatter = ToolFormatter(tool_format="default")
        result = formatter.apply(content="[]")
        self.assertEqual(result, [""])

    def test_apply_invalid_json_raises(self):
        formatter = ToolFormatter(tool_format="default")
        with self.assertRaises(RuntimeError):
            formatter.apply(content="not json")


class TestThinkingFormatter(unittest.TestCase):
    """Tests for ThinkingFormatter."""

    def test_apply_with_thought(self):
        formatter = ThinkingFormatter(slots=["{{content}}"])
        content = "<think\nthinking\n</think\nactual response"
        result = formatter.apply(content=content, thought_words=("<think\n", "\n</think\n"))
        self.assertIsInstance(result, list)
        # Should have reasoning_content and content
        self.assertTrue(len(result) >= 1)

    def test_apply_without_thought(self):
        formatter = ThinkingFormatter(slots=["{{content}}"])
        result = formatter.apply(content="just content", thought_words=("<think\n", "\n</think\n"))
        self.assertIsInstance(result, list)

    def test_apply_with_empty_content(self):
        formatter = ThinkingFormatter(slots=["{{content}}"])
        result = formatter.apply(content="", thought_words=("<think\n", "\n</think\n"))
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
