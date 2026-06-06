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

import json
import unittest

from paddleformers.datasets.template.tool_utils import (
    DefaultToolUtils,
    ERNIEToolUtils,
    ERNIEVLToolUtils,
    FunctionCall,
    GLM4MOEToolUtils,
    GLM4ToolUtils,
    GLM_5ToolUtils,
    Llama3ToolUtils,
    QwenToolUtils,
    get_tool_utils,
)


class TestFunctionCall(unittest.TestCase):
    """Tests for FunctionCall named tuple."""

    def test_creation(self):
        fc = FunctionCall(name="test_func", arguments='{"key": "value"}')
        self.assertEqual(fc.name, "test_func")
        self.assertEqual(fc.arguments, '{"key": "value"}')


class TestDefaultToolUtils(unittest.TestCase):
    """Tests for DefaultToolUtils."""

    def test_tool_formatter(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        result = DefaultToolUtils.tool_formatter(tools)
        self.assertIn("get_weather", result)
        self.assertIn("Tool Description: Get weather info", result)
        self.assertIn("location (string, required)", result)
        self.assertIn("should be one of [celsius, fahrenheit]", result)
        self.assertIn("Tool Name: get_weather", result)

    def test_tool_formatter_with_items(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search items",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ids": {
                                "type": "array",
                                "description": "List of IDs",
                                "items": {"type": "integer"},
                            },
                        },
                    },
                },
            }
        ]
        result = DefaultToolUtils.tool_formatter(tools)
        self.assertIn("each item should be integer", result)

    def test_tool_formatter_without_type_key(self):
        """Test tool_formatter when tool does not have 'type' key."""
        tools = [
            {
                "name": "search",
                "description": "Search function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "Query"},
                    },
                },
            }
        ]
        result = DefaultToolUtils.tool_formatter(tools)
        self.assertIn("search", result)

    def test_function_formatter(self):
        functions = [
            FunctionCall(name="get_weather", arguments='{"location": "NYC"}'),
            FunctionCall(name="search", arguments='{"q": "hello"}'),
        ]
        result = DefaultToolUtils.function_formatter(functions)
        self.assertIn("Action: get_weather", result)
        self.assertIn('Action Input: {"location": "NYC"}', result)
        self.assertIn("Action: search", result)


class TestQwenToolUtils(unittest.TestCase):
    """Tests for QwenToolUtils."""

    def test_tool_formatter(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "test func",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = QwenToolUtils.tool_formatter(tools)
        self.assertIn("Tools", result)
        self.assertIn("test", result)

    def test_tool_formatter_no_type_key(self):
        tools = [{"name": "test", "description": "test func", "parameters": {"type": "object", "properties": {}}}]
        result = QwenToolUtils.tool_formatter(tools)
        # Should wrap the tool in a type:function dict
        self.assertIn("test", result)

    def test_function_formatter(self):
        functions = [FunctionCall(name="test_func", arguments='{"key": "val"}')]
        result = QwenToolUtils.function_formatter(functions)
        self.assertIn("test_func", result)


class TestGLM4ToolUtils(unittest.TestCase):
    """Tests for GLM4ToolUtils."""

    def test_tool_formatter(self):
        tools = [
            {
                "name": "test_func",
                "description": "A test function",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        result = GLM4ToolUtils.tool_formatter(tools)
        self.assertIn("test_func", result)

    def test_function_formatter_single(self):
        functions = [FunctionCall(name="test_func", arguments='{"key": "val"}')]
        result = GLM4ToolUtils.function_formatter(functions)
        self.assertIn("test_func", result)

    def test_function_formatter_parallel_raises(self):
        """GLM-4 does not support parallel functions."""
        functions = [
            FunctionCall(name="f1", arguments='{"a": 1}'),
            FunctionCall(name="f2", arguments='{"b": 2}'),
        ]
        with self.assertRaises(ValueError):
            GLM4ToolUtils.function_formatter(functions)


class TestGLM4MOEToolUtils(unittest.TestCase):
    """Tests for GLM4MOEToolUtils."""

    def test_tool_formatter(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = GLM4MOEToolUtils.tool_formatter(tools)
        self.assertIn("test_func", result)

    def test_function_formatter_with_non_string_values(self):
        functions = [FunctionCall(name="test_func", arguments='{"key": 1}')]
        result = GLM4MOEToolUtils.function_formatter(functions)
        self.assertIn("test_func", result)


class TestGLM5ToolUtils(unittest.TestCase):
    """Tests for GLM_5ToolUtils."""

    def test_tool_formatter(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = GLM_5ToolUtils.tool_formatter(tools)
        self.assertIn("test_func", result)

    def test_function_formatter_no_newline_separators(self):
        """GLM-5 has no \\n separators in function call format."""
        functions = [
            FunctionCall(name="f1", arguments='{"a": 1}'),
            FunctionCall(name="f2", arguments='{"b": 2}'),
        ]
        result = GLM_5ToolUtils.function_formatter(functions)
        self.assertIn("f1", result)
        self.assertIn("f2", result)


class TestLlama3ToolUtils(unittest.TestCase):
    """Tests for Llama3ToolUtils."""

    def test_tool_formatter(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = Llama3ToolUtils.tool_formatter(tools)
        self.assertIn("test_func", result)

    def test_function_formatter_single(self):
        functions = [FunctionCall(name="test_func", arguments='{"key": "val"}')]
        result = Llama3ToolUtils.function_formatter(functions)
        parsed = json.loads(result)
        self.assertEqual(parsed["name"], "test_func")

    def test_function_formatter_multiple(self):
        functions = [
            FunctionCall(name="f1", arguments='{"a": 1}'),
            FunctionCall(name="f2", arguments='{"b": 2}'),
        ]
        result = Llama3ToolUtils.function_formatter(functions)
        parsed = json.loads(result)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 2)


class TestERNIEToolUtils(unittest.TestCase):
    """Tests for ERNIEToolUtils."""

    def test_tool_formatter(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = ERNIEToolUtils.tool_formatter(tools)
        self.assertIn("tool_list", result)
        self.assertIn("test_func", result)

    def test_function_formatter(self):
        functions = [FunctionCall(name="test_func", arguments='{"key": "val"}')]
        result = ERNIEToolUtils.function_formatter(functions)
        self.assertIn("test_func", result)


class TestERNIEVLToolUtils(unittest.TestCase):
    """Tests for ERNIEVLToolUtils."""

    def test_tool_formatter(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = ERNIEVLToolUtils.tool_formatter(tools)
        self.assertIn("tool_list", result)

    def test_function_formatter(self):
        functions = [FunctionCall(name="test_func", arguments='{"key": "val"}')]
        result = ERNIEVLToolUtils.function_formatter(functions)
        self.assertIn("test_func", result)


class TestGetToolUtils(unittest.TestCase):
    """Tests for get_tool_utils function."""

    def test_get_existing_tool(self):
        for name in ["default", "ernie", "ernie_vl", "qwen", "qwen3_5", "glm4", "glm4_moe", "glm_moe_dsa", "llama3"]:
            result = get_tool_utils(name)
            self.assertIsNotNone(result)

    def test_get_nonexistent_tool(self):
        with self.assertRaises(ValueError):
            get_tool_utils("nonexistent_tool")


if __name__ == "__main__":
    unittest.main()
