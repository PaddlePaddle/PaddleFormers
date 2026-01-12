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

# The file has been adapted from hiyouga LLaMA-Factory project
# Copyright (c) 2025 LLaMA-Factory
# Licensed under the Apache License - https://github.com/hiyouga/LLaMA-Factory/blob/main/LICENSE


import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple

from typing_extensions import override


class FunctionCall(NamedTuple):
    name: str
    arguments: str


DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)

QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)

ERNIE_TOOL_PROMPT = "\n\n<tool_list>\n[{tool_text}]\n</tool_list>"

ERNIE_VL_TOOL_PROMPT = "\n<tool_list>\n[{tool_text}]\n</tool_list>\n"


GLM4_TOOL_PROMPT = (
    "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱 AI 公司训练的语言模型 GLM-4 模型开发的，" "你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{tool_text}"
)

GLM4_MOE_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:"
    "\n<tool_call>{{function-name}}"
    "\n<arg_key>{{arg-key-1}}</arg_key>"
    "\n<arg_value>{{arg-value-1}}</arg_value>"
    "\n<arg_key>{{arg-key-2}}</arg_key>"
    "\n<arg_value>{{arg-value-2}}</arg_value>"
    "\n...\n</tool_call>\n"
)

LLAMA3_TOOL_PROMPT = (
    "Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
    "You have access to the following functions. To call a function, please respond with JSON for a function call. "
    """Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. """
    "Do not use variables.\n\n{tool_text}"
)


@dataclass
class ToolUtils(ABC):
    """Base class for tool utilities."""

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        r"""Generate the system message describing all the available tools."""
        ...

    @staticmethod
    @abstractmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        r"""Generate the assistant message including all the tool calls."""
        ...


class DefaultToolUtils(ToolUtils):
    r"""Default tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            tool = tool.get("function", "") if tool.get("type") == "function" else tool
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required, enum, items = "", "", ""
                if name in tool["parameters"].get("required", []):
                    required = ", required"

                if param.get("enum", None):
                    enum = ", should be one of [{}]".format(", ".join(param["enum"]))

                if param.get("items", None):
                    items = ", where each item should be {}".format(param["items"].get("type", ""))

                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name,
                    type=param.get("type", ""),
                    required=required,
                    desc=param.get("description", ""),
                    enum=enum,
                    items=items,
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        return "\n".join([f"Action: {name}\nAction Input: {arguments}" for name, arguments in functions])


class QwenToolUtils(ToolUtils):
    r"""Qwen 2.5 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return QWEN_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = [
            json.dumps({"name": name, "arguments": json.loads(arguments)}, ensure_ascii=False)
            for name, arguments in functions
        ]
        return "\n".join([f"<tool_call>\n{text}\n</tool_call>" for text in function_texts])


class GLM4ToolUtils(ToolUtils):
    r"""GLM-4 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool = tool.get("function", "") if tool.get("type") == "function" else tool
            tool_text += "\n\n## {name}\n\n{body}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(
                name=tool["name"], body=json.dumps(tool, indent=4, ensure_ascii=False)
            )

        return GLM4_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        if len(functions) > 1:
            raise ValueError("GLM-4 does not support parallel functions.")

        return f"{functions[0].name}\n{functions[0].arguments}"


class GLM4MOEToolUtils(QwenToolUtils):
    r"""GLM-4-MOE tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return GLM4_MOE_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_json = [
            {"func_name": name, "func_key_values": json.loads(arguments)} for name, arguments in functions
        ]
        function_texts = []
        for func in function_json:
            prompt = "\n<tool_call>" + func["func_name"]
            for key, value in func["func_key_values"].items():
                prompt += "\n<arg_key>" + key + "</arg_key>"
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                prompt += "\n<arg_value>" + value + "</arg_value>"
            function_texts.append(prompt)

        return "\n".join(function_texts)


class Llama3ToolUtils(ToolUtils):
    r"""Llama 3.x tool using template with `tools_in_user_message=False`.

    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
    """

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        date = datetime.now().strftime("%d %b %Y")
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += json.dumps(wrapped_tool, indent=4, ensure_ascii=False) + "\n\n"

        return LLAMA3_TOOL_PROMPT.format(date=date, tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_objects = [{"name": name, "parameters": json.loads(arguments)} for name, arguments in functions]
        return json.dumps(function_objects[0] if len(function_objects) == 1 else function_objects, ensure_ascii=False)


class ERNIEToolUtils(ToolUtils):
    r"""ERNIE 4.5 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text_list = []
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text_list.append(json.dumps(wrapped_tool, ensure_ascii=False))
        tool_text = ", ".join(tool_text_list)

        return ERNIE_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = [
            json.dumps({"name": name, "arguments": json.loads(arguments)}, ensure_ascii=False)
            for name, arguments in functions
        ]
        return "\n".join([f"<tool_call>\n{text}\n</tool_call>\n" for text in function_texts])


class ERNIEVLToolUtils(ToolUtils):
    r"""ERNIE VL 4.5 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text_list = []
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text_list.append(json.dumps(wrapped_tool, ensure_ascii=False))
        tool_text = ", ".join(tool_text_list)

        return ERNIE_VL_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = [
            json.dumps({"name": name, "arguments": json.loads(arguments)}, ensure_ascii=False)
            for name, arguments in functions
        ]
        return "\n".join([f"<tool_call{text}\n</tool_call>" for text in function_texts])


TOOLS = {
    "default": DefaultToolUtils(),
    "ernie": ERNIEToolUtils(),
    "ernie_vl": ERNIEVLToolUtils(),
    "qwen": QwenToolUtils(),
    "glm4": GLM4ToolUtils(),
    "glm4_moe": GLM4MOEToolUtils(),
    "llama3": Llama3ToolUtils(),
}


def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError(f"Tool utils `{name}` not found.")

    return tool_utils
