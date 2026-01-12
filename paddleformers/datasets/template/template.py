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

import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, unique
from typing import TYPE_CHECKING, Optional

from typing_extensions import override

from paddleformers.utils.log import logger

from .formatter import (
    EmptyFormatter,
    FunctionFormatter,
    StringFormatter,
    ThinkingFormatter,
    ToolFormatter,
)
from .grounding_plugin import get_grounding_plugin
from .mm_plugin import get_mm_plugin

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .formatter import SLOTS, Formatter
    from .grounding_plugin import BaseGroundingPlugin
    from .mm_plugin import BasePlugin


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    chat_sep: str
    suffix: list[str]
    stop_words: list[str]
    thought_words: tuple[str, str]
    efficient_eos: bool
    auto_add_bos: bool
    enable_thinking: Optional[bool]
    mm_plugin: "BasePlugin"
    grounding_plugin: "BaseGroundingPlugin"

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        r"""Return a single pair of token ids representing prompt and response respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        response_ids = encoded_messages[-1]
        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def add_thought(self, content: str = "") -> str:
        r"""Add empty thought to assistant message."""
        return f"{self.thought_words[0]}{self.thought_words[1]}" + content

    def remove_thought(self, content: str) -> str:
        r"""Remove thought from assistant message."""
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Get the token ids of thought words."""
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> list[int]:
        r"""Convert elements to token ids."""
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        r"""Encode formatted inputs to pairs of token ids.

        Turn 0: prefix + system + query        resp
        Turn t: query                          resp.
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
                if "tool_calls" in message:
                    elements += self.format_function.apply(
                        content=message["tool_calls"], thought_words=self.thought_words
                    )
                # Add chat sep to all except the last round
                if i < len(messages) - 1:
                    elements += [self.chat_sep]
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(content=message["content"], thought_words=self.thought_words)
                # Add chat sep to all except the last round
                if i < len(messages) - 1:
                    elements += [self.chat_sep]
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
        r"""Add or replace eos token to the tokenizer."""
        if tokenizer.eos_token == eos_token:
            return

        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            logger.warning(f"Add eos token: {tokenizer.eos_token}.")
        else:
            logger.warning(f"Replace eos token: {tokenizer.eos_token}.")

        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    def fix_special_tokens(self, tokenizer: "PreTrainedTokenizer") -> None:
        r"""Add eos token and pad token to the tokenizer."""
        stop_words = self.stop_words

        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Add pad token: {tokenizer.pad_token}")

        if stop_words:
            num_added_tokens = tokenizer.add_special_tokens(
                dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
            )
            logger.info("Add {} to stop words.".format(",".join(stop_words)))
            if num_added_tokens > 0:
                logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


@dataclass
class ReasoningTemplate(Template):
    r"""A template that add thought to assistant message."""

    @override
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        for i in range(1, len(messages) - 2, 2):
            messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if self.enable_thinking is False:  # remove all cot
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])

        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system, tools)
        if (
            self.thought_words[0].strip() not in messages[-1]["content"]
            and self.thought_words[1].strip() not in messages[-1]["content"]
        ):  # add empty cot
            if not self.enable_thinking:  # do not compute loss
                prompt_ids += self.get_thought_word_ids(tokenizer)
            else:  # do compute loss
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids

        return prompt_ids, response_ids

    @override
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:  # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded_messages = self._encode(tokenizer, messages, system, tools)
        for i in range(0, len(messages), 2):
            if (
                self.thought_words[0].strip() not in messages[i + 1]["content"]
                and self.thought_words[1].strip() not in messages[i + 1]["content"]
            ):  # add empty cot
                if not self.enable_thinking:  # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:  # do compute loss
                    encoded_messages[i + 1] = self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]

        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]


@dataclass
class Llama2Template(Template):
    r"""A template that fuse the system message to first user message."""

    @override
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: str,
        tools: str,
    ) -> list[list[int]]:
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
                if "tool_calls" in message:
                    elements += self.format_function.apply(content=message["tool_calls"])
                # Add chat sep to all except the last round
                if i < len(messages) - 1:
                    elements += [self.chat_sep]
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(content=message["content"])
                # Add chat sep to all except the last round
                if i < len(messages) - 1:
                    elements += [self.chat_sep]
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


@dataclass
class ErnieThinkingTemplate(ReasoningTemplate):
    r"""A template that fuse the system message to first user message."""

    @override
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: str,
        tools: str,
    ) -> list[list[int]]:
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if not system:
                    elements += ["<|im_start|>system\n<global_setting>\nthink_mode=True\n</global_setting>"]
                else:
                    elements += self.format_system.apply(content=system)
                if tools:
                    elements += self.format_tools.apply(content=tools)[0] if tools else ""
                elements += ["<|im_end|>\n\n"]

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"], thought_words=self.thought_words)
                if "tool_calls" in message:
                    elements += self.format_function.apply(
                        content=message["tool_calls"], thought_words=self.thought_words
                    )
                # Add chat sep to all except the last round
                if i < len(messages) - 1:
                    elements += [self.chat_sep]
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(content=message["content"], thought_words=self.thought_words)
                # Add chat sep to all except the last round
                if i < len(messages) - 1:
                    elements += [self.chat_sep]
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


TEMPLATES: dict[str, "Template"] = {}


def register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    suffix: Optional[list[str]] = None,
    stop_words: Optional[list[str]] = None,
    thought_words: Optional[tuple[str, str]] = None,
    efficient_eos: bool = True,
    chat_sep: str = "",
    auto_add_bos: bool = False,
    enable_thinking: Optional[bool] = True,
    mm_plugin: "BasePlugin" = get_mm_plugin(name="base"),
    grounding_plugin: "BaseGroundingPlugin" = get_grounding_plugin(name="base"),
    template_class: type["Template"] = Template,
) -> None:
    r"""Register a chat template.

    To add the following chat template:
    ```
    <s><user>user prompt here
    <model>model response here</s>
    <user>user prompt here
    <model>model response here</s>
    ```

    The corresponding code should be:
    ```
    register_template(
        name="custom",
        format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
        format_assistant=StringFormatter(slots=["{{content}}</s>\n"]),
        format_prefix=EmptyFormatter("<s>"),
    )
    ```
    """
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")

    default_slots = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=default_slots)
    if format_assistant is not None:
        default_function_formatter = FunctionFormatter(slots=format_assistant.slots, tool_format="default")
    else:
        default_function_formatter = FunctionFormatter(slots=default_slots, tool_format="default")

    default_tool_formatter = ToolFormatter(tool_format="default")
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        suffix=suffix or [],
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        efficient_eos=efficient_eos,
        chat_sep=chat_sep,
        auto_add_bos=auto_add_bos,
        enable_thinking=enable_thinking,
        mm_plugin=mm_plugin,
        grounding_plugin=grounding_plugin,
    )


def parse_template(tokenizer: "PreTrainedTokenizer") -> "Template":
    r"""Extract a chat template from the tokenizer."""

    def find_diff(short_str: str, long_str: str) -> str:
        i, j = 0, 0
        diff = ""
        while i < len(short_str) and j < len(long_str):
            if short_str[i] == long_str[j]:
                i += 1
                j += 1
            else:
                diff += long_str[j]
                j += 1

        return diff

    prefix = tokenizer.decode(tokenizer.encode(""))

    messages = [{"role": "system", "content": "{{content}}"}]
    system_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[len(prefix) :]

    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "{{content}}"}]
    user_slot_empty_system = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot_empty_system = user_slot_empty_system[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}]
    user_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot = user_slot[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}, {"role": "assistant", "content": "{{content}}"}]
    assistant_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assistant_slot = assistant_slot[len(prefix) + len(user_slot) :]
    template_class = ReasoningTemplate if "<think>" in assistant_slot else Template
    assistant_slot = assistant_slot.replace("<think>", "").replace("</think>", "").lstrip("\n")  # remove thought tags

    if len(user_slot) > len(user_slot_empty_system):
        default_system = find_diff(user_slot_empty_system, user_slot)
        sole_system = system_slot.replace("{{content}}", default_system, 1)
        user_slot = user_slot[len(sole_system) :]
    else:  # if defaut_system is empty, user_slot_empty_system will be longer than user_slot
        default_system = ""

    return template_class(
        format_user=StringFormatter(slots=[user_slot]),
        format_assistant=StringFormatter(slots=[assistant_slot]),
        format_system=StringFormatter(slots=[system_slot]),
        format_function=FunctionFormatter(slots=[assistant_slot], tool_format="default"),
        format_observation=StringFormatter(slots=[user_slot]),
        format_tools=ToolFormatter(tool_format="default"),
        format_prefix=EmptyFormatter(slots=[prefix]) if prefix else EmptyFormatter(),
        default_system=default_system,
        suffix=[],
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        efficient_eos=True,
        chat_sep="",
        auto_add_bos=False,
        enable_thinking=True,
        mm_plugin=get_mm_plugin(name="base"),
        grounding_plugin=get_grounding_plugin(name="base"),
    )


def get_template_and_fix_tokenizer(dataset_config) -> "Template":
    r"""Get chat template and fixes the tokenizer."""
    tokenizer = dataset_config["tokenizer"]
    if dataset_config["template"] is None:
        if isinstance(tokenizer.chat_template, str):
            logger.warning("`template` was not specified, try parsing the chat template from the tokenizer.")
            template = parse_template(tokenizer)
        else:
            logger.warning("`template` was not specified, use `empty` template.")
            template = TEMPLATES["empty"]  # placeholder
    else:
        if dataset_config["template"] not in TEMPLATES:
            raise ValueError(f"Template {dataset_config['template']} does not exist.")

        template = TEMPLATES[dataset_config["template"]]

    if dataset_config["tool_format"] is not None:
        default_slots = ["{{content}}"] if template.efficient_eos else ["{{content}}", {"eos_token"}]
        template.format_function = FunctionFormatter(slots=default_slots, tool_format=dataset_config["tool_format"])
        template.format_tools = ToolFormatter(tool_format=dataset_config["tool_format"])

    if dataset_config["default_system"] is not None:
        template.default_system = dataset_config["default_system"]

    template.fix_special_tokens(tokenizer)

    if not template.suffix:
        template.suffix = [tokenizer.eos_token]
        logger.warning("suffix is not specified, using eos token as suffix.")
    return template


register_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}", {"eos_token"}, "\nAssistant:"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n"]),
    format_system=StringFormatter(slots=["System: {{content}}", {"eos_token"}, "\n"]),
)

register_template(
    name="empty",
    format_assistant=StringFormatter(slots=["{{content}}"]),
)

# copied from chatml template
register_template(
    name="ernie",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n\n<|im_start|>assistant\n"]),
    format_assistant=ThinkingFormatter(slots=["<response>\n{{content}}\n</response>\n"]),
    format_system=StringFormatter(
        slots=[
            "<|im_start|>system\n<system_setting>\n{{content}}\n</system_setting>\n\n<global_setting>\nthink_mode=True\n</global_setting>"
        ]
    ),
    format_function=FunctionFormatter(slots=["\n{{content}}"], tool_format="ernie"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="ernie"),
    chat_sep="<|im_end|>\n\n",
    suffix=["<|im_end|>"],
    thought_words=("<think>", "</think>"),
    template_class=ErnieThinkingTemplate,
)

register_template(
    name="ernie_nothink",
    format_user=StringFormatter(slots=["User: {{content}}\nAssistant: "]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"]),
    chat_sep="<|end_of_sentence|>",
    suffix=["</s>"],
)

register_template(
    name="ernie_vl",
    format_user=StringFormatter(slots=["User: {{content}}\nAssistant: "]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_function=FunctionFormatter(slots=["{{content}}\n"], tool_format="ernie_vl"),
    format_observation=StringFormatter(slots=["User: <tool_output>\n{{content}}\n</tool_output>\n\nAssistant: "]),
    format_tools=ToolFormatter(tool_format="ernie_vl"),
    format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"]),
    chat_sep="<|end_of_sentence|>",
    suffix=["</s>"],
    mm_plugin=get_mm_plugin(name="ernie_vl", image_token="<|IMAGE_PLACEHOLDER|>", video_token="<|IMAGE_PLACEHOLDER|>"),
    template_class=ReasoningTemplate,
    thought_words=("\n<think>\n", "\n</think>\n\n"),
)

register_template(
    name="ernie_vl_nothink",
    format_user=StringFormatter(slots=["User: {{content}}\nAssistant: "]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"]),
    chat_sep="<|end_of_sentence|>",
    suffix=["</s>"],
    mm_plugin=get_mm_plugin(name="ernie_vl", image_token="<|IMAGE_PLACEHOLDER|>", video_token="<|IMAGE_PLACEHOLDER|>"),
    template_class=ReasoningTemplate,
    thought_words=("<think>\n", "\n</think>\n\n"),
)

register_template(
    name="paddleocr_vl",
    format_user=StringFormatter(slots=["User: {{content}}\nAssistant: "]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"]),
    chat_sep="<|end_of_sentence|>",
    mm_plugin=get_mm_plugin(name="paddleocr_vl", image_token="<|IMAGE_PLACEHOLDER|>"),
)

# copied from chatml template
register_template(
    name="qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    chat_sep="<|im_end|>\n",
    suffix=["<|im_end|>\n"],
)


# copied from qwen template
register_template(
    name="qwen3",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    suffix=["<|im_end|>\n"],
    chat_sep="<|im_end|>\n",
    template_class=ReasoningTemplate,
)


# copied from qwen template
register_template(
    name="qwen3_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    suffix=["<|im_end|>\n"],
    chat_sep="<|im_end|>\n",
)


# copied from qwen template
register_template(
    name="qwen2_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    suffix=["<|im_end|>\n"],
    chat_sep="<|im_end|>\n",
    mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


# copied from qwen template
register_template(
    name="qwen3_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    chat_sep="<|im_end|>\n",
    suffix=["<|im_end|>\n"],
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    template_class=ReasoningTemplate,
)


# copied from qwen template
register_template(
    name="qwen3_vl_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    chat_sep="<|im_end|>\n",
    suffix=["<|im_end|>\n"],
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


# copied from glm4 template
register_template(
    name="glm4_moe",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["[gMASK]<sop><|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4_moe"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4_moe"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    suffix=["<|user|>"],
    thought_words=("<think>", "</think>"),
    template_class=ReasoningTemplate,
)


# copied from glm4 template
register_template(
    name="glm4v",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    suffix=["<|user|>"],
    mm_plugin=get_mm_plugin(name="glm4v", image_token="<|image|>", video_token="<|video|>"),
    template_class=ReasoningTemplate,
)


# copied from glm4 template
register_template(
    name="glm4v_moe",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4_moe"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4_moe"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    suffix=["<|user|>"],
    mm_plugin=get_mm_plugin(name="glm4v", image_token="<|image|>", video_token="<|video|>"),
    template_class=ReasoningTemplate,
)

register_template(
    name="deepseek3",
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_user=StringFormatter(slots=["<｜User｜>{{content}}\n\n<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    chat_sep="<｜end▁of▁sentence｜>",
)


def _get_gpt_oss_prefix():
    today = datetime.now().strftime("%Y-%m-%d")
    return (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        f"Knowledge cutoff: 2024-06\nCurrent date: {today}\n\nReasoning: medium\n\n"
        "# Valid channels: analysis, commentary, final. "
        "Channel must be included for every message.<|end|>"
    )


register_template(
    name="gpt",
    format_user=StringFormatter(slots=["<|start|>user<|message|>{{content}}<|end|><|start|>assistant"]),
    format_assistant=StringFormatter(slots=["<|channel|>final<|message|>{{content}}"]),
    format_system=StringFormatter(slots=["<|start|>developer<|message|># Instructions\n\n{{content}}<|end|>"]),
    format_prefix=EmptyFormatter(slots=[_get_gpt_oss_prefix()]),
    chat_sep="<|end|>",
    default_system="You are ChatGPT, a large language model trained by OpenAI.",
    template_class=Template,
)

register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    suffix=["<|eot_id|>"],
    chat_sep="<|eot_id|>",
)


# copied from gemma template
register_template(
    name="gemma",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    chat_sep="<end_of_turn>\n",
    suffix=["<end_of_turn>"],
    template_class=Llama2Template,
)


register_template(
    name="phi4",
    format_user=StringFormatter(
        slots=["<|im_start|>user<|im_sep|>{{content}}<|im_end|><|im_start|>assistant<|im_sep|>"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|im_start|>system<|im_sep|>{{content}}<|im_end|>"]),
    suffix=["<|im_end|>"],
    chat_sep="<|im_end|>",
)
