# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Independent template system for datasets_v2.

Provides a simple, data-driven template definition and encoding mechanism.
Templates are defined as lists of "slots" (strings with {{content}} placeholders,
dicts for special token lookup, or sets for built-in token IDs).

Registration is one function call with plain strings — no Formatter classes needed.

Supports:
- Multi-turn SFT encoding (user/assistant pairs)
- Reasoning/Thinking mode (auto-add <think> tags)
- Tool calling (function role + observation role)
- Jinja-based encoding (using tokenizer's built-in chat_template)
- fix_special_tokens (auto fix eos/pad)
"""

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .tool_utils import FunctionCall, get_tool_utils

logger = logging.getLogger(__name__)

# A slot is one of:
#   str:  text fragment, may contain {{content}} placeholder
#   dict: {"token": "<special>"} → looked up by name in tokenizer vocab
#   set:  {"bos_token"} or {"eos_token"} → resolved to tokenizer's special token ID
Slot = Union[str, Dict[str, str], Set[str]]


@dataclass
class TemplateMeta:
    """Pure-data template definition."""

    name: str
    user: List[Slot]
    assistant: List[Slot]
    system: List[Slot] = field(default_factory=lambda: ["{{content}}"])
    prefix: List[Slot] = field(default_factory=list)
    # Observation role slot (for tool response messages)
    observation: Optional[List[Slot]] = None
    # Function role slot (for tool call messages from assistant)
    function: Optional[List[Slot]] = None
    chat_sep: str = ""
    default_system: str = ""
    suffix: List[str] = field(default_factory=list)
    stop_tokens: List[str] = field(default_factory=list)
    efficient_eos: bool = True
    # Reasoning/Thinking support
    thought_words: Tuple[str, str] = ("<think>\n", "\n</think>\n\n")
    enable_thinking: Optional[bool] = None  # None = not a reasoning template
    # Tool format name (references tool_utils.py)
    tool_format: Optional[str] = None


# ============================================================
# Registry
# ============================================================

_TEMPLATE_REGISTRY: Dict[str, TemplateMeta] = {}


def register_template(
    name: str,
    *,
    user: List[Slot],
    assistant: List[Slot],
    system: Optional[List[Slot]] = None,
    prefix: Optional[List[Slot]] = None,
    observation: Optional[List[Slot]] = None,
    function: Optional[List[Slot]] = None,
    chat_sep: str = "",
    default_system: str = "",
    suffix: Optional[List[str]] = None,
    stop_tokens: Optional[List[str]] = None,
    efficient_eos: bool = True,
    thought_words: Optional[Tuple[str, str]] = None,
    enable_thinking: Optional[bool] = None,
    tool_format: Optional[str] = None,
    exist_ok: bool = False,
) -> None:
    """Register a template by name.

    Example:
        register_template(
            "chatml",
            user=["<|im_start|>user\\n{{content}}<|im_end|>\\n<|im_start|>assistant\\n"],
            assistant=["{{content}}"],
            system=["<|im_start|>system\\n{{content}}<|im_end|>\\n"],
            chat_sep="<|im_end|>\\n",
            suffix=["<|im_end|>"],
        )
    """
    if not exist_ok and name in _TEMPLATE_REGISTRY:
        raise ValueError(f"Template '{name}' is already registered. Use exist_ok=True to overwrite.")
    _TEMPLATE_REGISTRY[name] = TemplateMeta(
        name=name,
        user=user,
        assistant=assistant,
        system=system or ["{{content}}"],
        prefix=prefix or [],
        observation=observation,
        function=function,
        chat_sep=chat_sep,
        default_system=default_system,
        suffix=suffix or [],
        stop_tokens=stop_tokens or [],
        efficient_eos=efficient_eos,
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        enable_thinking=enable_thinking,
        tool_format=tool_format,
    )


def get_template(name: str) -> TemplateMeta:
    """Look up a registered template by name.

    Raises:
        KeyError: if name not found.
    """
    if name not in _TEMPLATE_REGISTRY:
        available = ", ".join(sorted(_TEMPLATE_REGISTRY.keys()))
        raise KeyError(f"Template '{name}' not found. Available: {available}")
    return _TEMPLATE_REGISTRY[name]


def list_templates() -> List[str]:
    """Return sorted list of all registered template names."""
    return sorted(_TEMPLATE_REGISTRY.keys())


# ============================================================
# Slot substitution
# ============================================================


def _substitute_slots(slots: List[Slot], **kwargs: str) -> List[Slot]:
    """Replace {{key}} placeholders in string slots with values."""
    result: List[Slot] = []
    for slot in slots:
        if isinstance(slot, str):
            s = slot
            for key, value in kwargs.items():
                s = s.replace("{{" + key + "}}", value, 1)
            result.append(s)
        else:
            result.append(slot)
    return result


# ============================================================
# Slot → token IDs conversion
# ============================================================


def _slots_to_ids(tokenizer: Any, slots: List[Slot]) -> List[int]:
    """Convert a list of slots to token IDs using the given tokenizer."""
    token_ids: List[int] = []
    for slot in slots:
        if isinstance(slot, str):
            if len(slot) > 0:
                token_ids += tokenizer.encode(slot, add_special_tokens=False)
        elif isinstance(slot, dict):
            token_name = slot.get("token", "")
            if token_name:
                tid = tokenizer.convert_tokens_to_ids(token_name)
                if tid is not None:
                    token_ids.append(tid)
        elif isinstance(slot, set):
            if "bos_token" in slot and tokenizer.bos_token_id is not None:
                token_ids.append(tokenizer.bos_token_id)
            elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                token_ids.append(tokenizer.eos_token_id)
    return token_ids


# ============================================================
# Tool call formatting helpers
# ============================================================


def _format_function_content(content: str, tool_format: str, thought_words: Optional[Tuple[str, str]] = None) -> str:
    """Format function call content using tool_utils.

    Parses JSON tool calls and formats them according to the tool_format.
    """
    thought = None
    if thought_words and len(thought_words) == 2 and len(content) > 0:
        regex = re.compile(rf"{re.escape(thought_words[0])}(.*?){re.escape(thought_words[1])}", re.DOTALL)
        thought = re.search(regex, content)

    if thought:
        content = content.replace(thought.group(0), "")

    tool_utils = get_tool_utils(tool_format)
    functions: list[FunctionCall] = []
    try:
        tool_calls = json.loads(content)
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        for tool_call in tool_calls:
            if "type" in tool_call and tool_call["type"] == "function":
                tool_call = tool_call["function"]
            arguments = tool_call["arguments"]
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            functions.append(FunctionCall(tool_call["name"], arguments))

    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON format in function message: {str([content])}.")

    function_str = tool_utils.function_formatter(functions)
    if thought:
        function_str = thought.group(0) + function_str

    return function_str


def _format_tools_content(content: str, tool_format: str) -> str:
    """Format tools description using tool_utils."""
    tool_utils = get_tool_utils(tool_format)
    try:
        tools = json.loads(content)
        return tool_utils.tool_formatter(tools) if len(tools) != 0 else ""
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON format in tool description: {str([content])}.")


# ============================================================
# Reasoning/Thinking helpers
# ============================================================


def _remove_thought(content: str, thought_words: Tuple[str, str]) -> str:
    """Remove thought tags from content."""
    pattern = re.compile(f"{re.escape(thought_words[0])}(.*?){re.escape(thought_words[1])}", re.DOTALL)
    return re.sub(pattern, "", content).lstrip("\n")


_GLM5_TEMPLATES = {"glm_moe_dsa"}


def _get_thought_word_ids(tokenizer: Any, thought_words: Tuple[str, str], template_name: str = "") -> List[int]:
    """Get token IDs for empty thought. GLM5 uses only closing tag."""
    if template_name in _GLM5_TEMPLATES:
        return tokenizer.encode(thought_words[1], add_special_tokens=False)
    return tokenizer.encode(f"{thought_words[0]}{thought_words[1]}", add_special_tokens=False)


# ============================================================
# Core encoding
# ============================================================


def encode_multiturn(
    template: TemplateMeta,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    tools: Optional[str] = None,
) -> List[Tuple[List[int], List[int]]]:
    """Encode a multi-turn conversation into (prompt_ids, response_ids) pairs.

    Args:
        template: Template definition.
        tokenizer: Tokenizer with encode() and convert_tokens_to_ids().
        messages: List of {"role": "user"|"assistant"|"system"|"function"|"observation", "content": str}.
        system: Override system message. If None, uses template.default_system.
        tools: JSON string describing available tools. Appended to system message.

    Returns:
        List of (prompt_ids, response_ids) tuples, one per turn.
        prompt_ids includes user message tokens (and prefix/system for first turn).
        response_ids includes assistant response tokens.
    """
    # Extract system message if present as first message
    actual_messages = messages
    if messages and messages[0].get("role") == "system":
        system = system or messages[0]["content"]
        actual_messages = messages[1:]

    system = system or template.default_system

    encoded_messages: List[List[int]] = []
    for i, message in enumerate(actual_messages):
        elements: List[Slot] = []

        if i == 0:
            # Prefix (BOS, etc.)
            if template.prefix:
                elements += template.prefix
            # System message and tools description
            if system or tools:
                tool_text = ""
                if tools and template.tool_format:
                    tool_text = _format_tools_content(tools, template.tool_format)
                sys_content = (system or "") + tool_text
                elements += _substitute_slots(template.system, content=sys_content)

        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            elements += _substitute_slots(template.user, content=content)
        elif role == "assistant":
            # Check if this message has tool_calls
            if "tool_calls" in message and template.tool_format:
                func_content = _format_function_content(
                    message["tool_calls"], template.tool_format, template.thought_words
                )
                func_slots = template.function or template.assistant
                elements += _substitute_slots(func_slots, content=func_content)
            else:
                elements += _substitute_slots(template.assistant, content=content)
            if i < len(actual_messages) - 1 and template.chat_sep:
                elements.append(template.chat_sep)
        elif role in ("function", "tool_call"):
            # Function call from assistant
            if template.tool_format:
                func_content = _format_function_content(content, template.tool_format, template.thought_words)
                func_slots = template.function or template.assistant
                elements += _substitute_slots(func_slots, content=func_content)
            else:
                elements += _substitute_slots(template.assistant, content=content)
            if i < len(actual_messages) - 1 and template.chat_sep:
                elements.append(template.chat_sep)
        elif role in ("observation", "tool_response"):
            # Tool response
            obs_slots = template.observation or template.user
            elements += _substitute_slots(obs_slots, content=content)
        else:
            # Fallback: treat unknown roles as user
            elements += _substitute_slots(template.user, content=content)

        encoded_messages.append(_slots_to_ids(tokenizer, elements))

    # Pair up: (prompt, response) for each turn
    pairs: List[Tuple[List[int], List[int]]] = []
    for i in range(0, len(encoded_messages) - 1, 2):
        prompt_ids = encoded_messages[i]
        response_ids = encoded_messages[i + 1] if i + 1 < len(encoded_messages) else []
        pairs.append((prompt_ids, response_ids))

    return pairs


def encode_multiturn_reasoning(
    template: TemplateMeta,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    tools: Optional[str] = None,
) -> List[Tuple[List[int], List[int]]]:
    """Encode with reasoning/thinking support.

    Aligned with old ReasoningTemplate.encode_multiturn:
    - If enable_thinking is False: removes CoT from ALL assistant messages
    - For each turn without thought tags: inserts empty thought tokens
    - If enable_thinking is truthy: thought IDs go into response (trained on)
    - If enable_thinking is falsy: thought IDs go into prompt (not trained on)
    """
    messages = deepcopy(messages)
    thought_words = template.thought_words

    # Extract system
    actual_messages = messages
    if messages and messages[0].get("role") == "system":
        system = system or messages[0]["content"]
        actual_messages = messages[1:]

    # If enable_thinking is False, remove all CoT from all assistant messages
    if template.enable_thinking is False:
        for i in range(1, len(actual_messages), 2):
            if actual_messages[i].get("role") == "assistant":
                actual_messages[i]["content"] = _remove_thought(actual_messages[i]["content"], thought_words)

    # Rebuild messages with system
    if system:
        rebuild = [{"role": "system", "content": system}] + actual_messages
    else:
        rebuild = actual_messages

    # Encode using normal path
    pairs = encode_multiturn(template, tokenizer, rebuild, system=system, tools=tools)

    # Add empty thought to ALL turns that don't have thought tags
    for i in range(0, len(actual_messages), 2):
        if i + 1 >= len(actual_messages):
            break
        assistant_content = actual_messages[i + 1].get("content", "")
        has_thought = thought_words[0].strip() in assistant_content and thought_words[1].strip() in assistant_content
        pair_idx = i // 2
        if pair_idx >= len(pairs):
            break

        if not has_thought:
            thought_ids = _get_thought_word_ids(tokenizer, thought_words, template.name)
            prompt_ids, response_ids = pairs[pair_idx]
            if not template.enable_thinking:
                prompt_ids = prompt_ids + thought_ids
            else:
                response_ids = thought_ids + response_ids
            pairs[pair_idx] = (prompt_ids, response_ids)

    return pairs


# ============================================================
# fix_special_tokens
# ============================================================


def fix_special_tokens(tokenizer: Any, template: TemplateMeta) -> None:
    """Fix eos and pad tokens in the tokenizer based on template config.

    - If tokenizer has no eos_token, adds '<|endoftext|>'
    - If tokenizer has no pad_token, uses eos_token
    - Adds stop_tokens as additional special tokens
    """
    if tokenizer.eos_token_id is None:
        num_added = tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        logger.warning(f"Add eos token: {tokenizer.eos_token}.")
        if num_added > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad token: {tokenizer.pad_token}")

    if template.stop_tokens:
        num_added = tokenizer.add_special_tokens(
            dict(additional_special_tokens=template.stop_tokens), replace_extra_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(template.stop_tokens)))
        if num_added > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


# ============================================================
# parse_template: auto-detect from tokenizer's chat_template
# ============================================================


def parse_template(tokenizer: Any) -> TemplateMeta:
    """Extract a template from the tokenizer's built-in chat_template.

    Useful when no explicit template name is specified — auto-detect from Jinja.
    """

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

    # Detect reasoning template
    is_reasoning = "<think>" in assistant_slot
    assistant_slot = assistant_slot.replace("<think>", "").replace("</think>", "").lstrip("\n")

    if len(user_slot) > len(user_slot_empty_system):
        default_system = find_diff(user_slot_empty_system, user_slot)
        sole_system = system_slot.replace("{{content}}", default_system, 1)
        user_slot = user_slot[len(sole_system) :]
    else:
        default_system = ""

    eos_token = getattr(tokenizer, "eos_token", None) or ""
    efficient_eos = eos_token not in assistant_slot

    return TemplateMeta(
        name="_auto_parsed",
        user=[user_slot],
        assistant=[assistant_slot],
        system=[system_slot],
        prefix=[prefix] if prefix else [],
        default_system=default_system,
        suffix=[],
        stop_tokens=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        enable_thinking=True if is_reasoning else None,
        efficient_eos=efficient_eos,
    )


# ============================================================
# get_template_and_fix_tokenizer: high-level entry point
# ============================================================


def get_template_and_fix_tokenizer(
    tokenizer: Any,
    template_name: Optional[str] = None,
    tool_format: Optional[str] = None,
    default_system: Optional[str] = None,
) -> TemplateMeta:
    """Get template by name (or auto-parse) and fix tokenizer special tokens.

    Args:
        tokenizer: The tokenizer instance.
        template_name: Registered template name, or None to auto-parse.
        tool_format: Override the tool_format in the template.
        default_system: Override the default system message.

    Returns:
        The TemplateMeta instance (possibly modified).
    """
    if template_name is None:
        if isinstance(getattr(tokenizer, "chat_template", None), str):
            logger.warning("`template` was not specified, try parsing the chat template from the tokenizer.")
            template = parse_template(tokenizer)
        else:
            logger.warning("`template` was not specified, use `empty` template.")
            template = get_template("empty")
    else:
        template = get_template(template_name)

    # Override tool_format
    if tool_format is not None:
        template = TemplateMeta(
            name=template.name,
            user=template.user,
            assistant=template.assistant,
            system=template.system,
            prefix=template.prefix,
            observation=template.observation,
            function=template.function,
            chat_sep=template.chat_sep,
            default_system=template.default_system,
            suffix=template.suffix,
            stop_tokens=template.stop_tokens,
            efficient_eos=template.efficient_eos,
            thought_words=template.thought_words,
            enable_thinking=template.enable_thinking,
            tool_format=tool_format,
        )

    # Override default_system
    if default_system is not None:
        template.default_system = default_system

    # Ensure suffix is set
    if not template.suffix:
        template.suffix = [tokenizer.eos_token]
        logger.warning("suffix is not specified, using eos token as suffix.")

    # Fix special tokens
    fix_special_tokens(tokenizer, template)

    return template


# ============================================================
# Jinja-based encoding
# ============================================================


def encode_multiturn_jinja(
    tokenizer: Any,
    messages: List[Dict[str, str]],
) -> List[Tuple[List[int], List[int]]]:
    """Encode using the tokenizer's built-in chat_template (Jinja2).

    For models that ship with a chat_template in their tokenizer config.
    Encodes each turn incrementally to separate prompt and response IDs.

    Args:
        tokenizer: Tokenizer with apply_chat_template() method.
        messages: List of {"role": str, "content": str}.

    Returns:
        List of (prompt_ids, response_ids) tuples.
    """
    pairs: List[Tuple[List[int], List[int]]] = []

    # Build up conversation turn by turn
    accumulated: List[Dict[str, str]] = []
    prev_len = 0
    i = 0
    while i < len(messages):
        # Skip system at start — it gets included with first user turn
        if i == 0 and messages[i].get("role") == "system":
            accumulated.append(messages[i])
            i += 1
            continue

        # Expect user then assistant
        if i < len(messages) and messages[i].get("role") == "user":
            accumulated.append(messages[i])
            # Encode up to this user message with generation prompt
            prompt_ids = tokenizer.apply_chat_template(accumulated, add_generation_prompt=True, tokenize=True)

            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                accumulated.append(messages[i + 1])
                # Encode with assistant response included
                full_ids = tokenizer.apply_chat_template(accumulated, add_generation_prompt=False, tokenize=True)
                response_ids = full_ids[len(prompt_ids) :]
                pairs.append((prompt_ids if not pairs else prompt_ids[prev_len:], response_ids))
                prev_len = len(full_ids)
                i += 2
            else:
                # No response — just prompt
                pairs.append((prompt_ids if not pairs else prompt_ids[prev_len:], []))
                prev_len = len(prompt_ids)
                i += 1
        else:
            i += 1

    # Simpler approach: encode full conversation, then use incremental to split
    if not pairs:
        pairs = _encode_jinja_simple(tokenizer, messages)

    return pairs


def _encode_jinja_simple(
    tokenizer: Any,
    messages: List[Dict[str, str]],
) -> List[Tuple[List[int], List[int]]]:
    """Simple jinja encoding: encode incrementally turn by turn."""
    pairs: List[Tuple[List[int], List[int]]] = []
    prev_len = 0
    accumulated: List[Dict[str, str]] = []

    i = 0
    # Handle leading system message
    if messages and messages[0].get("role") == "system":
        accumulated.append(messages[0])
        i = 1

    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "user":
            accumulated.append(msg)
            prompt_ids_full = tokenizer.apply_chat_template(accumulated, add_generation_prompt=True, tokenize=True)
            prompt_ids = prompt_ids_full[prev_len:]

            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                accumulated.append(messages[i + 1])
                full_ids = tokenizer.apply_chat_template(accumulated, add_generation_prompt=False, tokenize=True)
                response_ids = full_ids[len(prompt_ids_full) :]
                pairs.append((prompt_ids, response_ids))
                prev_len = len(full_ids)
                i += 2
            else:
                pairs.append((prompt_ids, []))
                prev_len = len(prompt_ids_full)
                i += 1
        else:
            accumulated.append(msg)
            i += 1

    return pairs


# ============================================================
# Built-in template definitions
# ============================================================

# ChatML format (Qwen, Qwen2, Qwen3, etc.)
register_template(
    "chatml",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    assistant=["{{content}}"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    chat_sep="<|im_end|>\n",
    suffix=["<|im_end|>"],
    stop_tokens=["<|im_end|>", "<|endoftext|>"],
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
)

# Qwen3 (reasoning model, same format as chatml but with thinking)
register_template(
    "qwen3",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    assistant=["{{content}}"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    observation=[
        "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    ],
    function=["{{content}}"],
    chat_sep="<|im_end|>\n",
    suffix=["<|im_end|>\n"],
    enable_thinking=True,
    tool_format="qwen",
)

# Qwen3 without thinking
register_template(
    "qwen3_nothink",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    assistant=["{{content}}"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    observation=[
        "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    ],
    function=["{{content}}"],
    chat_sep="<|im_end|>\n",
    suffix=["<|im_end|>\n"],
    tool_format="qwen",
)

# Qwen3.5 variant (assistant includes im_end in slot)
register_template(
    "qwen3_5",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    assistant=["{{content}}<|im_end|>\n"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    observation=[
        "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    ],
    function=["{{content}}<|im_end|>\n"],
    stop_tokens=["<|im_end|>"],
    enable_thinking=True,
    tool_format="qwen3_5",
    efficient_eos=False,
)

# Qwen3.5 without thinking
register_template(
    "qwen3_5_nothink",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    assistant=["{{content}}<|im_end|>\n"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    observation=[
        "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    ],
    function=["{{content}}<|im_end|>\n"],
    stop_tokens=["<|im_end|>"],
    tool_format="qwen3_5",
    efficient_eos=False,
)

# Qwen2-VL / Qwen3-VL
register_template(
    "qwen2_vl",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    assistant=["{{content}}"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    observation=[
        "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    ],
    function=["{{content}}"],
    default_system="You are a helpful assistant.",
    chat_sep="<|im_end|>\n",
    suffix=["<|im_end|>\n"],
    tool_format="qwen",
)

register_template(
    "qwen3_vl",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    assistant=["{{content}}"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    observation=[
        "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    ],
    function=["{{content}}"],
    chat_sep="<|im_end|>\n",
    suffix=["<|im_end|>\n"],
    enable_thinking=True,
    tool_format="qwen",
)

register_template(
    "qwen3_vl_nothink",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    assistant=["{{content}}"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    observation=[
        "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    ],
    function=["{{content}}"],
    chat_sep="<|im_end|>\n",
    suffix=["<|im_end|>\n"],
    tool_format="qwen",
)

# Chatml EOS variant
register_template(
    "chatml_eos",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    assistant=["{{content}}<|im_end|>\n"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    stop_tokens=["<|im_end|>"],
    efficient_eos=False,
)

# Llama3 format
register_template(
    "llama3",
    prefix=[{"token": "<|begin_of_text|>"}],
    user=[
        "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ],
    assistant=["{{content}}<|eot_id|>"],
    system=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"],
    observation=[
        "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ],
    function=["{{content}}"],
    stop_tokens=["<|eot_id|>"],
    chat_sep="<|eot_id|>",
    efficient_eos=False,
    tool_format="llama3",
)

# DeepSeek v3 format (uses fullwidth vertical bars)
register_template(
    "deepseek3",
    user=["<｜User｜>{{content}}\n\n<｜Assistant｜>"],
    assistant=["{{content}}"],
    system=["{{content}}\n\n"],
    prefix=[{"token": "<｜begin▁of▁sentence｜>"}],
    stop_tokens=["<｜end▁of▁sentence｜>"],
    chat_sep="<｜end▁of▁sentence｜>",
)

# GLM4 format
register_template(
    "glm4",
    user=["<|user|>\n{{content}}<|assistant|>\n"],
    assistant=["\n{{content}}"],
    system=["<|system|>\n{{content}}"],
    observation=["<|observation|>\n{{content}}<|assistant|>"],
    function=["{{content}}"],
    stop_tokens=["<|endoftext|>", "<|user|>"],
    tool_format="glm4",
)

# GLM4 MOE
register_template(
    "glm4_moe",
    user=["<|user|>\n{{content}}<|assistant|>\n"],
    assistant=["\n{{content}}"],
    system=["<|system|>\n{{content}}"],
    observation=["<|observation|>\n{{content}}<|assistant|>"],
    function=["{{content}}"],
    prefix=["[gMASK]<sop>[gMASK]<sop>"],
    suffix=["<|user|>"],
    thought_words=("<think>", "</think>"),
    enable_thinking=True,
    tool_format="glm4_moe",
)

# GLM5 / GLM-MOE-DSA
register_template(
    "glm_moe_dsa",
    user=["<|user|>{{content}}<|assistant|>"],
    assistant=["{{content}}"],
    system=["[gMASK]<sop><|system|>{{content}}"],
    observation=["<|observation|>{{content}}<|assistant|>"],
    function=["{{content}}"],
    prefix=["[gMASK]<sop>"],
    suffix=["<|user|>"],
    thought_words=("<think>", "</think>"),
    enable_thinking=True,
    tool_format="glm_moe_dsa",
)

# GLM4V (vision)
register_template(
    "glm4v",
    user=["<|user|>\n{{content}}<|assistant|>"],
    assistant=["\n{{content}}"],
    system=["<|system|>\n{{content}}"],
    observation=["<|observation|>\n{{content}}<|assistant|>"],
    function=["{{content}}"],
    prefix=["[gMASK]<sop>"],
    suffix=["<|user|>"],
    enable_thinking=True,
    tool_format="glm4",
)

# GLM4V MOE (vision)
register_template(
    "glm4v_moe",
    user=["<|user|>\n{{content}}<|assistant|>\n"],
    assistant=["\n{{content}}"],
    system=["<|system|>\n{{content}}"],
    observation=["<|observation|>\n{{content}}<|assistant|>"],
    function=["{{content}}"],
    prefix=["[gMASK]<sop>"],
    suffix=["<|user|>"],
    stop_tokens=["<|user|>", "<|observation|>", "</answer>"],
    thought_words=("<think>", "</think>"),
    enable_thinking=True,
    tool_format="glm4_moe",
)

# GLM-OCR
register_template(
    "glm_ocr",
    user=["<|user|>\n{{content}}\n"],
    assistant=["{{content}}"],
    prefix=["[gMASK]<sop>"],
    chat_sep="<|assistant|>\n",
)

# ERNIE (Baidu) format
register_template(
    "ernie",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n\n<|im_start|>assistant\n"],
    assistant=["{{content}}"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    observation=[
        "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n\n<|im_start|>assistant\n"
    ],
    function=["\n{{content}}"],
    chat_sep="<|im_end|>\n\n",
    suffix=["<|im_end|>"],
    stop_tokens=["<|im_end|>"],
    thought_words=("<think>", "</think>"),
    enable_thinking=True,
    tool_format="ernie",
)

# ERNIE VL
register_template(
    "ernie_vl",
    user=["User: {{content}}\nAssistant: "],
    assistant=["{{content}}"],
    system=["{{content}}\n"],
    observation=["User: <tool_output>\n{{content}}\n</tool_output>\n\nAssistant: "],
    function=["\n{{content}}"],
    prefix=["<|begin_of_sentence|>"],
    chat_sep="<|end_of_sentence|>",
    suffix=["</s>"],
    thought_words=("\n<think>\n", "\n</think>\n\n"),
    enable_thinking=True,
    tool_format="ernie_vl",
)

# PaddleOCR VL
register_template(
    "paddleocr_vl",
    user=["User: {{content}}\nAssistant: "],
    assistant=["{{content}}"],
    system=["{{content}}\n"],
    prefix=["<|begin_of_sentence|>"],
    chat_sep="<|end_of_sentence|>",
)

# Gemma format
register_template(
    "gemma",
    user=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"],
    assistant=["{{content}}"],
    system=["{{content}}\n\n"],
    observation=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"],
    prefix=[{"bos_token"}],
    chat_sep="<end_of_turn>\n",
    suffix=["<end_of_turn>"],
    stop_tokens=["<end_of_turn>"],
)

# Phi-4 format
register_template(
    "phi4",
    user=["<|im_start|>user<|im_sep|>{{content}}<|im_end|><|im_start|>assistant<|im_sep|>"],
    assistant=["{{content}}"],
    system=["<|im_start|>system<|im_sep|>{{content}}<|im_end|>"],
    suffix=["<|im_end|>"],
    chat_sep="<|im_end|>",
)

# Kimi K2
register_template(
    "kimi_k2",
    user=["<｜User｜>{{content}}\n\n<｜Assistant｜>"],
    assistant=["{{content}}"],
    system=["{{content}}\n\n"],
    prefix=[{"bos_token"}],
    chat_sep="<｜end▁of▁sentence｜>",
    stop_tokens=["<｜end▁of▁sentence｜>"],
)

# Simple default format
register_template(
    "default",
    user=["Human: {{content}}", {"eos_token"}, "\nAssistant:"],
    assistant=["{{content}}", {"eos_token"}, "\n"],
    system=["System: {{content}}", {"eos_token"}, "\n"],
)

# Empty template — passthrough
register_template(
    "empty",
    user=["{{content}}"],
    assistant=["{{content}}"],
    efficient_eos=False,
)
