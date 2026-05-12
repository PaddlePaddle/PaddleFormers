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
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
    chat_sep: str = ""
    default_system: str = ""
    suffix: List[str] = field(default_factory=list)
    stop_tokens: List[str] = field(default_factory=list)
    efficient_eos: bool = True


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
    chat_sep: str = "",
    default_system: str = "",
    suffix: Optional[List[str]] = None,
    stop_tokens: Optional[List[str]] = None,
    efficient_eos: bool = True,
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
        chat_sep=chat_sep,
        default_system=default_system,
        suffix=suffix or [],
        stop_tokens=stop_tokens or [],
        efficient_eos=efficient_eos,
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
# Core encoding
# ============================================================


def encode_multiturn(
    template: TemplateMeta,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
) -> List[Tuple[List[int], List[int]]]:
    """Encode a multi-turn conversation into (prompt_ids, response_ids) pairs.

    Args:
        template: Template definition.
        tokenizer: Tokenizer with encode() and convert_tokens_to_ids().
        messages: List of {"role": "user"|"assistant"|"system", "content": str}.
        system: Override system message. If None, uses template.default_system.

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
            # System message
            if system:
                elements += _substitute_slots(template.system, content=system)

        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            elements += _substitute_slots(template.user, content=content)
        elif role == "assistant":
            elements += _substitute_slots(template.assistant, content=content)
            if i < len(actual_messages) - 1 and template.chat_sep:
                elements.append(template.chat_sep)
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
    # For v1, provide a straightforward implementation
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

# Qwen3.5 variant (assistant includes im_end in slot)
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
    stop_tokens=["<|eot_id|>"],
    efficient_eos=False,
)

# DeepSeek v3 format (uses fullwidth vertical bars)
register_template(
    "deepseek3",
    user=["<｜User｜>{{content}}\n\n<｜Assistant｜>"],
    assistant=["{{content}}"],
    system=["{{content}}\n\n"],
    prefix=[{"token": "<｜begin▁of▁sentence｜>"}],
    stop_tokens=["<｜end▁of▁sentence｜>"],
)

# GLM4 format
register_template(
    "glm4",
    user=["<|user|>\n{{content}}<|assistant|>\n"],
    assistant=["\n{{content}}"],
    system=["<|system|>\n{{content}}"],
    stop_tokens=["<|endoftext|>", "<|user|>"],
)

# Gemma format
register_template(
    "gemma",
    user=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"],
    assistant=["{{content}}"],
    system=["{{content}}\n"],
    chat_sep="<end_of_turn>\n",
    suffix=["<end_of_turn>"],
    stop_tokens=["<end_of_turn>"],
)

# ERNIE (Baidu) format
register_template(
    "ernie",
    user=["<|im_start|>user\n{{content}}<|im_end|>\n\n<|im_start|>assistant\n"],
    assistant=["{{content}}"],
    system=["<|im_start|>system\n{{content}}<|im_end|>\n"],
    chat_sep="<|im_end|>\n\n",
    suffix=["<|im_end|>"],
    stop_tokens=["<|im_end|>"],
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
)
