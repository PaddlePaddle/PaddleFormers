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

"""MessagesPreprocessor: handles datasets already in conversation format.

Supports: standard messages, ShareGPT format, various key naming conventions.
"""

import ast
from typing import Any, Callable, Dict, List, Optional, Union

from .base import BasePreprocessor


def default_repair_messages(s: Union[str, Any]) -> Any:
    """Parse stringified messages list back to Python objects."""
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


class MessagesPreprocessor(BasePreprocessor):
    """Preprocessor for datasets that already have a conversation structure.

    Handles various naming conventions:
        - role keys: 'role', 'from'
        - content keys: 'content', 'value'
        - user roles: 'user', 'human'
        - assistant roles: 'assistant', 'gpt', 'bot'

    Also handles ShareGPT paired format:
        [{'human': '...', 'gpt': '...'}, ...]

    Output: standard messages format.
    """

    def __init__(
        self,
        *,
        role_key: Optional[str] = None,
        content_key: Optional[str] = None,
        user_role: Optional[str] = None,
        assistant_role: Optional[str] = None,
        system_role: str = "system",
        repair_messages: Callable = default_repair_messages,
        inner_key: Optional[str] = None,
        columns: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        self.role_keys = ["role", "from"] if role_key is None else [role_key]
        self.content_keys = ["content", "value"] if content_key is None else [content_key]
        self.user_roles = ["user", "human"] if user_role is None else [user_role]
        self.assistant_roles = ["assistant", "gpt", "bot"] if assistant_role is None else [assistant_role]
        self.tool_call_roles = ["function_call"]
        self.tool_response_roles = ["function_response", "observation", "observations"]

        self.system_role = system_role
        self.repair_messages = repair_messages
        self.inner_key = inner_key

        for key in ["messages", "conversation", "conversations"]:
            self.columns[key] = "messages"
        for key in ["system", "system_prompt"]:
            self.columns[key] = "system"

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "rejected_messages" in row and row["rejected_messages"] is not None:
            sub = self.preprocess({"messages": row["rejected_messages"]})
            row["rejected_messages"] = sub["messages"] if sub else None

        messages = row["messages"]
        if self.inner_key is not None:
            messages = messages[self.inner_key]

        messages = self.repair_messages(messages)
        if not messages or isinstance(messages, str):
            return None

        self._to_std_key(messages, "role", self.role_keys)
        self._to_std_key(messages, "content", self.content_keys)

        system = row.pop("system", None)

        if self._is_sharegpt_format(messages[0]):
            messages = self._sharegpt_to_messages(messages, system)
        else:
            self._to_std_messages(messages, system)

        row["messages"] = messages
        return row

    # ================================================================
    # Format detection and conversion
    # ================================================================

    @staticmethod
    def _is_sharegpt_format(message: Dict[str, str]) -> bool:
        return "role" not in message and "content" not in message

    def _sharegpt_to_messages(self, messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
        """Convert ShareGPT paired format to standard messages."""
        self._to_std_key(messages, "user", self.user_roles)
        self._to_std_key(messages, "assistant", self.assistant_roles)
        new_messages = []
        if system is not None:
            new_messages.append({"role": "system", "content": system})
        for message in messages:
            new_messages.append({"role": "user", "content": message["user"]})
            new_messages.append({"role": "assistant", "content": message["assistant"]})
        return new_messages

    def _to_std_messages(self, messages: List[Dict[str, str]], system: Optional[str]) -> None:
        """Normalize role names in-place for standard messages format."""
        if messages[0]["role"] == self.system_role:
            messages[0]["role"] = "system"
        elif system is not None:
            messages.insert(0, {"role": "system", "content": system})

        for message in messages:
            role = message["role"]
            if role in self.user_roles:
                message["role"] = "user"
            elif role in self.assistant_roles:
                message["role"] = "assistant"
            elif role.replace("-", "_") in self.tool_call_roles:
                message["role"] = "tool_call"
            elif role.replace("-", "_") in self.tool_response_roles:
                message["role"] = "tool_response"

    @staticmethod
    def _to_std_key(messages: List[Dict[str, str]], std_key: str, optional_keys: List[str]) -> None:
        """Rename variant keys to a standard key in each message dict."""
        for message in messages:
            for key in optional_keys:
                if key in message:
                    message[std_key] = message.pop(key)
