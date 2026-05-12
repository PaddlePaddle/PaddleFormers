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

"""ResponsePreprocessor: handles query/response format datasets.

Corresponds to ms-swift's ResponsePreprocessor (preprocessor/core.py:356).
Supports: query + response + optional history + optional system.
"""

import ast
from typing import Any, Dict, List, Optional

from .base import BasePreprocessor


def history_to_messages(history: List[List[Optional[str]]], system: Optional[str] = None) -> List[Dict[str, str]]:
    """Convert history pairs to messages list.

    Args:
        history: [['query1', 'response1'], ['query2', 'response2'], ...]
                 Either element can be None (will be skipped).
        system: optional system prompt
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for pair in history:
        assert isinstance(pair, (list, tuple))
        if pair[0] is not None:
            messages.append({"role": "user", "content": pair[0]})
        if pair[1] is not None:
            messages.append({"role": "assistant", "content": pair[1]})
    return messages


class ResponsePreprocessor(BasePreprocessor):
    """Preprocessor for query/response format datasets.

    Handles datasets with columns like:
        - query (or: prompt, input, instruction, question, problem)
        - response (or: answer, output, targets, text, completion, ...)
        - system (optional)
        - history (optional): [['q1','r1'], ['q2','r2'], ...]

    Output: standard messages format.
    """

    SYSTEM_KEYS = ["system", "system_prompt"]
    QUERY_KEYS = ["query", "prompt", "input", "instruction", "question", "problem"]
    RESPONSE_KEYS = [
        "response",
        "answer",
        "output",
        "targets",
        "target",
        "answer_key",
        "answers",
        "solution",
        "text",
        "completion",
        "content",
    ]

    def __init__(self, *, columns: Optional[Dict[str, str]] = None, **kwargs) -> None:
        super().__init__(columns=columns, **kwargs)
        for key in self.SYSTEM_KEYS:
            self.columns[key] = "system"
        for key in self.QUERY_KEYS:
            self.columns[key] = "query"
        for key in self.RESPONSE_KEYS:
            self.columns[key] = "response"

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = row.pop("response", None)
        if isinstance(response, (list, tuple)):
            response = response[0]

        history = row.pop("history", None) or []
        query = row.pop("query", None)
        system = row.pop("system", None)

        if isinstance(history, str):
            history = ast.literal_eval(history)

        history.append([query, response])
        row["messages"] = history_to_messages(history, system)
        return row
