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

"""ErnieKit format preprocessor: {"src": [...], "tgt": [...]} → messages."""

from typing import Any, Dict, List

from .base import BasePreprocessor


class ErnieKitPreprocessor(BasePreprocessor):
    """Convert erniekit-format data to standard messages format.

    ErnieKit format: {"src": ["q1", "q2", ...], "tgt": ["a1", "a2", ...]}
    Output: {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, ...]}
    """

    def preprocess(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        src = row.get("src", [])
        tgt = row.get("tgt", [])

        if not src or not tgt:
            return []

        messages = []
        num_turns = min(len(src), len(tgt))
        for i in range(num_turns):
            messages.append({"role": "user", "content": src[i]})
            messages.append({"role": "assistant", "content": tgt[i]})

        return [{"messages": messages}]
