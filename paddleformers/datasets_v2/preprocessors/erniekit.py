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

"""ErnieKit format preprocessor: {"src": [...], "tgt": [...]} → messages.

Supports both SFT format and DPO format:
- SFT: {"src": [...], "tgt": [...]}
- DPO: {"src": [...], "tgt": [...], "response": [chosen, rejected], "sort": [2, 1]}
"""

import logging
from typing import Any, Dict, List, Optional

from .base import BasePreprocessor

logger = logging.getLogger(__name__)


class ErnieKitPreprocessor(BasePreprocessor):
    """Convert erniekit-format data to standard messages format.

    SFT format:
        Input:  {"src": ["q1", "q2", ...], "tgt": ["a1", "a2", ...]}
        Output: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}, ...]}

    DPO format:
        Input:  {"src": [...], "tgt": [...], "response": [chosen, rejected], "sort": [2, 1]}
        Output: {"messages": [prompt + chosen_response], "rejected_messages": [prompt + rejected_response]}
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        src = row.get("src", [])
        tgt = row.get("tgt", [])

        if not src:
            return []

        # Detect DPO format: presence of "response" and "sort" fields
        response = row.get("response")
        sort = row.get("sort")
        if response is not None and sort is not None:
            return self._preprocess_dpo(row, src, tgt, response, sort)

        # SFT format
        if not tgt:
            return []

        messages = []
        num_turns = min(len(src), len(tgt))
        for i in range(num_turns):
            messages.append({"role": "user", "content": src[i]})
            messages.append({"role": "assistant", "content": tgt[i]})

        return [{"messages": messages}]

    def _preprocess_dpo(
        self,
        row: Dict[str, Any],
        src: List[str],
        tgt: List[str],
        response: List,
        sort: List,
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert ErnieKit DPO format to standard messages + rejected_messages.

        ErnieKit DPO format:
            src: ["user_q1", "user_q2", ...]  (user turns, len = tgt + 1)
            tgt: ["asst_a1", ...]              (shared assistant turns)
            response: [resp_a, resp_b]         (two candidate responses)
            sort: [score_a, score_b]           (higher = better)
        """
        if isinstance(src, str):
            src = [src]
        if isinstance(tgt, str):
            tgt = [tgt]

        if len(response) != 2:
            logger.warning(f"[SKIP] DPO response must have 2 items, got {len(response)}")
            return []
        if len(sort) != 2:
            logger.warning(f"[SKIP] DPO sort must have 2 items, got {len(sort)}")
            return []
        if sort[0] == sort[1]:
            logger.warning(f"[SKIP] DPO sort values must differ, got {sort}")
            return []

        # Determine chosen/rejected based on sort scores
        if sort[0] > sort[1]:
            chosen_text = response[0]
            rejected_text = response[1]
        else:
            chosen_text = response[1]
            rejected_text = response[0]

        # Normalize response to string
        if isinstance(chosen_text, list):
            chosen_text = chosen_text[0] if chosen_text else ""
        if isinstance(rejected_text, list):
            rejected_text = rejected_text[0] if rejected_text else ""

        if not chosen_text or not rejected_text:
            logger.warning("[SKIP] DPO chosen or rejected response is empty")
            return []

        # Handle system message
        system = None
        is_system = row.get("is_system", 0)
        if is_system == 1 and len(src) > 0:
            system = src[0]
            src = src[1:]
            if tgt:
                tgt = tgt[1:]
        elif "system" in row:
            system = row["system"]

        # Build shared prompt messages
        prompt_messages = []
        if system:
            prompt_messages.append({"role": "system", "content": system})

        # src has one more element than tgt (the last user query)
        for idx in range(len(src)):
            prompt_messages.append({"role": "user", "content": src[idx]})
            if idx < len(tgt):
                prompt_messages.append({"role": "assistant", "content": tgt[idx]})

        # Build full chosen and rejected conversations
        chosen_messages = list(prompt_messages) + [{"role": "assistant", "content": chosen_text}]
        rejected_messages = list(prompt_messages) + [{"role": "assistant", "content": rejected_text}]

        return [{"messages": chosen_messages, "rejected_messages": rejected_messages}]
