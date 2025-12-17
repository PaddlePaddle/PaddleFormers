# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from transformers import AutoTokenizer


class SeparatorStyle(Enum):
    """Different separator style."""
    
    CHATML = auto()
    QWEN3VL = auto()


@dataclass
class Conversation:
    system: str | None
    roles: tuple[str, str]
    messages: list[list[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.CHATML
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    
    tokenizer_name_or_path: Any = None
    stop_str: str | list[str] = "<|im_end|>"
    stop_token_ids: list[int] = None
    
    skip_next: bool = False
    
    def process_chat_template(self, tokenzier_name_or_path, messages):
        tokenizer = AutoTokenizer.from_pretrained(tokenzier_name_or_path)
        if not self.system:
            chat = []
        else:
            chat = [{"role": "system", "content": self.system}]
        for role, message in messages:
            chat.append({"role": role.lower(), "content": message})
        res = tokenizer.apply_chat_template(chat, tokineize=False, add_vision_id=True, add_generation_prompt=False)
        return res
    
    def get_prompt(self):
        messages = self.messages
        if self.sep_style == SeparatorStyle.QWEN3VL:
            tokenizer_name_or_path = self.tokenizer_name_or_path or "Qwen/Qwen3-VL-2B-Instruct"
            res = self.process_chat_template(tokenizer_name_or_path, messages)
        elif self.sep_style == SeparatorStyle.CHATML:
            res = "" if not self.system else f"{self.system}{self.sep}\n"
            for role, message in messages:
                if message:
                    if isinstance(message, tuple):
                        message, images = message
                        message = f"{'<image>' * len(images)}{messages}"
                    res += f"{role}\n{message}{self.sep}\n"
                else:
                    res += f"{role}\n"
        else:
            raise ValueError(f"Invalid style {self.sep_style}")
        return res
    
    def append_message(self, role, message):
        self.messages.append([role, message])


conv_qwen3vl = Conversation(
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    version="qwen3vl",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.QWEN3VL,
    sep=""
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system 
Answer the questions.""",
    roles=("<|im_start|>user\n", "<im_start|>assistant\n"),
    version="",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>"
)

default_conversation = conv_qwen3vl
conv_templates = {
    "default": conv_qwen3vl,
    "quen3vl": conv_qwen3vl,
    "chatml_direct": conv_chatml_direct,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
