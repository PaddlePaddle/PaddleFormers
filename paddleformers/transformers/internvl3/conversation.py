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

from __future__ import annotations

import copy
import dataclasses
from enum import IntEnum, auto


class SeparatorStyle(IntEnum):
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: tuple[str, str] = ("<|im_start|>user\n", "<|im_start|>assistant\n")
    messages: list[list[str]] = dataclasses.field(default_factory=list)
    sep_style: SeparatorStyle = SeparatorStyle.MPT
    sep: str = "<|im_end|>\n"

    def get_prompt(self) -> str:
        ret = self.system_template.format(system_message=self.system_message) + self.sep
        for role, message in self.messages:
            if message:
                if isinstance(message, tuple):
                    message = message[0]
                ret += role + message + self.sep
            else:
                ret += role
        return ret

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        self.messages[-1][1] = message

    def copy(self):
        return copy.deepcopy(self)


_CONV_TEMPLATES = {}


def register_conv_template(template: Conversation, override: bool = False):
    if not override and template.name in _CONV_TEMPLATES:
        raise ValueError(f"{template.name} has been registered.")
    _CONV_TEMPLATES[template.name] = template


def get_conv_template(name: str) -> Conversation:
    return _CONV_TEMPLATES[name].copy()


register_conv_template(
    Conversation(
        name="Hermes-2",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
    )
)


register_conv_template(
    Conversation(
        name="internlm2-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
    )
)


register_conv_template(
    Conversation(
        name="phi3-chat",
        system_template="<|system|>\n{system_message}",
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|user|>\n", "<|assistant|>\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|end|>",
    )
)


register_conv_template(
    Conversation(
        name="internvl2_5",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)

register_conv_template(
    Conversation(
        name="internvl3",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )
)


__all__ = [
    "Conversation",
    "SeparatorStyle",
    "get_conv_template",
    "register_conv_template",
]
