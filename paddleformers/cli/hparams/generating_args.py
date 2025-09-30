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

from dataclasses import dataclass, field


class StreamOptions:
    """
    Class of stream options...see https://developers.feedly.com/v3/streams/
    note camel casing...this is on purpose so we can just use the __dict__ of the object
    to produce url parameters
    """

    def __init__(self, max_count: int = 100):
        self.count: int = 20
        self.ranked: str = "newest"
        self.unreadOnly: bool = False
        self.newerThan: int = None
        self._max_count = max_count
        self.continuation: str = None


@dataclass
class GeneratingArguments:
    """Generating Argument"""

    # base
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "The maximum numbers of tokens to generate"},
    )
    min_tokens: int = field(
        default=0,
        metadata={"help": "The minimum numbers of tokens to generate"},
    )
    temperature: float = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: float = field(
        default=0.7,
        metadata={
            "help": (
                "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
            )
        },
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={
            "help": "The penalty for controlling duplicate tokens is stricter than presence_penalty "
            "and will punish high-frequency duplicates"
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Control the penalty coefficient for generating repetitive content by the model. "
            "A positive value reduces the probability of repetitive topics occurring"
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "The coefficient for directly punishing repeatedly generated tokens "
            "(>1: Punish repetition, <1: Encourage repetition."
        },
    )
    stream: bool = field(
        default=True,
        metadata={"help": "Whether to start stream output"},
    )
    stream_options: StreamOptions = field(
        default=None,
        metadata={"help": "Relevant options for stream output"},
    )
    enable_thinking: bool = field(
        default=False, metadata={"help": "Whether enable thinking when using VL model."}
    )
