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
from typing import Callable


@dataclass
class MultiModalToken:
    """
    Base class for multi-modal tokens representing different media types.
    """
    token_str: str
    token_index: int
    media_type: str
    use_start_end: bool
    encoder_fn: Callable = None

@dataclass
class Qwen3VLImageToken(MultiModalToken):
    """Image token class."""
    token_str: str = "<|image_pad|>"
    token_index: int = 151655
    media_type: str = "image"
    use_start_end: bool = False

@dataclass
class Qwen3VLVideoToken(MultiModalToken):
    """Video token class."""
    token_str: str = "<|video_pad|>"
    token_index: int = 151656
    media_type: str = "video"
    use_start_end: bool = False


# Contants for token indexing and special token mapping.
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = Qwen3VLImageToken.token_index
VIDEO_TOKEN_INDEX = Qwen3VLVideoToken.token_index
OBJECT_REF_START_TOKEN_INDEX = 151646
OBJECT_REF_END_TOKEN_INDEX = 151647
BOX_START_TOKEN_INDEX = 151648
BOX_END_TOKEN_INDEX = 151649
QUAD_START_TOKEN_INDEX = 151650
QUAD_END_TOKEN_INDEX = 151651
VISION_START_TOKEN_INDEX = 151652
VISION_END_TOKEN_INDEX = 151653
PAD_TOKEN_INDEX = 151643
HF_IMAGE_TOKEN_INDEX = 151655
HF_VIDEO_TOKEN_INDEX = 151656

SPECIAL_TOKEN_MAP = [
    (Qwen3VLImageToken.token_str, Qwen3VLImageToken.token_index),
    (Qwen3VLVideoToken.token_str, Qwen3VLVideoToken.token_index),
    ("<|object_ref_start|>", OBJECT_REF_START_TOKEN_INDEX),
    ("<|object_ref_end|>", OBJECT_REF_END_TOKEN_INDEX),
    ("<|box_start|>", BOX_START_TOKEN_INDEX),
    ("<|box_end|>", BOX_END_TOKEN_INDEX),
    ("<|quad_start|>", QUAD_START_TOKEN_INDEX),
    ("<|quad_end|>", QUAD_END_TOKEN_INDEX),
    ("<|vision_start|>", VISION_START_TOKEN_INDEX),
    ("<|vision_end|>", VISION_END_TOKEN_INDEX),
]
