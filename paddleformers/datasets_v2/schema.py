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

"""Standard schema definitions for datasets.

All preprocessors normalize raw data into this schema.
After preprocessing, only fields defined here are retained.
"""

from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Literal, Optional, Union

from datasets import Dataset as HfMapDataset
from datasets import IterableDataset as HfIterableDataset

DATASET_TYPE = Union[HfMapDataset, HfIterableDataset]

# ============================================================
# Constants
# ============================================================

ROLES = ("system", "user", "assistant", "tool_call", "tool_response", "tool")
Role = Literal["system", "user", "assistant", "tool_call", "tool_response", "tool"]
# Core fields of a single sample
PAIR_KEYS = ["messages", "images", "videos", "audios", "tools", "objects"]
# Prefixed pair fields for preference learning
# Expands to: ['rejected_messages', 'rejected_images', ..., 'positive_videos', ...]
PREFIXED_PAIR_KEYS = list(
    chain.from_iterable([f"{prefix}_{k}" for k in PAIR_KEYS] for prefix in ["rejected", "positive", "negative"])
)
# Scalar fields
SCALAR_KEYS = ["rejected_response", "label", "channel", "margin", "teacher_prompt"]
STANDARD_KEYS = PAIR_KEYS + PREFIXED_PAIR_KEYS + SCALAR_KEYS

# ============================================================
# Type definitions
# ============================================================


class Message(dict):
    """A single message in a conversation.

    Keys:
        role: one of ROLES
        content: str
        loss: Optional[bool] — whether to compute loss on this turn
    """

    pass


class ImageMedia(dict):
    """Image media representation.

    Keys:
        bytes: Optional[bytes] — raw image bytes, None if using path
        path: str — file path to the image
    """

    pass


class GroundingObjects(dict):
    """Grounding annotation for object detection / referring.

    Keys:
        ref: List[str]
        bbox: List[List[float]] — each is [x1, y1, x2, y2] or [x, y]
        bbox_type: str
        image_id: int
    """

    pass


# ============================================================
# Standard row dataclass
# ============================================================


@dataclass
class StandardRow:
    """The canonical representation of a single sample after preprocessing."""

    # Core conversation
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # Multimodal
    images: Optional[List[Dict[str, Any]]] = None
    videos: Optional[List[str]] = None
    audios: Optional[List[str]] = None

    # Tool use
    tools: Optional[List[Dict[str, Any]]] = None

    # Grounding
    objects: Optional[Dict[str, Any]] = None

    # Preference (DPO / RLHF)
    rejected_response: Optional[str] = None
    rejected_messages: Optional[List[Dict[str, Any]]] = None
    rejected_images: Optional[List[Dict[str, Any]]] = None
    rejected_videos: Optional[List[str]] = None
    rejected_audios: Optional[List[str]] = None
    rejected_tools: Optional[List[Dict[str, Any]]] = None
    rejected_objects: Optional[Dict[str, Any]] = None

    # Embedding / Reranking
    positive_messages: Optional[List[Dict[str, Any]]] = None
    positive_images: Optional[List[Dict[str, Any]]] = None
    positive_videos: Optional[List[str]] = None
    positive_audios: Optional[List[str]] = None
    positive_tools: Optional[List[Dict[str, Any]]] = None
    positive_objects: Optional[Dict[str, Any]] = None

    negative_messages: Optional[List[Dict[str, Any]]] = None
    negative_images: Optional[List[Dict[str, Any]]] = None
    negative_videos: Optional[List[str]] = None
    negative_audios: Optional[List[str]] = None
    negative_tools: Optional[List[Dict[str, Any]]] = None
    negative_objects: Optional[Dict[str, Any]] = None

    # Classification / Reward
    label: Optional[int] = None
    channel: Optional[str] = None
    margin: Optional[float] = None

    # Distillation
    teacher_prompt: Optional[str] = None


# ============================================================
# Checks
# ============================================================


def check_messages(messages: List[Dict[str, Any]]) -> None:
    """Check that a messages list conforms to schema."""
    assert len(messages) > 0, "empty messages"
    for msg in messages:
        assert "role" in msg, f'message missing "role": {msg}'
        assert "content" in msg, f'message missing "content": {msg}'
        assert msg["role"] in ROLES, f'invalid role "{msg["role"]}", must be one of {ROLES}'
        assert msg["content"] is not None, f"message content is None: {msg}"
        extra_keys = set(msg.keys()) - {"role", "content", "loss"}
        assert not extra_keys, f"unexpected keys in message: {extra_keys}"


# ============================================================
# Type casting utilities
# ============================================================


def cast_images(images: Union[str, Dict, List]) -> List[Dict[str, Any]]:
    """Normalize image input to List[ImageMedia] format."""
    if isinstance(images, str):
        return [{"bytes": None, "path": images}]
    if isinstance(images, dict):
        return [images]
    if isinstance(images, list):
        result = []
        for img in images:
            if isinstance(img, str):
                result.append({"bytes": None, "path": img})
            else:
                result.append(img)
        return result
    raise TypeError(f"unsupported images type: {type(images)}")


def cast_media_list(media: Union[str, List[str]]) -> List[str]:
    """Normalize videos/audios to List[str] format."""
    if isinstance(media, str):
        return [media]
    return media


def remove_non_standard_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only standard keys in a row dict."""
    return {k: v for k, v in row.items() if k in STANDARD_KEYS}
