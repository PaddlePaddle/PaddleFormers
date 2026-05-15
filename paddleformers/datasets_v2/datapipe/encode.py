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

"""Single-sample SFT encoding: messages → input_ids + labels.

Takes a datasets_v2 row (with 'messages' column in OpenAI format) and produces
token IDs with loss masking, ready for collation into a training batch.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .template import TemplateMeta, encode_multiturn, encode_multiturn_jinja

logger = logging.getLogger(__name__)


@dataclass
class EncodeConfig:
    """Configuration for SFT encoding."""

    max_seq_len: int = 4096
    truncation: str = "right"  # "right" | "left"
    label_shift: bool = True  # causal LM: labels = labels[1:] + [-100]


@dataclass
class EncodedSample:
    """Output of encode_sft."""

    input_ids: List[int]
    labels: List[int]  # -100 for no-loss positions
    seq_len: int  # actual token count (before any padding)


def encode_sft(
    example: Dict[str, Any],
    tokenizer: Any,
    template: Optional[TemplateMeta],
    config: EncodeConfig,
) -> Optional[EncodedSample]:
    """Encode a single datasets_v2 row into token_ids + labels for SFT.

    Args:
        example: Dict with 'messages' key. Each message has:
            - role: "user" | "assistant" | "system"
            - content: str
            - loss (optional): bool, whether this assistant turn contributes to loss.
              Defaults to True for assistant turns.
        tokenizer: Tokenizer with encode() and convert_tokens_to_ids().
        template: TemplateMeta for custom encoding. If None, uses jinja path.
        config: EncodeConfig.

    Returns:
        EncodedSample or None if the sample is invalid.
    """
    messages = example.get("messages")
    if not messages or len(messages) < 2:
        return None

    # Extract per-turn loss mask (for assistant turns only)
    loss_mask = _extract_loss_mask(messages)

    # Encode messages → (prompt_ids, response_ids) pairs
    if template is not None:
        pairs = encode_multiturn(template, tokenizer, messages)
    else:
        pairs = encode_multiturn_jinja(tokenizer, messages)

    if not pairs:
        return None

    # Build token_ids and labels
    token_ids: List[int] = []
    labels: List[int] = []

    for turn_idx, (prompt_ids, response_ids) in enumerate(pairs):
        # Prompt tokens: no loss
        token_ids += prompt_ids
        labels += [-100] * len(prompt_ids)

        # Response tokens: loss if enabled for this turn
        token_ids += response_ids
        if turn_idx < len(loss_mask) and not loss_mask[turn_idx]:
            labels += [-100] * len(response_ids)
        else:
            labels += response_ids

    if not token_ids:
        return None

    # Append EOS if template uses efficient_eos
    if template and template.efficient_eos:
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            token_ids.append(eos_id)
            labels.append(eos_id)

    # Label shift: labels[i] = labels[i+1] for causal LM
    if config.label_shift:
        labels = labels[1:] + [-100]

    # Truncation
    if len(token_ids) > config.max_seq_len:
        if config.truncation == "right":
            token_ids = token_ids[: config.max_seq_len]
            labels = labels[: config.max_seq_len]
        elif config.truncation == "left":
            token_ids = token_ids[-config.max_seq_len :]
            labels = labels[-config.max_seq_len :]
        else:
            return None  # skip samples that exceed max_seq_len

    seq_len = len(token_ids)
    return EncodedSample(input_ids=token_ids, labels=labels, seq_len=seq_len)


def encode_pt(
    example: Dict[str, Any],
    tokenizer: Any,
    config: EncodeConfig,
) -> Optional[EncodedSample]:
    """Encode a single pretrain sample: plain text → token_ids with full loss.

    Matches the old pipeline's _process_pretraining_sequence behavior:
    1. Extract messages[0]["content"]
    2. Tokenize + append EOS
    3. Truncate to max_seq_len + 1
    4. input_ids = tokens[:-1], labels = tokens[1:]
    """
    messages = example.get("messages")
    if not messages:
        return None

    content = messages[0].get("content", "")
    if not content:
        return None

    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))
    tokens = tokens + [tokenizer.eos_token_id]

    if len(tokens) > config.max_seq_len + 1:
        tokens = tokens[: config.max_seq_len + 1]

    input_ids = tokens[:-1]
    labels = tokens[1:]

    return EncodedSample(input_ids=input_ids, labels=labels, seq_len=len(input_ids))


def _extract_loss_mask(messages: List[Dict[str, Any]]) -> List[bool]:
    """Extract per-assistant-turn loss mask from messages.

    Returns a list of booleans, one per assistant turn, indicating
    whether that turn should contribute to the loss.
    """
    mask: List[bool] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            mask.append(msg.get("loss", True))
    return mask


# ---------------------------------------------------------------------------
# VL (Vision-Language) encoding
# ---------------------------------------------------------------------------


@dataclass
class VLEncodedSample:
    """Output of encode_vl_sft. Extends EncodedSample with vision fields."""

    input_ids: List[int]
    labels: List[int]
    seq_len: int
    mm_inputs: Dict[str, Any]


def encode_vl_sft(
    example: Dict[str, Any],
    tokenizer: Any,
    template: Optional[TemplateMeta],
    config: EncodeConfig,
    processor: Any,
    mm_plugin: Any,
) -> Optional[VLEncodedSample]:
    """Encode a VL-SFT sample: process images + messages → token_ids + labels + mm_inputs.

    Args:
        example: Dict with 'messages' and 'images' keys.
        tokenizer: Tokenizer instance.
        template: TemplateMeta for encoding. If None, uses jinja path.
        config: EncodeConfig.
        processor: AutoProcessor (contains image_processor).
        mm_plugin: Multimodal plugin instance (from datasets/template/mm_plugin).

    Returns:
        VLEncodedSample or None if the sample is invalid.
    """
    messages = example.get("messages")
    if not messages or len(messages) < 2:
        return None

    images = example.get("images", [])
    if not images:
        return None

    # Normalize HF Image feature dicts back to path strings
    images = [img["path"] if isinstance(img, dict) else img for img in images]

    mm_inputs = mm_plugin.get_mm_inputs(images=images, videos=[], audios=[], processor=processor)

    messages = mm_plugin.process_messages(
        messages=messages,
        images=images,
        videos=[],
        audios=[],
        mm_inputs=mm_inputs,
        processor=processor,
    )

    loss_mask = _extract_loss_mask(messages)

    if template is not None:
        pairs = encode_multiturn(template, tokenizer, messages)
    else:
        pairs = encode_multiturn_jinja(tokenizer, messages)

    if not pairs:
        return None

    token_ids: List[int] = []
    labels: List[int] = []

    for turn_idx, (prompt_ids, response_ids) in enumerate(pairs):
        token_ids += prompt_ids
        labels += [-100] * len(prompt_ids)

        token_ids += response_ids
        if turn_idx < len(loss_mask) and not loss_mask[turn_idx]:
            labels += [-100] * len(response_ids)
        else:
            labels += response_ids

    if not token_ids:
        return None

    if template and template.efficient_eos:
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            token_ids.append(eos_id)
            labels.append(eos_id)

    labels = mm_plugin.process_tokens(token_ids, processor)

    if config.label_shift:
        labels = labels[1:] + [-100]

    if len(token_ids) > config.max_seq_len:
        if config.truncation == "right":
            token_ids = token_ids[: config.max_seq_len]
            labels = labels[: config.max_seq_len]
        elif config.truncation == "left":
            token_ids = token_ids[-config.max_seq_len :]
            labels = labels[-config.max_seq_len :]
        else:
            return None

    seq_len = len(token_ids)
    return VLEncodedSample(input_ids=token_ids, labels=labels, seq_len=seq_len, mm_inputs=mm_inputs)
