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

"""Single-sample encoding: messages → input_ids + labels."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .template import (
    TemplateMeta,
    encode_multiturn,
    encode_multiturn_jinja,
    encode_multiturn_reasoning,
)

logger = logging.getLogger(__name__)


@dataclass
class EncodeConfig:
    """Configuration for SFT encoding."""

    max_seq_len: int = 4096
    truncation: str = "right"  # "right" | "left" | "oral" | "delete"
    label_shift: bool = True
    auto_add_bos: bool = False
    placeholder_tokens: List[int] = field(default_factory=list)


@dataclass
class EncodedSample:
    """Output of encode_sft / encode_pt."""

    input_ids: List[int]
    labels: List[int]
    seq_len: int
    position_ids: List[int] = field(default_factory=list)


@dataclass
class VLEncodedSample:
    """Output of encode_vl_sft."""

    input_ids: List[int]
    labels: List[int]
    seq_len: int
    mm_inputs: Dict[str, Any]
    position_ids: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ERNIE_THINK_MODE_SYSTEM = "<global_setting>\nthink_mode=True\n</global_setting>"


def _inject_ernie_think_system(messages: List[Dict[str, Any]], template: TemplateMeta) -> List[Dict[str, Any]]:
    if not template.name.startswith("ernie"):
        return messages
    if template.enable_thinking is None:
        return messages
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": _ERNIE_THINK_MODE_SYSTEM}] + messages


def _extract_loss_mask(messages: List[Dict[str, Any]]) -> List[bool]:
    mask: List[bool] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            mask.append(msg.get("loss", True))
    return mask


def _dispatch_encode(
    template: Optional[TemplateMeta],
    tokenizer: Any,
    messages: List[Dict[str, Any]],
) -> List[Tuple[List[int], List[int]]]:
    if template is not None:
        if template.enable_thinking is not None:
            return encode_multiturn_reasoning(template, tokenizer, messages)
        else:
            return encode_multiturn(template, tokenizer, messages)
    else:
        return encode_multiturn_jinja(tokenizer, messages)


def _get_sep_token_len(template: Optional[TemplateMeta], tokenizer: Any) -> int:
    if template is not None and template.chat_sep:
        return len(tokenizer.encode(template.chat_sep, add_special_tokens=False))
    return 0


def _flatten_turns(
    pairs: List[Tuple[List[int], List[int]]],
    loss_mask: List[bool],
    sep_token_len: int,
) -> Tuple[List[int], List[int]]:
    """Flatten (prompt_ids, response_ids) pairs into token_ids + labels."""
    token_ids: List[int] = []
    labels: List[int] = []
    num_pairs = len(pairs)

    for turn_idx, (prompt_ids, response_ids) in enumerate(pairs):
        token_ids += prompt_ids
        labels += [-100] * len(prompt_ids)

        token_ids += response_ids
        if turn_idx < len(loss_mask) and not loss_mask[turn_idx]:
            labels += [-100] * len(response_ids)
        else:
            if sep_token_len > 0 and turn_idx != (num_pairs - 1) and len(response_ids) > sep_token_len:
                labels += response_ids[: len(response_ids) - sep_token_len] + [-100] * sep_token_len
            else:
                labels += response_ids

    return token_ids, labels


def _add_dynamic_eos(input_ids: List[int], labels: List[int], suffix_tokens_id: List[int]) -> None:
    suffix_len = len(suffix_tokens_id)
    if suffix_len == 0:
        return

    start = 0
    for i in range(1, len(labels) + 1):
        if labels[i - 1] >= 0 and i < len(labels) and labels[i] == -100:
            start = i
        elif start > 0 and labels[i - 1] == -100 and (i == len(labels) or labels[i] >= 0):
            length = i - start
            if length >= suffix_len and input_ids[start : start + suffix_len] == suffix_tokens_id:
                labels[start : start + suffix_len] = suffix_tokens_id


def _apply_dynamic_eos(token_ids: List[int], labels: List[int], template: Optional[TemplateMeta], tokenizer: Any):
    if template is not None and template.suffix:
        suffix_ids = tokenizer.encode(template.suffix[-1], add_special_tokens=False)
        _add_dynamic_eos(token_ids, labels, suffix_ids)


def _apply_efficient_eos(token_ids: List[int], labels: List[int], template: Optional[TemplateMeta], tokenizer: Any):
    if template and template.efficient_eos:
        if template.suffix:
            suffix_ids = tokenizer.encode(template.suffix[-1], add_special_tokens=False)
            token_ids.extend(suffix_ids)
            labels.extend(suffix_ids)
        elif tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)


def _apply_label_shift(labels: List[int], config: EncodeConfig) -> List[int]:
    if config.label_shift:
        return labels[1:] + [-100]
    return labels


def _truncate_placeholder_aware(
    input_ids: List[int],
    labels: List[int],
    max_seq_len: int,
    placeholder_set: Set[int],
    strategy: str = "right",
) -> Tuple[List[int], List[int]]:
    is_placeholder = [tok in placeholder_set for tok in input_ids]
    placeholder_idx = [i for i, v in enumerate(is_placeholder) if v]

    if len(placeholder_idx) >= max_seq_len:
        keep_idx = placeholder_idx[:max_seq_len]
    else:
        remain = max_seq_len - len(placeholder_idx)
        non_placeholder_idx = [i for i, v in enumerate(is_placeholder) if not v]

        if strategy == "left":
            extra_idx = non_placeholder_idx[-remain:]
        else:
            extra_idx = non_placeholder_idx[:remain]

        keep_idx = sorted(placeholder_idx + extra_idx)

    input_ids = [input_ids[i] for i in keep_idx]
    labels = [labels[i] for i in keep_idx]
    return input_ids, labels


def _apply_truncation(
    token_ids: List[int],
    labels: List[int],
    config: EncodeConfig,
) -> Optional[Tuple[List[int], List[int]]]:
    """Apply truncation. Returns None if strategy is 'delete'."""
    if len(token_ids) <= config.max_seq_len:
        return token_ids, labels

    if config.truncation == "delete":
        return None
    elif config.placeholder_tokens:
        return _truncate_placeholder_aware(
            token_ids, labels, config.max_seq_len, set(config.placeholder_tokens), config.truncation
        )
    elif config.truncation == "left":
        return token_ids[-config.max_seq_len :], labels[-config.max_seq_len :]
    else:
        return token_ids[: config.max_seq_len], labels[: config.max_seq_len]


def _apply_auto_bos(
    token_ids: List[int],
    labels: List[int],
    config: EncodeConfig,
    tokenizer: Any,
) -> Tuple[List[int], List[int]]:
    if config.auto_add_bos and tokenizer.bos_token_id is not None:
        if not token_ids or token_ids[0] != tokenizer.bos_token_id:
            token_ids = [tokenizer.bos_token_id] + token_ids
            labels = [-100] + labels
            if len(token_ids) > config.max_seq_len:
                token_ids = token_ids[: config.max_seq_len]
                labels = labels[: config.max_seq_len]
    return token_ids, labels


def _validate_and_build(
    token_ids: List[int],
    labels: List[int],
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    """Validate and return (token_ids, labels, position_ids) or None."""
    if not token_ids:
        return None
    if all(x == -100 for x in labels):
        logger.warning("[SKIP] all labels set to -100")
        return None
    position_ids = list(range(len(token_ids)))
    return token_ids, labels, position_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode_sft(
    example: Dict[str, Any],
    tokenizer: Any,
    template: Optional[TemplateMeta],
    config: EncodeConfig,
) -> Optional[EncodedSample]:
    """Encode a single SFT sample: messages → token_ids + labels."""
    messages = example.get("messages")
    if not messages or len(messages) < 2:
        return None

    if template is not None:
        messages = _inject_ernie_think_system(messages, template)

    loss_mask = _extract_loss_mask(messages)
    pairs = _dispatch_encode(template, tokenizer, messages)
    if not pairs:
        return None

    if config.truncation == "oral":
        return _encode_oral_truncation(pairs, loss_mask, tokenizer, template, config)

    sep_token_len = _get_sep_token_len(template, tokenizer)
    token_ids, labels = _flatten_turns(pairs, loss_mask, sep_token_len)
    if not token_ids:
        return None

    _apply_dynamic_eos(token_ids, labels, template, tokenizer)
    _apply_efficient_eos(token_ids, labels, template, tokenizer)
    labels = _apply_label_shift(labels, config)

    result = _apply_truncation(token_ids, labels, config)
    if result is None:
        return None
    token_ids, labels = result

    token_ids, labels = _apply_auto_bos(token_ids, labels, config, tokenizer)

    validated = _validate_and_build(token_ids, labels)
    if validated is None:
        return None
    token_ids, labels, position_ids = validated
    return EncodedSample(input_ids=token_ids, labels=labels, seq_len=len(token_ids), position_ids=position_ids)


def _encode_oral_truncation(
    pairs: List[Tuple[List[int], List[int]]],
    loss_mask: List[bool],
    tokenizer: Any,
    template: Optional[TemplateMeta],
    config: EncodeConfig,
) -> Optional[EncodedSample]:
    """Oral truncation: process turns in reverse order, newest first."""
    max_seq_len = config.max_seq_len
    sep_token_len = _get_sep_token_len(template, tokenizer)

    num_pairs = len(pairs)
    tokens_chunks: List[List[int]] = []
    labels_chunks: List[List[int]] = []
    cur_len = 0

    for turn_index in range(num_pairs - 1, -1, -1):
        prompt_ids, response_ids = pairs[turn_index]

        if len(response_ids) == 0:
            logger.warning("[SKIP] The length of encoded assistant tokens is 0")
            return None

        remaining_len = max_seq_len - cur_len
        if len(prompt_ids) + len(response_ids) > remaining_len:
            if len(prompt_ids) > remaining_len:
                break
            else:
                response_ids = response_ids[: remaining_len - len(prompt_ids)]

        labels_src = [-100] * len(prompt_ids)

        if turn_index < len(loss_mask) and not loss_mask[turn_index]:
            labels_target = [-100] * len(response_ids)
        else:
            if sep_token_len > 0 and turn_index != (num_pairs - 1) and len(response_ids) > sep_token_len:
                labels_target = list(response_ids[: len(response_ids) - sep_token_len]) + [-100] * sep_token_len
            else:
                labels_target = list(response_ids)

        tokens_chunks.append(list(prompt_ids) + list(response_ids))
        labels_chunks.append(labels_src + labels_target)
        cur_len += len(prompt_ids) + len(response_ids)

    tokens_chunks.reverse()
    labels_chunks.reverse()

    token_ids: List[int] = []
    labels: List[int] = []
    for tc in tokens_chunks:
        token_ids.extend(tc)
    for lc in labels_chunks:
        labels.extend(lc)

    if not token_ids:
        return None

    _apply_dynamic_eos(token_ids, labels, template, tokenizer)
    token_ids, labels = _apply_auto_bos(token_ids, labels, config, tokenizer)
    _apply_efficient_eos(token_ids, labels, template, tokenizer)

    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    labels = _apply_label_shift(labels, config)

    validated = _validate_and_build(token_ids, labels)
    if validated is None:
        return None
    token_ids, labels, position_ids = validated
    return EncodedSample(input_ids=token_ids, labels=labels, seq_len=len(token_ids), position_ids=position_ids)


def encode_vl_sft(
    example: Dict[str, Any],
    tokenizer: Any,
    template: Optional[TemplateMeta],
    config: EncodeConfig,
    processor: Any,
    mm_plugin: Any,
) -> Optional[VLEncodedSample]:
    """Encode a VL-SFT sample: process images + messages → token_ids + labels + mm_inputs."""
    messages = example.get("messages")
    if not messages or len(messages) < 2:
        return None

    images = example.get("images", [])
    if not images:
        return None

    images = [img["path"] if isinstance(img, dict) else img for img in images]

    mm_inputs = mm_plugin.get_mm_inputs(images=images, videos=[], audios=[], processor=processor)
    messages = mm_plugin.process_messages(
        messages=messages, images=images, videos=[], audios=[], mm_inputs=mm_inputs, processor=processor
    )

    if template is not None:
        messages = _inject_ernie_think_system(messages, template)

    loss_mask = _extract_loss_mask(messages)
    pairs = _dispatch_encode(template, tokenizer, messages)
    if not pairs:
        return None

    sep_token_len = _get_sep_token_len(template, tokenizer)
    token_ids, labels = _flatten_turns(pairs, loss_mask, sep_token_len)
    if not token_ids:
        return None

    _apply_dynamic_eos(token_ids, labels, template, tokenizer)
    _apply_efficient_eos(token_ids, labels, template, tokenizer)

    # VL-specific: mm_plugin overrides labels for image/video placeholder regions
    labels = mm_plugin.process_tokens(token_ids, processor)

    labels = _apply_label_shift(labels, config)

    # VL: oral not supported
    if len(token_ids) > config.max_seq_len:
        if config.truncation == "delete" or config.truncation == "oral":
            logger.warning("[SKIP] VL data too long, discarding")
            return None
        elif config.placeholder_tokens:
            token_ids, labels = _truncate_placeholder_aware(
                token_ids, labels, config.max_seq_len, set(config.placeholder_tokens), config.truncation
            )
        elif config.truncation == "left":
            token_ids = token_ids[-config.max_seq_len :]
            labels = labels[-config.max_seq_len :]
        else:
            token_ids = token_ids[: config.max_seq_len]
            labels = labels[: config.max_seq_len]

    token_ids, labels = _apply_auto_bos(token_ids, labels, config, tokenizer)

    position_ids = list(range(len(token_ids)))
    seq_len = len(token_ids)
    return VLEncodedSample(
        input_ids=token_ids, labels=labels, seq_len=seq_len, mm_inputs=mm_inputs, position_ids=position_ids
    )


def encode_pt(
    example: Dict[str, Any],
    tokenizer: Any,
    config: EncodeConfig,
) -> Optional[EncodedSample]:
    """Encode a single pretrain sample: plain text → token_ids with full loss."""
    messages = example.get("messages")
    if not messages:
        return None

    content = messages[0].get("content", "")
    if not content:
        return None

    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))
    if tokenizer.eos_token_id is not None:
        tokens = tokens + [tokenizer.eos_token_id]

    if len(tokens) < 2:
        return None

    if len(tokens) > config.max_seq_len + 1:
        tokens = tokens[: config.max_seq_len + 1]

    input_ids = tokens[:-1]
    labels = tokens[1:]
    position_ids = list(range(len(input_ids)))

    return EncodedSample(input_ids=input_ids, labels=labels, seq_len=len(input_ids), position_ids=position_ids)
