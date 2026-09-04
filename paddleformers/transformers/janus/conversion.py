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

import gc
from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any

import numpy as np
import paddle

from ..model_utils import load_state_dict

LANGUAGE_PREFIX = "language_model."
VISION_PREFIX = "vision_model."
ALIGNER_PREFIX = "aligner."
JANUS_PREFIXES = (LANGUAGE_PREFIX, VISION_PREFIX, ALIGNER_PREFIX)
EXPECTED_JANUS_PRO_7B_LANGUAGE_KEYS = 273
PROJECTION_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
VISION_LINEAR_SUFFIXES = (
    ".qkv.weight",
    ".q.weight",
    ".kv.weight",
    ".proj.weight",
    ".fc1.weight",
    ".fc2.weight",
)


def expected_language_keys(target_state_dict: Mapping[str, Any]) -> set[str]:
    """Return the language parameters expected by the Paddle wrapper."""

    return {key for key in target_state_dict if key.startswith(LANGUAGE_PREFIX)}


def expected_multimodal_keys(target_state_dict: Mapping[str, Any]) -> set[str]:
    """Return all language, vision, and aligner keys in a Janus target."""

    return {key for key in target_state_dict if key.startswith(JANUS_PREFIXES)}


def _transpose_2d(value):
    if isinstance(value, paddle.Tensor):
        return value.transpose([1, 0]).contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value.transpose([1, 0]))
    raise TypeError(f"unsupported tensor type {type(value).__name__}")


def _format_key_error(
    *,
    missing: Iterable[str] = (),
    duplicates: Iterable[str] = (),
    unexpected: Iterable[str] = (),
    shape_mismatches: Iterable[str] = (),
) -> str:
    details = []
    if missing:
        details.append("missing language keys: " + ", ".join(sorted(missing)))
    if duplicates:
        details.append("duplicate language keys: " + ", ".join(sorted(duplicates)))
    if unexpected:
        details.append("unexpected language keys: " + ", ".join(sorted(unexpected)))
    if shape_mismatches:
        details.append("shape mismatches: " + "; ".join(sorted(shape_mismatches)))
    return "; ".join(details)


def merge_shard_state_dicts(shards: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge checkpoint dictionaries while rejecting duplicate source keys."""

    merged = {}
    duplicates = set()
    for shard in shards:
        for key, value in shard.items():
            if key in merged:
                duplicates.add(key)
                continue
            merged[key] = value

    if duplicates:
        duplicate_language_keys = {key for key in duplicates if key.startswith(LANGUAGE_PREFIX)}
        if duplicate_language_keys:
            raise ValueError(_format_key_error(duplicates=duplicate_language_keys))
        raise ValueError("duplicate checkpoint keys: " + ", ".join(sorted(duplicates)))
    return merged


def convert_language_state_dict(
    source_state_dict: Mapping[str, Any], target_state_dict: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Filter and transpose a Janus language state dict for Paddle."""

    target_keys = expected_language_keys(target_state_dict)
    accepted = {}
    skipped = []
    unexpected = []
    shape_mismatches = []

    for key, value in source_state_dict.items():
        if not key.startswith(LANGUAGE_PREFIX):
            skipped.append(key)
            continue
        if key not in target_keys:
            unexpected.append(key)
            continue

        converted = value
        if value.ndim == 2 and any(key.endswith(f".{name}.weight") for name in PROJECTION_NAMES):
            converted = _transpose_2d(value)

        converted_shape = tuple(converted.shape)
        target_shape = tuple(target_state_dict[key].shape)
        if converted_shape != target_shape:
            shape_mismatches.append(f"{key}: got {converted_shape}, expected {target_shape}")
            continue
        accepted[key] = converted

    missing = target_keys - set(accepted)
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            _format_key_error(
                missing=missing,
                unexpected=unexpected,
                shape_mismatches=shape_mismatches,
            )
        )

    report = {
        "accepted_language_keys": len(accepted),
        "skipped_non_language_keys": len(skipped),
        "skipped_keys": sorted(skipped),
    }
    return accepted, report


def _needs_multimodal_transpose(key: str, value: Any) -> bool:
    if getattr(value, "ndim", None) != 2:
        return False
    if key.startswith(LANGUAGE_PREFIX):
        return any(key.endswith(f".{name}.weight") for name in PROJECTION_NAMES)
    if key.startswith(VISION_PREFIX):
        return key.endswith(VISION_LINEAR_SUFFIXES)
    if key.startswith(ALIGNER_PREFIX):
        return key.endswith(".weight")
    return False


def convert_janus_state_dict(
    source_state_dict: Mapping[str, Any], target_state_dict: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert the complete Janus understanding path.

    The generator-only tensors remain outside the target model and are listed
    explicitly in the report.  Any missing or unexpected understanding-path
    key is fatal, so a partial vision conversion cannot be published.
    """

    target_keys = expected_multimodal_keys(target_state_dict)
    accepted = {}
    skipped = []
    unexpected = []
    shape_mismatches = []

    for key, value in source_state_dict.items():
        if not key.startswith(JANUS_PREFIXES):
            skipped.append(key)
            continue
        if key not in target_keys:
            unexpected.append(key)
            continue

        converted = _transpose_2d(value) if _needs_multimodal_transpose(key, value) else value
        converted_shape = tuple(converted.shape)
        target_shape = tuple(target_state_dict[key].shape)
        if converted_shape != target_shape:
            shape_mismatches.append(f"{key}: got {converted_shape}, expected {target_shape}")
            continue
        accepted[key] = converted

    missing = target_keys - set(accepted)
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            _format_key_error(
                missing=missing,
                unexpected=unexpected,
                shape_mismatches=shape_mismatches,
            )
        )

    report = {
        "accepted_janus_keys": len(accepted),
        "accepted_language_keys": sum(key.startswith(LANGUAGE_PREFIX) for key in accepted),
        "accepted_vision_keys": sum(key.startswith(VISION_PREFIX) for key in accepted),
        "accepted_aligner_keys": sum(key.startswith(ALIGNER_PREFIX) for key in accepted),
        "skipped_non_janus_keys": len(skipped),
        "skipped_keys": sorted(skipped),
    }
    return accepted, report


def load_and_convert_shards(
    shard_paths: Iterable[str | PathLike[str]],
    target_state_dict: Mapping[str, Any],
    multimodal: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load PyTorch shards sequentially and convert the requested Janus path."""

    janus_state_dict = {}
    seen_keys = set()
    skipped_keys = []
    source_shards = []

    for shard_path in shard_paths:
        shard_path = str(shard_path)
        source_shards.append(shard_path)
        shard = load_state_dict(shard_path, convert_from_hf=True, transpose_weight_keys=None)
        duplicate_keys = seen_keys.intersection(shard)
        if duplicate_keys:
            duplicate_janus_keys = {key for key in duplicate_keys if key.startswith(JANUS_PREFIXES)}
            if duplicate_janus_keys:
                raise ValueError(_format_key_error(duplicates=duplicate_janus_keys))
            raise ValueError("duplicate checkpoint keys: " + ", ".join(sorted(duplicate_keys)))

        seen_keys.update(shard)
        accepted_prefixes = JANUS_PREFIXES if multimodal else (LANGUAGE_PREFIX,)
        for key, value in shard.items():
            if key.startswith(accepted_prefixes):
                janus_state_dict[key] = value
            else:
                skipped_keys.append(key)
        del shard
        gc.collect()

    converter = convert_janus_state_dict if multimodal else convert_language_state_dict
    converted, report = converter(janus_state_dict, target_state_dict)
    generator_keys = [key for key in skipped_keys if not key.startswith(JANUS_PREFIXES)]
    report.update(
        {
            "skipped_non_language_keys": len(skipped_keys),
            "skipped_non_janus_keys": len(skipped_keys),
            "skipped_generator_keys": len(generator_keys),
            "generator_keys": sorted(generator_keys),
            "skipped_keys": sorted(skipped_keys),
            "source_shards": source_shards,
            "source_checkpoint_keys": len(seen_keys),
        }
    )
    return converted, report


__all__ = [
    "EXPECTED_JANUS_PRO_7B_LANGUAGE_KEYS",
    "ALIGNER_PREFIX",
    "JANUS_PREFIXES",
    "LANGUAGE_PREFIX",
    "PROJECTION_NAMES",
    "VISION_LINEAR_SUFFIXES",
    "VISION_PREFIX",
    "convert_janus_state_dict",
    "convert_language_state_dict",
    "expected_language_keys",
    "expected_multimodal_keys",
    "load_and_convert_shards",
    "merge_shard_state_dicts",
]
