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
"""Canonical spelling of the ``use_accuracy_compatible`` accuracy target.

The field selects which reference the accuracy-compatible kernels reproduce
bit-for-bit: ``False`` keeps the throughput kernels, ``"megatron"`` (historically
spelled ``True``) aligns with Megatron-LM and ``"hf"`` aligns with the
HuggingFace/Torch reference.

The value arrives from YAML, a CLI flag, a dataclass default or a deserialized
``config.json``, and each of those layers spells "off" differently -- a YAML
``false`` can reach Python as the *string* ``"false"``, which is truthy. Roughly
a dozen call sites test the field for truthiness, so a non-canonical spelling
silently enables a different set of kernels. Everything that writes the field
onto a config therefore normalizes it here first; see
``LlmMetaConfig.set_llm_config``.

This mirrors ``paddlefleet.accuracy_target`` and must stay in sync with it. It is
deliberately *not* imported from there: PaddleFleet is an optional dependency and
these models must keep working without it, the same reason
``utils.hf_bitexact_clip`` and ``utils.optimizer`` carry their own ``"hf"``
literal. The accepted spellings are pinned by
``tests/utils/test_accuracy_target.py``, so a divergence shows up as a test
failure rather than as a silent change of kernel.
"""

from __future__ import annotations

from typing import Union

__all__ = [
    "ACCURACY_TARGETS",
    "ACCURACY_TARGET_HF",
    "ACCURACY_TARGET_MEGATRON",
    "AccuracyTarget",
    "normalize_accuracy_target",
    "targets_hf",
]

#: Bit-reproducible against Megatron-LM; the meaning of a bare ``True``.
ACCURACY_TARGET_MEGATRON = "megatron"

#: Bit-reproducible against the HuggingFace/Torch reference implementation.
ACCURACY_TARGET_HF = "hf"

ACCURACY_TARGETS = (ACCURACY_TARGET_MEGATRON, ACCURACY_TARGET_HF)

#: ``False`` (default kernels) or one of :data:`ACCURACY_TARGETS`.
AccuracyTarget = Union[bool, str]

#: Spellings of "off" and "on" that config layers produce when they stringify a
#: boolean. YAML, argparse and env plumbing all do this somewhere, and a value
#: that survived as ``"True"`` must not be mistaken for an unknown target.
_FALSE_WORDS = frozenset({"false", "0", "no", "off", "none", "null"})
_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})


def normalize_accuracy_target(value: AccuracyTarget) -> AccuracyTarget:
    """Canonicalize a ``use_accuracy_compatible`` value.

    ``True`` becomes ``"megatron"`` so the stored value always names its
    reference; every falsy input becomes a real ``False`` so the downstream
    truthiness tests are correct. Raises on an unknown target rather than
    silently degrading to the default kernels, which would turn a typo into a
    run that looks aligned but is not.
    """
    if not value:
        # Covers False, None, "" and 0 -- the YAML, CLI and dataclass layers each
        # produce a different spelling of "off" and they must not diverge.
        return False
    if value is True or value == 1:
        # ``True`` predates the "hf" target; a YAML scalar ``1`` means the same.
        return ACCURACY_TARGET_MEGATRON
    if isinstance(value, str):
        target = value.strip().lower()
        if target in ACCURACY_TARGETS:
            return target
        if target in _TRUE_WORDS:
            return ACCURACY_TARGET_MEGATRON
        if target in _FALSE_WORDS:
            return False
        raise ValueError(
            f"use_accuracy_compatible must be False, True, or one of " f"{list(ACCURACY_TARGETS)}; got {value!r}."
        )
    raise TypeError(f"use_accuracy_compatible must be a bool or str, got {type(value).__name__}.")


def targets_hf(value: AccuracyTarget) -> bool:
    """Whether ``value`` selects the HuggingFace/Torch reference."""
    return value == ACCURACY_TARGET_HF
