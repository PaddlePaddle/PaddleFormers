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

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import paddle
from paddle.distributed.flex_checkpoint.dcp.metadata import (
    LocalTensorMetadata,
    Metadata,
)
from paddle.distributed.flex_checkpoint.dcp.utils import create_hf_ckpt_metadata

from .checkpoint_dequant import CheckpointDequantizer, get_checkpoint_dequantizer

_PADDLE_METADATA_FILE_NAME = "flex-ckpt.auto_generated.metadata"
_HF_CONFIG_FILE_NAME = "config.json"
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Quantized checkpoint metadata model
#
# A descriptor defines shared ``HFQuantizationGroupSpec`` objects.  Each
# matched logical weight gets one ``HFQuantizedWeightSpec`` relation pointing
# to a group and listing its physical qweight/scale components.  ``QuanMetadata``
# keeps these relations together with both logical (DCP/AOA-facing) and
# physical (checkpoint-facing) metadata.
# -----------------------------------------------------------------------------


@dataclass
class HFQuantizationGroupSpec:
    """Shared quantization rules for all weights matched by one descriptor group."""

    name: str
    targets: tuple[re.Pattern[str], ...]
    quant_method: str
    value_format: str
    scale_format: str
    block_shape: tuple[int, ...]
    dequantizer: CheckpointDequantizer
    block_axes: tuple[int, ...] | None = None

    def configure_geometry(self, block_axes: tuple[int, ...]) -> None:
        self.block_axes = tuple(block_axes)
        self.dequantizer = self.dequantizer.configure_geometry(self.block_axes, self.block_shape)


@dataclass
class HFQuantizedWeightSpec:
    """Relation from one logical weight to its physical checkpoint components."""

    group_name: str
    logical_name: str
    logical_shape: tuple[int, ...]
    components: dict[str, str]


@dataclass
class QuanMetadata:
    """Unified metadata for a quantized HF checkpoint.

    ``groups`` stores shared quantization rules, ``relations`` maps each
    logical weight to its qweight/scale sources, and ``logical_metadata``
    exposes those virtual weights to DCP/AOA.  ``physical_metadata`` describes
    the tensors that actually exist in the HF safetensors checkpoint.
    """

    # Shared descriptor configuration: group name -> quantization rule.
    groups: dict[str, HFQuantizationGroupSpec]
    # Logical HF name -> physical qweight/scale relation.
    relations: dict[str, HFQuantizedWeightSpec]
    # Logical HF name -> virtual tensor metadata exposed to DCP/AOA.
    logical_metadata: dict[str, LocalTensorMetadata]
    # Physical checkpoint name -> safetensors storage metadata.
    physical_metadata: Metadata

    def __post_init__(self) -> None:
        for logical_key, spec in self.relations.items():
            group = self.groups.get(spec.group_name)
            if group is None:
                raise ValueError(f"Quantized weight {logical_key!r} refers to unknown group {spec.group_name!r}.")
            self._validate_relation(logical_key, spec, group)

    @staticmethod
    def _validate_relation(
        logical_key: str,
        spec: HFQuantizedWeightSpec,
        group: HFQuantizationGroupSpec,
    ) -> None:
        if spec.logical_name != logical_key:
            raise ValueError(f"Relation key {logical_key!r} does not match logical_name {spec.logical_name!r}.")
        if not spec.logical_shape or any(not isinstance(dim, int) or dim <= 0 for dim in spec.logical_shape):
            raise ValueError(f"Invalid logical shape for {logical_key!r}: {spec.logical_shape}.")
        if "qweight" not in spec.components:
            raise ValueError(f"Quantized weight {logical_key!r} does not define a qweight component.")
        if any(not role or not source_key for role, source_key in spec.components.items()):
            raise ValueError(f"Quantized weight {logical_key!r} contains an empty component role or source key.")
        if group.block_axes is None:
            raise ValueError(f"Quantization group {group.name!r} does not define block_axes.")
        if any(axis < 0 or axis >= len(spec.logical_shape) for axis in group.block_axes):
            raise ValueError(f"Block axes for group {group.name!r} are outside logical shape {spec.logical_shape}.")


# -----------------------------------------------------------------------------
# Checkpoint metadata helpers
# -----------------------------------------------------------------------------


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path!r}.")
    return value


def _physical_state_dict_metadata(
    physical_metadata: Metadata,
) -> dict[str, list[LocalTensorMetadata]]:
    state_dict_metadata = physical_metadata.state_dict_metadata
    if not isinstance(state_dict_metadata, dict) or not state_dict_metadata:
        raise ValueError("Paddle checkpoint Metadata contains no physical tensors.")
    return state_dict_metadata


def _physical_tensor_metadata(
    physical_metadata: Metadata,
    tensor_key: str,
) -> LocalTensorMetadata:
    tensor_items = _physical_state_dict_metadata(physical_metadata).get(tensor_key)
    if not tensor_items:
        raise KeyError(f"Physical tensor {tensor_key!r} does not exist in Paddle Metadata.")
    return tensor_items[0]


def _load_paddle_hf_metadata(checkpoint_path: str) -> Metadata:
    """Load or create Paddle physical metadata for an HF checkpoint."""
    metadata_path = os.path.join(checkpoint_path, _PADDLE_METADATA_FILE_NAME)
    if os.path.isfile(metadata_path):
        return paddle.load(metadata_path)
    return create_hf_ckpt_metadata(checkpoint_path)


def _expected_scale_shape(
    logical_shape: tuple[int, ...],
    block_axes: tuple[int, ...],
    block_shape: tuple[int, ...],
) -> tuple[int, ...]:
    axis_to_block = dict(zip(block_axes, block_shape))
    return tuple(
        math.ceil(dim / axis_to_block[axis]) if axis in axis_to_block else dim
        for axis, dim in enumerate(logical_shape)
    )


# -----------------------------------------------------------------------------
# Quantization descriptor validation and metadata construction
# -----------------------------------------------------------------------------


class QuanDescriptor:
    """Validate a quantization descriptor and build logical/physical relations.

    Descriptor parsing is intentionally separate from checkpoint loading:
    parsing creates shared group rules, while ``build_metadata`` binds those
    rules to the physical tensor names and shapes found in the checkpoint.
    """

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self._validate_descriptor()

    @classmethod
    def from_file(cls, path: str) -> "QuanDescriptor":
        return cls(_read_json(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QuanDescriptor":
        return cls(raw)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def _validate_descriptor(self) -> None:
        if not isinstance(self.raw, dict):
            raise ValueError("Quantization descriptor must contain a JSON object.")
        if self.raw.get("schema_version") != 1:
            raise ValueError("Unsupported quan_desc schema_version; expected 1.")
        raw_groups = self.raw.get("groups")
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError("Quantization descriptor must define a non-empty groups list.")
        pairing = self.raw.get("component_pairing")
        if not isinstance(pairing, dict):
            raise ValueError("quan_desc must define component_pairing as an object.")
        self.weight_suffix = pairing.get("weight_suffix")
        self.scale_suffix = pairing.get("scale_suffix")
        if not isinstance(self.weight_suffix, str) or not isinstance(self.scale_suffix, str):
            raise ValueError("quan_desc component_pairing must define string weight_suffix and scale_suffix.")
        if not self.weight_suffix or not self.scale_suffix:
            raise ValueError("quan_desc component suffixes must be non-empty strings.")
        self.logical_name_suffix = self.raw.get("logic_name_suffix")
        if not isinstance(self.logical_name_suffix, str) or not self.logical_name_suffix:
            raise ValueError("quan_desc must define a non-empty string logic_name_suffix.")
        compiled_groups = tuple(self._compile_group(group) for group in raw_groups)
        group_names = [group.name for group in compiled_groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError(f"quan_desc group names must be unique, got {group_names}.")
        self._groups = compiled_groups

    @classmethod
    def _compile_group(cls, group: dict[str, Any]) -> HFQuantizationGroupSpec:
        if not isinstance(group, dict):
            raise ValueError("Each quan_desc group must be an object.")
        name = group.get("name")
        method = group.get("quant_method")
        value_format = group.get("value_format")
        scale_format = group.get("scale_format")
        block_shape = group.get("block_shape")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each quan_desc group must define a non-empty name.")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"Quan_desc group {name!r} must define quant_method.")
        method = method.strip().lower()
        if not isinstance(value_format, str) or not value_format.strip():
            raise ValueError(f"Quan_desc group {name!r} must define value_format.")
        value_format = value_format.strip().lower()
        if not isinstance(scale_format, str) or not scale_format.strip():
            raise ValueError(f"Quan_desc group {name!r} must define scale_format.")
        scale_format = scale_format.strip().lower()
        try:
            dequantizer = get_checkpoint_dequantizer(method).configure_formats(value_format, scale_format)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid quan_desc formats for group {name!r}: {exc}") from exc
        if (
            not isinstance(block_shape, list)
            or not block_shape
            or any(not isinstance(size, int) or size <= 0 for size in block_shape)
        ):
            raise ValueError(f"Quan_desc group {name!r} must define a positive integer block_shape list.")
        return HFQuantizationGroupSpec(
            name=name,
            targets=cls._descriptor_patterns(group),
            quant_method=method,
            value_format=value_format,
            scale_format=scale_format,
            block_shape=tuple(block_shape),
            dequantizer=dequantizer,
        )

    @staticmethod
    def _descriptor_patterns(group: dict[str, Any]) -> tuple[re.Pattern[str], ...]:
        """Compile one group's ``targets`` entries into match patterns.

        Two forms are accepted, and the ``re:`` form is the preferred one:

        - ``"re:<regex>"`` -- everything after ``re:`` is used as a regular
          expression.  Real checkpoints name weights per layer and per expert,
          so this is the only form able to describe a whole weight family; the
          in-tree DeepSeek-V4 and Kimi-K3 descriptors both use a single anchored
          ``re:`` target.  Anchor these patterns yourself (``^`` / ``$``) to keep
          the match tight, since they are applied with :meth:`re.Pattern.search`.
        - a plain string -- a literal, fully qualified physical tensor name,
          matched in full.  This is only a supplementary form, for the occasional
          one-off weight that carries no layer or expert index.

        Literal targets are anchored with ``\\A`` / ``\\Z`` here.  ``re.escape``
        alone would keep the literal from being read as a regex but would still
        leave ``search`` free to match a substring, so ``layers.1.attn.wq_a.weight``
        would also match ``prefix.layers.1.attn.wq_a.weight`` and a shorter
        ``layers.1`` would match ``layers.11...``, dragging weight/scale pairs the
        descriptor never declared into the transform.  Anchoring makes a literal
        target denote exactly one tensor name.
        """
        raw_targets = group.get("targets")
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        if (
            not isinstance(raw_targets, list)
            or not raw_targets
            or any(not isinstance(item, str) for item in raw_targets)
        ):
            raise ValueError("Each quan_desc group must define a non-empty string targets list.")
        patterns = []
        for target in raw_targets:
            expression = target[3:] if target.startswith("re:") else rf"\A{re.escape(target)}\Z"
            try:
                patterns.append(re.compile(expression))
            except re.error as exc:
                raise ValueError(f"Invalid quan_desc target pattern {target!r}: {exc}") from exc
        return tuple(patterns)

    @staticmethod
    def _infer_block_axes(
        logical_shape: tuple[int, ...],
        scale_shape: tuple[int, ...],
        block_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        if len(logical_shape) != len(scale_shape):
            raise ValueError(f"Cannot infer block axes for ranks {len(logical_shape)} and {len(scale_shape)}.")
        if not block_shape or len(block_shape) > len(logical_shape):
            raise ValueError(f"Block rank {len(block_shape)} is invalid for logical shape {logical_shape}.")
        for axes in permutations(range(len(logical_shape)), len(block_shape)):
            if _expected_scale_shape(logical_shape, axes, block_shape) == scale_shape:
                return axes
        raise ValueError(f"Cannot infer block axes for logical shape {logical_shape} and scale shape {scale_shape}.")

    def build_metadata(
        self,
        physical_metadata: Metadata,
        output_dtype: paddle.dtype = paddle.bfloat16,
    ) -> QuanMetadata:
        source_keys = _physical_state_dict_metadata(physical_metadata)
        relations: dict[str, HFQuantizedWeightSpec] = {}
        groups: dict[str, HFQuantizationGroupSpec] = {}
        matched_scales: set[str] = set()
        for weight_key in sorted(source_keys):
            if not weight_key.endswith(self.weight_suffix):
                continue
            matches = [group for group in self._groups if any(pattern.search(weight_key) for pattern in group.targets)]
            if len(matches) > 1:
                raise ValueError(
                    f"Quantized weight {weight_key!r} matches multiple quan_desc groups: "
                    f"{[group.name for group in matches]}"
                )
            if not matches:
                continue
            group = matches[0]
            name = group.name
            scale_key = weight_key[: -len(self.weight_suffix)] + self.scale_suffix
            if scale_key not in source_keys:
                raise ValueError(
                    f"Quan_desc group {name!r} matched {weight_key!r}, but paired scale {scale_key!r} is missing."
                )
            weight_shape = tuple(_physical_tensor_metadata(physical_metadata, weight_key).global_shape)
            logical_shape = group.dequantizer.logical_shape(weight_shape)
            if group.block_axes is None:
                scale_shape = tuple(_physical_tensor_metadata(physical_metadata, scale_key).global_shape)
                group.configure_geometry(self._infer_block_axes(logical_shape, scale_shape, group.block_shape))
            logical_name = weight_key[: -len(self.weight_suffix)] + self.logical_name_suffix
            if logical_name in relations:
                raise ValueError(f"Quan_desc produced duplicate logical weight name {logical_name!r}.")
            relations[logical_name] = HFQuantizedWeightSpec(
                logical_name=logical_name,
                logical_shape=logical_shape,
                components={"qweight": weight_key, "scale": scale_key},
                group_name=group.name,
            )
            groups[group.name] = group
            matched_scales.add(scale_key)
        unmatched = [
            key
            for key in source_keys
            if key.endswith(self.scale_suffix)
            and key[: -len(self.scale_suffix)] + self.weight_suffix in source_keys
            and key not in matched_scales
        ]
        if unmatched:
            raise ValueError(
                f"quan_desc did not match all quantized weight/scale pairs; unmatched examples: {unmatched[:5]}"
            )
        if not relations:
            raise ValueError("quan_desc matched no quantized weight/scale pairs.")
        dtype = str(output_dtype).split(".")[-1]
        logical_metadata = {
            logical_key: LocalTensorMetadata(
                global_offset=(0,) * len(spec.logical_shape),
                local_shape=spec.logical_shape,
                global_shape=spec.logical_shape,
                dtype=dtype,
            )
            for logical_key, spec in relations.items()
        }
        return QuanMetadata(
            groups=groups,
            relations=relations,
            logical_metadata=logical_metadata,
            physical_metadata=physical_metadata,
        )


# -----------------------------------------------------------------------------
# DCP load transform
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HFReadPlan:
    """Map one logical shard to the physical qweight/scale reads it needs.

    ``mode == "local"`` means the physical components can be read from
    block-aligned slices.  ``mode == "global"`` means the components are read
    in full and the logical output is sliced after dequantization.
    """

    mode: str
    logical_slice: LocalTensorMetadata
    source_slices: dict[str, LocalTensorMetadata]

    @property
    def logical_global_shape(self) -> tuple[int, ...]:
        return tuple(self.logical_slice.global_shape)

    @property
    def logical_local_shape(self) -> tuple[int, ...]:
        return tuple(self.logical_slice.local_shape)

    @property
    def logical_global_offset(self) -> tuple[int, ...]:
        return tuple(self.logical_slice.global_offset)


class HFDequantLoadTransform:
    """Expose quantized HF tensors as logical tensors to Paddle DCP.

    ``QuanMetadata`` owns descriptor validation and logical/physical relations.
    This class consumes that model to answer DCP's metadata/source-key
    requests, plan physical slices, and invoke the group-shared dequantizer.
    """

    def __init__(self, quan_metadata: QuanMetadata):
        self.quan_metadata = quan_metadata
        self._read_plans: dict[str, HFReadPlan] = {}

    def logical_metadata(self) -> dict[str, LocalTensorMetadata]:
        return dict(self.quan_metadata.logical_metadata)

    def _relation(
        self,
        logical_key: str,
    ) -> tuple[HFQuantizedWeightSpec, HFQuantizationGroupSpec]:
        try:
            spec = self.quan_metadata.relations[logical_key]
        except KeyError as exc:
            raise KeyError(f"Logical weight {logical_key!r} is not managed by this load transform.") from exc
        return spec, self.quan_metadata.groups[spec.group_name]

    def source_keys(self, logical_key: str) -> list[str]:
        spec, _ = self._relation(logical_key)
        return list(dict.fromkeys(spec.components.values()))

    def read_plan(
        self,
        logical_key: str,
        target_shard_metadata: LocalTensorMetadata,
        force_global: bool = False,
    ) -> HFReadPlan:
        """Build and cache a local physical read plan when it is safe.

        The first implementation deliberately requires block/group-aligned
        local boundaries.  Unaligned, transposed, or otherwise irregular
        layouts use the existing global path instead of guessing a scale
        origin.

        ``target_shard_metadata`` describes the shard of the logical tensor
        this rank holds, so a distributed target reports its local shape here,
        not its global one.  The plan is cached because ``apply()`` validates
        the dequantized shape against it.
        """
        spec, group = self._relation(logical_key)

        global_shape = tuple(target_shard_metadata.global_shape)
        local_shape = tuple(target_shard_metadata.local_shape)
        global_offset = tuple(target_shard_metadata.global_offset)
        source_metadata = self.quan_metadata.physical_metadata.state_dict_metadata
        dequantizer = group.dequantizer
        logical_axes = group.block_axes
        block_shape = group.block_shape
        axis_to_block = dict(zip(logical_axes, block_shape))
        if global_shape != spec.logical_shape:
            raise ValueError(
                f"Target shape mismatch for {logical_key!r}: "
                f"descriptor={spec.logical_shape}, target={global_shape}."
            )
        aligned = not force_global and dequantizer.logical_shard_is_aligned(
            spec.logical_shape, local_shape, global_offset
        )
        mode = "local" if aligned and local_shape != spec.logical_shape else "global"
        source_slices: dict[str, LocalTensorMetadata] = {}
        for role, source_key in spec.components.items():
            source_items = source_metadata.get(source_key)
            if not source_items:
                raise ValueError(f"Read plan for {logical_key!r} is missing source metadata for {source_key!r}.")
            physical_metadata = source_items[0]
            if role == "qweight":
                physical_shape = dequantizer.physical_qweight_shape(spec.logical_shape)
                if mode == "local":
                    physical_offset, physical_local_shape = dequantizer.physical_qweight_slice(
                        global_offset,
                        local_shape,
                    )
            elif role == "scale":
                physical_shape = _expected_scale_shape(spec.logical_shape, logical_axes, block_shape)
                if mode == "local":
                    # Map a block-aligned logical shard to the scale grid.  A
                    # shard ending at the logical tensor boundary may include
                    # one partial block, hence ceil on its local extent.
                    scale_factors = tuple(axis_to_block.get(axis, 1) for axis in range(len(global_offset)))
                    physical_offset = tuple(offset // factor for offset, factor in zip(global_offset, scale_factors))
                    physical_local_shape = tuple(
                        math.ceil(size / factor) for size, factor in zip(local_shape, scale_factors)
                    )
            else:
                raise ValueError(f"Unsupported component role {role!r} for {logical_key!r}.")
            if mode == "global":
                physical_offset = (0,) * len(physical_shape)
                physical_local_shape = physical_shape
            if tuple(physical_metadata.global_shape) != tuple(physical_shape):
                raise ValueError(
                    f"Read plan shape mismatch for {source_key!r}: description="
                    f"{tuple(physical_shape)}, checkpoint="
                    f"{tuple(physical_metadata.global_shape)}."
                )
            source_slices[source_key] = LocalTensorMetadata(
                global_shape=tuple(physical_shape),
                global_offset=tuple(physical_offset),
                local_shape=tuple(physical_local_shape),
                dtype=physical_metadata.dtype,
            )

        plan = HFReadPlan(
            mode=mode,
            logical_slice=LocalTensorMetadata(
                global_offset=global_offset if mode == "local" else (0,) * len(spec.logical_shape),
                local_shape=local_shape if mode == "local" else spec.logical_shape,
                dtype=self.quan_metadata.logical_metadata[logical_key].dtype,
                global_shape=spec.logical_shape,
            ),
            source_slices=source_slices,
        )
        self._read_plans[logical_key] = plan
        return plan

    def apply(
        self,
        logical_key: str,
        source_tensors: dict[str, paddle.Tensor],
        output_dtype: paddle.dtype,
    ) -> paddle.Tensor:
        spec, group = self._relation(logical_key)

        components = {role: source_tensors[source_key] for role, source_key in spec.components.items()}
        plan = self._read_plans.get(logical_key)
        output = group.dequantizer.dequantize(components, output_dtype)
        expected_shape = spec.logical_shape
        if plan is not None and plan.mode == "local":
            expected_shape = plan.logical_local_shape
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"Invalid dequantized shape for {logical_key!r}: expected {expected_shape}, got {tuple(output.shape)}."
            )
        return output


# -----------------------------------------------------------------------------
# Public transform builder
# -----------------------------------------------------------------------------


def hf_checkpoint_is_quantized(checkpoint_path: str) -> bool:
    """Report whether an HF checkpoint declares quantized weights.

    The checkpoint's own ``config.json`` is the authority: HuggingFace writes a
    ``quantization_config`` block exactly when the weights on disk are
    quantized.  Reading it here means no training argument has to repeat that
    fact, and a model class that owns a ``_gen_hf_quan_config()`` descriptor can
    still load its unquantized releases through the plain path.
    """
    config_path = os.path.join(checkpoint_path, _HF_CONFIG_FILE_NAME)
    if not os.path.isfile(config_path):
        return False
    quantization_config = _read_json(config_path).get("quantization_config")
    return isinstance(quantization_config, dict) and bool(quantization_config)


def build_hf_dequant_load_transform(
    checkpoint_path: str,
    quan_config: dict[str, Any] | None = None,
) -> HFDequantLoadTransform | None:
    """Build the HF dequantization transform from model-defined rules.

    ``quan_config`` is the descriptor returned by a model's parameterless
    ``_gen_hf_quan_config()`` method.  Checkpoint files only provide physical
    tensor metadata; quantization rules are owned by the model definition.
    Without a descriptor there is nothing to dequantize, so the caller gets
    ``None`` and loads the checkpoint through the plain path.
    """
    if quan_config is None:
        return None
    logger.info(f"load hf quantizated models at {checkpoint_path}.")
    quan_desc = QuanDescriptor.from_dict(quan_config)

    physical_metadata = _load_paddle_hf_metadata(checkpoint_path)
    quan_metadata = quan_desc.build_metadata(
        physical_metadata=physical_metadata,
        output_dtype=paddle.bfloat16,
    )

    return HFDequantLoadTransform(
        quan_metadata=quan_metadata,
    )
