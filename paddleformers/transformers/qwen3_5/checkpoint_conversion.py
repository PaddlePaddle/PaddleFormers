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

"""Deterministic source-layout normalization for Qwen3.5 non-fused experts.

FlexCheckpoint AOA split primitives preserve the split dimension.  Official
Qwen3.5 main-layer MoE tensors are grouped as ``[experts, ...]``, while the
accuracy-compatible non-fused Fleet implementation owns one 2-D parameter per
expert.  This module creates a byte-preserving, reversible per-expert view of
those *official* tensors before AOA loading.  It never synthesizes weights and
is not used by grouped/fused expert configurations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import paddle
from safetensors import safe_open
from safetensors.paddle import save_file

_LAYOUT = "qwen3_5-per-expert-hf/v1"
_MANIFEST = "paddle_nonfused_conversion.json"
_INDEX = "model.safetensors.index.json"
_GROUPED_PATTERN = re.compile(
    r"^(model\.language_model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$"
)
_PER_EXPERT_PATTERN = re.compile(
    r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_GROUPED_OUTPUT_PATTERN = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _cache_is_valid(cache_dir: Path, source_index_sha256: str) -> bool:
    manifest_path = cache_dir / _MANIFEST
    index_path = cache_dir / _INDEX
    if not manifest_path.is_file() or not index_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, TypeError):
        return False
    if manifest.get("schema") != _LAYOUT or manifest.get("source_index_sha256") != source_index_sha256:
        return False
    if manifest.get("output_index_sha256") != _sha256(index_path):
        return False
    for name, receipt in manifest.get("output_shards", {}).items():
        shard = cache_dir / name
        if not shard.is_file() or shard.stat().st_size != receipt.get("size"):
            return False
        if _sha256(shard) != receipt.get("sha256"):
            return False
    return True


def _split_grouped_tensor(key: str, tensor: paddle.Tensor, num_experts: int) -> dict[str, paddle.Tensor]:
    match = _GROUPED_PATTERN.fullmatch(key)
    if match is None:
        return {key: tensor}
    prefix, kind = match.groups()
    if tensor.ndim != 3 or tensor.shape[0] != num_experts:
        raise ValueError(
            f"{key} must have shape [num_experts, *, *] with num_experts={num_experts}, got {tensor.shape}"
        )

    converted: dict[str, paddle.Tensor] = {}
    if kind == "gate_up_proj":
        if tensor.shape[1] % 2:
            raise ValueError(f"{key} gate/up dimension must be even, got {tensor.shape}")
        gate_size = tensor.shape[1] // 2
        for expert_id in range(num_experts):
            expert = tensor[expert_id]
            converted[f"{prefix}.{expert_id}.gate_proj.weight"] = expert[:gate_size].contiguous()
            converted[f"{prefix}.{expert_id}.up_proj.weight"] = expert[gate_size:].contiguous()
    else:
        for expert_id in range(num_experts):
            converted[f"{prefix}.{expert_id}.down_proj.weight"] = tensor[expert_id].contiguous()
    return converted


def _build_cache(source_dir: Path, cache_dir: Path, num_experts: int, source_index_sha256: str) -> None:
    source_index = _read_json(source_dir / _INDEX)
    source_weight_map = source_index["weight_map"]
    grouped_keys = sorted(key for key in source_weight_map if _GROUPED_PATTERN.fullmatch(key))
    expected_grouped = 4  # two minimum-complete language layers × gate_up/down
    if len(grouped_keys) != expected_grouped:
        raise ValueError(
            f"expected {expected_grouped} grouped main-expert tensors for Qwen3.5 minimum-complete, "
            f"found {len(grouped_keys)}: {grouped_keys}"
        )

    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp-{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    output_weight_map: dict[str, str] = {}
    transformations: list[dict[str, Any]] = []
    try:
        by_shard: dict[str, list[str]] = {}
        for key, shard_name in source_weight_map.items():
            by_shard.setdefault(shard_name, []).append(key)

        for shard_name, shard_keys in sorted(by_shard.items()):
            source_shard = source_dir / shard_name
            output_shard = tmp_dir / shard_name
            grouped_in_shard = [key for key in shard_keys if _GROUPED_PATTERN.fullmatch(key)]
            if not grouped_in_shard:
                shutil.copy2(source_shard, output_shard)
                output_weight_map.update({key: shard_name for key in shard_keys})
                continue

            tensors: dict[str, paddle.Tensor] = {}
            with safe_open(source_shard, framework="paddle", device="cpu") as handle:
                metadata = handle.metadata()
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    converted = _split_grouped_tensor(key, tensor, num_experts)
                    tensors.update(converted)
                    if key in grouped_in_shard:
                        transformations.append(
                            {
                                "source": key,
                                "source_shape": list(tensor.shape),
                                "source_dtype": str(tensor.dtype).replace("paddle.", ""),
                                "destinations": sorted(converted),
                                "operation": (
                                    "split expert axis, split gate/up axis"
                                    if key.endswith("gate_up_proj")
                                    else "split expert axis"
                                ),
                            }
                        )
            save_file(tensors, output_shard, metadata=metadata)
            for key in tensors:
                output_weight_map[key] = shard_name

        output_index = dict(source_index)
        output_index["weight_map"] = dict(sorted(output_weight_map.items()))
        (tmp_dir / _INDEX).write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n")

        output_shards = {
            name: {"sha256": _sha256(tmp_dir / name), "size": (tmp_dir / name).stat().st_size}
            for name in sorted(set(output_weight_map.values()))
        }
        manifest = {
            "schema": _LAYOUT,
            "source": str(source_dir),
            "source_index_sha256": source_index_sha256,
            "source_tensor_count": len(source_weight_map),
            "source_payload_bytes": source_index.get("metadata", {}).get("total_size"),
            "num_experts": num_experts,
            "transformations": transformations,
            "output_tensor_count": len(output_weight_map),
            "output_payload_bytes": output_index.get("metadata", {}).get("total_size"),
            "output_index_sha256": _sha256(tmp_dir / _INDEX),
            "output_shards": output_shards,
            "reversible": True,
            "not_a_weight_source": "All output bytes are partitions of the four listed official grouped tensors or byte-identical copies.",
        }
        (tmp_dir / _MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp_dir.rename(cache_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _tensor_payload_bytes(tensor: paddle.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _finalize_grouped_expert_checkpoint_local(
    save_dir: Path, num_experts: int
) -> dict[str, Any]:
    """Reassemble normalized main-layer expert keys after inverse AOA save."""
    index_path = save_dir / _INDEX
    if index_path.is_file():
        index = _read_json(index_path)
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        shard_paths = [save_dir / name for name in shard_names]
    else:
        canonical = save_dir / "model.safetensors"
        shard_paths = [canonical] if canonical.is_file() else sorted(save_dir.glob("model*.safetensors"))
    if not shard_paths or any(not path.is_file() for path in shard_paths):
        raise FileNotFoundError(f"no complete safetensors checkpoint found in {save_dir}")

    tensors: dict[str, paddle.Tensor] = {}
    metadata: dict[str, str] = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="paddle", device="cpu") as handle:
            if not metadata and handle.metadata() is not None:
                metadata = handle.metadata()
            for key in handle.keys():
                if key in tensors:
                    raise ValueError(f"duplicate checkpoint tensor {key!r} across {shard_paths}")
                tensors[key] = handle.get_tensor(key)

    before_count = len(tensors)
    before_payload_bytes = sum(_tensor_payload_bytes(tensor) for tensor in tensors.values())
    per_layer: dict[int, dict[int, dict[str, paddle.Tensor]]] = {}
    consumed_keys: list[str] = []
    for key, tensor in tensors.items():
        match = _PER_EXPERT_PATTERN.fullmatch(key)
        if match is None:
            continue
        _, layer_text, expert_text, projection = match.groups()
        layer_id = int(layer_text)
        expert_id = int(expert_text)
        per_layer.setdefault(layer_id, {}).setdefault(expert_id, {})[projection] = tensor
        consumed_keys.append(key)

    if not per_layer:
        grouped_keys = sorted(key for key in tensors if _GROUPED_OUTPUT_PATTERN.fullmatch(key))
        if grouped_keys:
            return {
                "schema": "qwen3_5-grouped-expert-save/v1",
                "status": "already_grouped",
                "tensor_count": before_count,
                "payload_bytes": before_payload_bytes,
                "grouped_keys": grouped_keys,
            }
        raise ValueError("saved checkpoint has neither normalized per-expert nor grouped main-layer tensors")

    expected_experts = set(range(num_experts))
    grouped_receipts: list[dict[str, Any]] = []
    grouped_tensors: dict[str, paddle.Tensor] = {}
    for layer_id, experts in sorted(per_layer.items()):
        if set(experts) != expected_experts:
            raise ValueError(
                f"layer {layer_id} expert coverage mismatch: expected {sorted(expected_experts)}, got {sorted(experts)}"
            )
        for expert_id, projections in experts.items():
            if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} projection coverage mismatch: {sorted(projections)}"
                )
            if projections["gate_proj"].shape != projections["up_proj"].shape:
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} gate/up shapes differ: "
                    f"{projections['gate_proj'].shape} vs {projections['up_proj'].shape}"
                )

        prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
        gate_up = paddle.stack(
            [
                paddle.concat(
                    [experts[expert_id]["gate_proj"], experts[expert_id]["up_proj"]],
                    axis=0,
                )
                for expert_id in range(num_experts)
            ],
            axis=0,
        ).contiguous()
        down = paddle.stack(
            [experts[expert_id]["down_proj"] for expert_id in range(num_experts)],
            axis=0,
        ).contiguous()
        grouped_tensors[f"{prefix}.gate_up_proj"] = gate_up
        grouped_tensors[f"{prefix}.down_proj"] = down
        grouped_receipts.append(
            {
                "layer": layer_id,
                "gate_up_shape": list(gate_up.shape),
                "down_shape": list(down.shape),
                "dtype": str(gate_up.dtype).replace("paddle.", ""),
            }
        )

    for key in consumed_keys:
        tensors.pop(key)
    overlap = set(tensors).intersection(grouped_tensors)
    if overlap:
        raise ValueError(f"grouped checkpoint keys already exist: {sorted(overlap)}")
    tensors.update(grouped_tensors)
    after_payload_bytes = sum(_tensor_payload_bytes(tensor) for tensor in tensors.values())
    if after_payload_bytes != before_payload_bytes:
        raise ValueError(
            f"expert regrouping changed payload bytes: {before_payload_bytes} -> {after_payload_bytes}"
        )
    expected_after_count = before_count - len(consumed_keys) + len(grouped_tensors)
    if len(tensors) != expected_after_count:
        raise AssertionError(f"unexpected regrouped tensor count: {len(tensors)} != {expected_after_count}")

    output_path = save_dir / "model.safetensors"
    temporary = save_dir / f".model.safetensors.tmp-{os.getpid()}"
    save_file(tensors, temporary, metadata=metadata)
    with safe_open(temporary, framework="paddle", device="cpu") as handle:
        output_keys = list(handle.keys())
        output_payload_bytes = sum(
            int(math.prod(handle.get_slice(key).get_shape()))
            * {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}[
                handle.get_slice(key).get_dtype()
            ]
            for key in output_keys
        )
    if len(output_keys) != len(tensors) or output_payload_bytes != before_payload_bytes:
        temporary.unlink(missing_ok=True)
        raise ValueError("reopened grouped checkpoint inventory does not match the in-memory transaction")

    temporary.replace(output_path)
    for old_path in shard_paths:
        if old_path != output_path:
            old_path.unlink(missing_ok=True)
    index_path.unlink(missing_ok=True)
    receipt = {
        "schema": "qwen3_5-grouped-expert-save/v1",
        "status": "grouped",
        "source_files": [path.name for path in shard_paths],
        "output": output_path.name,
        "output_sha256": _sha256(output_path),
        "tensor_count_before": before_count,
        "tensor_count_after": len(output_keys),
        "payload_bytes": before_payload_bytes,
        "num_experts": num_experts,
        "consumed_per_expert_keys": len(consumed_keys),
        "grouped_tensors": grouped_receipts,
        "reversible": True,
    }
    _atomic_json(save_dir / "paddle_grouped_checkpoint_conversion.json", receipt)
    return receipt


def finalize_grouped_expert_checkpoint(
    save_dir: str | os.PathLike[str], config
) -> dict[str, Any]:
    """Collectively finalize a non-fused Qwen3.5 save into official grouped names."""
    save_path = Path(save_dir).resolve()
    text_config = getattr(config, "text_config", config)
    num_experts = int(getattr(text_config, "num_experts"))
    result_path = save_path / "paddle_grouped_checkpoint_conversion.json"
    error_path = save_path / ".paddle_grouped_checkpoint_conversion.error.json"
    rank = paddle.distributed.get_rank()
    world_size = paddle.distributed.get_world_size()
    if rank == 0:
        result_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)
        try:
            _finalize_grouped_expert_checkpoint_local(save_path, num_experts)
        except Exception as exc:
            _atomic_json(
                error_path,
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
    if world_size > 1:
        paddle.distributed.barrier()
    if error_path.is_file():
        failure = _read_json(error_path)
        raise RuntimeError(
            "Qwen3.5 grouped checkpoint finalization failed: "
            f"{failure.get('error_type')}: {failure.get('error')}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"Qwen3.5 grouped checkpoint receipt missing: {result_path}")
    return _read_json(result_path)


def prepare_nonfused_expert_checkpoint(
    pretrained_model_name_or_path: str | os.PathLike[str], config
) -> tuple[str, dict[str, Any] | None]:
    """Return an AOA-loadable path for non-fused official Qwen3.5 experts.

    Rank zero materializes and verifies the deterministic cache; a distributed
    barrier makes every other rank consume that exact completed transaction.
    """

    if getattr(config, "moe_expert_fusion", True):
        return str(pretrained_model_name_or_path), None

    source_dir = Path(pretrained_model_name_or_path).resolve()
    index_path = source_dir / _INDEX
    if not index_path.is_file():
        raise FileNotFoundError(f"Qwen3.5 non-fused conversion requires {_INDEX}: {index_path}")
    source_index = _read_json(index_path)
    grouped_keys = [key for key in source_index.get("weight_map", {}) if _GROUPED_PATTERN.fullmatch(key)]
    if not grouped_keys:
        # Already normalized input is accepted only with its auditable marker.
        marker = source_dir / _MANIFEST
        if not marker.is_file() or _read_json(marker).get("schema") != _LAYOUT:
            raise ValueError("non-fused Qwen3.5 checkpoint has no grouped experts and no validated conversion marker")
        config._checkpoint_source_layout = _LAYOUT
        getattr(config, "text_config", config)._checkpoint_source_layout = _LAYOUT
        return str(source_dir), _read_json(marker)

    source_index_sha256 = _sha256(index_path)
    cache_dir = source_dir.with_name(f"{source_dir.name}-paddle-nonfused-v1")
    text_config = getattr(config, "text_config", config)
    num_experts = int(getattr(text_config, "num_experts"))
    rank = paddle.distributed.get_rank()
    world_size = paddle.distributed.get_world_size()

    if rank == 0 and not _cache_is_valid(cache_dir, source_index_sha256):
        _build_cache(source_dir, cache_dir, num_experts, source_index_sha256)
    if world_size > 1:
        paddle.distributed.barrier()
    if not (cache_dir / _MANIFEST).is_file():
        raise RuntimeError(f"Qwen3.5 non-fused conversion did not produce {cache_dir / _MANIFEST}")

    manifest = _read_json(cache_dir / _MANIFEST)
    if manifest.get("source_index_sha256") != source_index_sha256:
        raise RuntimeError("Qwen3.5 non-fused conversion source digest changed across ranks")
    config._checkpoint_source_layout = _LAYOUT
    getattr(config, "text_config", config)._checkpoint_source_layout = _LAYOUT
    return str(cache_dir), manifest


__all__ = [
    "finalize_grouped_expert_checkpoint",
    "prepare_nonfused_expert_checkpoint",
]
