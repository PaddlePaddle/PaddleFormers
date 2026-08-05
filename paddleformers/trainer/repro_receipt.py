# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Fail-closed machine receipts for model-reproduction training runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import paddle
from safetensors import safe_open

from .trainer_callback import TrainerCallback

_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _paddle_input_array(value):
    if isinstance(value, paddle.Tensor):
        shape = list(value.shape)
        stride = list(value.strides) if hasattr(value, "strides") else None
        logical_dtype = str(value.dtype).replace("paddle.", "")
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        shape = list(value.shape)
        stride = [int(item // value.itemsize) for item in value.strides] if value.ndim else []
        logical_dtype = str(value.dtype)
        array = value
    else:
        return None
    storage_dtype = str(array.dtype)
    return np.ascontiguousarray(array), shape, stride, logical_dtype, storage_dtype


def _is_paddlefleet_column_parallel_linear(layer) -> bool:
    """Recognize Fleet ColumnParallelLinear subclasses without importing Fleet."""
    return any(
        base.__name__ == "ColumnParallelLinear"
        and base.__module__ == "paddlefleet.tensor_parallel.layers"
        for base in type(layer).__mro__
    )


def _write_model_input_receipt(
    output_dir, framework, rank, inputs, labels, step, phase, file_prefix="model_inputs"
):
    receipt_dir = output_dir / "model_inputs"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    tensors = {}
    for name in sorted(inputs):
        if name == "labels":
            continue
        converted = _paddle_input_array(inputs[name])
        if converted is None:
            continue
        array, shape, stride, logical_dtype, storage_dtype = converted
        array_name = f"t{len(arrays)}"
        arrays[array_name] = array
        tensors[name] = {
            "array": array_name, "shape": shape, "stride": stride, "dtype": logical_dtype,
            "storage_dtype": storage_dtype, "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    converted = _paddle_input_array(labels)
    if converted is not None:
        array, shape, stride, logical_dtype, storage_dtype = converted
        array_name = f"t{len(arrays)}"
        arrays[array_name] = array
        tensors["labels"] = {
            "array": array_name, "shape": shape, "stride": stride, "dtype": logical_dtype,
            "storage_dtype": storage_dtype, "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    npz_path = receipt_dir / f"{file_prefix}_rank{rank}.npz"
    np.savez(npz_path, **arrays)
    _atomic_json(receipt_dir / f"{file_prefix}_rank{rank}.json", {
        "schema": "model-facing-input-receipt/v1", "framework": framework, "rank": rank,
        "step": int(step), "phase": phase, "tensor_count": len(tensors), "tensors": tensors, "npz": npz_path.name,
    })


def _first_paddle_tensor(value):
    if isinstance(value, paddle.Tensor):
        return value
    if isinstance(value, dict):
        if "hidden_states" in value:
            tensor = _first_paddle_tensor(value["hidden_states"])
            if tensor is not None:
                return tensor
        for key in sorted(value):
            tensor = _first_paddle_tensor(value[key])
            if tensor is not None:
                return tensor
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_paddle_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _write_named_internal_receipt(output_dir, framework, rank, boundary, module_name, module_class, calls):
    receipt_dir = output_dir / "internal_boundaries"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    call_receipts = []
    for call_index, call in enumerate(calls):
        tensors = {}
        for role, converted in call.items():
            array, shape, stride, logical_dtype, storage_dtype = converted
            array_name = f"c{call_index}_{role}"
            arrays[array_name] = array
            tensors[role] = {
                "array": array_name,
                "shape": shape,
                "stride": stride,
                "dtype": logical_dtype,
                "storage_dtype": storage_dtype,
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        call_receipts.append({"call_index": call_index, "tensors": tensors})
    npz_path = receipt_dir / f"{boundary}_rank{rank}.npz"
    np.savez(npz_path, **arrays)
    _atomic_json(receipt_dir / f"{boundary}_rank{rank}.json", {
        "schema": "internal-component-receipt/v1",
        "framework": framework,
        "rank": rank,
        "step": 0,
        "boundary": boundary,
        "module_name": module_name,
        "module_class": module_class,
        "call_count": len(call_receipts),
        "calls": call_receipts,
        "npz": npz_path.name,
    })


def _write_internal_boundary_receipt(output_dir, rank, boundary, module_name, module_class, calls):
    receipt_dir = output_dir / "internal_boundaries"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    call_receipts = []
    for call_index, call in enumerate(calls):
        call_receipt = {"call_index": call_index}
        for role in ("input", "output"):
            array, shape, stride, logical_dtype, storage_dtype = call[role]
            array_name = f"c{call_index}_{role}"
            arrays[array_name] = array
            call_receipt[role] = {
                "array": array_name,
                "shape": shape,
                "stride": stride,
                "dtype": logical_dtype,
                "storage_dtype": storage_dtype,
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        call_receipts.append(call_receipt)
    npz_path = receipt_dir / f"{boundary}_rank{rank}.npz"
    np.savez(npz_path, **arrays)
    _atomic_json(receipt_dir / f"{boundary}_rank{rank}.json", {
        "schema": "internal-boundary-receipt/v1",
        "framework": "paddle",
        "rank": rank,
        "step": 0,
        "boundary": boundary,
        "module_name": module_name,
        "module_class": module_class,
        "call_count": len(call_receipts),
        "calls": call_receipts,
        "npz": npz_path.name,
    })


def _checkpoint_inventory(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    inventory: dict[str, dict[str, Any]] = {}
    payload_bytes = 0
    with safe_open(path, framework="paddle", device="cpu") as handle:
        for key in handle.keys():
            tensor_slice = handle.get_slice(key)
            shape = list(tensor_slice.get_shape())
            dtype = tensor_slice.get_dtype()
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"unsupported safetensors dtype {dtype!r} for {key!r}")
            numel = math.prod(shape)
            payload_bytes += numel * _DTYPE_BYTES[dtype]
            inventory[key] = {"shape": shape, "dtype": dtype}
    return inventory, payload_bytes


class ReproReceiptCallback(TrainerCallback):
    """Record unrounded losses, runtime ownership, and a canonical checkpoint."""

    def __init__(self, model_dir: str | os.PathLike[str]):
        self.model_dir = Path(model_dir).resolve()
        self.profile_path = self.model_dir / "repro_profile.json"
        self.config_path = self.model_dir / "config.json"
        self.losses_by_step: dict[int, float] = {}
        self.output_dir: Path | None = None
        self.profile: dict[str, Any] | None = None
        self._model_inputs_captured = False
        self._collated_inputs_captured = False
        self.capture_model_inputs = os.environ.get("REPRO_CAPTURE_MODEL_INPUTS", "1") == "1"
        self.internal_boundary = os.environ.get("REPRO_CAPTURE_INTERNAL_BOUNDARIES", "").strip()
        self._internal_boundary_calls = []
        self._internal_boundary_handle = None
        self._internal_boundary_restore = None

    @staticmethod
    def _is_write_rank(state) -> bool:
        return bool(getattr(state, "is_world_process_zero", paddle.distributed.get_rank() == 0))

    def _write_losses(self) -> None:
        if self.output_dir is None:
            raise RuntimeError("reproduction receipt output directory is not initialized")
        ordered = [self.losses_by_step[step] for step in sorted(self.losses_by_step)]
        _atomic_json(self.output_dir / "loss.json", {"framework": "paddle", "losses": ordered})

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        del control, kwargs
        self.output_dir = Path(args.output_dir).resolve()
        if not self.profile_path.is_file():
            raise FileNotFoundError(f"reproduction receipt requires {self.profile_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"reproduction receipt requires {self.config_path}")
        self.profile = json.loads(self.profile_path.read_text(encoding="utf-8"))

        if model is None:
            raise RuntimeError("reproduction receipt requires the concrete runtime model")
        runtime_layers = model.sublayers(include_self=True)
        module_classes = Counter(
            f"{type(layer).__module__}.{type(layer).__qualname__}" for layer in runtime_layers
        )
        forbidden_modules = set()
        inactive_fused_wrappers = set()
        for layer in runtime_layers:
            name = f"{type(layer).__module__}.{type(layer).__qualname__}"
            lowered = name.lower()
            if "transformer_engine" in lowered or lowered.startswith("paddlefleet_ops."):
                forbidden_modules.add(name)
            elif "fused" in lowered:
                # This legacy-named wrapper contains only the eager mask/softmax
                # implementation. It is admissible only with its fusion switch off.
                if name == "paddlefleet.fusions.fused_softmax.FusedScaleMaskSoftmax" and not bool(
                    getattr(layer, "scaled_masked_softmax_fusion", True)
                ):
                    inactive_fused_wrappers.add(name)
                else:
                    forbidden_modules.add(name)
        forbidden_modules = sorted(forbidden_modules)
        inactive_fused_wrappers = sorted(inactive_fused_wrappers)
        if forbidden_modules:
            raise RuntimeError(f"reproduction receipt found forbidden fused/custom modules: {forbidden_modules}")

        if paddle.distributed.get_rank() == 0:
            output_dir = Path(args.output_dir).resolve()
            shutil.rmtree(output_dir / "model_inputs", ignore_errors=True)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
        self._setup_internal_boundary_receipt(model)
        if not self._is_write_rank(state):
            return
        for name in ("env.json", "loss.json", "repro_metrics.jsonl"):
            (self.output_dir / name).unlink(missing_ok=True)
        text_config = getattr(getattr(model, "config", None), "text_config", getattr(model, "config", None))
        config_flags = {}
        for name in (
            "apply_rope_fusion",
            "bias_activation_fusion",
            "bias_dropout_fusion",
            "masked_softmax_fusion",
            "moe_expert_fusion",
            "moe_permute_fusion",
            "moe_router_fusion",
            "moe_shared_expert_overlap",
            "qk_norm_fusion",
            "sequence_parallel",
            "sigmoid_gate_fusion",
            "use_accuracy_compatible",
            "use_fused_linear_cross_entropy",
            "use_unified_moe",
        ):
            config_flags[name] = getattr(text_config, name, getattr(args, name, None))

        device_name = None
        if paddle.device.is_compiled_with_cuda():
            device_name = paddle.device.cuda.get_device_name()
        env = {
            "framework": "paddle",
            "framework_version": paddle.__version__,
            "device": "cuda",
            "device_name": device_name,
            "dtype": "bfloat16",
            "model_id": self.profile["source_model_id"],
            "revision": self.profile["source_revision"],
            "model_config_sha256": self.profile["source_config_sha256"],
            "profile_config_sha256": _sha256_file(self.config_path),
            "profile": self.profile["schema"],
            "profile_tensor_count": self.profile["tensor_count"],
            "profile_payload_bytes": self.profile["payload_bytes"],
            "weights_loaded": True,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "paddle_cuda": paddle.version.cuda(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "paddlefleet_ops_no_ext": os.environ.get("PADDLEFLEET_OPS_NO_EXT"),
            "flags_cudnn_deterministic": os.environ.get("FLAGS_cudnn_deterministic"),
            "flags_embedding_deterministic": os.environ.get("FLAGS_embedding_deterministic"),
            "flags_use_accuracy_compatible_kernel": os.environ.get("FLAGS_use_accuracy_compatible_kernel"),
            "world_size": paddle.distributed.get_world_size(),
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
            "context_parallel_size": args.context_parallel_size,
            "expert_model_parallel_size": args.expert_model_parallel_size,
            "expert_tensor_model_parallel_size": args.expert_tensor_model_parallel_size,
            "sharding_parallel_size": args.sharding_parallel_size,
            "sharding": str(args.sharding),
            "sequence_parallel": args.sequence_parallel,
            "use_accuracy_compatible": args.use_accuracy_compatible,
            "deterministic_mode": args.deterministic_mode,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "optimizer": type(optimizer).__module__ + "." + type(optimizer).__qualname__ if optimizer is not None else None,
            "learning_rate": args.learning_rate,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "runtime_parameter_count_rank0": sum(int(parameter.numel()) for parameter in model.parameters()),
            "runtime_trainable_parameter_count_rank0": sum(
                int(parameter.numel()) for parameter in model.parameters() if not parameter.stop_gradient
            ),
            "module_class_counts_rank0": dict(sorted(module_classes.items())),
            "inactive_fused_wrapper_classes": inactive_fused_wrappers,
            "forbidden_module_classes": forbidden_modules,
            "config_flags": config_flags,
            "accuracy_environment": {
                name: os.environ.get(name)
                for name in (
                    "PADDLEFLEET_ACCURACY_GDN_IN_PROJ_ORDER",
                    "PADDLEFLEET_ACCURACY_GDN_CAUSAL_CONV1D",
                    "PADDLEFLEET_ACCURACY_GDN_CONV_FP32_ACCUM",
                    "PADDLEFLEET_ACCURACY_GDN_QK_L2",
                    "PADDLEFLEET_ACCURACY_GDN_RECURRENCE",
                )
            },
        }
        _atomic_json(self.output_dir / "env.json", env)

    def _setup_internal_boundary_receipt(self, model):
        if not self.internal_boundary:
            return
        supported_boundaries = {
            "text_embedding_lookup",
            "language_layer0",
            "language_layer0_input_norm",
            "language_layer0_gdn",
            "language_layer0_gdn_in_proj",
            "language_layer0_gdn_recurrence",
            "language_layer0_gdn_out_norm",
            "language_layer0_gdn_out_proj",
            "vision_patch_embed",
            "vision_pos_lookup",
            "vision_block0",
            "vision_merger",
        }
        if self.internal_boundary not in supported_boundaries:
            raise ValueError(f"unsupported internal reproduction boundary: {self.internal_boundary!r}")
        rank = paddle.distributed.get_rank()
        receipt_dir = self.output_dir / "internal_boundaries"
        if rank == 0:
            shutil.rmtree(receipt_dir, ignore_errors=True)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
        if self.internal_boundary == "text_embedding_lookup":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if type(layer).__name__ == "VocabParallelEmbedding"
            ]
        elif self.internal_boundary == "language_layer0":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name == "model.language_model._layers.1"
                and type(layer).__name__ == "TransformerLayer"
            ]
        elif self.internal_boundary == "language_layer0_input_norm":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name == "model.language_model._layers.1.input_layernorm"
                and type(layer).__name__ == "Qwen3_5RMSNorm"
            ]
        elif self.internal_boundary == "language_layer0_gdn":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name == "model.language_model._layers.1.self_attn"
                and type(layer).__name__ == "GatedDeltaNet"
            ]
        elif self.internal_boundary == "language_layer0_gdn_in_proj":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name == "model.language_model._layers.1.self_attn.in_proj"
                and _is_paddlefleet_column_parallel_linear(layer)
            ]
        elif self.internal_boundary == "language_layer0_gdn_recurrence":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name == "model.language_model._layers.1.self_attn"
                and type(layer).__name__ == "GatedDeltaNet"
            ]
        elif self.internal_boundary == "language_layer0_gdn_out_norm":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name == "model.language_model._layers.1.self_attn.out_norm"
                and type(layer).__name__ == "RMSNorm"
            ]
        elif self.internal_boundary == "language_layer0_gdn_out_proj":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name == "model.language_model._layers.1.self_attn.out_proj"
                and type(layer).__name__ == "RowParallelLinear"
            ]
        elif self.internal_boundary == "vision_patch_embed":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name.endswith(".patch_embed")
                and type(layer).__name__ in {"Conv3D", "AccuracyCompatiblePatchProjection"}
            ]
        elif self.internal_boundary == "vision_pos_lookup":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name == "model.visual._layers.0.pos_embed"
            ]
        elif self.internal_boundary == "vision_block0":
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if name == "model.visual._layers.1"
            ]
        else:
            matches = [
                (name, layer)
                for name, layer in model.named_sublayers(include_self=True)
                if type(layer).__name__ == "Qwen3VLVisionPathMerger"
            ]
        if len(matches) != 1:
            found = [f"{name}:{type(layer).__module__}.{type(layer).__qualname__}" for name, layer in matches]
            raise RuntimeError(
                f"{self.internal_boundary} receipt requires exactly one runtime module, found {found}"
            )
        module_name, module = matches[0]
        module_class = f"{type(module).__module__}.{type(module).__qualname__}"

        if self.internal_boundary == "language_layer0_gdn_recurrence":
            observer_name = "_repro_recurrence_observer"
            if hasattr(module, observer_name):
                raise RuntimeError("GDN recurrence observer slot is already occupied")

            def capture_recurrence(**tensors):
                required = {
                    "query_pre_norm",
                    "key_pre_norm",
                    "query",
                    "key",
                    "value",
                    "alpha",
                    "softplus_input",
                    "g",
                    "beta",
                    "core_output",
                }
                if set(tensors) != required or any(tensor is None for tensor in tensors.values()):
                    raise RuntimeError(
                        f"GDN recurrence receipt requires exactly {sorted(required)}, got {sorted(tensors)}"
                    )
                self._internal_boundary_calls.append(
                    {name: _paddle_input_array(tensor) for name, tensor in tensors.items()}
                )
                _write_named_internal_receipt(
                    self.output_dir,
                    "paddle",
                    rank,
                    self.internal_boundary,
                    module_name,
                    module_class,
                    self._internal_boundary_calls,
                )

            setattr(module, observer_name, capture_recurrence)
            self._internal_boundary_restore = lambda: delattr(module, observer_name)
            return

        def capture(_module, inputs, output):
            input_tensor = _first_paddle_tensor(inputs)
            output_tensor = _first_paddle_tensor(output)
            if input_tensor is None or output_tensor is None:
                raise RuntimeError(f"{self.internal_boundary} receipt requires tensor input and output")
            self._internal_boundary_calls.append({
                "input": _paddle_input_array(input_tensor),
                "output": _paddle_input_array(output_tensor),
            })
            _write_internal_boundary_receipt(
                self.output_dir, rank, self.internal_boundary, module_name, module_class, self._internal_boundary_calls
            )

        self._internal_boundary_handle = module.register_forward_post_hook(capture)

    def _close_internal_boundary_receipt(self):
        if self._internal_boundary_handle is not None:
            self._internal_boundary_handle.remove()
            self._internal_boundary_handle = None
        if self._internal_boundary_restore is not None:
            self._internal_boundary_restore()
            self._internal_boundary_restore = None
        if self.internal_boundary and not self._internal_boundary_calls:
            raise RuntimeError(f"internal boundary {self.internal_boundary!r} produced no first-step calls")
        if self.internal_boundary == "language_layer0_gdn_recurrence" and len(self._internal_boundary_calls) != 1:
            raise RuntimeError(
                "GDN recurrence receipt requires exactly one first-step call, "
                f"got {len(self._internal_boundary_calls)}"
            )

    def on_model_inputs(self, args, state, control, inputs=None, labels=None, phase=None, **kwargs):
        del control, kwargs
        if not self.capture_model_inputs or self._model_inputs_captured:
            return
        rank = paddle.distributed.get_rank()
        _write_model_input_receipt(
            self.output_dir, "paddle", rank, inputs or {}, labels, state.global_step, phase
        )
        self._model_inputs_captured = True

    def on_load_data_end(self, args, state, control, inputs=None, **kwargs):
        del control, kwargs
        if not self.capture_model_inputs or self._collated_inputs_captured:
            return
        collated_labels = inputs.get("labels") if isinstance(inputs, dict) else None
        _write_model_input_receipt(
            self.output_dir, "paddle", paddle.distributed.get_rank(), inputs or {}, collated_labels,
            state.global_step, "collated", file_prefix="collated_inputs"
        )
        self._collated_inputs_captured = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        logs = logs or {}
        if "repro_raw_loss" not in logs:
            return
        step = int(logs.get("global_step", state.global_step))
        loss = float(logs["repro_raw_loss"])
        if not math.isfinite(loss):
            raise ValueError(f"non-finite reproduction loss at step {step}: {loss}")
        previous = self.losses_by_step.setdefault(step, loss)
        if previous != loss:
            raise RuntimeError(f"conflicting raw losses for step {step}: {previous} vs {loss}")
        self._close_internal_boundary_receipt()
        if not self._is_write_rank(state):
            return
        self._write_losses()
        if self.output_dir is None:
            raise RuntimeError("reproduction receipt output directory is not initialized")
        record = {"step": step, "loss": loss}
        for key, value in logs.items():
            if key == "learning_rate" or key == "loss_md5" or key.endswith("_loss") or key.endswith("_loss_md5"):
                record[key] = value
        with (self.output_dir / "repro_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()

    def record_initial_checkpoint(self, args, checkpoint_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
        """Collectively validate the official-layout parameters before step one."""
        self.output_dir = Path(args.output_dir).resolve()
        if self.profile is None:
            if not self.profile_path.is_file():
                raise FileNotFoundError(f"reproduction receipt requires {self.profile_path}")
            self.profile = json.loads(self.profile_path.read_text(encoding="utf-8"))

        checkpoint_path = Path(checkpoint_dir).resolve()
        error: dict[str, str] | None = None
        manifest: dict[str, Any] | None = None
        if paddle.distributed.get_rank() == 0:
            try:
                source = checkpoint_path / "model.safetensors"
                conversion_receipt = checkpoint_path / "paddle_grouped_checkpoint_conversion.json"
                if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
                    raise ValueError(f"initial checkpoint must be one regular non-hard-linked file: {source}")
                if not conversion_receipt.is_file() or conversion_receipt.is_symlink():
                    raise ValueError(f"initial grouped checkpoint conversion receipt missing: {conversion_receipt}")
                inventory, payload_bytes = _checkpoint_inventory(source)
                expected_count = int(self.profile["tensor_count"])
                if len(inventory) != expected_count:
                    raise ValueError(f"initial checkpoint tensor count mismatch: {len(inventory)} != {expected_count}")
                leaked = sorted(key for key in inventory if "._c" in key or ".grouped_gemm_experts." in key)
                if leaked:
                    raise ValueError(f"initial checkpoint contains intermediate/grouped runtime names: {leaked}")
                manifest = {
                    "schema": "paddle-repro-initial-checkpoint/v1",
                    "framework": "paddle",
                    "file": source.name,
                    "bytes": source.stat().st_size,
                    "sha256": _sha256_file(source),
                    "nlink": source.stat().st_nlink,
                    "tensor_count": len(inventory),
                    "payload_bytes": payload_bytes,
                    "tensor_names_sha256": hashlib.sha256(
                        ("\n".join(sorted(inventory)) + "\n").encode("utf-8")
                    ).hexdigest(),
                    "dtype_counts": dict(sorted(Counter(item["dtype"] for item in inventory.values()).items())),
                    "profile_tensor_count": expected_count,
                    "profile_payload_bytes": int(self.profile["payload_bytes"]),
                    "source_runtime_payload_delta": int(self.profile["payload_bytes"]) - payload_bytes,
                    "global_step": 0,
                    "conversion_receipt": "paddle_grouped_checkpoint_conversion.json",
                }
                _atomic_json(checkpoint_path / "initial_checkpoint_manifest.json", manifest)
            except Exception as exc:
                error = {"error_type": type(exc).__name__, "error": str(exc)}

        if paddle.distributed.get_world_size() > 1:
            payload = [error]
            paddle.distributed.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error is not None:
            raise RuntimeError(
                f"Paddle initial reproduction receipt failed: {error['error_type']}: {error['error']}"
            )
        return manifest

    def _validate_losses(self, expected_steps: int) -> None:
        expected = list(range(1, expected_steps + 1))
        actual = sorted(self.losses_by_step)
        if actual != expected:
            raise RuntimeError(f"reproduction receipt expected raw losses for steps {expected}, got {actual}")

    def finalize_checkpoint(self, args, state) -> dict[str, Any] | None:
        """Validate and relocate the final grouped checkpoint on every rank."""
        self._validate_losses(int(args.max_steps))
        if self.output_dir is None or self.profile is None:
            raise RuntimeError("reproduction receipt was not initialized by on_train_begin")

        error: dict[str, str] | None = None
        manifest: dict[str, Any] | None = None
        if self._is_write_rank(state):
            try:
                source = self.output_dir / "model.safetensors"
                conversion_receipt = self.output_dir / "paddle_grouped_checkpoint_conversion.json"
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f"checkpoint must be a regular file: {source}")
                if source.stat().st_nlink != 1:
                    raise ValueError(f"checkpoint must not be hard-linked: {source}")
                if not conversion_receipt.is_file() or conversion_receipt.is_symlink():
                    raise ValueError(f"grouped checkpoint conversion receipt missing: {conversion_receipt}")

                inventory, payload_bytes = _checkpoint_inventory(source)
                expected_count = int(self.profile["tensor_count"])
                if len(inventory) != expected_count:
                    raise ValueError(f"checkpoint tensor count mismatch: {len(inventory)} != {expected_count}")
                leaked = sorted(key for key in inventory if "._c" in key or ".grouped_gemm_experts." in key)
                if leaked:
                    raise ValueError(f"checkpoint contains intermediate/grouped runtime names: {leaked}")

                target = self.output_dir / "checkpoint"
                temporary = self.output_dir / f".checkpoint.tmp-{os.getpid()}"
                shutil.rmtree(temporary, ignore_errors=True)
                temporary.mkdir(parents=True)
                source.replace(temporary / source.name)
                conversion_receipt.replace(temporary / conversion_receipt.name)
                for name in ("config.json", "preprocessor_config.json"):
                    candidate = self.output_dir / name
                    if candidate.is_file() and not candidate.is_symlink():
                        shutil.copy2(candidate, temporary / name)
                shutil.copy2(self.profile_path, temporary / self.profile_path.name)

                checkpoint_file = temporary / source.name
                manifest = {
                    "schema": "paddle-repro-checkpoint/v1",
                    "framework": "paddle",
                    "file": source.name,
                    "bytes": checkpoint_file.stat().st_size,
                    "sha256": _sha256_file(checkpoint_file),
                    "nlink": checkpoint_file.stat().st_nlink,
                    "tensor_count": len(inventory),
                    "payload_bytes": payload_bytes,
                    "tensor_names_sha256": hashlib.sha256(
                        ("\n".join(sorted(inventory)) + "\n").encode("utf-8")
                    ).hexdigest(),
                    "dtype_counts": dict(sorted(Counter(item["dtype"] for item in inventory.values()).items())),
                    "profile_tensor_count": expected_count,
                    "profile_payload_bytes": int(self.profile["payload_bytes"]),
                    "source_runtime_payload_delta": int(self.profile["payload_bytes"]) - payload_bytes,
                    "conversion_receipt": "paddle_grouped_checkpoint_conversion.json",
                }
                _atomic_json(temporary / "checkpoint_manifest.json", manifest)
                shutil.rmtree(target, ignore_errors=True)
                temporary.replace(target)
                self._write_losses()
            except Exception as exc:
                error = {"error_type": type(exc).__name__, "error": str(exc)}

        if paddle.distributed.get_world_size() > 1:
            payload = [error]
            paddle.distributed.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error is not None:
            raise RuntimeError(
                f"Paddle reproduction receipt finalization failed: {error['error_type']}: {error['error']}"
            )
        return manifest


__all__ = ["ReproReceiptCallback"]
