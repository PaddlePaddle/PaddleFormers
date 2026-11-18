# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Strict streaming conversion from the original local safetensors checkpoint."""
import json
from pathlib import Path

import numpy as np
import paddle
from paddle import nn
from safetensors import safe_open

from .configuration import FalconPerceptionConfig


def load_local_model(model_class, path, *, dtype="float32", config=None, **kwargs):
    # Remote HF code is deliberately never executed by the Paddle implementation.
    for key in ("local_files_only", "trust_remote_code"):
        kwargs.pop(key, None)
    # AutoModel supplies resource-resolution options even for local models.
    kwargs.pop("cache_dir", None)
    kwargs.pop("download_hub", None)
    subfolder = kwargs.pop("subfolder", "") or ""
    if kwargs:
        raise TypeError(f"Unsupported local loader options: {sorted(kwargs)}")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("Supported dtypes: float32, bfloat16 (experimental).")
    directory = (Path(path).expanduser() / subfolder).resolve()
    if not directory.is_dir():
        raise ValueError(f"Expected a local model directory: {directory}")
    if config is None:
        values = json.loads((directory / "config.json").read_text())
        config = FalconPerceptionConfig(**values)
    config._name_or_path = str(directory)
    # Construct in FP32; normalizing ops and complex frequency buffers must not be
    # blanket-cast. Cast individual real parameters after strict weight loading.
    previous = paddle.get_default_dtype()
    try:
        paddle.set_default_dtype("float32")
        model = model_class(config)
    finally:
        paddle.set_default_dtype(previous)
    state = model.state_dict()
    linear_keys = {
        f"{name}.weight" if name else "weight"
        for name, layer in model.named_sublayers(include_self=True)
        if isinstance(layer, nn.Linear)
    }
    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
    seen = set()
    loaded = []
    # Validate the entire key/shape mapping before mutating any parameter.
    for shard in shards:
        with safe_open(str(directory / shard), framework="np") as handle:
            for key in handle.keys():
                if key in seen:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                seen.add(key)
                if key not in state:
                    raise ValueError(f"Unexpected checkpoint tensor: {key}")
                shape = handle.get_slice(key).get_shape()
                target_shape = shape[::-1] if key in linear_keys else shape
                if list(state[key].shape) != list(target_shape):
                    raise ValueError(f"Shape mismatch for {key}: source {shape}, target {list(state[key].shape)}")
    missing = sorted(set(state) - seen)
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {missing}")
    with paddle.no_grad():
        for shard in shards:
            with safe_open(str(directory / shard), framework="np") as handle:
                for key in handle.keys():
                    value = handle.get_tensor(key)
                    if key in linear_keys:
                        value = np.ascontiguousarray(value.T)
                    state[key].set_value(value)
                    loaded.append(key)
    if dtype != "float32":
        # nn.Layer.to(dtype=...) handles real parameters; preserve complex RoPE.
        model.to(dtype=dtype)
    model.eval()
    model.weight_loading_info = {
        "num_loaded": len(loaded),
        "num_transposed": len(linear_keys),
        "missing_keys": [],
        "unexpected_keys": [],
        "source": str(directory),
        "dtype": dtype,
    }
    return model
