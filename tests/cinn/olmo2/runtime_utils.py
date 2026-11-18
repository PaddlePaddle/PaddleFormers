# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Runtime checks and shared fixed-input timing for the OLMo2 CINN entry."""

import json
import time

import numpy as np
import paddle


def validate_inference_checkpoint(inference_only, checkpoint):
    if inference_only and (checkpoint is None or not checkpoint.is_file()):
        raise ValueError("Inference requires an existing --initial-checkpoint file")


def check_loss(loss, step):
    value = float(loss)
    if not np.isfinite(value):
        raise FloatingPointError(f"Nonfinite loss at step {step}: {value}")
    return value


def benchmark_forward(net, inputs, output, scope, save_logits=False):
    durations = []
    with paddle.no_grad():
        for iteration in range(40):
            paddle.device.synchronize()
            start = time.perf_counter()
            result = net(inputs)
            paddle.device.synchronize()
            if iteration >= 10:
                durations.append(1000 * (time.perf_counter() - start))
        if save_logits:
            np.save(output / "logits.npy", result.numpy())
    (output / "inference.json").write_text(
        json.dumps(
            {
                "median_ms": float(np.median(durations)),
                "times_ms": durations,
                "warmup": 10,
                "iterations": 30,
                "scope": scope,
            },
            indent=2,
        )
    )
