#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Compare Paddle MiMo outputs with a saved NumPy reference bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import paddle

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from paddleformers.transformers import MiMoForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True, help="Path to reference.npz saved by a HF-side script.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--load-checkpoint-format", default="naive", choices=["naive", "flex_checkpoint"])
    parser.add_argument("--no-convert-from-hf", action="store_true")
    parser.add_argument("--load-via-cpu", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--ignore-mismatched-sizes", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ref = np.load(args.reference)
    input_ids = ref["input_ids"]
    attention_mask = ref["attention_mask"] if "attention_mask" in ref.files else None
    reference_logits = ref["reference_logits"]
    reference_generated = ref["reference_generated"] if "reference_generated" in ref.files else None

    paddle.set_device(args.device)
    model = MiMoForCausalLM.from_pretrained(
        args.model,
        dtype=args.dtype,
        load_checkpoint_format=args.load_checkpoint_format,
        convert_from_hf=not args.no_convert_from_hf,
        load_via_cpu=args.load_via_cpu,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    model.eval()

    inputs = {"input_ids": paddle.to_tensor(input_ids)}
    if attention_mask is not None:
        inputs["attention_mask"] = paddle.to_tensor(attention_mask)

    with paddle.no_grad():
        outputs = model(**inputs)
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            decode_strategy="greedy_search",
        )

    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    paddle_logits = logits.astype("float32").numpy()
    diff = np.abs(reference_logits - paddle_logits)

    if isinstance(generated, tuple):
        generated = generated[0]
    generated_np = generated.numpy()
    input_len = int(input_ids.shape[1])
    if generated_np.shape[1] > input_len:
        paddle_new_tokens = generated_np[:, input_len : input_len + args.topk].tolist()
    else:
        paddle_new_tokens = generated_np[:, : args.topk].tolist()

    reference_new_tokens = None
    first_tokens_match = None
    if reference_generated is not None:
        reference_new_tokens = reference_generated[:, input_len : input_len + args.topk].tolist()
        first_tokens_match = reference_new_tokens == paddle_new_tokens

    result = {
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "reference_first_tokens": reference_new_tokens,
        "paddle_first_tokens": paddle_new_tokens,
        "first_tokens_match": first_tokens_match,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
