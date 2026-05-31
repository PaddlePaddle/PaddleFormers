#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Compare MiMo PaddleFormers logits and greedy tokens with Transformers."""

from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np
import paddle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="XiaomiMiMo/MiMo-7B-Base")
    parser.add_argument("--prompt", default="Solve: If a train travels 60 miles in 2 hours, what is its speed?")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--torch-device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--load-checkpoint-format", default="naive", choices=["naive", "sharding_io", "flex_checkpoint"]
    )
    parser.add_argument("--no-convert-from-hf", action="store_true")
    parser.add_argument("--load-via-cpu", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    text = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    torch_inputs = tokenizer([text], return_tensors="pt")

    torch_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.torch_device,
        trust_remote_code=True,
    )
    torch_model.eval()
    torch_forward_inputs = {key: value.to(torch_model.device) for key, value in torch_inputs.items()}
    with torch.inference_mode():
        torch_outputs = torch_model(**torch_forward_inputs)
        torch_generated = torch_model.generate(
            **torch_forward_inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    torch_logits = torch_outputs.logits.detach().cpu().float().numpy()
    input_len = int(torch_inputs["input_ids"].shape[1])
    torch_new_tokens = torch_generated[:, input_len : input_len + args.topk].detach().cpu().numpy().tolist()

    del torch_model, torch_outputs, torch_generated, torch_forward_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from paddleformers.transformers import MiMoForCausalLM

    paddle.set_device(args.device)
    paddle_model = MiMoForCausalLM.from_pretrained(
        args.model,
        dtype=args.dtype,
        load_checkpoint_format=args.load_checkpoint_format,
        convert_from_hf=not args.no_convert_from_hf,
        load_via_cpu=args.load_via_cpu,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    paddle_model.eval()
    paddle_inputs = {key: paddle.to_tensor(value.detach().cpu().numpy()) for key, value in torch_inputs.items()}
    with paddle.no_grad():
        paddle_outputs = paddle_model(**paddle_inputs)
        paddle_generated = paddle_model.generate(
            **paddle_inputs,
            max_new_tokens=args.max_new_tokens,
            decode_strategy="greedy_search",
        )

    paddle_logits = paddle_outputs.logits.astype("float32").numpy()
    diff = np.abs(torch_logits - paddle_logits)
    print(f"max_diff: {diff.max():.8f}")
    print(f"mean_diff: {diff.mean():.8f}")

    if isinstance(paddle_generated, tuple):
        paddle_generated = paddle_generated[0]
    if paddle_generated.shape[1] > input_len:
        paddle_new_tokens = paddle_generated[:, input_len : input_len + args.topk].numpy().tolist()
    else:
        paddle_new_tokens = paddle_generated[:, : args.topk].numpy().tolist()
    print(f"torch_first_{args.topk}_tokens: {torch_new_tokens}")
    print(f"paddle_first_{args.topk}_tokens: {paddle_new_tokens}")
    print(f"first_{args.topk}_tokens_match: {torch_new_tokens == paddle_new_tokens}")


if __name__ == "__main__":
    main()
