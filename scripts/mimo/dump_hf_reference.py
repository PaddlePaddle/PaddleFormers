#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Dump a Transformers reference bundle for MiMo text-only alignment."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="XiaomiMiMo/MiMo-7B-Base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="Solve: If a train travels 60 miles in 2 hours, what is its speed?")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer([text], return_tensors="pt")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    forward_inputs = {key: value.to(model.device) for key, value in inputs.items()}

    outputs = model(**forward_inputs)
    generated = model.generate(**forward_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    arrays = {key: value.cpu().numpy() for key, value in inputs.items()}
    arrays["reference_logits"] = outputs.logits.detach().cpu().float().numpy()
    arrays["reference_generated"] = generated.detach().cpu().numpy()
    np.savez(os.path.join(args.output_dir, "reference.npz"), **arrays)

    input_len = int(inputs["input_ids"].shape[1])
    new_tokens = generated[:, input_len : input_len + args.topk].detach().cpu().numpy().tolist()
    decoded = tokenizer.batch_decode(
        generated[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    metadata = {
        "model": args.model,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "text": text,
        "input_length": input_len,
        "topk": args.topk,
        "torch_first_tokens": new_tokens,
        "decoded": decoded,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved reference bundle to: {args.output_dir}")
    print(f"torch_first_{args.topk}_tokens: {new_tokens}")
    print(f"decoded: {decoded[0] if decoded else ''}")


if __name__ == "__main__":
    main()
