#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Dump a HF Qwen2 reference bundle from a local MiMo tiny checkpoint.

MiMo's main decoder path is Qwen2-compatible. This helper ignores MTP-only
weights and is meant for local smoke alignment of tiny/reduced checkpoints, not
as a substitute for full MiMo acceptance against the official HF model.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="Solve: If a train travels 60 miles in 2 hours, what is its speed?")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
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

    config = Qwen2Config.from_pretrained(args.model, trust_remote_code=True)
    state_path = os.path.join(args.model, "model-00001-of-00001.safetensors")
    state_dict = load_file(state_path)
    state_dict = {key: value for key, value in state_dict.items() if ".mtp_layers." not in key}

    model = Qwen2ForCausalLM(config).to(dtype=dtype_map[args.dtype], device=args.device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing Qwen2 weights: {missing}")
    unexpected = [key for key in unexpected if ".mtp_layers." not in key]
    if unexpected:
        raise RuntimeError(f"Unexpected Qwen2 weights: {unexpected}")
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
    decoded = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    metadata = {
        "model": args.model,
        "reference_model": "Qwen2ForCausalLM",
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
    print(f"Saved tiny Qwen2 reference bundle to: {args.output_dir}")
    print(f"torch_first_{args.topk}_tokens: {new_tokens}")
    print(f"decoded: {decoded[0] if decoded else ''}")


if __name__ == "__main__":
    main()
