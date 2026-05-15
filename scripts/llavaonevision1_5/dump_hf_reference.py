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
"""Dump Transformers reference inputs/logits for LLaVA-OneVision-1.5.

This validation helper intentionally depends on PyTorch and Transformers. It is
kept outside the Paddle model implementation so the Paddle port stays pure.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct")
    parser.add_argument(
        "--image",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    )
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument("--output-dir", default="./llavaonevision1_5_reference")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-pixels", type=int, default=768 * 768)
    return parser.parse_args()


def build_messages(args):
    content = []
    if not args.text_only:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": args.prompt})
    return [{"role": "user", "content": content}]


def tensor_to_numpy(value):
    if value.dtype in (torch.float16, torch.bfloat16):
        return value.detach().cpu().float().numpy()
    return value.detach().cpu().numpy()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    messages = build_messages(args)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, max_pixels=args.max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    forward_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**forward_inputs)
        generated = model.generate(**forward_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    arrays = {key: tensor_to_numpy(value) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    arrays["reference_logits"] = outputs.logits.detach().cpu().float().numpy()
    arrays["reference_generated"] = generated.detach().cpu().numpy()
    np.savez(os.path.join(args.output_dir, "reference.npz"), **arrays)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_len : input_len + args.topk].detach().cpu().numpy().tolist()
    decoded = processor.batch_decode(
        [ids[input_len:] for ids in generated],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "image": None if args.text_only else args.image,
        "text": text,
        "input_length": input_len,
        "topk": args.topk,
        "torch_first_tokens": new_tokens,
        "decoded": decoded,
        "input_keys": sorted(arrays.keys()),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved reference bundle to: {args.output_dir}")
    print(f"torch_first_{args.topk}_tokens: {new_tokens}")
    print(f"decoded: {decoded[0] if decoded else ''}")


if __name__ == "__main__":
    main()
