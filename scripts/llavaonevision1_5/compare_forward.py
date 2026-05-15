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
"""Compare LLaVA-OneVision-1.5 PaddleFormers logits with Transformers.

This script intentionally imports both frameworks. It is a validation tool, not
PaddleFormers model implementation code.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np
import paddle
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct")
    parser.add_argument(
        "--image",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    )
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-pixels", type=int, default=768 * 768)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    return parser.parse_args()


def build_messages(args):
    content = []
    if not args.text_only:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": args.prompt})
    return [{"role": "user", "content": content}]


def torch_to_paddle_inputs(torch_inputs):
    paddle_inputs = {}
    for key, value in torch_inputs.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            paddle_inputs[key] = paddle.to_tensor(value.detach().cpu().numpy())
        elif isinstance(value, list):
            converted = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    converted.append(paddle.to_tensor(item.detach().cpu().numpy()))
                else:
                    converted.append(item)
            paddle_inputs[key] = converted
        else:
            paddle_inputs[key] = value
    return paddle_inputs


def main():
    args = parse_args()

    messages = build_messages(args)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, max_pixels=args.max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    torch_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    torch_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    torch_model.eval()
    torch_device = next(torch_model.parameters()).device
    torch_forward_inputs = {
        key: value.to(torch_device) if isinstance(value, torch.Tensor) else value
        for key, value in torch_inputs.items()
    }

    with torch.inference_mode():
        torch_outputs = torch_model(**torch_forward_inputs)
        torch_generated = torch_model.generate(**torch_forward_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    torch_logits = torch_outputs.logits.detach().cpu().float().numpy()
    input_len = torch_inputs["input_ids"].shape[1]
    torch_new_tokens = torch_generated[:, input_len : input_len + args.topk].detach().cpu().numpy().tolist()

    del torch_model, torch_outputs, torch_generated, torch_forward_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from paddleformers.transformers import LLaVAOneVision1_5ForConditionalGeneration

    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    paddle_model = LLaVAOneVision1_5ForConditionalGeneration.from_pretrained(args.model)
    paddle_model.eval()
    paddle_inputs = torch_to_paddle_inputs(torch_inputs)

    with paddle.no_grad():
        paddle_outputs = paddle_model(**paddle_inputs)
        paddle_generated = paddle_model.generate(
            **paddle_inputs,
            max_new_tokens=args.max_new_tokens,
            decode_strategy="greedy_search",
            repetition_penalty=args.repetition_penalty,
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
