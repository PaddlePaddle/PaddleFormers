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
"""Compare Paddle LLaVA-OneVision-1.5 outputs with a dumped HF reference."""

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

from paddleformers.transformers import LLaVAOneVision1_5ForConditionalGeneration

INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct")
    parser.add_argument("--reference-dir", default="./llavaonevision1_5_reference")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-checkpoint-format", default="flex_checkpoint")
    parser.add_argument("--no-convert-from-hf", action="store_true")
    parser.add_argument("--load-via-cpu", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--disable-fused-rms-norm", action="store_true")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--logits-to-keep", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Generation repetition penalty used by the HF reference generation_config.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paddle.set_device(args.device if args.device == "cpu" or paddle.is_compiled_with_cuda() else "cpu")

    reference_path = os.path.join(args.reference_dir, "reference.npz")
    metadata_path = os.path.join(args.reference_dir, "metadata.json")
    reference = np.load(reference_path)
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    paddle_inputs = {key: paddle.to_tensor(reference[key]) for key in INPUT_KEYS if key in reference.files}
    reference_logits = reference["reference_logits"].astype("float32")
    if args.logits_to_keep:
        reference_logits = reference_logits[:, -args.logits_to_keep :, :]
    reference_generated = reference["reference_generated"]

    model = LLaVAOneVision1_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=args.dtype,
        load_checkpoint_format=args.load_checkpoint_format,
        convert_from_hf=not args.no_convert_from_hf,
        load_via_cpu=args.load_via_cpu,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
    )
    if args.disable_fused_rms_norm or args.device == "cpu":
        model.config.text_config.fuse_rms_norm = False
        model.model.language_model.config.fuse_rms_norm = False
    if args.attn_implementation is not None:
        model.config.text_config._attn_implementation = args.attn_implementation
        model.model.language_model.config._attn_implementation = args.attn_implementation
        model.config.vision_config._attn_implementation = args.attn_implementation
        model.model.visual.config._attn_implementation = args.attn_implementation
    model.eval()

    with paddle.no_grad():
        outputs = model(**paddle_inputs, logits_to_keep=args.logits_to_keep)
        generated = None
        if not args.skip_generate:
            generated = model.generate(
                **paddle_inputs,
                max_new_tokens=args.max_new_tokens,
                decode_strategy="greedy_search",
                repetition_penalty=args.repetition_penalty,
            )

    paddle_logits = outputs.logits.astype("float32").numpy()
    if paddle_logits.shape != reference_logits.shape:
        raise ValueError(f"logits shape mismatch: Paddle {paddle_logits.shape}, HF {reference_logits.shape}")

    diff = np.abs(paddle_logits - reference_logits)
    print(f"max_diff: {diff.max():.8f}")
    print(f"mean_diff: {diff.mean():.8f}")
    last_diff = np.abs(paddle_logits[:, -1, :] - reference_logits[:, -1, :])
    print(f"last_max_diff: {last_diff.max():.8f}")
    print(f"last_mean_diff: {last_diff.mean():.8f}")

    if generated is not None:
        input_len = int(metadata.get("input_length", paddle_inputs["input_ids"].shape[1]))
        if isinstance(generated, tuple):
            generated = generated[0]
        if generated.shape[1] > input_len:
            paddle_tokens = generated[:, input_len : input_len + args.topk].numpy().tolist()
        else:
            paddle_tokens = generated[:, : args.topk].numpy().tolist()
        reference_tokens = reference_generated[:, input_len : input_len + args.topk].tolist()
        print(f"hf_first_{args.topk}_tokens: {reference_tokens}")
        print(f"paddle_first_{args.topk}_tokens: {paddle_tokens}")
        print(f"first_{args.topk}_tokens_match: {reference_tokens == paddle_tokens}")


if __name__ == "__main__":
    main()
