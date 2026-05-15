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
"""Create a tiny random LLaVA-OneVision-1.5 checkpoint for CI/CE smoke tests."""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import paddle
from transformers import AutoTokenizer

from paddleformers.transformers import LLaVAOneVision1_5ForConditionalGeneration, Llavaonevision1_5Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./tiny-random-llavaonevision1_5")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--tokenizer-dir", default=None, help="Optional tokenizer directory to copy into the tiny repo.")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size; useful when tokenizer length differs from model config vocab_size.")
    parser.add_argument("--text-hidden-size", type=int, default=32)
    parser.add_argument("--text-intermediate-size", type=int, default=64)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--text-attention-heads", type=int, default=4)
    parser.add_argument("--text-kv-heads", type=int, default=2)
    parser.add_argument("--vision-hidden-size", type=int, default=32)
    parser.add_argument("--vision-intermediate-size", type=int, default=64)
    parser.add_argument("--vision-depth", type=int, default=2)
    parser.add_argument("--vision-heads", type=int, default=4)
    parser.add_argument("--max-position-embeddings", type=int, default=64)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--no-rope-scaling", action="store_true", help="Use rope_scaling=None like the original 8B config.")
    parser.add_argument("--image-token-id", type=int, default=None)
    parser.add_argument("--video-token-id", type=int, default=None)
    parser.add_argument("--vision-start-token-id", type=int, default=None)
    return parser.parse_args()


def build_tiny_config(
    vocab_size=99,
    text_hidden_size=32,
    text_intermediate_size=64,
    text_layers=2,
    text_attention_heads=4,
    text_kv_heads=2,
    vision_hidden_size=32,
    vision_intermediate_size=64,
    vision_depth=2,
    vision_heads=4,
    max_position_embeddings=64,
    rope_theta=10000.0,
    rope_scaling=None,
    image_token_id=None,
    video_token_id=None,
    vision_start_token_id=None,
):
    text_head_dim = text_hidden_size // text_attention_heads
    if image_token_id is None:
        image_token_id = min(151646, vocab_size - 3)
    if video_token_id is None:
        video_token_id = min(151647, vocab_size - 2)
    if vision_start_token_id is None:
        vision_start_token_id = min(151648, vocab_size - 1)
    return Llavaonevision1_5Config(
        text_config={
            "vocab_size": vocab_size,
            "hidden_size": text_hidden_size,
            "intermediate_size": text_intermediate_size,
            "num_hidden_layers": text_layers,
            "num_attention_heads": text_attention_heads,
            "num_key_value_heads": text_kv_heads,
            "head_dim": text_head_dim,
            "max_position_embeddings": max_position_embeddings,
            "max_window_layers": text_layers,
            "rope_theta": rope_theta,
            "rope_scaling": rope_scaling,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "_attn_implementation": "eager",
        },
        vision_config={
            "depth": vision_depth,
            "hidden_size": vision_hidden_size,
            "embed_dim": vision_hidden_size,
            "intermediate_size": vision_intermediate_size,
            "num_heads": vision_heads,
            "in_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "text_hidden_size": text_hidden_size,
            "_attn_implementation": "eager",
        },
        vocab_size=vocab_size,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
    )


def main():
    args = parse_args()
    paddle.seed(args.seed)
    paddle.set_default_dtype(args.dtype)
    vocab_size = 99
    if args.tokenizer_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        vocab_size = len(tokenizer)
    if args.vocab_size is not None:
        vocab_size = args.vocab_size

    if args.text_hidden_size % args.text_attention_heads != 0:
        raise ValueError("--text-hidden-size must be divisible by --text-attention-heads.")
    if args.text_attention_heads % args.text_kv_heads != 0:
        raise ValueError("--text-attention-heads must be divisible by --text-kv-heads.")
    if args.vision_hidden_size % args.vision_heads != 0:
        raise ValueError("--vision-hidden-size must be divisible by --vision-heads.")

    rope_scaling = None if args.no_rope_scaling else {"type": "mrope", "mrope_section": [1, 1, 2]}
    config = build_tiny_config(
        vocab_size=vocab_size,
        text_hidden_size=args.text_hidden_size,
        text_intermediate_size=args.text_intermediate_size,
        text_layers=args.text_layers,
        text_attention_heads=args.text_attention_heads,
        text_kv_heads=args.text_kv_heads,
        vision_hidden_size=args.vision_hidden_size,
        vision_intermediate_size=args.vision_intermediate_size,
        vision_depth=args.vision_depth,
        vision_heads=args.vision_heads,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        rope_scaling=rope_scaling,
        image_token_id=args.image_token_id,
        video_token_id=args.video_token_id,
        vision_start_token_id=args.vision_start_token_id,
    )
    model = LLaVAOneVision1_5ForConditionalGeneration(config)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    if args.tokenizer_dir:
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "chat_template.jinja",
            "preprocessor_config.json",
            "processor_config.json",
            "image_processor_config.json",
        ]:
            src = os.path.join(args.tokenizer_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(args.output_dir, name))
    print(f"Saved tiny random checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
