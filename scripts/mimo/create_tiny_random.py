#!/usr/bin/env python3
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
"""Create a tiny or reduced-depth random MiMo checkpoint for CI/CE smoke tests."""

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

from paddleformers.transformers import MiMoConfig, MiMoForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./tiny-random-mimo")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--safe-serialization", action="store_true", help="Save safetensors instead of Paddle native pdparams."
    )
    parser.add_argument(
        "--save-to-hf", action="store_true", help="Export HF-style keys instead of Paddle native keys."
    )
    parser.add_argument("--save-checkpoint-format", default="naive", choices=["naive", "flex_checkpoint"])
    parser.add_argument(
        "--tokenizer-dir", default=None, help="Optional tokenizer directory to copy into the tiny repo."
    )
    parser.add_argument("--vocab-size", type=int, default=99)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--intermediate-size", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-key-value-heads", type=int, default=2)
    parser.add_argument("--num-nextn-predict-layers", type=int, default=1)
    parser.add_argument("--max-position-embeddings", type=int, default=128)
    parser.add_argument("--rope-theta", type=float, default=640000.0)
    parser.add_argument(
        "--full-width",
        action="store_true",
        help="Use MiMo-7B hidden width and only reduce depth. Requires a large-memory device to instantiate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paddle.seed(args.seed)
    paddle.set_default_dtype(args.dtype)

    vocab_size = args.vocab_size
    if args.tokenizer_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
        vocab_size = len(tokenizer)

    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_attention_heads = args.num_attention_heads
    num_key_value_heads = args.num_key_value_heads
    if args.full_width:
        hidden_size = 4096
        intermediate_size = 11008
        num_attention_heads = 32
        num_key_value_heads = 8
        vocab_size = max(vocab_size, 151680)

    if hidden_size % num_attention_heads != 0:
        raise ValueError("--hidden-size must be divisible by --num-attention-heads.")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("--num-attention-heads must be divisible by --num-key-value-heads.")

    config = MiMoConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_nextn_predict_layers=args.num_nextn_predict_layers,
        max_position_embeddings=args.max_position_embeddings,
        max_window_layers=args.num_hidden_layers,
        use_sliding_window=False,
        sliding_window=args.max_position_embeddings,
        head_dim=hidden_size // num_attention_heads,
        rope_theta=args.rope_theta,
        tie_word_embeddings=False,
        _attn_implementation="eager",
    )

    model = MiMoForCausalLM(config)
    model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
        save_to_hf=args.save_to_hf,
        save_checkpoint_format=args.save_checkpoint_format,
    )

    if args.tokenizer_dir:
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "generation_config.json",
        ]:
            src = os.path.join(args.tokenizer_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(args.output_dir, name))
    print(f"Saved MiMo random checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
