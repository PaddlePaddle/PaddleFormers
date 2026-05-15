#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import argparse
import time

import paddle

from paddleformers.transformers import LLaVAOneVision1_5ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLaVA-OneVision-1.5 inference in PaddleFormers.")
    parser.add_argument("--model", required=True, help="Paddle checkpoint path or model name.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--use-to-static", action="store_true", help="Wrap forward with paddle.jit.to_static.")
    parser.add_argument("--load-checkpoint-format", default="naive", choices=["naive", "flex_checkpoint"])
    parser.add_argument("--no-convert-from-hf", action="store_true")
    parser.add_argument("--load-via-cpu", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default=None)
    parser.add_argument(
        "--benchmark-mode",
        default="full",
        choices=["full", "manual-lm-head", "backbone"],
        help="Use a static-friendly manual lm_head path or benchmark only the backbone.",
    )
    parser.add_argument("--allow-partial-graph", action="store_true", help="Use full_graph=False for to_static.")
    return parser.parse_args()


def sync(device):
    if device == "gpu":
        paddle.device.synchronize()


def manual_lm_head(model, hidden_states):
    logits = paddle.matmul(hidden_states, model.lm_head.weight, transpose_y=True)
    bias = getattr(model.lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias
    return logits


def main():
    args = parse_args()
    paddle.set_device(args.device)
    paddle.set_grad_enabled(False)

    model = LLaVAOneVision1_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=args.dtype,
        load_checkpoint_format=args.load_checkpoint_format,
        convert_from_hf=not args.no_convert_from_hf,
        load_via_cpu=args.load_via_cpu,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
    )
    model.eval()
    if args.attn_implementation is not None:
        model.config.text_config._attn_implementation = args.attn_implementation
        model.model.language_model.config._attn_implementation = args.attn_implementation
        model.config.vision_config._attn_implementation = args.attn_implementation
        model.model.visual.config._attn_implementation = args.attn_implementation
    def forward_full(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    def forward_manual_lm_head(input_ids, attention_mask):
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        return manual_lm_head(model, outputs.last_hidden_state)

    def forward_backbone(input_ids, attention_mask):
        return model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True).last_hidden_state

    if args.benchmark_mode == "full":
        forward_fn = forward_full
    elif args.benchmark_mode == "manual-lm-head":
        forward_fn = forward_manual_lm_head
    else:
        forward_fn = forward_backbone

    if args.use_to_static:
        forward_fn = paddle.jit.to_static(forward_fn, full_graph=not args.allow_partial_graph)

    vocab_size = int(getattr(model.config.text_config, "vocab_size", model.config.vocab_size))
    input_ids = paddle.randint(0, vocab_size, shape=[args.batch_size, args.seq_len], dtype="int64")
    attention_mask = paddle.ones([args.batch_size, args.seq_len], dtype="int64")

    for _ in range(args.warmup_steps):
        forward_fn(input_ids, attention_mask)
    sync(args.device)

    start = time.perf_counter()
    for _ in range(args.steps):
        forward_fn(input_ids, attention_mask)
    sync(args.device)
    elapsed = time.perf_counter() - start

    tokens = args.batch_size * args.seq_len * args.steps
    print(f"use_to_static: {args.use_to_static}")
    print(f"benchmark_mode: {args.benchmark_mode}")
    print(f"attn_implementation: {args.attn_implementation}")
    print(f"steps: {args.steps}")
    print(f"elapsed_sec: {elapsed:.6f}")
    print(f"tokens_per_sec: {tokens / elapsed:.2f}")


if __name__ == "__main__":
    main()
