#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

MODEL=${MODEL:-./.cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2}
DEVICE=${DEVICE:-gpu}
SEQ_LEN=${SEQ_LEN:-64}
STEPS=${STEPS:-50}
WARMUP_STEPS=${WARMUP_STEPS:-10}
DTYPE=${DTYPE:-bfloat16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}
BENCHMARK_MODE=${BENCHMARK_MODE:-manual-lm-head}

COMMON_ARGS=(
  --model "$MODEL"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --seq-len "$SEQ_LEN"
  --steps "$STEPS"
  --warmup-steps "$WARMUP_STEPS"
  --load-checkpoint-format flex_checkpoint
  --no-convert-from-hf
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --benchmark-mode "$BENCHMARK_MODE"
)

echo "== Reduced dynamic inference =="
python scripts/llavaonevision1_5/benchmark_inference.py "${COMMON_ARGS[@]}"

echo "== Reduced to_static inference =="
python scripts/llavaonevision1_5/benchmark_inference.py "${COMMON_ARGS[@]}" --use-to-static
