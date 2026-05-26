#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(pwd)}
CACHE_DIR=${CACHE_DIR:-$ROOT_DIR/.cache}
TINY_SRC=${TINY_SRC:-$ROOT_DIR/tiny-random-llavaonevision1_5}
TINY_DST=${TINY_DST:-$CACHE_DIR/llavaonevision1_5/tiny-random-llavaonevision1_5}
TOKENIZER_SRC=${TOKENIZER_SRC:-}
PREPARE_REDUCED=${PREPARE_REDUCED:-0}
REDUCED_SRC=${REDUCED_SRC:-$ROOT_DIR/reduced-depth-4l-fullwidth-random-v2}
REDUCED_DST=${REDUCED_DST:-$CACHE_DIR/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2}

if [[ ! -d "$TINY_SRC" ]] || [[ -n "$TOKENIZER_SRC" && ! -f "$TINY_SRC/tokenizer.json" ]]; then
  args=(--output-dir "$TINY_SRC")
  if [[ -n "$TOKENIZER_SRC" ]]; then
    args+=(--tokenizer-dir "$TOKENIZER_SRC")
  fi
  python "$ROOT_DIR/scripts/llavaonevision1_5/create_tiny_random.py" "${args[@]}"
fi

mkdir -p "$TINY_DST"
cp -a "$TINY_SRC"/. "$TINY_DST"/

echo "Prepared tiny checkpoint for CE:"
echo "$TINY_DST"

if [[ "$PREPARE_REDUCED" = "1" ]]; then
  if [[ ! -d "$REDUCED_SRC" ]]; then
    args=(
      --output-dir "$REDUCED_SRC"
      --text-hidden-size 4096
      --text-intermediate-size 12288
      --text-layers 4
      --text-attention-heads 32
      --text-kv-heads 8
      --vision-hidden-size 1024
      --vision-intermediate-size 4096
      --vision-depth 4
      --vision-heads 16
    )
    if [[ -n "$TOKENIZER_SRC" ]]; then
      args+=(--tokenizer-dir "$TOKENIZER_SRC")
    fi
    python "$ROOT_DIR/scripts/llavaonevision1_5/create_tiny_random.py" "${args[@]}"
  fi
  mkdir -p "$REDUCED_DST"
  cp -a "$REDUCED_SRC"/. "$REDUCED_DST"/
  echo "Prepared reduced checkpoint for local validation:"
  echo "$REDUCED_DST"
fi
