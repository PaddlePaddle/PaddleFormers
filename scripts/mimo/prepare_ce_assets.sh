#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(pwd)}
CACHE_DIR=${CACHE_DIR:-$ROOT_DIR/.cache}
TINY_DST=${TINY_DST:-$CACHE_DIR/mimo/tiny-random-mimo}
REDUCED_DST=${REDUCED_DST:-$CACHE_DIR/mimo/reduced-depth-4l-fullwidth-random}
TOKENIZER_DIR=${TOKENIZER_DIR:-}
DTYPE=${DTYPE:-float32}
PYTHON=${PYTHON:-python3}

mkdir -p "$(dirname "$TINY_DST")" "$(dirname "$REDUCED_DST")"

tiny_args=(--output-dir "$TINY_DST")
if [[ -n "$TOKENIZER_DIR" ]]; then
  tiny_args+=(--tokenizer-dir "$TOKENIZER_DIR")
fi

"$PYTHON" "$ROOT_DIR/scripts/mimo/create_tiny_random.py" "${tiny_args[@]}"

reduced_args=(
  --output-dir "$REDUCED_DST"
  --num-hidden-layers 4
  --max-position-embeddings 512
  --dtype "$DTYPE"
  --full-width
)
if [[ -n "$TOKENIZER_DIR" ]]; then
  reduced_args+=(--tokenizer-dir "$TOKENIZER_DIR")
fi

"$PYTHON" "$ROOT_DIR/scripts/mimo/create_tiny_random.py" "${reduced_args[@]}"

echo "Prepared MiMo CE assets:"
echo "  tiny:    $TINY_DST"
echo "  reduced: $REDUCED_DST"
