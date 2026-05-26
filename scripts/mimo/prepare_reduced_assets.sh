#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-./.cache/mimo/reduced-depth-4l-fullwidth-random}
TOKENIZER_DIR=${TOKENIZER_DIR:-}
LAYERS=${LAYERS:-4}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-512}
DTYPE=${DTYPE:-float32}

ARGS=(
  --output-dir "$OUTPUT_DIR"
  --num-hidden-layers "$LAYERS"
  --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
  --dtype "$DTYPE"
  --full-width
)

if [[ -n "$TOKENIZER_DIR" ]]; then
  ARGS+=(--tokenizer-dir "$TOKENIZER_DIR")
fi

python scripts/mimo/create_tiny_random.py "${ARGS[@]}"
echo "Prepared MiMo reduced-depth full-width asset at: $OUTPUT_DIR"
