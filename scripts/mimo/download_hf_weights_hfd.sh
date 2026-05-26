#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

REPO_ID=${REPO_ID:-XiaomiMiMo/MiMo-7B-Base}
LOCAL_DIR=${LOCAL_DIR:-/sda/data/Lichenyang/models/MiMo-7B-Base}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
HFD_SCRIPT=${HFD_SCRIPT:-/tmp/hfd.sh}
TOOL=${TOOL:-wget}
THREADS=${THREADS:-4}

export HF_ENDPOINT

mkdir -p "$LOCAL_DIR"

if [ ! -s "$HFD_SCRIPT" ]; then
  wget -O "$HFD_SCRIPT" "$HF_ENDPOINT/hfd/hfd.sh"
  chmod +x "$HFD_SCRIPT"
fi

echo "Using HF_ENDPOINT=$HF_ENDPOINT"
echo "Repo: $REPO_ID"
echo "Local dir: $LOCAL_DIR"
echo "Tool: $TOOL, threads: $THREADS"

bash "$HFD_SCRIPT" "$REPO_ID" \
  --local-dir "$LOCAL_DIR" \
  --tool "$TOOL" \
  -x "$THREADS" \
  --include "*.safetensors" \
  --include "*.json" \
  --include "*.py" \
  --include "tokenizer*" \
  --include "vocab.json" \
  --include "merges.txt"

echo "Download finished. File summary:"
find "$LOCAL_DIR" -maxdepth 1 -type f -printf "%f\t%k KiB\n" | sort
du -sh "$LOCAL_DIR"
