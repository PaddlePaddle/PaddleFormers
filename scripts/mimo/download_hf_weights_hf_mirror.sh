#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOCAL_DIR=${LOCAL_DIR:-/sda/data/Lichenyang/models/MiMo-7B-Base}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
REPO_ID=${REPO_ID:-XiaomiMiMo/MiMo-7B-Base}

export HF_ENDPOINT

mkdir -p "$LOCAL_DIR"

FILES=(
  config.json
  configuration_mimo.py
  generation_config.json
  merges.txt
  model.safetensors.index.json
  modeling_mimo.py
  tokenizer.json
  tokenizer_config.json
  vocab.json
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
)

echo "Using HF mirror endpoint: $HF_ENDPOINT"
echo "Repo: $REPO_ID"
echo "Local dir: $LOCAL_DIR"

for file in "${FILES[@]}"; do
  url="$HF_ENDPOINT/$REPO_ID/resolve/main/$file"
  echo "Downloading $file"
  wget -c --tries=0 --retry-connrefused --read-timeout=30 --connect-timeout=15 \
    --no-check-certificate \
    -O "$LOCAL_DIR/$file" \
    "$url"
done

echo "Download finished. File summary:"
find "$LOCAL_DIR" -maxdepth 1 -type f -printf "%f\t%k KiB\n" | sort
du -sh "$LOCAL_DIR"
