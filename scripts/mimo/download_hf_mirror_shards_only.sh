#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

LOCAL_DIR=${LOCAL_DIR:-./models/MiMo-7B-Base}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
REPO_ID=${REPO_ID:-XiaomiMiMo/MiMo-7B-Base}

mkdir -p "$LOCAL_DIR"

SHARDS=(
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
)

echo "Using endpoint: $HF_ENDPOINT"
echo "Repo: $REPO_ID"
echo "Local dir: $LOCAL_DIR"

for shard in "${SHARDS[@]}"; do
  url="$HF_ENDPOINT/$REPO_ID/resolve/main/$shard"
  echo "Downloading shard: $shard"
  wget -c \
    --tries=0 \
    --retry-connrefused \
    --read-timeout=30 \
    --connect-timeout=15 \
    --no-check-certificate \
    -O "$LOCAL_DIR/$shard" \
    "$url"
done

echo "Shard summary:"
find "$LOCAL_DIR" -maxdepth 1 -name "model-*.safetensors" -type f -printf "%f\t%s bytes\n" | sort
du -sh "$LOCAL_DIR"
