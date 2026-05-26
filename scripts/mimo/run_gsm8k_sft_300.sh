#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(pwd)}
CONFIG_YAML=${CONFIG_YAML:-$ROOT_DIR/examples/config/sft/mimo_gsm8k_300.yaml}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-XiaomiMiMo/MiMo-7B-Base}
DATA_DIR=${DATA_DIR:-$ROOT_DIR/data/gsm8k_erniekit}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/checkpoints/mimo-gsm8k-sft-300}
LOG_FILE=${LOG_FILE:-$ROOT_DIR/mimo_gsm8k_sft_300.log}
TMP_CONFIG=${TMP_CONFIG:-/tmp/mimo_gsm8k_sft_300.yaml}

cp "$CONFIG_YAML" "$TMP_CONFIG"
python "$ROOT_DIR/scripts/llavaonevision1_5/patch_yaml_top_level.py" "$TMP_CONFIG" \
  --set "train_dataset_path=$DATA_DIR/train.jsonl" \
  --set "eval_dataset_path=$DATA_DIR/test.jsonl" \
  --set "model_name_or_path=$MODEL_NAME_OR_PATH" \
  --set "output_dir=$OUTPUT_DIR"

paddleformers-cli train "$TMP_CONFIG" 2>&1 | tee "$LOG_FILE"
