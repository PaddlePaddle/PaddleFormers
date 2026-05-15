#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(pwd)}
CONFIG_YAML=${CONFIG_YAML:-$ROOT_DIR/examples/config/sft/llavaonevision1_5_gsm8k_300.yaml}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/sda/data/Lichenyang/llavaonevision1_5_paddle_native_naive_bf16_v2}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-$ROOT_DIR/data/gsm8k_erniekit/train.jsonl}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-$ROOT_DIR/data/gsm8k_erniekit/test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/checkpoints/llavaonevision1_5-gsm8k-sft-300}
LOG_FILE=${LOG_FILE:-$ROOT_DIR/llavaonevision1_5_gsm8k_sft_300.log}
GPUS=${GPUS:-0}

python "$ROOT_DIR/scripts/llavaonevision1_5/patch_yaml_top_level.py" "$CONFIG_YAML" \
  --set "model_name_or_path=$MODEL_NAME_OR_PATH" \
  --set "train_dataset_path=$TRAIN_DATASET_PATH" \
  --set "eval_dataset_path=$EVAL_DATASET_PATH" \
  --set "output_dir=$OUTPUT_DIR"

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=$GPUS

master=$(hostname -i)
port=${MASTER_PORT:-36677}

NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port paddleformers-cli train "$CONFIG_YAML" 2>&1 | tee "$LOG_FILE"
