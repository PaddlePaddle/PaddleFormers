#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(pwd)}
BASE_CONFIG=${BASE_CONFIG:-$ROOT_DIR/examples/config/sft/mimo_gsm8k_reduced_depth_fullwidth_300.yaml}
GPUS=${GPUS:-0,1,2,3}
MAX_STEPS=${MAX_STEPS:-30}
DYNAMIC_CONFIG=${DYNAMIC_CONFIG:-/tmp/mimo_train_dynamic.yaml}
STATIC_CONFIG=${STATIC_CONFIG:-/tmp/mimo_train_static.yaml}
DYNAMIC_LOG=${DYNAMIC_LOG:-$ROOT_DIR/mimo_gsm8k_reduced_depth_fullwidth_compile_train_dynamic.log}
STATIC_LOG=${STATIC_LOG:-$ROOT_DIR/mimo_gsm8k_reduced_depth_fullwidth_compile_train_static.log}

run_case() {
  local config_file=$1
  local to_static=$2
  local output_dir=$3
  local logging_dir=$4
  local log_file=$5
  local port=$6

  cp "$BASE_CONFIG" "$config_file"
  python "$ROOT_DIR/scripts/llavaonevision1_5/patch_yaml_top_level.py" "$config_file" \
    --set "max_steps=$MAX_STEPS" \
    --set "do_eval=false" \
    --set "eval_steps=1000" \
    --set "save_steps=1000000" \
    --set "output_dir=$output_dir" \
    --set "logging_dir=$logging_dir" \
    --set "to_static=$to_static"

  export FLAGS_embedding_deterministic=1
  export FLAGS_cudnn_deterministic=1
  export FLAGS_use_stride_compute_kernel=False
  export CUDA_VISIBLE_DEVICES=$GPUS

  NNODES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=$port \
    paddleformers-cli train "$config_file" 2>&1 | tee "$log_file"
}

run_case "$DYNAMIC_CONFIG" false \
  ./checkpoints/mimo-gsm8k-reduced-depth-fullwidth-compile-dynamic \
  ./vdl_log_mimo_reduced_depth_fullwidth_compile_dynamic \
  "$DYNAMIC_LOG" 36792

run_case "$STATIC_CONFIG" true \
  ./checkpoints/mimo-gsm8k-reduced-depth-fullwidth-compile-static \
  ./vdl_log_mimo_reduced_depth_fullwidth_compile_static \
  "$STATIC_LOG" 36793
