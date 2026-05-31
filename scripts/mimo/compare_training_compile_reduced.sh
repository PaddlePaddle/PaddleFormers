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
DYNAMIC_OUTPUT_DIR=${DYNAMIC_OUTPUT_DIR:-./checkpoints/mimo-gsm8k-reduced-depth-fullwidth-compile-dynamic}
STATIC_OUTPUT_DIR=${STATIC_OUTPUT_DIR:-./checkpoints/mimo-gsm8k-reduced-depth-fullwidth-compile-static}
DYNAMIC_LOGGING_DIR=${DYNAMIC_LOGGING_DIR:-./vdl_log_mimo_reduced_depth_fullwidth_compile_dynamic}
STATIC_LOGGING_DIR=${STATIC_LOGGING_DIR:-./vdl_log_mimo_reduced_depth_fullwidth_compile_static}
PADDLEFORMERS_DIST_LOG=${PADDLEFORMERS_DIST_LOG:-/tmp/mimo_assets/dist_log}
export PADDLEFORMERS_DIST_LOG
mkdir -p "$PADDLEFORMERS_DIST_LOG"
RUN_CASES=${RUN_CASES:-both}

if [[ -n "${PYTHON_RECURSION_LIMIT:-}" ]]; then
  SITE_HOOK_DIR=${SITE_HOOK_DIR:-/tmp/mimo_assets/python_site}
  mkdir -p "$SITE_HOOK_DIR"
  cat > "$SITE_HOOK_DIR/sitecustomize.py" <<EOF
import sys
sys.setrecursionlimit(${PYTHON_RECURSION_LIMIT})
EOF
  export PYTHONPATH="$SITE_HOOK_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

run_case() {
  local config_file=$1
  local to_static=$2
  local output_dir=$3
  local logging_dir=$4
  local log_file=$5
  local port=$6

  cp "$BASE_CONFIG" "$config_file"
  python "$ROOT_DIR/scripts/mimo/patch_yaml_top_level.py" "$config_file" \
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

if [[ "$RUN_CASES" == "both" || "$RUN_CASES" == "dynamic" ]]; then
  run_case "$DYNAMIC_CONFIG" false \
    "$DYNAMIC_OUTPUT_DIR" \
    "$DYNAMIC_LOGGING_DIR" \
    "$DYNAMIC_LOG" 36792
fi

if [[ "$RUN_CASES" == "both" || "$RUN_CASES" == "static" ]]; then
  run_case "$STATIC_CONFIG" true \
    "$STATIC_OUTPUT_DIR" \
    "$STATIC_LOGGING_DIR" \
    "$STATIC_LOG" 36793
fi
