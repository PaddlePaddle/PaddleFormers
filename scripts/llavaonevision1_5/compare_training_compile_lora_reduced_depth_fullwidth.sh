#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/sda/data/Lichenyang/PaddleFormers}
IMAGE=${IMAGE:-ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5}
BASE_CONFIG=${BASE_CONFIG:-/root/PaddleFormers/examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300.yaml}
GPUS=${GPUS:-2}
MAX_STEPS=${MAX_STEPS:-300}
LOGGING_STEPS=${LOGGING_STEPS:-20}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
DYNAMIC_LOG=${DYNAMIC_LOG:-llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_compile_train_dynamic_300.log}
STATIC_LOG=${STATIC_LOG:-llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_compile_train_static_300.log}

run_case() {
  local name=$1
  local to_static=$2
  local log_file=$3
  local tmp_config="/tmp/llavaonevision1_5_lora_${name}_compile.yaml"

  cp "${BASE_CONFIG}" "${tmp_config}"
  python /root/PaddleFormers/scripts/llavaonevision1_5/patch_yaml_top_level.py "${tmp_config}" \
    --set "max_steps=${MAX_STEPS}" \
    --set "do_eval=false" \
    --set "eval_steps=1000000" \
    --set "save_steps=1000000" \
    --set "logging_steps=${LOGGING_STEPS}" \
    --set "recompute_granularity=${RECOMPUTE_GRANULARITY}" \
    --set "output_dir=/root/PaddleFormers/checkpoints/llavaonevision1_5-gsm8k-reduced-depth-fullwidth-lora-compile-${name}-300" \
    --set "logging_dir=./vdl_log_llavaonevision1_5_reduced_depth_fullwidth_lora_compile_${name}_300" \
    --set "to_static=${to_static}"

  paddleformers-cli train "${tmp_config}" 2>&1 | tee "${log_file}"
}

cd "${ROOT_DIR}"

sg docker -c "docker run --rm --gpus all \
  --ipc host --net host --privileged --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /sda/data/Lichenyang:/root \
  -w /root/PaddleFormers \
  -e CUDA_VISIBLE_DEVICES=${GPUS} \
  -e NVIDIA_VISIBLE_DEVICES=${GPUS} \
  -e PYTHONPATH=/root/PaddleFormers \
  -e BASE_CONFIG=${BASE_CONFIG} \
  -e MAX_STEPS=${MAX_STEPS} \
  -e LOGGING_STEPS=${LOGGING_STEPS} \
  -e RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY} \
  -e DYNAMIC_LOG=${DYNAMIC_LOG} \
  -e STATIC_LOG=${STATIC_LOG} \
  -e FLAGS_embedding_deterministic=1 \
  -e FLAGS_cudnn_deterministic=1 \
  -e FLAGS_use_stride_compute_kernel=False \
  ${IMAGE} \
  /bin/bash -lc 'set -euo pipefail; $(declare -f run_case); run_case dynamic false ${DYNAMIC_LOG}; run_case static true ${STATIC_LOG}'"
