#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/sda/data/Lichenyang/PaddleFormers}
IMAGE=${IMAGE:-ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5}
CONFIG=${CONFIG:-examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300.yaml}
LOG=${LOG:-llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300.log}
GPUS=${GPUS:-1}

cd "${ROOT_DIR}"

sg docker -c "docker run --rm --gpus all \
  --ipc host --net host --privileged --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /sda/data/Lichenyang:/root \
  -w /root/PaddleFormers \
  -e CUDA_VISIBLE_DEVICES=${GPUS} \
  -e NVIDIA_VISIBLE_DEVICES=${GPUS} \
  -e PYTHONPATH=/root/PaddleFormers \
  -e FLAGS_embedding_deterministic=1 \
  -e FLAGS_cudnn_deterministic=1 \
  ${IMAGE} \
  /bin/bash -lc 'paddleformers-cli train ${CONFIG} 2>&1 | tee ${LOG}'"
