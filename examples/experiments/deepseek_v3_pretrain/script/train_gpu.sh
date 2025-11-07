#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

#export FLAGS_shard_bypass_dygraph_optimizer=1
export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=162

#export NVSHMEM_IB_ENABLE_IBGDA=true
##export NVSHMEM_DISABLE_P2P=1
export NVSHMEM_BOOTSTRAP=UID

unset NVSHMEM_HCA_LIST 
unset NVSHMEM_ENABLE_NIC_PE_MAPPING

export PYTHONPATH=../../../:$PYTHONPATH

export CUDA_PATH=/usr/local/cuda-12.9

# Flags for allocator
export FLAGS_large_pool_auto_growth_chunk_size_in_mb=128
export FLAGS_small_pool_auto_growth_chunk_size_in_mb=10
export FLAGS_small_pool_size_in_mb=1
export FLAGS_samll_pool_pre_alloc_in_mb=250
export FLAGS_large_pool_pre_alloc_in_mb=61440
export FLAGS_deep_ep_comm_prealloc_in_mb=3500
export FLAGS_share_tensor_for_grad_tensor_holder=1
export FLAGS_use_default_stream=false
export USE_DS_GEMM=true

python3.10 -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    --master <master_ip>:<port> \
    --nnodes 8 \
    --run_mode=collective \
    ${script:-run_pretrain.py}  \
    $@

# Training command by paddleformers-cli
# NOTE: Please use "pre_alloc_memory" in pretrain_argument.yaml instead of setting "export FLAGS_large_pool_pre_alloc_in_mb". 
# paddleformers-cli train $@