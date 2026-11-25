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

nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

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
# export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME==xgbe0

unset NVSHMEM_HCA_LIST 
unset NVSHMEM_ENABLE_NIC_PE_MAPPING

LAUNCH_CMD=`python script/selective_launch.py 36677`
if [[ -z "$LAUNCH_CMD" ]]; then
    exit 0
fi

# export PYTHONPATH=../../../:$PYTHONPATH

export PATH=/opt/nvidia/nsight-systems/2025.3.1/bin/:$PATH

export DSV3_USE_FP8_GEMM=true
export DSV3_USE_ATTEN_RECOMPUTE=true
export FA_VERSION=3
export CUDA_PATH=/usr/local/cuda-12.9
export FLAGS_share_tensor_for_grad_tensor_holder=1
export FLAGS_use_default_stream=false
export DSV3_USE_FP8_DISPATCH=true
export USE_DS_GEMM=false

bash script/kill_process.sh

# source /root/paddlejob/workspace/env_run/zhangbo/env_ds/bin/activate
source /root/paddlejob/workspace/env_run/chenxi/chenxi_py3.10/bin/activate

export FLAGS_large_pool_auto_growth_chunk_size_in_mb=500
export FLAGS_small_pool_auto_growth_chunk_size_in_mb=20
export FLAGS_small_pool_size_in_mb=10

export FLAGS_samll_pool_pre_alloc_in_mb=500
export FLAGS_large_pool_pre_alloc_in_mb=61440

export DSV3_FAST_PRETRAIN=true
# nsys profile --stats=true -t cuda,nvtx -o test_no_quant_cache --force-overwrite true \
python3.10 -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    $LAUNCH_CMD \
    --run_mode=collective \
    ${script:-run_pretrain.py}  \
    $@
