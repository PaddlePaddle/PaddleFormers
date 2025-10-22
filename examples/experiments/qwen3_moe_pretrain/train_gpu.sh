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

LOG_DIR=output/trainer_$rank/paddle_distributed_logs
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

LAUNCH_CMD=`python script/selective_launch.py 36677`
if [[ -z "$LAUNCH_CMD" ]]; then
    exit 0
fi

export PYTHONPATH=../:$PYTHONPATH
export CUDA_PATH=/usr/local/cuda-12.9

export USE_DEEPEP=1


# bash script/kill_process.sh 
# export LD_LIBRARY_PATH=/usr/local/nccl:/usr/local/cuda/compat:/usr/local/lib:/home/opt/nvidia_lib:/usr/local/cuda/lib64:/usr/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu
# export FLAGS_benchmark=1
# export FLAGS_call_stack_level=3
# export GLOG_v=6
# export GLOG_vmodule=process_group_nccl=3
# export FLAGS_use_system_allocator=1
# export FLAGS_check_cuda_error=1
export USE_DEEPEP=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# source /root/paddlejob/gpfs/hushenwei/hushenwei_env/bin/activate
# export http_proxy=agent.baidu.com:8188
# export https_proxy=agent.baidu.com:8188
# export TRAINER_INSTANCES=$(hostname -I | awk '{print $1}')

# source /root/paddlejob/workspace/env_run/hushenwei/hushenwei_env/bin/activate


# debug
# export CUDA_VISIBLE_DEVICES=1,2,3,4
# export FLAGS_benchmark=1
# export FLAGS_call_stack_level=3
# export GLOG_v=6
# export GLOG_vmodule=process_group_nccl=3
# export FLAGS_use_system_allocator=1
# export FLAGS_check_cuda_error=1

# export TRAINER_INSTANCES=$(hostname -I | awk '{print $1}')

python3.10 -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    $LAUNCH_CMD \
    --run_mode=collective \
    ${script:-run_pretrain.py}  \
    $@
