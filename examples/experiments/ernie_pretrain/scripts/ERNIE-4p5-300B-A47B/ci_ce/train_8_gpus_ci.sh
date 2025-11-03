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

export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
unset GLOG_vmodule GLOG_v
export PADDLE_DISABLE_CUDNN_FA=1
export FLAGS_use_auto_growth_pinned_allocator=True
export FLAGS_pipeline_nccl_comm_init_option=1
export FLAGS_sharding_v2_check_zero_padding=1
export FLAGS_use_paddle_recall_error=0
export FLAGS_tcp_max_syn_backlog=16384
export FLAGS_call_stack_level=2

SM=`nvidia-smi --query-gpu=compute_cap --format=csv | tail -n 1 | sed 's/\.//g'`
if [ $SM -eq 90 ]
then
    export FLAGS_flash_attn_version=3
else
    export FLAGS_flash_attn_version=2
fi

export PYTHONPATH=$PYTHONPATH:./ernie

python -m paddle.distributed.launch \
    --master $hostname:36677 \
    --nnodes 1 \
    --run_mode=collective \
    ${script:-ernie/pretrain.py}  \
    --config yamls/ERNIE-4p5-300B-A47B/ci_ce/pretrain_8_gpus_ci.yaml
