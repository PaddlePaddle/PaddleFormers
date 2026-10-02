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

# export PYTHONPATH=/root/paddlejob/share-storage/gpfs/system-public/lizhenxing/GLM5/PaddleFleet/src:${PYTHONPATH}

source /root/paddlejob/share-storage/gpfs/system-public/lizhenxing/midtrain/PaddleFleet/.venv/bin/activate

# export CUDA_VISIBLE_DEVICES=7
export FLAGS_use_accuracy_compatible_kernel=1
export SAVE_TENSOR_SUBDIRS="moe_layer"

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

unset TRAINER_INSTANCES
unset TRAINER_IP_PORT_LIST
unset TRAINERS_NUM
unset TRAINER_HOSTS_NUM
unset TRAINER_INSTANCES_NUM

export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=162
export NVSHMEM_BOOTSTRAP=UID

unset NVSHMEM_HCA_LIST
unset NVSHMEM_ENABLE_NIC_PE_MAPPING

# 单机运行配置
MASTER_ADDR_PORT="127.0.0.1:36677"
RANK=0
NNODES=1
RANK_ID=0

MASTER_ADDR=$(echo "$MASTER_ADDR_PORT" | cut -d':' -f1)

rm -rf output/logs/*

export CUDA_PATH=/usr/local/cuda-12.9

export FA_VERSION=3
export FLAGS_share_tensor_for_grad_tensor_holder=1
export FLAGS_use_default_stream=false

export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1

# bash scripts/kill_process.sh

MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RANK=$RANK NNODES=$NNODES \
PADDLEFORMERS_DIST_LOG=output/logs/paddle_distributed_logs_${RANK_ID} \
paddleformers-cli train scripts/longjob/FT_test/tiny_glm45_elastic_fc_2nodes.yaml 2>&1 | tee scripts/longjob/FT_test/pf_glm45_a2.log
