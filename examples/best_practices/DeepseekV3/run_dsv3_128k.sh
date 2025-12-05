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

source /root/formers_venv/bin/activate #python环境存放位置
export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=162
export NVSHMEM_BOOTSTRAP=UID

unset NVSHMEM_HCA_LIST 
unset NVSHMEM_ENABLE_NIC_PE_MAPPING

# 修改num_nodes， your_master_addr， your_master_port， 为实际配置
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli train /root/PaddleFormers/examples/config/dsv3_128k_config.yaml #配置存放路径