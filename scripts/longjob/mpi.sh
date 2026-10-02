#!/bin/bash

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

export master=`head -1 ${TRAIN_WORKSPACE}/hostfile |awk '{print $1}'`

if [[ ${IS_STANDALONE:-1} -eq 0 ]]; then
    mpirun \
        --allow-run-as-root \
        -tag-output -timestamp-output \
        -mca btl_tcp_if_exclude docker0,lo,matrixdummy0,matrix0 \
        -pernode \
        --bind-to none \
        -x iplist=${TRAINER_IP_LIST} \
        -x PATH \
        -x LD_LIBRARY_PATH \
        -x NCCL_DEBUG=INFO  \
        -x NCCL_ERROR_FILE=/root/paddlejob/workspace/log/err.nccl.%p.log \
        -x script \
        -x pt_args \
        -x PYTHONPATH \
        -x PAIMON_CONFIG \
        -x restore_ckpt \
        -x RANDOM_PORT=100 \
        -x gpus=8 \
        -x master=${master} \
        -x restore_state \
        -x copyrun_root_dir \
        -x expr_name \
        -x outdir \
        $@
    echo 'mpirun finished at' `date '+%Y-%m-%d %T'`
else
    $@
fi
