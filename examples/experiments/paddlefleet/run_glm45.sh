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
set -eux
# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset NVSHMEM_HCA_LIST
unset CUDA_DEVICE_MAX_CONNECTIONS

unset GLOG_vmodule GLOG_v
###
# export PYTHONUNB
UFFERED=1
rank=$PADDLE_TRAINER_ID
nnodes=$PADDLE_TRAINERS_NUM
# source /root/paddlejob/tmpspace/wangxiangzhe/erniebot_fc/script/utils.sh
for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

# 加速pin memory save ckpt时间
# 保证集群稳定性的配置，跟性能无关
export NCCL_IB_QPS_PER_CONNECTION=8 
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
# 开启AR功能
export NCCL_IB_ADAPTIVE_ROUTING=1
# 关闭 H 卡 CUDNN FA 功能
export PADDLE_DISABLE_CUDNN_FA=1
# export FLAGS_embedding_deterministic=1
# export FLAGS_cudnn_deterministic=1
# 使用BCCL，需要配合镜像版本 >= FleetY10.1.0
export LD_LIBRARY_PATH=/usr/local/bccl/lib:$LD_LIBRARY_PATH
# 增加tcp_syn_max_backlog, 避免建联失败
export FLAGS_tcp_max_syn_backlog=16384

# 保证先launch的kernel先抢占SM, 提高tpsp_comm_overlap效率
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 确定性flag
export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1 

# 错误发生时打印更全的堆栈信息
export FLAGS_call_stack_level=2
export FLAGS_use_stride_compute_kernel=False
echo "rank (calculated): $rank"    
START_RANK=0  # 改成真正执行的机器号
END_RANK=4 # 改成真正执行的机器号

if [[ $rank -lt $START_RANK ]]; then
   exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
   exit 0
fi

#bash ../../kill_process.sh 
rank=$(($rank-$START_RANK))
nnodes=$(($END_RANK-$START_RANK))
master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
port=36677


rm core.* -rf

root_path="/root/paddlejob/share-storage/gpfs/system-public/wangruting/wangruting"
export PYTHONPATH=$root_path/PaddleFleet/src #修改为自己的paddlefleet路径
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export GLOG_vmodule=nodes=3
#export GLOG_vmodule=dygraph_functions=6
rm -rf /root/paddlejob/tmpspace/checkpoints/glm_sft_ckps 

rm -rf glm_fleet2_log$rank
export PADDLEFORMERS_DIST_LOG=glm_fleet_log$rank
#python3.10 ./examples/experiments/paddlefleet/run_pretrain.py ./examples/experiments/paddlefleet/glm45.json
# python3.10  -m paddle.distributed.launch --device "0,1" ./examples/run_finetune.py ./examples/experiments/paddlefleet/lora.yaml
NNODES=${nnodes} MASTER_ADDR=${master} MASTER_PORT=${port} paddleformers-cli train ./examples/experiments/paddlefleet/lora.yaml
# ./examples/config/sft/full_tp_pp.yaml


# paddleformers-cli train ./examples/experiments/paddlefleet/lora.yaml
# python3.10 -u -m paddle.distributed.launch --master $master:$port --nnodes $nnodes  --rank $rank  ./examples/experiments/paddlefleet/run_pretrain.py ./examples/experiments/paddlefleet/glm45.json \
#   --output_dir $root_path/PaddleFormers/examples/experiments/paddlefleet/outputs # 改成自己的保存模型目录

# python3.10 -m paddle.distributed.launch \
#    --log_dir $root_path/outputs/output_$rank/paddle_distributed_logs \
#    --master $master:$port \
#    --nnodes $nnodes \
#    --rank $rank \
#    --run_mode=collective \
#    ${script:-run_finetune.py}  \
#    $@


bash ../mv_safetensor.sh /root/paddlejob/tmpspace/checkpoints/glm_sft_ckps ./checkpoints/glm_fleet_lora_ckps