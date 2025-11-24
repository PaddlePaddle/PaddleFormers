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

START_RANK=0 # 改成真正执行的机器号
END_RANK=2 # 改成真正执行的机器号

if [[ $rank -lt $START_RANK ]]; then
   exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
   exit 0
fi
nnodes=$(($END_RANK-$START_RANK))
master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
port=36677

rank=$(($rank-$START_RANK))
#bash script/kill_process.sh 
#sleep 5

python -m paddle.distributed.launch \
   --log_dir ./outputs/output_$rank/paddle_distributed_logs \
   --master $master:$port \
   --nnodes $nnodes \
   --rank $rank \
   --run_mode=collective \
   run_pretrain.py glm45.json \
   --output_dir . # 改成自己的保存模型目录