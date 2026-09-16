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

# llama 模型数据下载
# mkdir -p data
# wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
# wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx

rm -rf output
GLOG_v=6 nohup sh script/train_gpu.sh config/pretrain_argument.json > run.log 2>&1 & 

