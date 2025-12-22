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

set -exo pipefail
export root_dir=$(pwd)

source PaddleFleet/.venv/bin/activate
cd PaddleFormers
git pull origin pull/3200/head
cd -

wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/glm45/glm45_fleet.12-18.tar --no-check-certificate
tar -xf glm45_fleet.12-18.tar # glm45_fleet
cd $root_dir/glm45_fleet
# cd $CACHE_DIR/glm
export cur_dir=$(pwd)

config_yaml=$root_dir/PaddleFormers/tests/config/ci/glm45_pt.yaml

yq eval '.train_dataset_path = strenv(cur_dir) + "/data/pre-training/train.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/pre-training/eval.jsonl"
    | .model_name_or_path = strenv(cur_dir) + "/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/vdl_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints"' \
   $config_yaml > ${config_yaml}.tmp
mv ${config_yaml}.tmp $config_yaml

rm -rf ./outputs
rm -rf paddleformers_dist_log
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

unset http_proxy https_proxy

# coverage run -m paddle.distributed.launch \
#    --log_dir ./log \
#    --master $master:$port \
#    --nnodes 1 \
#    --rank 0 \
#    --run_mode=collective \
#    run_pretrain.py $config_json \
#    --output_dir ./checkpoint 2>&1 | tee ./glm45.log

NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./glm45_pt.log

echo "
10 11.70277214
" > ./glm45_pt_multi_card_gt_loss.txt

python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --compare_step 10 \
   --log_file ./glm45_pt.log \
   --gt_file ./glm45_pt_multi_card_gt_loss.txt
