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

cd $root_dir/glm45_fleet
export cur_dir=$(pwd)

# prepare dpo data
wget https://paddle-qa.bj.bcebos.com/fleet/fleet_dpo.tar
tar -xf fleet_dpo.tar

config_dpo_yaml=$root_dir/PaddleFormers/tests/config/ci/glm45_dpo.yaml

config_json=$CACHE_DIR/glm45/GLM-4.5-Air/config.json

yq '.train_dataset_path = strenv(cur_dir) + "/dpo_data/dpo_train.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/dpo_data/dpo_eval.jsonl"
    | .model_name_or_path = strenv(CACHE_DIR) + "/glm45/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/glm_full_dpo_vdl_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints/glm_full_dpo_ckpts"' \
   $config_dpo_yaml > ${config_dpo_yaml}.tmp
mv ${config_dpo_yaml}.tmp $config_dpo_yaml

rm -rf ./outputs
rm -rf paddleformers_dist_log
master=$(hostname -i)
port=36677

export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

unset http_proxy https_proxy

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

set +e
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_dpo_yaml 2>&1 | tee ./glm45_dpo.log
sft_exit_code=$?
if [ $sft_exit_code -ne 0 ]; then
   echo "GLM4.5 multi-cards training failed, try to check the log file"
   python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./glm45_dpo.log "***** eval metrics *****"
   sft_check_exit_code=$?
   if [ $sft_check_exit_code -ne 0 ]; then
     echo "Failed to find 'Training completed' in log file."
     exit 1
   else
     echo "Log check passed."
   fi
fi

set -e
echo "
10 0.46693826
" > ./glm45_dpo_multi_card_gt_loss.txt

python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --compare_step 10 \
   --log_file ./glm45_dpo.log \
   --gt_file ./glm45_dpo_multi_card_gt_loss.txt

echo -e "\033[34mdpo is over, lora is about to start\033[0m"