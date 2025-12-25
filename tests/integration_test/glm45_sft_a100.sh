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

config_sft_yaml=$cur_dir/tp4_sp_ep4_pp2_sft.yaml
config_lora_yaml=$cur_dir/tp4_sp_ep4_pp2_lora.yaml

config_json=$CACHE_DIR/glm45/GLM-4.5-Air/config.json

yq '.moe_router_force_load_balancing = true
    | .train_dataset_path = strenv(cur_dir) + "/data/sft/train.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/sft/dev.jsonl"
    | .model_name_or_path = strenv(CACHE_DIR) + "/glm45/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/glm_full_pp_vdl_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints/glm_full_pp_ckpts"' \
   $config_sft_yaml > ${config_sft_yaml}.tmp
mv ${config_sft_yaml}.tmp $config_sft_yaml

yq '.moe_router_force_load_balancing = true
    | .train_dataset_path = strenv(cur_dir) + "/data/sft/train.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/sft/dev.jsonl"
    | .model_name_or_path = strenv(CACHE_DIR) + "/glm45/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/glm_full_single_lora_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints/glm_single_lora_ckps"' \
   $config_lora_yaml > ${config_lora_yaml}.tmp
mv ${config_lora_yaml}.tmp $config_lora_yaml

jq '.num_hidden_layers = 2' \
    $config_json > ${config_json}.tmp
mv ${config_json}.tmp $config_json

rm -rf ./outputs
rm -rf paddleformers_dist_log
master=$(hostname -i)
port=36677

export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

unset http_proxy https_proxy


set +e
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_sft_yaml 2>&1 | tee ./glm45_sft.log

sft_exit_code=$?
if [ $sft_exit_code -ne 0 ]; then
   echo "GLM4.5 multi-cards training failed, try to check the log file"
   python $root_dir/PaddleFleet/ci/check_log_for_exitcode.py ./glm45_sft.log
   sft_check_exit_code=$?
   if [ $sft_check_exit_code -ne 0 ]; then
     echo "Failed to find 'Training completed' in log file."
     exit 1
   else
     echo "Log check passed."
   fi
else
   echo "SFT Test passed."
fi

echo -e "\033[34msft is over, lora is about to start\033[0m"

NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_lora_yaml 2>&1 | tee ./glm45_lora.log

lora_exit_code=$?
if [ $lora_exit_code -ne 0 ]; then
   echo "GLM4.5 multi-cards training failed, try to check the log file"
   python $root_dir/PaddleFleet/ci/check_log_for_exitcode.py ./glm45_lora.log
   lora_check_exit_code=$?
   if [ $lora_check_exit_code -ne 0 ]; then
     echo "Failed to find 'Training completed' in log file."
     exit 1
   else
     echo "Log check passed."
   fi
else
    echo "LORA Test passed."
fi
