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

if [ -f 'PaddleFleet/.venv/bin/activate' ]; then
   source PaddleFleet/.venv/bin/activate
fi

export root_dir=$(pwd)

export config_yaml=$root_dir/PaddleFormers/tests/config/ci/glm45_pt_grouped_gemm.yaml
export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/pt

yq eval '.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/eval.jsonl"
    | .model_name_or_path = strenv(CACHE_DIR) + "/glm45/GLM-4.5-Air"
    | .logging_dir = strenv(data_dir) + "/vdl_log"
    | .output_dir = strenv(data_dir) + "/checkpoints"' \
   $config_yaml > ${config_yaml}.tmp
mv ${config_yaml}.tmp $config_yaml

rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

unset http_proxy https_proxy
rm -rf checkpoint/
rm -rf outputs/

log_file=glm45_pt_grouped_gemm.txt
gt_loss_file=glm45_multi_cards_grouped_gemm_gt_loss.txt

set +e
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./${log_file}

exit_code=$?
if [ $exit_code -ne 0 ]; then
   echo "Training failed with exit code $exit_cod, see ./${log_file} for details."
   python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./${log_file}
   check_result=$?
   if [ $check_result -ne 0 ]; then
       echo "Failed to find 'Training completed' in log file."
       exit 1
   else
       echo "Log check passed."
   fi
else
   echo "Test passed."
fi


wget --no-proxy --no-check-certificate https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/latest/${gt_loss_file}
if [ $? -ne 0 ]; then
  echo "To request precision checks for new models, please contact swgu98."
  exit 1
fi

log_loss_file=${log_file%.*}_loss.${log_file##*.}
python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --compare_step 10 \
   --log_file ./${log_file} \
   --log_loss_file ./${log_loss_file} \
   --gt_file ./${gt_loss_file}

if [ $? -ne 0 ]; then
  export repo_name=$(echo $GITHUB_REPO_NAME | awk -F'/' '{print $2}')
  pushd $root_dir/PaddleFormers
  bash $root_dir/PaddleFormers/tests/integration_test/check_precision_approval.sh
  if [ $? -ne 0 ]; then
    echo -e "\033[31mThe precision has been changed and requires approvals.\033[0m"
    exit 1
  fi
  popd
  rm ${gt_loss_file} && mv ${log_loss_file} ${gt_loss_file}
  if [ ! -f precision_list.txt ]; then
    wget --no-proxy --no-check-certificate https://paddle-github-action.cdn.bcebos.com/PaddleFleet/precision/${repo_name}_${PR_ID}/precision_list.txt
  fi
  if [ $? -ne 0 ]; then
    wget --no-proxy --no-check-certificate https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/latest/precision_list.txt
    python $root_dir/bos/BosClient.py precision_list.txt paddle-github-action/PaddleFleet/precision/${repo_name}_${PR_ID}
  fi
  python $root_dir/bos/BosClient.py ${gt_loss_file} paddle-github-action/PaddleFleet/precision/${repo_name}_${PR_ID}
  cat ${gt_loss_file}
fi