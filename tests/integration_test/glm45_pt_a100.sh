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

if [ -f 'PaddleFleet/.venv/bin/activate' ]; then
   source PaddleFleet/.venv/bin/activate
fi

wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/glm45/glm45_fleet.12-18.tar --no-check-certificate
tar -xf glm45_fleet.12-18.tar # glm45_fleet
cd $root_dir/glm45_fleet
export cur_dir=$(pwd)

config_yaml=$root_dir/PaddleFormers/tests/config/ci/glm45_pt.yaml
export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/pt

yq eval '.expert_model_parallel_size = 1
    | .num_hidden_layers = 2
    | .per_device_train_batch_size = 1
    | .use_expert_parallel = false
    | .stage1_overlap = false
    | .train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/eval.jsonl"
    | .model_name_or_path = strenv(cur_dir) + "/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/vdl_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints"' \
  $config_yaml > ${config_yaml}.tmp
mv ${config_yaml}.tmp $config_yaml

rm -rf checkpoints/
rm -rf vdl_log/
master=$(hostname -i)
port=36677

unset http_proxy https_proxy

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

log_file=glm45_pt_a100.txt
gt_loss_file=glm45_pt_multi_card_a100_gt_loss.txt

set +e
FLAGS_use_stride_compute_kernel=False NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./${log_file}


exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Test failed with exit code $exit_code, check the log: ./${log_file}"
    python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./${log_file} "***** train metrics *****"
    check_log_exit_code=$?
    if [ $check_log_exit_code -ne 0 ]; then
        echo "Failed to find 'Training completed' in log file."
        exit 1
    else
        echo "Log check passed"
    fi
else
    echo "Test passed."
fi

# export repo_name=$(echo $GITHUB_REPO_NAME | awk -F'/' '{print $2}')
export repo_name=PaddleFleet
# if [[ "${PP}" == "rel" ]]; then
#   export pppatch="_PPrel"
# fi
# if [[ "${PF}" == rel* ]]; then
#   export pfpatch="rel"
# fi
wget --no-proxy --no-check-certificate https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/${repo_name}${pfpatch}${pppatch}_latest/${gt_loss_file}
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
  pushd $root_dir/PaddleFormers
  source /root/proxy
  bash $root_dir/PaddleFormers/tests/integration_test/check_precision_approval.sh
  if [ $? -ne 0 ]; then
    echo -e "\033[31mThe precision has been changed and requires approvals.\033[0m"
    exit 1
  fi
  popd
  rm ${gt_loss_file} && mv ${log_loss_file} ${gt_loss_file}
  if [ ! -f precision_list.txt ]; then
    wget --no-proxy --no-check-certificate https://paddle-github-action.cdn.bcebos.com/PaddleFleet/precision/${repo_name}${pfpatch}${pppatch}/${PR_ID}/precision_list.txt
    if [ $? -ne 0 ]; then
      wget --no-proxy --no-check-certificate https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/${repo_name}${pfpatch}${pppatch}_latest/precision_list.txt
      python $root_dir/bos/BosClient.py precision_list.txt paddle-github-action/PaddleFleet/precision/${repo_name}${pfpatch}${pppatch}/${PR_ID}
    fi
  fi
  python $root_dir/bos/BosClient.py ${gt_loss_file} paddle-github-action/PaddleFleet/precision/${repo_name}${pfpatch}${pppatch}/${PR_ID}
fi
