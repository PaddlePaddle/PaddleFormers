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

set -exo pipefail
export root_dir=$(pwd)

step=$1

if [ "$step" == "single" ]; then
  export config_yaml=$root_dir/PaddleFormers/tests/config/ci/llavaonevision1_5_sft_single.yaml
  export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/sft
  export model_name_or_path=$CACHE_DIR/llavaonevision1_5/tiny-random-llavaonevision1_5
  export output_dir=$root_dir/checkpoints/llavaonevision1_5-single
fi

if [ "$step" == "lora_single" ]; then
  export config_yaml=$root_dir/PaddleFormers/tests/config/ci/llavaonevision1_5_lora_single.yaml
  export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/sft
  export model_name_or_path=$CACHE_DIR/llavaonevision1_5/tiny-random-llavaonevision1_5
  export output_dir=$root_dir/checkpoints/llavaonevision1_5-lora-single
fi

if [ -z "${config_yaml:-}" ]; then
  echo "Unsupported step: $step"
  exit 1
fi

python $root_dir/PaddleFormers/scripts/llavaonevision1_5/patch_yaml_top_level.py $config_yaml \
  --set "train_dataset_path=$data_dir/train.jsonl" \
  --set "eval_dataset_path=$data_dir/eval.jsonl" \
  --set "model_name_or_path=$model_name_or_path" \
  --set "output_dir=$output_dir"

rm -rf ./outputs
rm -rf paddleformers_dist_log
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False

unset http_proxy https_proxy

git config --global --add safe.directory $root_dir/PaddleFormers || true

if [ "$step" == "single" ]; then
  log_file=llavaonevision1_5_sft_single_card.txt
  gt_loss_file=llavaonevision1_5_sft_single_card_gt_loss.txt
else
  log_file=llavaonevision1_5_${step}_card.txt
  gt_loss_file=llavaonevision1_5_${step}_card_gt_loss.txt
fi

set +e
if command -v coverage >/dev/null 2>&1; then
  train_cmd="coverage run $(which paddleformers-cli) train $config_yaml"
else
  train_cmd="$(which paddleformers-cli) train $config_yaml"
fi
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port bash -c "$train_cmd" 2>&1 | tee ./${log_file}

exit_code=$?
if [ $exit_code -ne 0 ]; then
   echo "llavaonevision1_5 sft training failed, try to check the log file"
   python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./${log_file} "***** train metrics *****"
   check_exit_code=$?
   if [ $check_exit_code -ne 0 ]; then
     echo "Failed to find 'Training completed' in log file."
     exit 1
   else
     echo "Log check passed."
   fi
else
    echo "Test passed."
fi

if [ "${SKIP_PRECISION_CHECK:-0}" = "1" ]; then
  echo "Skip precision check for local smoke."
  exit 0
fi

export repo_name=PaddleFleet
export REPO_NAME=$(echo $GITHUB_REPO_NAME | awk -F'/' '{print $2}')

wget --no-proxy --no-check-certificate https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/${REPO_NAME}${pfpatch}${pppatch}_latest/${gt_loss_file}
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
  if [ "${BRANCH}" != "develop" ]; then
    echo "please update precision in develop and rerun this workflow"
    exit 1
  fi
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
    wget --no-proxy --no-check-certificate https://paddle-github-action.cdn.bcebos.com/PaddleFleet/precision/${REPO_NAME}${pfpatch}${pppatch}/${PR_ID}/precision_list.txt
    if [ $? -ne 0 ]; then
      wget --no-proxy --no-check-certificate https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/${repo_name}${pfpatch}${pppatch}_latest/precision_list.txt
      python $root_dir/bos/BosClient.py precision_list.txt paddle-github-action/PaddleFleet/precision/${REPO_NAME}${pfpatch}${pppatch}/${PR_ID}
    fi
  fi
  python $root_dir/bos/BosClient.py ${gt_loss_file} paddle-github-action/PaddleFleet/precision/${REPO_NAME}${pfpatch}${pppatch}/${PR_ID}
fi
