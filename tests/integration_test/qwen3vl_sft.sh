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

step=$1
machine=$2



if [[ "$step" == "moe" ]]; then
    export config_yaml=$root_dir/PaddleFormers/tests/config/ci/qwen3vl_sft_moe.yaml
    export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/sft-vl
    export model_name_or_path=$CACHE_DIR/qwen3vl/tiny-random-qwen3vlmoev2
    export output_dir=$root_dir/checkpoints/qwen3vl-moe
    if [ $machine == "a100" ]; then
        export config_yaml=$root_dir/PaddleFormers/tests/config/ci/qwen3vl_sft_moe_a100.yaml
    fi
elif [[ "$step" == "tp8" ]]; then
    export config_yaml=$root_dir/PaddleFormers/tests/config/ci/qwen3vl_sft.yaml
    export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/sft-vl
    export model_name_or_path=$CACHE_DIR/qwen3vl/tiny-random-qwen3vlv2
    export output_dir=$root_dir/checkpoints/qwen3vl-sft
elif [ $step == "fsdp" ]; then
  export config_yaml=$root_dir/PaddleFormers/tests/config/ci/qwen3vl_sft_fsdp.yaml
  export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/sft-vl
  export model_name_or_path=$CACHE_DIR/qwen3vl/tiny-random-qwen3vlv2
  export output_dir=$root_dir/checkpoints/qwen3vl-fsdp
fi

if [[ ! -d $data_dir/DoclingMatix ]]; then
  wget https://paddleformers.bj.bcebos.com/datasets/DoclingMatix.tar.gz
  tar -xf DoclingMatix.tar.gz -C $data_dir
  rm -rf DoclingMatix.tar.gz
fi

yq eval '.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .model_name_or_path = strenv(model_name_or_path)
    | .output_dir = strenv(output_dir)' \
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

log_file=qwen3vl_sft_${machine}_${step}_multi_card.txt
gt_loss_file=qwen3vl_sft_${machine}_${step}_multi_card_gt_loss.txt

set +e
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./${log_file}

exit_code=$?
if [ $exit_code -ne 0 ]; then
   echo "qwen sft training failed, try to check the log file"
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

# export repo_name=$(echo $GITHUB_REPO_NAME | awk -F'/' '{print $2}')
export repo_name=PaddleFleet
export REPO_NAME=$(echo $GITHUB_REPO_NAME | awk -F'/' '{print $2}')
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