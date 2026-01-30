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

python -c "
infile = '$root_dir/PaddleFormers/paddleformers/transformers/qwen3_moe/modeling.py'
print(infile)
outfile = infile + '.new'
with open(infile) as fin:
    lines = fin.readlines()
with open(outfile, 'w') as fout:
    i = 0
    while i < len(lines):
        line = lines[i]
        pad = line[:len(line)-len(line.lstrip())]
        if line.lstrip().startswith('config.fuse_rms_norm = True'):
            fout.write(pad + 'config.fuse_rms_norm = True\n')
            fout.write(pad + 'config.router_aux_loss_coef = None\n')
        else:
            fout.write(line)
        i += 1
"
mv $root_dir/PaddleFormers/paddleformers/transformers/qwen3_moe/modeling.py.new $root_dir/PaddleFormers/paddleformers/transformers/qwen3_moe/modeling.py

config_yaml=$root_dir/PaddleFormers/tests/config/ci/qwen3_pt.yaml
yq eval '
  .save_steps = 100 |
  .input_dir = "1.0 '"${CACHE_DIR}"'/glm45/data/pre-training/llama_openwebtext_100k" |
  .model_name_or_path = "'"${CACHE_DIR}"'/qwen/Qwen3-30B-A3B-Base"
' "$config_yaml" -i

cat $config_yaml
rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False

unset http_proxy https_proxy

log_file=qwen3_single_card.txt
gt_loss_file=qwen3_single_card_gt_loss.txt

set +e
# coverage run run_pretrain.py $config_json 2>&1 | tee ./qwen3_single_card.log
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./${log_file}

exit_code=$?
if [ $exit_code -ne 0 ]; then
      echo "Qwen3-30B-A3B single card training failed, try to check the log ./${log_file}"
      python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./${log_file} "***** train metrics *****"
      check_exit_code=$?
      if [ $check_exit_code -ne 0 ]; then
         echo "Log check failed."
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