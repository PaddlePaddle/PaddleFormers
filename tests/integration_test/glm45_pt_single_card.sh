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
infile = '$root_dir/PaddleFormers/paddleformers/transformers/glm4_moe/modeling.py'
print(infile)
outfile = infile + '.new'
with open(infile) as fin:
    lines = fin.readlines()
with open(outfile, 'w') as fout:
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i+1] if i+1 < len(lines) else ''
        pad = line[:len(line)-len(line.lstrip())]
        if line.lstrip().startswith('class Glm4MoeForCausalLMFleet(Glm4MoePreTrainedModel)') and next_line.strip().startswith('is_fleet'):
            fout.write(pad + 'class Glm4MoeForCausalLM(Glm4MoePreTrainedModel)' + line.lstrip()[len('class Glm4MoeForCausalLMFleet(Glm4MoePreTrainedModel)'):])
        elif line.lstrip().startswith('class Glm4MoeForCausalLM(Glm4MoePreTrainedModel)') and next_line.strip().startswith('_tied_weights_keys'):
            fout.write(pad + 'class Glm4MoeForCausalLMFleet(Glm4MoePreTrainedModel)' + line.lstrip()[len('class Glm4MoeForCausalLM(Glm4MoePreTrainedModel)'):])
        elif line.lstrip().startswith('class Glm4MoeForCausalLMPipeFleet(Glm4MoePreTrainedModel') and next_line.strip().startswith('is_fleet'):
            fout.write(pad + 'class Glm4MoeForCausalLMPipe(Glm4MoePreTrainedModel' + line.lstrip()[len('class Glm4MoeForCausalLMPipeFleet(Glm4MoePreTrainedModel'):])
        elif line.lstrip().startswith('class Glm4MoeForCausalLMPipe(GeneralModelForCausalLMPipe)') and next_line.strip().startswith('config_class'):
            fout.write(pad + 'class Glm4MoeForCausalLMPipeFleet(GeneralModelForCausalLMPipe)' + line.lstrip()[len('class Glm4MoeForCausalLMPipe(GeneralModelForCausalLMPipe)'):])
        else:
            fout.write(line)
        i += 1
"
mv $root_dir/PaddleFormers/paddleformers/transformers/glm4_moe/modeling.py.new $root_dir/PaddleFormers/paddleformers/transformers/glm4_moe/modeling.py

config_yaml=$root_dir/PaddleFormers/tests/config/ci/glm45_single_pt-test.yaml 


rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
unset http_proxy https_proxy

log_file=glm45_single_card.txt
gt_loss_file=glm45_single_card_gt_loss.txt

set +e
# coverage run run_pretrain.py $config_json 2>&1 | tee ./glm45_single_card.log
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./${log_file}

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "GLM4.5 single card training failed, try to check the log file"
    python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./${log_file}
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


wget --no-proxy --no-check-certificate https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/latest/${gt_loss_file}
if [ $? -ne 0 ]; then
  echo "To request precision checks for new models, please contact swgu98."
  exit 1
fi

log_loss_file=${log_file%.*}_loss.${log_file##*.}
python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --log_file ./${log_file} \
   --log_loss_file ./${log_loss_file} \
   --gt_file ./${gt_loss_file}

cat ${log_loss_file}

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
fi
