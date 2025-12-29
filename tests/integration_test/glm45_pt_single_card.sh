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

source PaddleFleet/.venv/bin/activate

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

set +e
# coverage run run_pretrain.py $config_json 2>&1 | tee ./glm45_single_card.log
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./glm45_single_card.log

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "GLM4.5 single card training failed, try to check the log file"
    python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./glm45_single_card.log
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


set -e
echo "
1 12.10422421
2 12.05354404
3 12.03884697
4 12.03464031
5 12.02043915
6 12.00771523
7 11.95508194
8 11.96421719
9 11.97694683
10 11.96971035
" > ./glm45_single_card_gt_loss.txt

export FLAGS_use_stride_compute_kernel=False


python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --log_file ./glm45_single_card.log \
   --gt_file ./glm45_single_card_gt_loss.txt
