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
infile = '$root_dir/PaddleFormers/paddleformers/transformers/qwen3_moe/modeling.py'
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
        if line.lstrip().startswith('class Qwen3MoeForCausalLMFleet(Qwen3MoePretrainedModel)') and next_line.strip().startswith('is_fleet'):
            fout.write(pad + 'class Qwen3MoeForCausalLM(Qwen3MoePretrainedModel)' + line.lstrip()[len('class Qwen3MoeForCausalLMFleet(Qwen3MoePretrainedModel)'):])
        elif line.lstrip().startswith('class Qwen3MoeForCausalLM(Qwen3MoePretrainedModel)') and next_line.strip().startswith('enable_to_static_method'):
            fout.write(pad + 'class Qwen3MoeForCausalLMFleet(Qwen3MoePretrainedModel)' + line.lstrip()[len('class Qwen3MoeForCausalLM(Qwen3MoePretrainedModel)'):])
        elif line.lstrip().startswith('class Qwen3MoeForCausalLMPipeFleet(Qwen3MoePretrainedModel') and next_line.strip().startswith('is_fleet'):
            fout.write(pad + 'class Qwen3MoeForCausalLMPipe(Qwen3MoePretrainedModel' + line.lstrip()[len('class Qwen3MoeForCausalLMPipeFleet(Qwen3MoePretrainedModel'):])
        elif line.lstrip().startswith('class Qwen3MoeForCausalLMPipe(GeneralModelForCausalLMPipe)') and next_line.strip().startswith('config_class'):
            fout.write(pad + 'class Qwen3MoeForCausalLMPipeFleet(GeneralModelForCausalLMPipe)' + line.lstrip()[len('class Qwen3MoeForCausalLMPipe(GeneralModelForCausalLMPipe)'):])
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

set +e
# coverage run run_pretrain.py $config_json 2>&1 | tee ./qwen3_single_card.log
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./qwen3_single_card.log

exit_code=$?
if [ $exit_code -ne 0 ]; then
      echo "Qwen3-30B-A3B single card training failed, try to check the log ./qwen3_single_card.log"
      python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./qwen3_single_card.log
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


set -e
echo "
1 12.28671932
2 12.26999569
3 12.29613113
4 12.30189323
5 12.12231159
6 12.26899815
7 12.25399399
8 12.03514481
9 11.77854824
10 11.82082653
" > ./qwen3_single_card_gt_loss.txt

python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --log_file ./qwen3_single_card.log \
   --gt_file ./qwen3_single_card_gt_loss.txt
