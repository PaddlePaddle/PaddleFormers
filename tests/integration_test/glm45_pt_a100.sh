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

source PaddleFleet/.venv/bin/activate

wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/glm45/glm45_fleet.12-18.tar --no-check-certificate
tar -xf glm45_fleet.12-18.tar # glm45_fleet
cd $root_dir/glm45_fleet
export cur_dir=$(pwd)

config_yaml=$root_dir/PaddleFormers/tests/config/ci/glm45_pt.yaml

yq eval '.expert_model_parallel_size = 1
    | .num_hidden_layers = 2
    | .per_device_train_batch_size = 1
    | .use_expert_parallel = false
    | .train_dataset_path = strenv(cur_dir) + "/data/pre-training/train.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/pre-training/eval.jsonl"
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

set +e
FLAGS_use_stride_compute_kernel=False NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./glm45_pt_a100.log


exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Test failed with exit code $exit_code, check the log: ./glm45_pt_a100.log"
    python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./glm45_pt_a100.log
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

set -e

echo "
10 12.69604778
" > ./glm45_pt_multi_card_gt_loss.txt

python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --compare_step 10 \
   --log_file ./glm45_pt_a100.log \
   --gt_file ./glm45_pt_multi_card_gt_loss.txt
