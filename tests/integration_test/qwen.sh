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

if [[ ! -d $CACHE_DIR/Qwen3-30B-A3B ]]; then
    pushd $CACHE_DIR
    wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/Qwen/Qwen3-30B-A3B.tar.gz --no-check-certificate
    tar xf Qwen3-30B-A3B.tar.gz
    popd
fi

if [[ "$step" == "pt" ]]; then
    pushd $root_dir/PaddleFormers
    git reset --hard HEAD
    popd
    python <<EOF
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
EOF
    mv $root_dir/PaddleFormers/paddleformers/transformers/qwen3_moe/modeling.py.new $root_dir/PaddleFormers/paddleformers/transformers/qwen3_moe/modeling.py
fi

source PaddleFleet/.venv/bin/activate

if [[ "$step" == "pt" ]]; then
    export config_yaml=$root_dir/PaddleFormers/tests/config/ci/qwen3_multicard_pt.yaml
    export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/pt
    export model_name_or_path=$CACHE_DIR/Qwen3-30B-A3B
    export output_dir=$root_dir/checkpoints/qwen-pt
elif [[ "$step" == "sft" ]]; then
    export config_yaml=$root_dir/PaddleFormers/tests/config/ci/qwen3_multicard_sft.yaml
    export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/sft
    export model_name_or_path=$root_dir/checkpoints/qwen-pt
    export output_dir=$root_dir/checkpoints/qwen-sft
else
    export config_yaml=$root_dir/PaddleFormers/tests/config/ci/qwen3_multicard_lora.yaml
    export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/sft
    export model_name_or_path=$root_dir/checkpoints/qwen-sft
    export output_dir=$root_dir/checkpoints/qwen-lora
fi

yq eval '.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/eval.jsonl"
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

set +e
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./qwen_$step.log

exit_code=$?
if [ $exit_code -ne 0 ]; then
   echo "qwen multi-cards training failed, try to check the log file"
   python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./qwen_${step}.log "***** train metrics *****"
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

python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --compare_step 10 \
   --log_file ./qwen_${step}.log \
   --gt_file $root_dir/PaddleFormers/tests/integration_test/precision/qwen_${step}_multi_card_gt_loss.txt
