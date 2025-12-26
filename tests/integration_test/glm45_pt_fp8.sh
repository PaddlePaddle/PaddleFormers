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
cd $root_dir/PaddleFormers/examples/experiments/paddlefleet

config_json="glm45_fp8.json"

jq --arg cache "$CACHE_DIR" \
   '.expert_model_parallel_size = 8
    | .save_steps = 100
    | .input_dir = "1.0 \($cache)/glm45/data/pre-training/llama_openwebtext_100k"
    | .model_name_or_path = "\($cache)/glm45/GLM-4.5-Air"' \
   $config_json > $config_json.tmp
mv $config_json.tmp $config_json

# use fp8 Provider
sed -i 's/GLM_muiti_cards/GLM_muiti_cards_fp8/' glm45_fp8.json
sed -i 's/num_hidden_layers: int = 10/num_hidden_layers: int = 3/g' $root_dir/PaddleFormers/examples/experiments/paddlefleet/glm45_provider.py
sed -i 's/\[0\] \* 1 + \[1\] \* 9/\[0\] \* 1 + \[1\] \* 2/g' $root_dir/PaddleFormers/examples/experiments/paddlefleet/glm45_provider.py

rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False

unset http_proxy https_proxy
rm -rf checkpoint/
rm -rf outputs/

set +e
coverage run -m paddle.distributed.launch \
   --log_dir ./log \
   --master $master:$port \
   --nnodes 1 \
   --rank 0 \
   --run_mode=collective \
   run_pretrain.py $config_json \
   --output_dir ./checkpoint 2>&1 | tee ./glm45_fp8.log

exit_code=$?
if [ $exit_code -ne 0 ]; then
   echo "Training failed with exit code $exit_cod, see ./glm45_fp8.log for details."
   python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./glm45_fp8.log
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

set -e
echo "
20 10.29259872
" > ./glm45_multi_cards_fp8_gt_loss.txt

python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --compare_step 20 \
   --log_file ./glm45_fp8.log \
   --gt_file ./glm45_multi_cards_fp8_gt_loss.txt
