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

config_json="glm45_single_card.json"

jq --arg cache "$CACHE_DIR" \
   '.save_steps = 100
    | .input_dir = "1.0 \($cache)/glm45/data/pre-training/llama_openwebtext_100k"
    | .model_name_or_path = "\($cache)/glm45/GLM-4.5-Air"' \
   $config_json > $config_json.tmp
mv $config_json.tmp $config_json


rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
unset http_proxy https_proxy

set +e
coverage run run_pretrain.py $config_json 2>&1 | tee ./glm45_single_card.log

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "GLM4.5 single card training failed, try to check the log file"
    python $root_dir/PaddleFleet/ci/check_log_for_exitcode.py ./glm45_single_card.log
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
1 12.10431099
2 12.05327988
3 12.03885174
4 12.03460503
5 12.02089691
6 12.00885010
7 11.95639896
8 11.96551323
9 11.97876358
10 11.97223091
" > ./glm45_single_card_gt_loss.txt

export FLAGS_use_stride_compute_kernel=False


python $root_dir/PaddleFleet/ci/integration_test/check_loss.py \
   --log_file ./glm45_single_card.log \
   --gt_file ./glm45_single_card_gt_loss.txt
