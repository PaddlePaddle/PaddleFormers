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

root_dir=$(pwd)
config_yaml=$root_dir/PaddleFormers/tests/config/ci/minicpm_sft_single.yaml
data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/sft
model_name_or_path=${CACHE_DIR}/minicpm/tiny-random-minicpm
output_dir=$root_dir/checkpoints/minicpm-sft-single

yq eval '.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/eval.jsonl"
    | .model_name_or_path = strenv(model_name_or_path)
    | .output_dir = strenv(output_dir)' \
   "$config_yaml" > "${config_yaml}.tmp"
mv "${config_yaml}.tmp" "$config_yaml"

rm -rf ./outputs
rm -rf paddleformers_dist_log

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=0

unset http_proxy https_proxy

log_file=minicpm_sft_single_card.txt

set +e
coverage run "$(which paddleformers-cli)" train "$config_yaml" 2>&1 | tee ./"${log_file}"
exit_code=$?
if [ $exit_code -ne 0 ]; then
   echo "MiniCPM single-card SFT failed, try to check the log file"
   python "$root_dir/PaddleFormers/tests/check_log_for_exitcode.py" ./"${log_file}" "***** train metrics *****"
   check_exit_code=$?
   if [ $check_exit_code -ne 0 ]; then
     echo "Failed to find train metrics in log file."
     exit 1
   fi
fi

echo "Test passed."
