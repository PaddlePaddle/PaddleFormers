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

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config_template=$repo_dir/tests/config/ci/minicpm_sft_single.yaml
config_yaml=$(mktemp /tmp/minicpm_sft_single.XXXXXX.yaml)
trap 'rm -f "$config_yaml"' EXIT
export data_dir=$repo_dir/tests/fixtures/dummy/sft
export model_name_or_path=${MODEL_NAME_OR_PATH:-${CACHE_DIR}/minicpm/tiny-random-minicpm}
export output_dir=${OUTPUT_DIR:-/tmp/minicpm-sft-single}

yq eval '.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/eval.jsonl"
    | .model_name_or_path = strenv(model_name_or_path)
    | .output_dir = strenv(output_dir)' \
   "$config_template" > "$config_yaml"

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

unset http_proxy https_proxy

log_file=minicpm_sft_single_card.txt
train_command=("$(which paddleformers-cli)" train "$config_yaml")
if command -v coverage >/dev/null 2>&1; then
   train_command=(coverage run "${train_command[@]}")
fi

set +e
"${train_command[@]}" 2>&1 | tee ./"${log_file}"
exit_code=$?
if [ $exit_code -ne 0 ]; then
   echo "MiniCPM single-card SFT failed, see ${log_file}."
   exit $exit_code
fi

echo "Test passed."
