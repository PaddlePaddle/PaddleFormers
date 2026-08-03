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

# TODO ，前期不在 .github/workflows/fleet-model-test.yml 中生效，避免直接卡死流程
#  TODO，提交PR的时候，会提交loss对比材料

set -exo pipefail
export root_dir=$(pwd)

if [ -f 'PaddleFleet/.venv/bin/activate' ]; then
   source PaddleFleet/.venv/bin/activate
fi

config_sft_yaml=$root_dir/PaddleFormers/tests/config/ci/internlm3_sft.yaml

if [[ ! -f "$config_sft_yaml" ]]; then
  echo "Config file not found: $config_sft_yaml"
  exit 1
fi

rm -rf ./outputs
rm -rf paddleformers_dist_log
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

unset http_proxy https_proxy

log_file=internlm3_sft.txt
gt_loss_file=internlm3_sft_multi_card_gt_loss.txt

set +e
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_sft_yaml 2>&1 | tee ./${log_file}