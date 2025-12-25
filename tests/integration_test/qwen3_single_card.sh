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

ls -lah $CACHE_DIR/qwen/Qwen3-30B-A3B-Base

config_sft_yaml=$root_dir/PaddleFormers/tests/config/ci/qwen_single.yaml

yq '.train_dataset_path = strenv(cur_dir) + "/data/sft/train.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/sft/dev.jsonl"
    | .model_name_or_path = strenv(CACHE_DIR) + "/glm45/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/glm_full_pp_vdl_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints/glm_full_pp_ckpts"' \
   $config_sft_yaml > ${config_sft_yaml}.tmp
mv ${config_sft_yaml}.tmp $config_sft_yaml

rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False

unset http_proxy https_proxy

set +e
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_lora_yaml 2>&1 | tee ./qwen3_single_card.log

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
1 10.57088089
2 10.57881927
3 10.56455803
4 10.55170441
5 10.55012321
6 10.53712845
7 10.52390480
8 10.52836990
9 10.54636002
10 10.52686119
" > ./qwen3_single_card_gt_loss.txt

python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --log_file ./qwen3_single_card.log \
   --gt_file ./qwen3_single_card_gt_loss.txt
