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

export config_yaml=$root_dir/PaddleFormers/tests/config/ci/glm45_pt_grouped_gemm.yaml
export data_dir=$root_dir/PaddleFormers/tests/fixtures/dummy/pt

yq eval '.train_dataset_path = strenv(data_dir) + "/train.jsonl"
    | .eval_dataset_path = strenv(data_dir) + "/eval.jsonl"
    | .model_name_or_path = strenv(CACHE_DIR) + "/glm45/GLM-4.5-Air"
    | .logging_dir = strenv(data_dir) + "/vdl_log"
    | .output_dir = strenv(data_dir) + "/checkpoints"' \
   $config_yaml > ${config_yaml}.tmp
mv ${config_yaml}.tmp $config_yaml

rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

unset http_proxy https_proxy
rm -rf checkpoint/
rm -rf outputs/

set +e
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./glm45_pt_grouped_gemm.log

exit_code=$?
if [ $exit_code -ne 0 ]; then
   echo "Training failed with exit code $exit_cod, see ./glm45_pt_grouped_gemm.log for details."
   python $root_dir/PaddleFormers/tests/check_log_for_exitcode.py ./glm45_pt_grouped_gemm.log
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
10 12.13314247
" > ./glm45_multi_cards_grouped_gemm_gt_loss.txt

python $root_dir/PaddleFormers/tests/integration_test/check_loss.py \
   --compare_step 10 \
   --log_file ./glm45_pt_grouped_gemm.log \
   --gt_file ./glm45_multi_cards_grouped_gemm_gt_loss.txt
