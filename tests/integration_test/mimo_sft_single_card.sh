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
export REPO_DIR=${REPO_DIR:-$(pwd)}
export CACHE_DIR=${CACHE_DIR:-$REPO_DIR/.cache}
export PADDLEFORMERS_DIST_LOG=${PADDLEFORMERS_DIST_LOG:-$REPO_DIR/paddleformers_dist_log}

step=${1:-single}

if [ "$step" == "single" ]; then
  export config_yaml=${config_yaml:-$REPO_DIR/tests/config/ci/mimo_sft_single.yaml}
  export data_dir=${data_dir:-$REPO_DIR/tests/fixtures/dummy/sft}
  export model_name_or_path=${model_name_or_path:-$CACHE_DIR/mimo/tiny-random-mimo}
  export output_dir=${output_dir:-$REPO_DIR/checkpoints/mimo-single}
fi

if [ -z "${config_yaml:-}" ]; then
  echo "Unsupported step: $step"
  exit 1
fi

python $REPO_DIR/scripts/llavaonevision1_5/patch_yaml_top_level.py $config_yaml \
  --set "train_dataset_path=$data_dir/train.jsonl" \
  --set "eval_dataset_path=$data_dir/eval.jsonl" \
  --set "model_name_or_path=$model_name_or_path" \
  --set "output_dir=$output_dir"

rm -rf ./outputs
mkdir -p "$PADDLEFORMERS_DIST_LOG"
master=$(hostname -i)
port=36777

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_use_stride_compute_kernel=False

unset http_proxy https_proxy

git config --global --add safe.directory $REPO_DIR || true

log_file=mimo_sft_single_card.txt
gt_loss_file=mimo_sft_single_card_gt_loss.txt

set +e
if command -v coverage >/dev/null 2>&1; then
  train_cmd="coverage run $(which paddleformers-cli) train $config_yaml"
else
  train_cmd="$(which paddleformers-cli) train $config_yaml"
fi
NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port bash -c "$train_cmd" 2>&1 | tee ./${log_file}

exit_code=$?
if [ $exit_code -ne 0 ]; then
   echo "mimo sft training failed, try to check the log file"
   python $REPO_DIR/tests/check_log_for_exitcode.py ./${log_file} "***** train metrics *****"
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

if [ "${SKIP_PRECISION_CHECK:-0}" = "1" ]; then
  echo "Skip precision check for local smoke."
  exit 0
fi

export repo_name=PaddleFleet
export REPO_NAME=$(echo $GITHUB_REPO_NAME | awk -F'/' '{print $2}')

wget --no-proxy --no-check-certificate https://xly-devops.cdn.bcebos.com/PaddleFleet/precision/${REPO_NAME}${pfpatch}${pppatch}_latest/${gt_loss_file}
if [ $? -ne 0 ]; then
  echo "To request precision checks for new models, please contact swgu98."
  exit 1
fi

log_loss_file=${log_file%.*}_loss.${log_file##*.}
python $REPO_DIR/tests/integration_test/check_loss.py \
   --compare_step 10 \
   --log_file ./${log_file} \
   --log_loss_file ./${log_loss_file} \
   --gt_file ./${gt_loss_file}
