#!/usr/bin/env bash

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

set -euo pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root_dir"

config_yaml="$root_dir/tests/config/ci/mistral_sft_loss_vs_swift_300.yaml"
paddle_log="${1:-$root_dir/mistral_sft_paddle_300.log}"
swift_log="${2:-}"
num_hidden_layers="${3:-2}"
mistral_src="${MISTRAL_SRC_DIR:-/sda/yuqifan/PaddleFormer/mistral-7B}"
slim_model_dir="${SLIM_MODEL_DIR:-/tmp/mistral-7B-l${num_hidden_layers}-dev}"
max_seq_len_override="${MAX_SEQ_LEN_OVERRIDE:-}"

master="${MASTER_ADDR:-127.0.0.1}"
port="${MASTER_PORT:-37777}"

export PYTHONPATH="$root_dir:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NNODES="${NNODES:-1}"
export MASTER_ADDR="$master"
export MASTER_PORT="$port"

echo "[INFO] Preparing slim mistral model: layers=${num_hidden_layers}"
python "$root_dir/tests/integration_test/prepare_mistral_slim_model.py" \
  --src_dir "$mistral_src" \
  --dst_dir "$slim_model_dir" \
  --num_hidden_layers "$num_hidden_layers" \
  --force

echo "[INFO] Running PaddleFormers SFT for 300 steps..."
train_cmd=(
  python -u -m paddleformers.cli.cli train "$config_yaml"
  model_name_or_path="$slim_model_dir"
  output_dir="./checkpoints/mistral-sft-full-l${num_hidden_layers}"
  logging_dir="./vdl_log_mistral_l${num_hidden_layers}"
)
if [[ -n "$max_seq_len_override" ]]; then
  train_cmd+=("max_seq_len=$max_seq_len_override")
fi
"${train_cmd[@]}" 2>&1 | tee "$paddle_log"

echo "[INFO] Paddle log written to: $paddle_log"

if [[ -z "$swift_log" ]]; then
  echo "[INFO] No swift log provided. Skip loss comparison."
  echo "[INFO] To compare later:"
  echo "python $root_dir/tests/integration_test/compare_loss_with_swift.py --paddle_log $paddle_log --swift_log <swift_log> --max_steps 300"
  exit 0
fi

if [[ ! -f "$swift_log" ]]; then
  echo "[ERROR] swift log does not exist: $swift_log"
  exit 1
fi

echo "[INFO] Comparing Paddle vs ms-swift loss..."
python "$root_dir/tests/integration_test/compare_loss_with_swift.py" \
  --paddle_log "$paddle_log" \
  --swift_log "$swift_log" \
  --max_steps 300 \
  --dump_json "$root_dir/mistral_loss_diff_300.json"
