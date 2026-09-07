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

: "${SWIFT_BIN:?Set SWIFT_BIN to the ms-swift executable.}"
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to the HF MiniCPM checkpoint.}"
: "${TRAIN_DATASET_PATH:?Set TRAIN_DATASET_PATH to GSM8K train jsonl.}"
: "${EVAL_DATASET_PATH:?Set EVAL_DATASET_PATH to GSM8K eval jsonl.}"

OUTPUT_DIR="${OUTPUT_DIR:-./ms_swift/minicpm-1b-gsm8k}"
CACHE_DIR="${CACHE_DIR:-./ms_swift/cache}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/minicpm_ms_swift_gsm8k.log}"
GPUS="${GPUS:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-100}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_PATH}")" "${CACHE_DIR}/modelscope" "${CACHE_DIR}/matplotlib"

CUDA_VISIBLE_DEVICES="${GPUS}" \
XDG_CACHE_HOME="${CACHE_DIR}" \
MODELSCOPE_CACHE="${CACHE_DIR}/modelscope" \
MPLCONFIGDIR="${CACHE_DIR}/matplotlib" \
"${SWIFT_BIN}" sft \
  --model_type minicpm-1b-sft-chat \
  --model_id_or_path "${MODEL_NAME_OR_PATH}" \
  --template_type minicpm \
  --custom_train_dataset_path "${TRAIN_DATASET_PATH}" \
  --custom_val_dataset_path "${EVAL_DATASET_PATH}" \
  --sft_type full \
  --dtype bf16 \
  --use_flash_attn false \
  --max_length "${MAX_LENGTH}" \
  --batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate 1e-5 \
  --warmup_steps "${WARMUP_STEPS}" \
  --weight_decay 0.0 \
  --adam_beta2 0.999 \
  --max_steps "${MAX_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps 1 \
  --output_dir "${OUTPUT_DIR}" \
  --add_output_dir_suffix false \
  --report_to none \
  --save_total_limit 1 \
  --seed 23 \
  --dataset_seed 23 \
  --check_model_is_latest false \
  --disable_tqdm true \
  2>&1 | tee "${LOG_PATH}"
