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

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a Paddle native MiniCPM checkpoint.}"
: "${TRAIN_DATASET_PATH:?Set TRAIN_DATASET_PATH to GSM8K train jsonl.}"
: "${EVAL_DATASET_PATH:?Set EVAL_DATASET_PATH to GSM8K eval jsonl.}"

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/minicpm-1b-sft}"
LOGGING_DIR="${LOGGING_DIR:-./vdl_log/minicpm-1b-sft}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-100}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
GPUS="${GPUS:-0}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/minicpm_1b_sft.yaml}"
TRAIN_DATASET_TYPE="${TRAIN_DATASET_TYPE:-messages}"
EVAL_DATASET_TYPE="${EVAL_DATASET_TYPE:-messages}"

cat > "${CONFIG_PATH}" <<EOF
train_dataset_type: ${TRAIN_DATASET_TYPE}
eval_dataset_type: ${EVAL_DATASET_TYPE}
train_dataset_path: ${TRAIN_DATASET_PATH}
train_dataset_prob: "1.0"
eval_dataset_path: ${EVAL_DATASET_PATH}
eval_dataset_prob: "1.0"
max_seq_len: 1024
packing: false
mix_strategy: concat
template_backend: jinja

model_name_or_path: ${MODEL_NAME_OR_PATH}
_attn_implementation: eager
convert_from_hf: false

stage: SFT
fine_tuning: full
continue_training: true
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 1
max_steps: ${MAX_STEPS}
eval_steps: ${EVAL_STEPS}
evaluation_strategy: steps
save_steps: ${SAVE_STEPS}
save_strategy: steps
save_to_hf: false
logging_steps: 1
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
logging_dir: ${LOGGING_DIR}
output_dir: ${OUTPUT_DIR}
disable_tqdm: true
eval_accumulation_steps: 16
warmup_steps: 20
learning_rate: 1.0e-5
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: ""
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2
save_checkpoint_format: sharding_io
load_checkpoint_format: sharding_io
EOF

CUDA_VISIBLE_DEVICES="${GPUS}" paddleformers-cli train "${CONFIG_PATH}"
