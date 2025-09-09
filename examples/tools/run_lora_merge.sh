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

export PYTHONPATH=$(dirname "$0")/..:$PYTHONPATH

python -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    mergekit.py \
    --lora_model_path "../checkpoints/qwen2_hf_lora_ckpts" \
    --model_name_or_path "/root/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d" \
    --output_path "../checkpoints/merge_qwen2_hf_lora_model_multi_gpus" \
