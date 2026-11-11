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
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CINN_RESOURCE_DIR:?Provide matching assets/ and data/}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR (weights are not uploaded)}"
export CINN_RESOURCE_DIR
MODE=${MODE:-dynamic}
TASK=${TASK:-all}
[[ -z ${RESUME:-} ]] || { echo 'FastVLM CINN entry does not support RESUME; start a new run.' >&2; exit 2; }
case "$MODE" in dynamic|cinn) ;; *) exit 2 ;; esac
case "$TASK" in train|infer|all) ;; *) exit 2 ;; esac
PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
"$PYTHON_BIN" "$here/verify_assets.py" "$CHECKPOINT_DIR"
output_root=${RESULTS_DIR:-$here/results}
mkdir -p "$output_root"
run_task() {
    local task=$1
    local output="$output_root/$MODE-$task"
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 3; }
    local args=(--mode "$MODE" --split --output "$output")
    if [[ "$task" = infer ]]; then
        args+=(--infer --checkpoint "$CHECKPOINT_DIR/inference.pdparams")
    else
        args+=(--checkpoint "$CHECKPOINT_DIR/initial.pdparams" --steps "${STEPS:-20}")
    fi
    "$PYTHON_BIN" "$here/run_model.py" "${args[@]}" 2>&1 | tee "$output.log"
    if [[ "$MODE" = cinn ]]; then
        grep -q 'Compiling subgraph with CINN backend' "$output.log" || { echo 'Missing CINN compile marker' >&2; exit 4; }
    fi
}
if [[ "$TASK" = all ]]; then run_task train; run_task infer; else run_task "$TASK"; fi
