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

coverage_train() {
    local coverage_args=()
    if [[ -n "${COVERAGE_RCFILE:-}" ]]; then
        coverage_args+=(--rcfile="${COVERAGE_RCFILE}")
    elif [[ -f /paddle/scripts/fleet_ci/.coveragerc ]]; then
        coverage_args+=(--rcfile=/paddle/scripts/fleet_ci/.coveragerc)
    fi

    coverage run "${coverage_args[@]}" "$(which paddleformers-cli)" train "$@"
}
