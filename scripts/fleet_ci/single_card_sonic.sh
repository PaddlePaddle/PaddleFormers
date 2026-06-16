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


test_cases_list=(
    "tests/fleet/single_card_tests/custom_ops/sonic_moe/test_count_cumsum.py"
    "tests/fleet/single_card_tests/custom_ops/sonic_moe/test_moe.py"
)




export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export RUN_IN_PADDLE_CI=0

run_count=0
failed_tests=()
for test_file in "${test_cases_list[@]}"; do
    filename=$(basename "$test_file")

    echo "Running single card test: $test_file"
    run_count=$((run_count + 1))
    coverage run -m pytest -s "$test_file"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Test FAILED: $test_file, see log for details..."
        python $work_dir/scripts/fleet_ci/check_log_for_exitcode.py "./$(basename ${test_file%.*})_single_card.log" "OK"
        check_exit_code=${PIPESTATUS[0]}
        if [ $check_exit_code -ne 0 ]; then
            failed_tests+=("$test_file")
            echo "Log check failed for $test_file."
        else
            echo "Log check passed for $test_file."
        fi
    else
        echo "Test PASSED: $test_file"
    fi
done

echo "======================================"
echo -e "\033[34mTests executed: $run_count\033[0m"
if [ ${#failed_tests[@]} -eq 0 ]; then
    echo -e "\033[32mAll single card tests passed!\033[0m"
    echo "======================================"
else
    echo -e "::error:: Some single card tests failed:"
    for fail in "${failed_tests[@]}"; do
        echo -e "::error:: \033[31m- $fail\033[0m"
    done
    echo "======================================"
    exit 1
fi
