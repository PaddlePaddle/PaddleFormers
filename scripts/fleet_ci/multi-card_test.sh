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

disable_file="$work_dir/tests/fleet/multi_card_tests/disable_multi-card_uts.txt"
test_dir="$work_dir/tests/fleet/multi_card_tests"
yaml_config="$work_dir/tests/fleet/test_configs.yaml"

disabled=()
if [ -f "$disable_file" ]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        disabled+=("$line")
    done < "$disable_file"
fi

echo -e "\033[34mDisabled tests:\033[0m ${disabled[@]}"

is_disabled() {
    local test=$1
    for d in "${disabled[@]}"; do
        if [[ "$test" == "$d" ]]; then
            return 0
        fi
    done
    return 1
}

shopt -s extglob
shopt -s globstar

parse_yaml_patterns() {
    local yaml="$1"
    # pattern|num_gpus
    yq -r '.tests[] | .test_case[] as $pat | .products[] | "\($pat)|\(.num_gpus)"' "$yaml"
}

get_num_gpus_for_test() {
    local filepath=$1    # "tests/fleet/multi_card_tests/xxx/yy.py"
    local pattern num

    while IFS='|' read -r pattern num; do
        if [[ "$filepath" == $pattern ]]; then
            echo "$num"
            return 0
        fi
    done < <(parse_yaml_patterns "$yaml_config")
    echo "8"
}

gen_gpus_arg() {
    local num_gpus=$1
    local gpus=""
    for ((i=0; i<num_gpus; i++)); do
        if [[ $i -ne 0 ]]; then
            gpus+=","
        fi
        gpus+="$i"
    done
    echo "$gpus"
}

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_tcp_store_using_libuv=0

run_count=0
failed_tests=()
for test_file in $(find $test_dir -type f -name "test_*.py"); do
    rel_path="${test_file#$work_dir/}"
    filename=$(basename "$test_file")
    if is_disabled "$filename"; then
        echo "Skipping disabled test: $filename"
        continue
    fi

    num_gpus=$(get_num_gpus_for_test "$rel_path")
    gpus_arg=$(gen_gpus_arg "$num_gpus")
    echo "Running multi-card test: $test_file with $num_gpus GPUs ($gpus_arg)"

    run_count=$((run_count + 1))
    coverage run -m paddle.distributed.launch --gpus "$gpus_arg" "$test_file" | tee "./$(basename ${test_file%.*})_multi_card.log"
    check_exit_code=${PIPESTATUS[0]}
    if [ $check_exit_code -ne 0 ]; then
        echo "Test FAILED: $test_file, see log for details..."
        python $work_dir/scripts/fleet_ci/check_log_for_exitcode.py "./$(basename ${test_file%.*})_multi_card.log" "OK"
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
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
    echo -e "\033[32mAll multi-card tests passed!\033[0m"
    echo "======================================"
else
    echo -e "::error:: Some multi-card tests failed:"
    for fail in "${failed_tests[@]}"; do
        echo -e "::error:: \033[31m- $fail\033[0m"
    done
    echo "======================================"
    exit 1
fi
