#!/usr/bin/env bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

set -e
export FLAGS_enable_CI=${1-False}
export FLAGS_enable_CE=${2-False}
export update_baseline_models=${3-False}
export BRANCH=${4-develop}
export PR_NUMBER=${5-0000}

export nlp_dir=/workspace/PaddleFormers
export log_path=/workspace/PaddleFormers/model_unittest_logs
export model_unittest_path=/workspace/PaddleFormers/scripts/regression
cd $nlp_dir
mkdir -p $log_path

init_env() {
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_cudnn_deterministic=1
    export HF_ENDPOINT=https://hf-mirror.com

    # for CI/CE
    if [ -f "./scripts/regression/config.yaml" ]; then
      mv ./scripts/regression/config.yaml ./scripts/regression/config_origin.yaml
    fi

    if echo "${FLAGS_enable_CE}" | grep -q "CE_Release"; then
        echo "CE_Release: install paddle release + fleet release + formers release"
        bash ./scripts/regression/install_requirements.sh "${FLAGS_enable_CE}"
        cd ./scripts/regression
        wget https://paddle-qa.bj.bcebos.com/paddleformers/ce_release_config/config.yaml
        python merge_configs.py --origin_config config_origin.yaml --update_config config.yaml --output config.yaml 2>&1 | tee /tmp/merge_output.txt
        cd -

    elif echo "${FLAGS_enable_CE}" | grep -q "CE_Develop"; then

        echo "CE_Develop: install paddle develop + fleet develop + formers develop"
        bash ./scripts/regression/install_requirements.sh "${FLAGS_enable_CE}"
        # donwload configs
        cd ./scripts/regression
        wget https://paddle-qa.bj.bcebos.com/paddleformers/ce_develop_config/config.yaml
        python merge_configs.py --origin_config config_origin.yaml --update_config config.yaml --output config.yaml 2>&1 | tee /tmp/merge_output.txt
        cd -
    elif [[ "${FLAGS_enable_CI}" == "True" ]] && [[ "${BRANCH}" == "develop" ]];then
        echo "CI: install paddle stable + fleet stable + develop formers"
        bash ./scripts/regression/install_requirements.sh ${FLAGS_enable_CI}
        # donwload configs
        cd ./scripts/regression
        wget https://paddle-qa.bj.bcebos.com/paddleformers/ci_develop_config/config.yaml
        python merge_configs.py --origin_config config_origin.yaml --update_config config.yaml --output config.yaml 2>&1 | tee /tmp/merge_output.txt
        cd -
    else
        # CI Release
        echo "CI: install paddle stable + fleet stable + release formers"
        bash ./scripts/regression/install_requirements.sh ${FLAGS_enable_CI}
        cd ./scripts/regression
        wget https://paddle-qa.bj.bcebos.com/paddleformers/ci_release_config/config.yaml
        python merge_configs.py --origin_config config_origin.yaml --update_config config.yaml --output config.yaml 2>&1 | tee /tmp/merge_output.txt
        cd -
    fi
    grep "^new_models=" /tmp/merge_output.txt || true
    new_models=$(grep "^new_models=" /tmp/merge_output.txt | cut -d'=' -f2)

    if [[ "$new_models" != "" ]] && [[ "$new_models" != "false" ]] && [[ "$new_models" != "False" ]]; then
        if [[ "$update_baseline_models" == "false" ]] || [[ "$update_baseline_models" == "False" ]]; then

            update_baseline_models="$new_models"
        else
            update_baseline_models="$update_baseline_models $new_models"
        fi
        echo "Updated baseline models: $update_baseline_models"
    else
        echo "No new models found, keeping existing: $update_baseline_models"
    fi
   
}
upload_baseline(){
    cp -r /home/models/bos/* ./
    rm -rf upload
    mkdir upload 
    cp scripts/regression/config.yaml upload/
    mv scripts/regression/config.yaml config_${PR_NUMBER}.yaml 
    cp config_${PR_NUMBER}.yaml upload/

    if echo "${FLAGS_enable_CE}" | grep -q "CE_Release"; then
        python upload.py upload "paddle-qa/paddleformers/ce_release_config/"
    elif echo "${FLAGS_enable_CE}" | grep -q "CE_Develop"; then
        python upload.py upload "paddle-qa/paddleformers/ce_develop_config/"
    elif [[ "${FLAGS_enable_CI}" == "True" ]] && [[ "$BRANCH" == "develop" ]];then
        python upload.py upload "paddle-qa/paddleformers/ci_develop_config/"
    else
        python upload.py upload "paddle-qa/paddleformers/ci_release_config/"
    fi
}
print_info() {
    if [ $1 -ne 0 ]; then
        cat ${log_path}/model_unittest.log | grep -v "Fail to fscanf: Success" \
            | grep -v "SKIPPED" | grep -v "warning" > ${log_path}/model_unittest_FAIL.log
        tail -n 1 ${log_path}/model_unittest.log >> ${log_path}/model_unittest_FAIL.log
        echo -e "\033[31m ${log_path}/model_unittest_FAIL \033[0m"
        cat ${log_path}/model_unittest_FAIL.log
        if [ $1 -eq 124 ]; then
            echo "\033[32m [failed-timeout] Test case execution was terminated after exceeding the ${running_time} min limit."
        fi
    else
        tail -n 1 ${log_path}/model_unittest.log
        echo -e "\033[32m ${log_path}/model_unittest_SUCCESS \033[0m"
    fi
}

get_diff_TO_case(){
declare -a model_array=()
for file_name in `git diff --numstat ${BRANCH} -- |awk '{print $NF}'`;do
    ext="${file_name##*.}"
    echo "file_name: ${file_name}, ext: ${file_name##*.}"

    # Check if file is in transformer directories (don't check file existence, rely on git diff)
    if [[ "$file_name" == "paddleformers/transformers/"* ]] || [[ "$file_name" == "tests/transformers/"* ]]; then
        model_name=$(echo "$file_name" | sed -n 's#.*paddleformers/transformers/\([^/]*\)/.*#\1#p')
        if [ -z "$model_name" ]; then
            model_name=$(echo "$file_name" | sed -n 's#.*tests/transformers/\([^/]*\)/.*#\1#p')
        fi
        if [ -n "$model_name" ]; then
            if [[ ! " ${model_array[*]} " =~ " ${model_name} " ]]; then
                model_array+=("$model_name")
            fi
        fi
    fi
done

if [ ${#model_array[@]} -gt 0 ]; then
    models=$(IFS=,; echo "${model_array[*]}")
    echo "Models to test: $models"
else
    models="glm4_moe"
    echo "No transformer changes detected, using default model: $models"
fi

}

init_env
if [[ "$update_baseline_models" != "false" ]] && [[ "$update_baseline_models" != "False" ]]; then
    echo "Update baseline models: $update_baseline_models"
    models=$update_baseline_models
elif [[ ${FLAGS_enable_CI} == "True" ]];then
    if [[ "$PR_NUMBER" == "0" ]]; then
        models="all"
    else
        get_diff_TO_case
    fi
elif [[ ${FLAGS_enable_CE} != "False" ]];then
    models="all"
fi

if [[ ${FLAGS_enable_CI} == "True" ]] || [[ ${FLAGS_enable_CE} != "False" ]];then
    cd ${nlp_dir}
    unset http_proxy && unset https_proxy
    set +e
    echo "Check nvidia-smi"
    nvidia-smi
    echo "Check paddle device count"
    python -c "import paddle; print(paddle.device.device_count())"
    echo "Regression model: ${models}, Update baseline models: ${update_baseline_models}"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export FLAGS_tcp_store_using_libuv=0
    PYTHONPATH=$(pwd) \
    COVERAGE_SOURCE=paddleformers \
    python -m pytest -s -v --alluredir=result --models=${models} --update-baseline=${update_baseline_models} scripts/regression/test_models.py > ${log_path}/model_unittest.log 2>&1
    exit_code=$?
    print_info $exit_code model_unittest
    if [[ $exit_code -eq 0 ]] && [[ "$update_baseline_models" != "false" ]] && [[ "$update_baseline_models" != "False" ]]; then
        upload_baseline   
    else
        echo " fix error, first"
    fi
else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    exit_code=0
fi
exit $exit_code