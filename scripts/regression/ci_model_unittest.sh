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
export paddle=$1
export FLAGS_enable_CE=${2-false}
export paddleformers_code_path=/workspace/PaddleFormers
export log_path=${paddleformers_code_path}/unittest_logs
export model_unittest_path=/workspace/PaddleFormers/scripts/regression
cd $paddleformers_code_path
mkdir -p $log_path


set_env() {
    export NVIDIA_TF32_OVERRIDE=0 
    export FLAGS_cudnn_deterministic=1
    export HF_ENDPOINT=https://hf-mirror.com
    export FLAGS_use_cuda_managed_memory=true

    # for CE
    if [[ ${FLAGS_enable_CE} == "true" ]];then
        export CE_TEST_ENV=1
        export RUN_SLOW_TEST=1
        export PYTHONPATH=${paddleformers_code_path}:${paddleformers_code_path}/llm:${PYTHONPATH}
    fi
}

print_info() {
    if [ $1 -ne 0 ]; then
        cat ${log_path}/model_unittest.log | grep -v "Fail to fscanf: Success" \
            | grep -v "SKIPPED" | grep -v "warning" > ${log_path}/model_unittest_FAIL.log
        tail -n 1 ${log_path}/model_unittest.log >> ${log_path}/model_unittest_FAIL.log
        echo -e "\033[31m ${log_path}/model_unittest_FAIL \033[0m"
        cat ${log_path}/model_unittest_FAIL.log
        if [ -n "${AGILE_JOB_BUILD_ID}" ]; then
            cp ${log_path}/model_unittest_FAIL.log ${PPNLP_HOME}/upload/model_unittest_FAIL.log.${AGILE_PIPELINE_BUILD_ID}.${AGILE_JOB_BUILD_ID}
            cd ${PPNLP_HOME} && python upload.py ${PPNLP_HOME}/upload 'paddlenlp/PaddleNLP_CI/PaddleNLP-CI-Model-Unittest-GPU'
            rm -rf upload/* && cd -
        fi
        if [ $1 -eq 124 ]; then
            echo "\033[32m [failed-timeout] Test case execution was terminated after exceeding the ${running_time} min limit."
        fi
    else
        tail -n 1 ${log_path}/model_unittest.log
        echo -e "\033[32m ${log_path}/model_unittest_SUCCESS \033[0m"
    fi
}

get_diff_TO_case(){
    export FLAGS_enable_CI=false
    if [ -z "${AGILE_COMPILE_BRANCH}" ]; then
        # Scheduled Regression Test
        FLAGS_enable_CI=true
    else
        for file_name in `git diff --numstat ${AGILE_COMPILE_BRANCH} -- |awk '{print $NF}'`;do
            ext="${file_name##*.}"
            echo "file_name: ${file_name}, ext: ${file_name##*.}"
            
            if [ ! -f ${file_name} ];then # Delete Files for a Pull Request
                continue
            elif [[ "$ext" == "md" || "$ext" == "rst" || "$file_name" == docs/* ]]; then
                continue
            else
                FLAGS_enable_CI=true
            fi
        done
    fi
}

get_diff_TO_case
set_env
if [[ ${FLAGS_enable_CI} == "true" ]] || [[ ${FLAGS_enable_CE} == "true" ]];then
    cd ${paddleformers_code_path}
    echo ' Testing all model unittest cases '
    unset http_proxy && unset https_proxy
    set +e
    echo "Check paddle Cuda Version"
    python -c "import paddle; print(paddle.version.cuda()); print(paddle.version.cudnn()); print(paddle.is_compiled_with_cuda())"
    echo "Check docker Cuda Version"
    nvcc -V  
    echo "Check nvidia-smi"
    nvidia-smi
    echo "Check paddle GPU count"
    python -c "import paddle; print(paddle.device.device_count())"
    PYTHONPATH=$(pwd) \
    COVERAGE_SOURCE=paddleformers \
    python -m pytest -s -v ${model_unittest_path} > ${log_path}/model_unittest.log 2>&1
    exit_code=$?
    print_info $exit_code 
else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    exit_code=0
fi
exit $exit_code