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
export nlp_dir=/workspace/PaddleFormers
export log_path=/workspace/PaddleFormers/unittest_logs
cd $nlp_dir
if [ ! -d "unittest_logs" ];then
    mkdir unittest_logs
fi
mkdir -p $log_path
export PYTEST_EXECUTE_FLAG_FILE=${3}
echo "PYTEST_EXECUTE_FLAG_FILE is ${PYTEST_EXECUTE_FLAG_FILE}"
if [ -f "${PYTEST_EXECUTE_FLAG_FILE}" ]; then
    rm "${PYTEST_EXECUTE_FLAG_FILE}"
fi
dir_name=$(dirname "${PYTEST_EXECUTE_FLAG_FILE}")
mkdir -p "${dir_name}"
AGILE_COMPILE_BRANCH=$4


install_requirements() {
    python -m pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip config --user set global.trusted-host pypi.tuna.tsinghua.edu.cn
    python -m pip uninstall paddlepaddle paddlepaddle_gpu paddlefleet -y
    # python -m pip install --no-cache-dir ${paddle} --no-dependencies --progress-bar off
    python setup.py bdist_wheel > /dev/null
    uv pip install dist/p****.whl --system --prerelease=allow -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/ --index-strategy unsafe-best-match
    python -c "import paddle;print('paddle');print(paddle.__version__);print(paddle.version.show())" >> ${log_path}/commit_info.txt
    uv pip install -r tests/requirements.txt --system -i https://pypi.tuna.tsinghua.edu.cn/simple --index-strategy unsafe-best-match
    python -c "from paddleformers import __version__; print('paddleformers version:', __version__)" >> ${log_path}/commit_info.txt
    python -c "import paddleformers; print('paddleformers commit:',paddleformers.version.commit)" >> ${log_path}/commit_info.txt
    python -m pip list >> ${log_path}/commit_info.txt
}

set_env() {
    export NVIDIA_TF32_OVERRIDE=0 
    export FLAGS_cudnn_deterministic=1
    export HF_ENDPOINT=https://hf-mirror.com
    export FLAGS_use_cuda_managed_memory=true

    # for CE
    if [[ ${FLAGS_enable_CE} == "true" ]];then
        export CE_TEST_ENV=1
        export RUN_SLOW_TEST=1
        unset PF_HOME
        export PYTHONPATH=${nlp_dir}:${nlp_dir}/llm:${PYTHONPATH}
    fi
}

print_info() {
    if [ $1 -ne 0 ]; then
        cat ${log_path}/unittest.log | grep -v "Fail to fscanf: Success" \
            | grep -v "SKIPPED" | grep -v "warning" > ${log_path}/unittest_FAIL.log
        tail -n 1 ${log_path}/unittest.log >> ${log_path}/unittest_FAIL.log
        echo -e "\033[31m ${log_path}/unittest_FAIL \033[0m"
        cat ${log_path}/unittest_FAIL.log
        if [ -n "${AGILE_JOB_BUILD_ID}" ]; then
            cp ${log_path}/unittest_FAIL.log ${PPNLP_HOME}/upload/unittest_FAIL.log.${AGILE_PIPELINE_BUILD_ID}.${AGILE_JOB_BUILD_ID}
            cd ${PPNLP_HOME} && python upload.py ${PPNLP_HOME}/upload 'paddlenlp/PaddleNLP_CI/PaddleNLP-CI-Unittest-GPU'
            rm -rf upload/* && cd -
        fi
        if [ $1 -eq 124 ]; then
            echo "\033[32m [failed-timeout] Test case execution was terminated after exceeding the ${running_time} min limit."
        fi
    else
        tail -n 1 ${log_path}/unittest.log
        echo -e "\033[32m ${log_path}/unittest_SUCCESS \033[0m"
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
    install_requirements
    cd ${nlp_dir}
    echo ' Testing all unittest cases '
    unset http_proxy && unset https_proxy
    set +e
    DOWNLOAD_SOURCE=aistudio WAIT_UNTIL_DONE=True \
    PYTHONPATH=$(pwd) \
    COVERAGE_SOURCE=paddleformers \
    python -m pytest -v -s -n 8 \
        --dist no \
        --maxfail=1 \
        --retries 3 --retry-delay 1 \
        --timeout 200 --durations 20 \
        --alluredir=result \
        --cov=paddleformers \
        --cov-report=xml:coverage.xml > ${log_path}/unittest.log 2>&1
    exit_code=$?
    print_info $exit_code unittest
    echo -e "\033[35m ---- Set PYTEST_EXECUTE_FLAG_FILE  \033[0m"
    touch ${PYTEST_EXECUTE_FLAG_FILE}
    if [ -n "${AGILE_JOB_BUILD_ID}" ]; then
        cd ${nlp_dir}
        echo -e "\033[35m ---- Generate Allure Report  \033[0m"
        unset http_proxy && unset https_proxy
        cp scripts/unit_test/gen_allure_report.py ./
        python gen_allure_report.py > /dev/null
        echo -e "\033[35m ---- Report: https://xly.bce.baidu.com/ipipe/ipipe-report/report/${AGILE_JOB_BUILD_ID}/report/  \033[0m"
    else
        echo "AGILE_JOB_BUILD_ID is empty, skip generate allure report"
    fi
else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    exit_code=0
fi
exit $exit_code