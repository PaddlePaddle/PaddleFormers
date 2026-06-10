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
unset http_proxy && unset https_proxy
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
    start_ts=$(date +%s)
    python -m pip config --user set global.trusted-host pypi.org
    python -m pip config --user set global.index-url https://pypi.org/simple
    python -m pip uninstall paddlepaddle paddlepaddle_gpu paddlefleet -y
    python -m pip install -U --no-cache-dir transformers -i https://pypi.org/simple  > /dev/null
    cd /home/models/my_packages && dpkg -i *.deb > /dev/null
    cd -
    # python -m pip install --no-cache-dir ${paddle} --no-dependencies --progress-bar off
    # echo "paddlepaddle-gpu @ https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-Gcc11-SelfBuiltPypiUse/cbf3469113cd76b7d5f4cba7b8d7d5f55d9e9911/paddlepaddle_gpu-3.3.0-cp310-cp310-linux_x86_64.whl" >> requirements.txt
    python setup.py bdist_wheel > /dev/null
    if [ $FLAGS_enable_CE == "true" ];then
        python -m pip install dist/*.whl 
        #fleet develop
        python -m pip install --pre paddlefleet --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/ 
        #paddlefleet_ops develop
        echo "Download PaddleFleet form https://paddle-qa.bj.bcebos.com/CodeSync/develop/PaddleFleet.tar"
        wget -q --no-proxy  https://paddle-qa.bj.bcebos.com/CodeSync/develop/PaddleFleet.tar --no-check-certificate
        rm -rf PaddleFleet && tar xf PaddleFleet.tar && rm -rf PaddleFleet.tar
        local commit=$(python -c "import paddlefleet; print(paddlefleet.version.commit)" 2>/dev/null || echo "")
        if [[ -z "${commit}" ]]; then
            echo "Warning: failed to get paddlefleet commit from env, skip git reset"
        else
            echo "Reset PaddleFleet to commit: ${commit}"
        fi
        cd PaddleFleet
        if [[ -n "${commit}" ]]; then
            git reset --hard ${commit}
        fi
        bash scripts/install_ops_wheel.sh
        cd -
        #paddle develop
        python -m pip uninstall paddlepaddle-gpu -y
        wget -q $paddle
        python -m pip install paddlepaddle_gpu-0.0.0-cp312-cp312-linux_x86_64.whl --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/ 
    else
        # fleet_locked paddle_locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/
        #paddlefleet_ops for fleet_locked
        echo "Download PaddleFleet form https://paddle-qa.bj.bcebos.com/CodeSync/release/0.3/PaddleFleet.tar"
        wget -q --no-proxy  https://paddle-qa.bj.bcebos.com/CodeSync/release/0.3/PaddleFleet.tar --no-check-certificate
        rm -rf PaddleFleet && tar xf PaddleFleet.tar && rm -rf PaddleFleet.tar
        local commit=$(python -c "import paddlefleet; print(paddlefleet.version.commit)" 2>/dev/null || echo "")
        if [[ -z "${commit}" ]]; then
            echo "Warning: failed to get paddlefleet commit from env, skip git reset"
        else
            echo "Reset PaddleFleet to commit: ${commit}"
        fi
        cd PaddleFleet
        if [[ -n "${commit}" ]]; then
            git reset --hard ${commit}
        fi
        bash scripts/install_ops_wheel.sh
        cd -
    fi
    pip install -r tests/requirements.txt -i https://pypi.org/simple 

    echo "paddle commit:"
    python -c "import paddle; print(paddle.version.commit)"
    echo "paddlefleet commit:"
    python -c "import paddlefleet; print(paddlefleet.version.commit)"
    echo "paddlefleet_ops commit:"
    python -c "from paddlefleet_ops import __version__; print(__version__)"
    echo "paddleformers commit:"
    python -c "import paddleformers; print(paddleformers.version.commit)"

    python -c "import paddle; print('paddle commit:',paddle.version.commit)" >> ${log_path}/commit_info.txt
    python -c "import paddle;print('paddle');print(paddle.__version__);print(paddle.version.show())" >> ${log_path}/commit_info.txt
    python -c "from paddleformers import __version__; print('paddleformers version:', __version__)" >> ${log_path}/commit_info.txt
    python -c "import paddleformers; print('paddleformers commit:',paddleformers.version.commit)" >> ${log_path}/commit_info.txt
    python -c "from paddlefleet_ops import __version__; print('paddlefleet_ops version:', __version__)" >> ${log_path}/commit_info.txt
    python -c "import paddlefleet; print('paddlefleet commit:',paddlefleet.version.commit)" >> ${log_path}/commit_info.txt
    python -m pip list >> ${log_path}/commit_info.txt
    end_ts=$(date +%s)
    echo -e "\033[32m install requirements cost $((end_ts - start_ts))s \033[0m"
}

set_env() {
    export NVIDIA_TF32_OVERRIDE=0 
    export FLAGS_cudnn_deterministic=1
    export FLAGS_use_cuda_managed_memory=true
    export HF_ENDPOINT=https://hf-mirror.com

    # for CE
    # if [[ ${FLAGS_enable_CE} == "true" ]];then
    #     export CE_TEST_ENV=1
    #     export RUN_SLOW_TEST=1
    #     unset PF_HOME
    #     export PYTHONPATH=${nlp_dir}:${nlp_dir}/llm:${PYTHONPATH}
    # fi
}

print_info() {
    if [ $1 -ne 0 ]; then
        cat ${log_path}/unittest.log | grep -v "Fail to fscanf: Success" \
            | grep -v "SKIPPED" | grep -v "warning" > ${log_path}/unittest_FAIL.log
        cat ${log_path}/unittest_FAIL.log
        tail -n 1 ${log_path}/unittest.log >> ${log_path}/unittest_FAIL.log
        echo -e "\033[31m ${log_path}/unittest_FAIL \033[0m"
        tail -n 1 ${log_path}/unittest_FAIL.log
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
        [[ -f "$file_name" ]] || continue
        if [[ "$ext" == "py" ]] || [[ "$ext" == "yml" ]] || [[ "$file_name" == "requirements.txt" ]]; then
            FLAGS_enable_CI=true
            break
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
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
    set +e
    export PYTHONFAULTHANDLER=1
    
    DOWNLOAD_SOURCE=aistudio WAIT_UNTIL_DONE=True PADDLEFORMERS_TESTING=True \
    PYTHONPATH=$(pwd) \
    COVERAGE_SOURCE=paddleformers \
    timeout 60m \
    python -m pytest -v -s -n 1 \
        --dist no \
        --maxfail=10 \
        --timeout 200 --durations 20 \
        --alluredir=result \
        --cov=paddleformers \
        --cov-report=xml:coverage.xml > ${log_path}/unittest.log 2>&1
    exit_code=$?
    print_info $exit_code unittest
    echo -e "\033[35m ---- Set PYTEST_EXECUTE_FLAG_FILE  \033[0m"
    touch ${PYTEST_EXECUTE_FLAG_FILE}
else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    exit_code=0
fi
exit $exit_code
