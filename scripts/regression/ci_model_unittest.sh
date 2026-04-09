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

install_requirements() {
    start_ts=$(date +%s)
    python -m pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip config --user set global.trusted-host pypi.tuna.tsinghua.edu.cn
    python -m pip uninstall paddlepaddle paddlepaddle_gpu paddlefleet -y
    # Fix later 
    # python -m pip install -U --no-cache-dir transformers
    python -m pip install -r requirements.txt -i https://pypi.org/simple
    # echo "paddlepaddle-gpu @ https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-Gcc11-SelfBuiltPypiUse/cbf3469113cd76b7d5f4cba7b8d7d5f55d9e9911/paddlepaddle_gpu-3.3.0-cp310-cp310-linux_x86_64.whl" >> requirements.txt
    python setup.py bdist_wheel > /dev/null
    pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/
    echo "paddlefleet commit:"
    python -c "import paddlefleet; print(paddlefleet.version.commit)"
    python -c "import paddle;print('paddle');print(paddle.__version__);print(paddle.version.show())" >> ${log_path}/commit_info.txt
    pip install -r tests/requirements.txt
    python -c "from paddleformers import __version__; print('paddleformers version:', __version__)" >> ${log_path}/commit_info.txt
    python -c "import paddleformers; print('paddleformers commit:',paddleformers.version.commit)" >> ${log_path}/commit_info.txt
    python -m pip list >> ${log_path}/commit_info.txt
    end_ts=$(date +%s)
    echo -e "\033[32m install requirements cost $((end_ts - start_ts))s \033[0m"
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
    models="glm_moe"
    echo "No transformer changes detected, using default model: $models"
fi

}

init_env
if [[ "$update_baseline_models" != "false" ]] && [[ "$update_baseline_models" != "False" ]]; then
    echo "Update baseline models: $update_baseline_models"
    models=$update_baseline_models
elif [[ ${FLAGS_enable_CI} == "True" ]];then
    get_diff_TO_case
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
    python -m pytest -s -v --models=${models} --update-baseline=${update_baseline_models} scripts/regression/test_models.py > ${log_path}/model_unittest.log 2>&1
    exit_code=$?
    print_info $exit_code model_unittest
    upload_baseline 
else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    exit_code=0
fi
exit $exit_code