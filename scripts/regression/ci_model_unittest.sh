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

export nlp_dir=/workspace/PaddleFormers
export log_path=/workspace/PaddleFormers/model_unittest_logs
export model_unittest_path=/workspace/PaddleFormers/scripts/regression
export AGILE_COMPILE_BRANCH=$AGILE_COMPILE_BRANCH
cd $nlp_dir
mkdir -p $log_path

install_requirements() {
    local ce_branch=${1:-""}
    start_ts=$(date +%s)
    python -m pip uninstall paddlepaddle paddlepaddle_gpu paddlefleet paddleformers -y
    rm -rf ./build ./dist ./paddleformers.egg-info/
    # Todo: fix later 
    # python -m pip install -U --no-cache-dir transformers -i https://pypi.org/simple > /dev/null
    python -m pip install -r requirements.txt -i https://pypi.org/simple 
    if [[ $ce_branch="release" ]]; then
        #fleet
        wget -q https://paddle-github-action.bj.bcebos.com/PaddleFleet/release/0.2/latest/cu126/paddlefleet-0.0.0-cp310-cp310-linux_x86_64.whl
        pip install  paddlefleet-0.0.0-cp310-cp310-linux_x86_64.whl --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/ -i https://pypi.org/simple 
        pip uninstall paddlepaddle-gpu -y
        #paddle
        wget -q https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Gpu-Cuda12.6-Cudnn9.5-Trt10.5-Mkl-Avx-Gcc11-SelfBuiltPypiUse//latest/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
        pip install paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl  --index-url=https://www.paddlepaddle.org.cn/packages/nightly/cu126/
        #formers
        python setup.py bdist_wheel  > /dev/null
        python -m pip install ./dist/*.whl 
    elif [[ $ce_branch="develop" ]]; then
        #fleet
        python -m pip install --pre paddlefleet --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/ -i https://pypi.org/simple 
        python -m pip uninstall paddlepaddle-gpu -y
        #paddle
        wget -q https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda12.6-Cudnn9.5-Trt10.5-Mkl-Avx-Gcc11-SelfBuiltPypiUse/latest/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
        python -m pip install paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/ 
        #formers
        python setup.py bdist_wheel  > /dev/null
        python -m pip install ./dist/*.whl 
    else
        python setup.py bdist_wheel > /dev/null
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/
    fi
   
    echo "paddlefleet commit:"
    python -c "import paddlefleet; print(paddlefleet.version.commit)"
    python -c "import paddle;print('paddle');print(paddle.__version__);print(paddle.version.show())" >> ${log_path}/commit_info.txt
    python -c "from paddleformers import __version__; print('paddleformers version:', __version__)" >> ${log_path}/commit_info.txt
    python -c "import paddleformers; print('paddleformers commit:',paddleformers.version.commit)" >> ${log_path}/commit_info.txt
    python -m pip install -r tests/requirements.txt -i https://pypi.org/simple 
    python -m pip list >> ${log_path}/commit_info.txt
    end_ts=$(date +%s)
    echo -e "\033[32m install requirements cost $((end_ts - start_ts))s \033[0m"
}

set_env() {
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_cudnn_deterministic=1
    export HF_ENDPOINT=https://hf-mirror.com

    # for CI/CE
    if [ -f "./scripts/regression/config.yaml" ]; then
      mv ./scripts/regression/config.yaml ./scripts/regression/config.yaml.bak
    fi

    if [[ "${FLAGS_enable_CE}" == "CE_Release" ]];then
        echo "CE_Release: install paddle release + fleet release + formers release"
        install_requirements "${release}"
        # donwload configs
        cd ./scripts/regression
        wget https://paddle-qa.bj.bcebos.com/paddleformers/ce_release_config/config.yaml 
        # update configs
        python merge_configs.py --origin_config config_origin.yaml --update_config config.yaml
        cd -
    elif [[ "${FLAGS_enable_CE}" == "CE_Develop" ]];then
        echo "CE_Develop: install paddle develop + fleet develop + formers develop"
        install_requirements "${develop}"
        # donwload configs
        cd ./scripts/regression
        wget https://paddle-qa.bj.bcebos.com/paddleformers/ce_develop_config/config.yaml 
        # update configs
        python merge_configs.py --origin_config config_origin.yaml --update_config config.yaml
        cd -
    elif [[ "${FLAGS_enable_CI}" == "True" ]];then
        echo "CI: install paddle stable + fleet stable + formers"
        install_requirements
        # donwload configs
        cd ./scripts/regression
        wget https://paddle-qa.bj.bcebos.com/paddleformers/ci_config/config.yaml 
        # update configs
        python merge_configs.py --origin_config config_origin.yaml --update_config config.yaml
        cd -

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
for file_name in `git diff --numstat ${AGILE_COMPILE_BRANCH} -- |awk '{print $NF}'`;do
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
                FLAGS_enable_CI=True
                echo "Detected model: $model_name"
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

set_env
# 如果外部传入了 models，则跳过自动检测，使用外部传入的值
if [[ "$update_baseline_models" != "false" ]] && [[ "$update_baseline_models" != "False" ]]; then
    echo "Update baseline models: $update_baseline_models"
    models=$update_baseline_models
else
    get_diff_TO_case
fi

if [[ ${FLAGS_enable_CI} == "True" ]] || [[ ${FLAGS_enable_CE} == "CE_Develop" ]]|| [[ ${FLAGS_enable_CE} == "CE_Release" ]];then
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

else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    exit_code=0
fi
exit $exit_code