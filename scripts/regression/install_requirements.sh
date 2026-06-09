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

install_requirements() {
    local ce_branch=${1:-"false"}
    start_ts=$(date +%s)
    python -m pip uninstall paddlepaddle paddlepaddle_gpu paddlefleet paddleformers -y
    rm -rf ./build ./dist ./paddleformers.egg-info/
    python -m pip config --user set global.trusted-host pypi.org
    python -m pip config --user set global.index-url https://pypi.org/simple
    # Todo: fix later 
    # python -m pip install -U --no-cache-dir transformers -i https://pypi.org/simple > /dev/null
    python -m pip install -r requirements.txt -i https://pypi.org/simple 
    if [[ "$ce_branch" == "CE_Release_cu129_py312_nightly" ]]; then # nightly regerssion
        #formers - build wheel first
        python setup.py bdist_wheel  > /dev/null
        #fleet paddle locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/
        #paddlefleet_ops  todo:release
        install_ops_wheel_dev
        #formers - reinstall
        python -m pip install ./dist/*.whl 
    elif [[ "$ce_branch" == "CE_Develop_cu132_py312" ]]; then # nightly regerssion
        #fleet
        python -m pip install --pre paddlefleet --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu132/  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu132/ -i https://pypi.org/simple 
        #paddlefleet_ops
        install_ops_wheel_dev
        #paddle
        python -m pip uninstall paddlepaddle-gpu -y
        wget -q https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc11-Cuda132-Cudnn920-Trt1016-Py312-Compile/latest/paddlepaddle_gpu-0.0.0-cp312-cp312-linux_x86_64.whl
        python -m pip install paddlepaddle_gpu-0.0.0-cp312-cp312-linux_x86_64.whl --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu132 
        #formers
        python setup.py bdist_wheel  > /dev/null
        python -m pip install ./dist/*.whl
    elif [[ "$ce_branch" == "CE_Release_cu132_py312" ]]; then
        #formers - build wheel first
        python setup.py bdist_wheel  > /dev/null
        #fleet paddle locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu132/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu132/
        #paddlefleet_ops
        install_ops_wheel_dev
        #formers - reinstall
        python -m pip install ./dist/*.whl
    elif [[ "$ce_branch" == "CE_Develop_cu130_py313" ]]; then # nightly regerssion
        #fleet
        python -m pip install --pre paddlefleet --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/ -i https://pypi.org/simple 
        #paddlefleet_ops
        install_ops_wheel_dev
        #paddle
        python -m pip uninstall paddlepaddle-gpu -y
        wget -q https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda130-Cudnn913-Trt1013-Mkl-Avx-Gcc11-SelfBuiltPypiUse/latest/paddlepaddle_gpu-0.0.0-cp313-cp313-linux_x86_64.whl
        python -m pip install paddlepaddle_gpu-0.0.0-cp313-cp313-linux_x86_64.whl --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/ 
        #formers
        python setup.py bdist_wheel  > /dev/null
        python -m pip install ./dist/*.whl
    elif [[ "$ce_branch" == "CE_Develop_cu130_py312" ]]; then # nightly regerssion
        #fleet
        python -m pip install --pre paddlefleet --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/ -i https://pypi.org/simple 
        #paddlefleet_ops
        install_ops_wheel_dev
        #paddle
        python -m pip uninstall paddlepaddle-gpu -y
        wget -q https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda130-Cudnn913-Trt1013-Mkl-Avx-Gcc11-SelfBuiltPypiUse/latest/paddlepaddle_gpu-0.0.0-cp312-cp312-linux_x86_64.whl
        python -m pip install paddlepaddle_gpu-0.0.0-cp312-cp312-linux_x86_64.whl --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/ 
        #formers
        python setup.py bdist_wheel  > /dev/null
        python -m pip install ./dist/*.whl
    elif [[ "$ce_branch" == "CE_Release_cu130_py313" ]]; then # release regerssion
        #formers - build wheel first
        python setup.py bdist_wheel  > /dev/null
        #fleet paddle locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/
        #paddlefleet_ops
        install_ops_wheel_dev
        #formers - reinstall
        python -m pip install ./dist/*.whl
    elif [[ "$ce_branch" == "CE_Release_cu130_py312" ]]; then # release regerssion
        #formers - build wheel first
        python setup.py bdist_wheel  > /dev/null
        #fleet paddle locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/
        #paddlefleet_ops
        install_ops_wheel_dev
        #formers - reinstall
        python -m pip install ./dist/*.whl
     elif [[ "$ce_branch" == "CE_Release_cu130_py311" ]]; then # release regerssion
        #formers - build wheel first
        python setup.py bdist_wheel  > /dev/null
        #fleet paddle locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/
        #paddlefleet_ops
        install_ops_wheel_dev
        #formers - reinstall
        python -m pip install ./dist/*.whl
    elif [[ "$ce_branch" == "CE_Release_cu129_py313" ]]; then # release regerssion
        #formers - build wheel first
        python setup.py bdist_wheel  > /dev/null
        #fleet paddle locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/
        #paddlefleet_ops
        install_ops_wheel_dev
        #formers - reinstall
        python -m pip install ./dist/*.whl 
    elif [[ "$ce_branch" == "CE_Release_cu129_py312_weekly" ]]; then # release regerssion
        #formers - build wheel first
        python setup.py bdist_wheel  > /dev/null
        #fleet paddle locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/
        #paddlefleet_ops
        install_ops_wheel_dev
        #formers - reinstall
        python -m pip install ./dist/*.whl  
    elif [[ "$ce_branch" == "CE_Release_cu126_py310" ]]; then # release regerssion
        #formers - build wheel first
        python setup.py bdist_wheel  > /dev/null
        #fleet paddle locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/
        #paddlefleet_ops
        install_ops_wheel_dev
        #formers - reinstall
        python -m pip install ./dist/*.whl    
    else
        echo "Install CI ENV: Cuda129+Python312"
        #formers - build wheel first
        python setup.py bdist_wheel > /dev/null
        #fleet paddle locked
        pip install "$(ls -t dist/*.whl | head -1)[paddlefleet]" -i https://pypi.org/simple --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/
        #paddlefleet_ops
        install_ops_wheel_dev
    fi
    python -m pip install -r tests/requirements.txt -i https://pypi.org/simple 

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

install_ops_wheel_dev(){
    echo "Download PaddleFleet form https://paddle-qa.bj.bcebos.com/CodeSync/develop/PaddleFleet.tar"
    wget -q --no-proxy  https://paddle-qa.bj.bcebos.com/CodeSync/develop/PaddleFleet.tar --no-check-certificate
    rm -rf PaddleFleet && tar xf PaddleFleet.tar && rm -rf PaddleFleet.tar
    cd PaddleFleet && bash scripts/install_ops_wheel.sh && cd -
}

install_ops_wheel_release(){
    echo "Download PaddleFleet form https://paddle-qa.bj.bcebos.com/CodeSync/release/0.3/PaddleFleet.tar"
    wget -q --no-proxy  https://paddle-qa.bj.bcebos.com/CodeSync/release/0.3/PaddleFleet.tar --no-check-certificate
    rm -rf PaddleFleet && tar xf PaddleFleet.tar && rm -rf PaddleFleet.tar
    cd PaddleFleet && bash scripts/install_ops_wheel.sh && cd -
}

# Call the function with the first argument (ce_branch), default to "false" for CI env: Cuda126+Python310
install_requirements "${1:-false}"