#!/usr/bin/env bash

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

set -e
export formers_dir=/workspace/PaddleFormers
mkdir -p /workspace/PaddleFormers/build_logs
export log_path=/workspace/PaddleFormers/build_logs
mkdir -p /workspace/PaddleFormers/upload
upload_path=/workspace/PaddleFormers/upload
export Build_list=()

python -m pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --user set global.trusted-host pypi.tuna.tsinghua.edu.cn

get_diff_case(){
    git diff --numstat --name-only HEAD~1 HEAD
    for file_name in `git diff --numstat --name-only HEAD~1 HEAD`;do
        arr_file_name=(${file_name//// })
        if [[ "${arr_file_name[0]}" == ".github" || "${arr_file_name[0]}" == "scripts" || "${arr_file_name[0]}" == "tests" ]]; then
            continue
        else
            Build_list[${#Build_list[@]}]="paddleformers"
        fi
    done
    echo ${Build_list[*]}
}

install_paddle(){
    echo -e "\033[35m ---- Install paddlepaddle-gpu  \033[0m"
    python -m pip uninstall paddlepaddle -y
    python -m pip install --user ${paddle} --no-cache-dir;
    python -c "import paddle;print('paddle');print(paddle.__version__); \
        print(paddle.version.show())" >> ${log_path}/commit_info.txt
}

paddleformers_build (){
    echo -e "\033[35m ---- build latest paddleformers  \033[0m"
    cd $formers_dir
    rm -rf build/
    rm -rf dist/
    rm -rf paddleformers.egg-info/

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python setup.py bdist_wheel
    python -m pip install --ignore-installed  dist/p****.whl --force-reinstall --no-dependencies
    python -c "import paddleformers; print('paddleformers commit:',paddleformers.version.commit)" >> ${log_path}/commit_info.txt

    cp $formers_dir/dist/p****.whl ${upload_path}/
    cp $formers_dir/dist/p****.whl ${upload_path}/paddleformers-0.0.0-py3-none-any.whl
}

install_paddleformers(){
    echo "install_formers_develop"
    python -m pip install --user https://paddleformers.bj.bcebos.com/wheels/paddleformers-0.0.0-py3-none-any.whl --no-cache-dir
    python -c "import paddleformers; print('paddleformers commit:',paddleformers.version.commit)" >> ${log_path}/commit_info.txt
}

contain_case(){
    local e
    for e in "${@:2}";do
        if [[ "$e" == "$1" ]];then
            return 1
        fi
    done
    return 0
}

### main
cd ${formers_dir}
get_diff_case
Build_list=($(awk -v RS=' ' '!a[$1]++' <<< ${Build_list[*]}))
if [[ ${#Build_list[*]} -ne 0 ]];then
    echo -e "\033[31m ---- Build_list length: ${#Build_list[*]}, cases: ${Build_list[*]} \033[0m"
    echo -e "\033[31m ============================= \033[0m"
    # install_paddle
    if [[ $(contain_case paddleformers ${Build_list[@]}; echo $?) -eq 1 ]];then
        paddleformers_build
    fi
else
    echo -e "\033[32m Don't need build any whl  \033[0m"
fi


echo -e "\033[32m ---- make PaddleFormers.tar.gz  \033[0m"
cd ${formers_dir}
git checkout develop
cd /workspace
tar -zcf PaddleFormers.tar.gz PaddleFormers/
mv PaddleFormers.tar.gz ${upload_path}/


# upload in .github/workflows/ce-build-whl.yml
# if [ -e "${upload_path}" ] && [ "$(ls -A "${upload_path}/")" ]; then
#     cd ${upload_path} && ls -A "${upload_path}"
#     python /workspace/../../../bos/BosClient.py ${upload_path} 'paddleformers/wheels'
#     rm -rf ${upload_path}
#     echo -e "\033[32m upload wheels SUCCESS \033[0m"
# fi