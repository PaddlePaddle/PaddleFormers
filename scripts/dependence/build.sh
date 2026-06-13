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

python -m pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --user set global.trusted-host pypi.tuna.tsinghua.edu.cn

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
    python setup.py bdist_wheel

    echo "install_formers_develop_whl"
    cd ../
    python -m pip install --upgrade pip
    python -m pip install --ignore-installed  PaddleFormers/dist/p****.whl --no-cache-dir --force-reinstall --no-dependencies
    cd -
    echo "waiting for import paddleformers..."
    python -c "import paddleformers; print('paddleformers commit:',paddleformers.version.commit)" >> ${log_path}/commit_info.txt
    commit=$(python -c "import paddleformers; print(paddleformers.version.commit)")
    commit=${commit:-unknown}

    whl_file=$(ls $formers_dir/dist/paddleformers-*.whl)
    base_name=$(basename $whl_file)

    # 复制原始 whl
    cp "$whl_file" "${upload_path}/${base_name}"

    # 固定名称备份，支持 dev 和 post
    if echo "$base_name" | grep -q "\.post[0-9]*"; then
        cp "$whl_file" "${upload_path}/paddleformers-0.0.0.post0-py3-none-any.whl"
        new_name=$(echo $base_name | sed "s/\.post[0-9]\+/&+${commit}/")
    elif echo "$base_name" | grep -q "\.dev[0-9]*"; then
        cp "$whl_file" "${upload_path}/paddleformers-0.0.0.dev0-py3-none-any.whl"
        new_name=$(echo $base_name | sed "s/\.dev[0-9]\+/&+${commit}/")
    else
        cp "$whl_file" "${upload_path}/paddleformers-0.0.0-py3-none-any.whl"
        new_name="${base_name%.whl}+${commit}.whl"
    fi

    echo "commit whl: $new_name"
    cp "$whl_file" "${upload_path}/${new_name}"

    echo "install_formers_commit_whl"
    python -m pip install --user "${upload_path}/${new_name}" --no-cache-dir --force-reinstall --no-dependencies
    python -c "import paddleformers; print('paddleformers commit:',paddleformers.version.commit)" >> ${log_path}/commit_info.txt
}

# main
cd ${formers_dir}
echo -e "\033[32m ---- build paddleformers whl  \033[0m"
paddleformers_build

echo -e "\033[32m ---- make PaddleFormers.tar.gz  \033[0m"
cd ${formers_dir}
# paddleformer.tar only include the develop branch
if [ -n "$BRANCH" ] && [ "$BRANCH" = "develop" ]; then
    echo "Checkout branch $BRANCH"
    cd /workspace
    tar -zcf PaddleFormers.tar.gz PaddleFormers/
    mv PaddleFormers.tar.gz ${upload_path}/
else
    echo "No BRANCH specified, skip checkout"
fi