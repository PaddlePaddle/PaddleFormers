#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

export nlp_dir=/workspace/PaddleNLP/
mkdir ${nlp_dir}/logs
mkdir ${nlp_dir}/model_logs
mkdir ${nlp_dir}/unittest_logs
p0case="refactor_training_loop"

# Install Paddle FleetY
install_paddle(){
    echo -e "\033[35m ---- Install paddlepaddle-gpu  \033[0m"
    python -m pip install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
    python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
}
# Install paddlenlp 
nlp_build (){
    cd ${nlp_dir}
    rm -rf build/
    rm -rf paddlenlp.egg-info/
    rm -rf ppdiffusers.egg-info/
    rm -rf paddle_pipelines.egg-info/
    rm -rf dist/

    python -m pip install -r requirements.txt
    python -m pip install rouge
    python setup.py bdist_wheel
    python -m pip install dist/p****.whl
}
install_paddle
nlp_build
pip list

# run  ci case
echo -e "\033[35m ======= Check refactor_training_loop BRANCH ========= \033[0m"
set +e
echo -e "\033[35m ---- start run case  \033[0m"
bash ${nlp_dir}/scripts/regression/ci_case.sh ${p0case} ${cudaid1} ${cudaid2}
echo -e "\033[35m ---- end run P0case  \033[0m"

# analysis log
cd ${nlp_dir}/model_logs
FF=`ls *FAIL*|wc -l`
EXCODE=0
if [ "${FF}" -gt "0" ];then
    P0case_EXCODE=1
    EXCODE=2
else
    P0case_EXCODE=0
fi
if [ $P0case_EXCODE -ne 0 ] ; then
    echo -e "\033[31m ---- P0case Failed number: ${FF} \033[0m"
    ls *_FAIL*
else
    echo -e "\033[32m ---- P0case Success \033[0m"
fi
    
cd ${nlp_dir}/unittest_logs
UF=`ls *FAIL*|wc -l`
if [ "${UF}" -gt "0" ];then
    UT_EXCODE=1
    EXCODE=3
else
    UT_EXCODE=0
fi
if [ $UT_EXCODE -ne 0 ] ; then
    echo -e "\033[31m ---- Unittest Failed \033[0m"
    ls *_FAIL*
else
    echo -e "\033[32m ---- Unittest Success \033[0m"
fi
exit $EXCODE