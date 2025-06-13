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

export Testcase=$3
export cudaid2=$2
export cudaid1=$1
export nlp_dir=${PWD}
mkdir ${nlp_dir}/logs
mkdir ${nlp_dir}/model_logs
mkdir ${nlp_dir}/unittest_logs
export log_path=${nlp_dir}/logs
####################################
# for paddlenlp env
python -c 'import sys; print(sys.version_info[:])'
python -c 'import nltk; nltk.download("punkt")'
set -x
python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
nlp1_build (){
    echo -e "\033[35m ---- only install paddlenlp \033[0m"
    python -m pip install paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
}
nlp2_build (){
    echo -e "\033[35m ---- build and install paddlenlp  \033[0m"
    rm -rf build/
    rm -rf paddlenlp.egg-info/
    rm -rf dist/

    python -m pip install -r requirements.txt
    python setup.py bdist_wheel
    python -m pip install -U dist/paddlenlp****.whl
}
nlp2_build
python -c 'from visualdl import LogWriter'
pip list
set +x
####################################
# run p0case
export P0case_list=()
export P0case_time=0
export all_P0case_time=0
declare -A all_P0case_dic
get_diff_TO_P0case(){
if [[ ${Testcase} =~ "all" ]];then
    P0case_list=(msra_ner glue bert skep bigbird gpt ernie-1.0 xlnet ofa  squad tinybert lexical_analysis seq2seq \
    word_embedding ernie-ctm distilbert stacl transformer simbert pointer_summarizer question_matching ernie-csc \
    nptag clue taskflow transformers fast_generation ernie-3.0 fast_transformer fast_gpt llama)
elif [[ ${Testcase} =~ "p0" ]];then
    P0case_list=(glue bert skep gpt ernie-1.0 transformer clue)
else
    P0case_list=${Testcase}
fi
}
get_diff_TO_P0case
    echo -e "\033[35m =======CI Check P0case========= \033[0m"
    echo -e "\033[35m ---- P0case_list length: ${#P0case_list[*]}, cases: ${P0case_list[*]} \033[0m"
    set +e
    echo -e "\033[35m ---- start run P0case  \033[0m"
    case_num=1
    for p0case in ${P0case_list[*]};do
        echo -e "\033[35m ---- running P0case $case_num/${#P0case_list[*]}: ${p0case} \033[0m"
        bash ${nlp_dir}/scripts/regression/ci_case.sh ${p0case} ${cudaid1} ${cudaid2}
        let case_num++
    done
    echo -e "\033[35m ---- end run P0case  \033[0m"
cd ${nlp_dir}
upload(){
if [ -f '/ssd1/paddlenlp/bos/upload.py' ];then
    cp -r /ssd1/paddlenlp/bos/* ./
    tar -zcvf model_logs.tar model_logs/
    mkdir upload && mv model_logs.tar upload
    python upload.py upload 'paddle-qa/paddlenlp'
else
    echo 'No upload script found'
fi
}
upload
cd model_logs/
FF=`ls *_FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    P0case_EXCODE=1
else
    P0case_EXCODE=0
fi
if [ $P0case_EXCODE -ne 0 ] ; then
    cd model_logs/
    FF=`ls *_FAIL*|wc -l`
    echo -e "\033[31m ---- P0case failed number: ${FF} \033[0m"
    ls *_FAIL*
    exit $P0case_EXCODE
else
    echo -e "\033[32m ---- P0case Success \033[0m"
fi
