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
####################################
set -e
export python=$1
export paddle=$2
export nlp_dir=/workspace/PaddleNLP
mkdir -p /workspace/PaddleNLP/model_logs
export log_path=/workspace/PaddleNLP/model_logs
export P0case_list=()
declare -A all_P0case_dic
declare -A Build_list
target_lists_for_llm=(
    "paddlenlp/transformers"
    "paddlenlp/experimental/transformers/"
    "paddlenlp/data"
    "paddlenlp/datasets"
    "paddlenlp/generation"
    "paddlenlp/peft"
    "paddlenlp/mergekit"
    "paddlenlp/quantization"
    "paddlenlp/trainer"
    "paddlenlp/trl"
    "llm"
    "tests/llm"
    "csrc"
    "scripts/regression"
)
all_P0case_dic=(["msra_ner"]=15 
    ["glue"]=2 
    ["bert"]=2 
    ["skep"]=10 
    ["bigbird"]=2
    ["ernie-1.0"]=2 ["ernie"]=2 
    ["ofa"]=2 
    ["albert"]=2 
    ["lexical_analysis"]=5
    ["transformer"]=5
    ["question_matching"]=5 
    ["ernie-csc"]=5
    ["clue"]=5
    ["taskflow"]=5
    ["ernie-3.0"]=5 
    ["uie"]=5 
    ["ernie-layout"]=5  ["ernie_layout"]=5
    ["ernie_csc"]=5 
    ["segment_parallel_utils"]=5 
    ["ring_flash_attention"]=5
    ["llm"]=5)
####################################

python -m pip config --user set global.index http://pip.baidu-int.com/search/
python -m pip config --user set global.index-url http://pip.baidu-int.com/simple
python -m pip config --user set global.trusted-host pip.baidu-int.com
# Install paddlepaddle-gpu
install_paddle(){
    echo -e "\033[35m ---- Install paddlepaddle-gpu  \033[0m"
    python -m pip install --user -r scripts/regression/requirements_ci.txt
    python -m pip uninstall paddlepaddle -y
    python -m pip install --user ${paddle} --no-cache-dir;
    python -c "import paddle;print('paddle');print(paddle.__version__);print(paddle.version.show())" >> ${log_path}/commit_info.txt
    python -c 'from visualdl import LogWriter'
}
# Install paddlenlp func
nlp_build (){
    cd $1
    rm -rf build/
    rm -rf paddlenlp.egg-info/
    rm -rf ppdiffusers.egg-info/
    rm -rf paddle_pipelines.egg-info/
    rm -rf dist/

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python setup.py bdist_wheel
    python -m pip install --ignore-installed  dist/p****.whl
}
install_external_ops(){
    echo -e "\033[31m ---- Install extern_ops  \033"
    export PYTHONPATH=${nlp_dir}:$PYTHONPATH
    cd ${nlp_dir}/slm/model_zoo/gpt-3/external_ops
    python setup.py install
    python -c "import fused_ln;";
    cd ${nlp_dir}
}
####################################
# get diff case
cd ${nlp_dir}
get_diff_TO_case(){
for file_name in `git diff --numstat ${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    dir3=${arr_file_name[2]}
    dir4=${arr_file_name[3]}
    file_item=$dir1/$dir2/$dir3/$dir4
    ext="${file_name##*.}"
    echo "file_name:"${file_name}, "dir1:"${dir1}, "dir2:"${dir2},"dir3:"${dir3},".xx:" ${file_name##*.}
    echo "ext: ${file_name##*.}"
    if [ ! -f ${file_name} ];then # 针对pr删掉文件
        continue
    elif [[ "$ext" == "md" || "$ext" == "rst" || "$file_name" == docs/* ]]; then
        continue
    elif [[ "${AGILE_COMPILE_BRANCH}" == "refactor-training-loop" ]];then # 针对特定分支
        P0case_list[${#P0case_list[*]}]=gpt
    else
         # 判断是否命中 target_lists_for_llm 列表-执行llm
        for ((i=0; i<${#target_lists_for_llm[@]}; i++)); do 
            if [[ "${file_item}" == *"${target_lists_for_llm[i]}"* ]];then
                P0case_list[${#P0case_list[*]}]=llm
            fi
        done
        # 其他 case 判断
        if [[ ${dir1} =~ "scripts" ]];then # API 升级
            if [[ ${dir2} =~ "should_deploy" ]];then # 针对发版mini test
                P0case_list[${#P0case_list[*]}]=transformer
            fi  
        elif [[ ${dir1} =~ "paddlenlp" ]];then # API 升级
            Build_list[${dir1}]="paddlenlp" # 影响编包
            if [[ ${dir2} =~ "__init__" ]];then # 针对发版mini test
                P0case_list[${#P0case_list[*]}]=bert
            elif [[ -n "${all_P0case_dic[$dir2]}" ]]; then
                P0case_list[${#P0case_list[*]}]=${dir2}
            elif [[ ${dir2} =~ "transformers" ]];then
                if [[ -n "${all_P0case_dic[$dir3]}" ]];then
                    P0case_list[${#P0case_list[*]}]=${dir3}
                fi
            elif [[ ${dir2} =~ "taskflow" ]];then # ce case
                P0case_list[${#P0case_list[*]}]=taskflow
            fi
        elif [[ "${dir1}" =~ "slm" && "${dir2}" =~ "examples" ]];then # 模型升级
            if [[ -n "${all_P0case_dic[$dir2]}" ]];then
                P0case_list[${#P0case_list[*]}]=${dir2}
            elif [[ -n "${all_P0case_dic[$dir3]}" ]];then
                P0case_list[${#P0case_list[*]}]=${dir3}
            fi
        elif [[ "${dir1}" =~ "slm" && "${dir2}" =~ "model_zoo" ]];then # 模型升级
            if [[ -n "${all_P0case_dic[$dir2]}" ]];then
                P0case_list[${#P0case_list[*]}]=${dir2}
            fi
        elif [[ ${dir1} =~ "csrc" ]];then # 推理改动
            Build_list[${dir1}]="paddlenlp_ops" # 影响推理编包
        elif [[ ${dir1} =~ "requirements" ]];then # 依赖改动
            Build_list[${dir1}]="paddlenlp" # 影响paddlenlp编包
        else
            continue
        fi
    fi
done
}
get_diff_TO_case
P0case_list=($(awk -v RS=' ' '!a[$1]++' <<< ${P0case_list[*]}))
####################################
# build latest paddlenlp/paddlenlp_ops whl and install
if [[ ${#Build_list[*]} -ne 0 ]];then
    install_paddle
    echo -e "\033[32m start build ${Build_list[*]} whl \033[0m"
    for build_pkg in ${Build_list[*]};do
        if [[ ${build_pkg} == "paddlenlp" ]];then
            echo -e "\033[35m ---- build ${GIT_PR_ID} paddlenlp  \033[0m"
            nlp_build ${nlp_dir}
        elif [[ ${build_pkg} == "paddlenlp_ops" ]];then
            echo -e "\033[35m ---- build ${GIT_PR_ID} paddlenlp_ops  \033[0m"
            export http_proxy=${proxy} && export https_proxy=${proxy}
            cd ${nlp_dir}/csrc
            bash tools/build_wheel.sh
            unset http_proxy && unset https_proxy
        else
            echo -e "\033[35m ---- build ${GIT_PR_ID} ${build_pkg}  \033[0m"
        fi  
    done
else
   echo -e "\033[32m Don't need build whl  \033[0m"
fi
###################################
if [[ ${#P0case_list[*]} -ne 0 ]];then
    cd ${nlp_dir}
    # Install paddle
    if [[ ${#Build_list[*]} -eq 0 ]];then
        install_paddle
    else
        echo "install_paddle done"
    fi
    # Install paddlenlp
    if [ ! -f ./dist/p****.whl ];then
        echo "install_nlp_develop"
        python -m pip install --user https://paddlenlp.bj.bcebos.com/wheels/paddlenlp-ci-py3-none-any.whl --no-cache-dir
    else
        echo "install_nlp_pr done"
    fi
    # install paddlenlp_ops
    if [ ! -f ./csrc/gpu_dist/p****.whl ];then
        echo "install_paddlenlp_ops_develop"
        python -m pip install --user https://paddlenlp.bj.bcebos.com/wheels/paddlenlp_ops-ci-py3-none-any.whl --no-cache-dir
    else
        echo "install_paddlenlp_ops_pr done"
    fi
    # install fused_ln
    install_external_ops
    python -c "from paddlenlp import __version__; print('paddlenlp version:', __version__)" >> ${log_path}/commit_info.txt
    python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)" >> ${log_path}/commit_info.txt
    python -m pip list >> ${log_path}/commit_info.txt

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
    ####################################
    cd ${nlp_dir}
    echo -e "\033[35m ---- Generate Allure Report  \033[0m"
    unset http_proxy && unset https_proxy
    cp scripts/regression/gen_allure_report.py ./
    python gen_allure_report.py > /dev/null
    echo -e "\033[35m ---- Report: https://xly.bce.baidu.com/ipipe/ipipe-report/report/${AGILE_JOB_BUILD_ID}/report/  \033[0m"
    ####################################
    # run coverage
    # cd ${nlp_dir}/tests/
    # bash run_coverage.sh
    # Coverage_EXCODE=$? || true
    # mv ./htmlcov ${nlp_dir}/coverage_logs/
    # if [ $Coverage_EXCODE -ne 0 ] ; then
    #     echo -e "\033[31m ---- Coverage Failed \033[0m"
    # else
    #     echo -e "\033[32m ---- Coverage Success \033[0m"
    # fi
    ####################################
else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    EXCODE=0
fi
exit $EXCODE
