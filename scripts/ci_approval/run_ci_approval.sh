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

failed_num=0
echo_list=()
approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/PaddleNLP/pulls/${GIT_PR_ID}/reviews?per_page=10000`

function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}

function check_approval(){
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    echo ${person_num}
    APPROVALS=`echo ${approval_line}|python check_pr_approval.py $1 $person_num`
    echo ${APPROVALS}
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}

for file_name in `git diff --numstat upstream/${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    dir3=${arr_file_name[2]}
    dir4=${arr_file_name[3]}
    echo "file_name:"${file_name}, "dir1:"${dir1}, "dir2:"${dir2},"dir3:"${dir3},".xx:" ${file_name##*.}
    if [[ ${file_name} =~ "paddlenlp/trainer/training_args.py" ]] || [[ ${file_name} =~ "paddlenlp/trainer/trainer.py" ]] || [[ ${file_name} =~ "llm/run_pretrain.py" ]] || [[ ${file_name} =~ "llm/run_finetune.py" ]];then
        echo_line="You must have two RD: one from(ZHUI, wawltor),one from(ForFishes,sneaxiy,zhiqiu) approval for the changes of training_args.py/trainer.py/run_pretrain.py "
        check_approval 2 ZHUI wawltor ForFishes sneaxiy zhiqiu
    elif [[ ${dir1} =~ "paddlenlp" ]];then
        if [[ ${dir2} =~ "transformers" ]];then
            echo_line="You must have one RD (wawltor) approval for the changes of transformers "
            check_approval 1 wawltor 
        elif [[ ${dir2} =~ "taskflow" ]];then
            echo_line="You must have one RD (w5688414(Recommend),DesmonDay,wawltor) approval for the changes of taskflow"
            check_approval 1 w5688414 DesmonDay wawltor 
        elif [[ ${dir3} =~ "trainer" ]];then
            echo_line="You must have one RD (ZHUI(Recommend), wawltor) approval for the changes of trainer"
            check_approval 1 ZHUI wawltor 
        elif [[ ${dir3} =~ "fast_transformer" ]] || [[ ${dir4} =~ "FasterTransformer" ]] ;then
            echo_line="You must have one RD (guoshengCS(Recommend), wawltor) approval for the changes of FT or FG"
            check_approval 1 guoshengCS wawltor 
        elif [[ ${dir3} =~ "llama" ]] || [[ ${dir3} =~ "gpt" ]];then
            echo_line="You must have two RD: one from(ZHUI, wawltor),one from(ForFishes,sneaxiy,zhiqiu) approval for the changes of llm/llama/auto_parallel/ "
            check_approval 2 ZHUI wawltor ForFishes sneaxiy zhiqiu
        fi
    elif [[ ${dir1} =~ "model_zoo" ]];then # 
        if [[ ${dir2} =~ "gpt-3" ]] ;then
            echo_line="You must have two RD: one from(ZHUI, wawltor),one from(ForFishes,sneaxiy,zhiqiu) approval for the changes of model_zoo/gpt-3 "
            check_approval 2 ZHUI wawltor ForFishes sneaxiy zhiqiu
        fi
    elif [[ ${dir1} =~ "llm" ]];then 
        if [[ ${dir3} =~ "auto_parallel" ]] ;then
            echo_line="You must have two RD: one from(ZHUI, wawltor),one from(ForFishes,sneaxiy,zhiqiu) approval for the changes of llm/llama/auto_parallel/ "
            check_approval 2 ZHUI wawltor ForFishes sneaxiy zhiqiu
        else 
            echo_line="You must have one RD (wj-Mcat(Recommend), wawltor) approval for the changes of llm"
            check_approval 1 wj-Mcat lugimzzz DesmonDay wawltor
        fi
    elif [[ ${dir1} =~ "tests" ]];then 
        if [[ ${dir2} =~ "transformers" ]] ;then
            echo_line="You must have one RD (wawltor) approval for the changes of transformers "
            check_approval 1 wawltor
        elif [[ ${dir2} =~ "taskflow" ]] || [[ ${dir2} =~ "prompt" ]];then
            echo_line="You must have one RD (w5688414(Recommend),, wawltor) approval for the changes of taskflow"
            check_approval 1 w5688414 wawltor 
        elif [[ ${dir2} =~ "llm" ]] ;then
            echo_line="You must have one RD (wj-Mcat(Recommend), wawltor) approval for the changes of tests/llm/"
            check_approval 1 wj-Mcat lugimzzz gongel wtmlon wawltor 
        elif [[ ${dir2} =~ "trainer" ]] ;then
            echo_line="You must have one RD (ZHUI(Recommend), wawltor) approval for the changes of trainer"
            check_approval 1 ZHUI wawltor 
        elif [[ ${dir2} =~ "ops" ]] || [[ ${dir2} =~ "embedding" ]];then
            echo_line="You must have one RD (guoshengCS(Recommend),, wawltor) approval for the changes of ops or embedding"
            check_approval 1 guoshengCS wawltor  
        elif [[ ${dir2} =~ "model_zoo" ]] ;then
            echo_line="You must have one RD (wawltor(Recommend)) approval for the changes of model_zoo"
            check_approval 1 wawltor 
        elif [[ ${dir2} =~ "cli" ]] || [[ ${dir2} =~ "generation" ]];then
            echo_line="You must have one RD (wj-Mcat(Recommend), wawltor) approval for the changes of llm/generation"
            check_approval 1 wj-Mcat wawltor 
        elif [[ ${dir2} =~ "test_tipc" ]];then
            echo_line="You must have one RD (wj-Mcat(Recommend),lugimzzz, wawltor) approval for the changes of test_tipc"
            check_approval 1 wj-Mcat lugimzzz wawltor  
        elif [[ ${dir2} =~ "dataaug" ]]|| [[ ${dir2} =~ "peft" ]];then
            echo_line="You must have one RD (lugimzzz(Recommend), wawltor) approval for the changes of dataaug/peft"
            check_approval 1 lugimzzz wawltor 
        elif [[ ${dir2} =~ "data" ]]|| [[ ${dir2} =~ "dataset" ]];then
            echo_line="You must have one RD (KB-Ding(Recommend), wawltor) approval for the changes of data/dataset"
            check_approval 1 KB-Ding wawltor  
        elif [[ ${dir2} =~ "layers" ]] || [[ ${dir2} =~ "metrics" ]] || [[ ${file_name} =~ "tests/llm/test_pretrain.py" ]];then
            echo_line="You must have one RD (ZHUI(Recommend),DesmonDay,wawltor) approval for the changes of layers/metrics/tests/llm/test_pretrain.py"
            check_approval 1 ZHUI DesmonDay wawltor 
        elif [[ ${dir2} =~ "utils" ]] ;then
            echo_line="You must have one RD (wawltor(Recommend)) approval for the changes of tests/utils"
            check_approval 1 wawltor
        fi
    elif [[ ${dir1} =~ "pipelines" ]];then 
        echo_line="You must have one RD (w5688414(Recommend), wawltor) approval for the changes of pipelines"
        check_approval 1 w5688414 junnyu wawltor 
    elif [[ ${dir1} =~ "ppdiffusers" ]];then 
        echo_line="You must have one RD (junnyu(Recommend), wawltor) approval for the changes of pipelines"
        check_approval 1 w5688414 junnyu wawltor 
    else
        continue
    fi
done

if [ -n "${echo_list}" ];then
    echo "**************************************************************"
    echo "Please find RD for approval."
    echo -e "${echo_list[@]}"
    echo "There are ${failed_num} approved errors."
    echo "**************************************************************"
    exit 1
else
    echo "**************************************************************"
    echo "CI APPROVAL PASSED."
    echo "**************************************************************"
    exit 0
fi