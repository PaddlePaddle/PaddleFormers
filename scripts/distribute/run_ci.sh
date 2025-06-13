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
export paddle=$1
export nlp_dir=/workspace/PaddleNLP
mkdir -p /workspace/case_logs
export log_path=/workspace/case_logs
export case_list=()

global_total_count=0
global_success_count=0
global_exit_250_arr=()
global_runtime_fail_arr=()
global_verification_fail_arr=()

target_lists_for_gpt=(
    "slm/model_zoo/gpt-3"
    "llm/auto_parallel/gpt-3"
    "paddlenlp/transformers/gpt"
    "scripts/distribute"
)

target_lists_for_llama=(
    "llm/auto_parallel/llama"
    "paddlenlp/trainer/auto_trainer.py"
    "paddlenlp/transformers/llama"
    "scripts/distribute"
)

target_lists_for_deepseek=(
    "llm/auto_parallel/deepseek-v3"
    "paddlenlp/trainer/auto_trainer.py"
    "paddlenlp/transformers/deepseek_v2/modeling_auto.py"
    "paddlenlp/transformers/deepseek_v2/modeling.py"
    "paddlenlp/transformers/deepseek_v3/modeling_auto.py"
    "paddlenlp/transformers/moe_layer_auto.py"
    "paddlenlp/transformers/moe_gate_auto.py"
    "scripts/distribute"
)

target_path_for_ci_scripts="scripts/distribute"

####################################
install_paddle(){
    echo -e "\033[31m ---- Install paddlepaddle-gpu  \033"
    python -m pip install --no-cache-dir --user ${paddle} --force-reinstall --no-dependencies;
    python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
}

install_paddlenlp(){
    echo -e "\033[31m ---- Install paddlenlp by set PYTHONPATH  \033"
    export PYTHONPATH=${nlp_dir}:$PYTHONPATH
    # python -m pip install -r ${nlp_dir}/requirements.txt
    sed -i -e "s/paddlenlp/#paddlenlp/g" model_zoo/gpt-3/requirements.txt
    # export http_proxy=${proxy} && export https_proxy=${proxy}
    # python -m pip uninstall paddlenlp -y
    # rm -rf build/ && rm -rf paddlenlp.egg-info/ && rm -rf dist/
    # python -m pip install --ignore-installed -r requirements.txt
    # python -m pip install --ignore-installed -r requirements-dev.txt
    # python setup.py install
    # python setup.py build_ext
    # python setup.py bdist_wheel
    # unset http_proxy && unset https_proxy
    # cd -
    # python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)";
}

install_external_ops(){
    echo -e "\033[31m ---- Install extern_ops  \033"
    export PYTHONPATH=${nlp_dir}:$PYTHONPATH
    cd ${nlp_dir}/slm/model_zoo/gpt-3/external_ops
    python setup.py install
    python -c "import fused_ln;";
}

function is_a100() {
    if [ $(nvidia-smi|grep A100|wc -l)  -ne 0 ];then
        echo 1 # A100
    else
        echo 0 # not A100
    fi
}

IS_A100=$(is_a100) 

####################################
get_diff_TO_case(){
    cd ${nlp_dir}
    if [ $IS_A100 -ne 0 ];then
        for file_name in `git diff --numstat upstream/${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
            arr_file_name=(${file_name//// })
            dir1=${arr_file_name[0]}
            dir2=${arr_file_name[1]}
            dir3=${arr_file_name[2]}
            dir4=${arr_file_name[3]}
            file_item=$dir1/$dir2/$dir3/$dir4
            echo "file_name:"${file_name}, "path:"${file_item}
            if [ ! -f ${file_name} ];then # 针对pr删掉文件
                continue
            elif [[ ${file_name##*.} == "md" ]] || [[ ${file_name##*.} == "rst" ]] || [[ ${dir1} == "docs" ]];then
                continue
            else
                for ((i=0; i<${#target_lists_for_gpt[@]}; i++)); do
                    if [[ ! ${dir3} =~ "benchmarks" ]] && [[ ${file_item} == *${target_lists_for_gpt[i]}* ]];then
                        case_list[${#case_list[*]}]=gpt-3_auto
                        case_list[${#case_list[*]}]=gpt-3_dygraph
                    fi
                done
                for ((i=0; i<${#target_lists_for_llama[@]}; i++)); do
                    if [[ ${file_item} == *${target_lists_for_llama[i]}* ]];then
                        case_list[${#case_list[*]}]=llama_auto
                    fi
                done
                for ((i=0; i<${#target_lists_for_deepseek[@]}; i++)); do
                    if [[ ${file_item} == *${target_lists_for_deepseek[i]}* ]];then
                        case_list[${#case_list[*]}]=deepseek_auto
                    fi
                done
            fi
        done
    else
        for file_name in `git diff --numstat upstream/${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
            arr_file_name=(${file_name//// })
            dir1=${arr_file_name[0]}
            dir2=${arr_file_name[1]}
            dir3=${arr_file_name[2]}
            dir4=${arr_file_name[3]}
            file_item=$dir1/$dir2/$dir3/$dir4
            echo "file_name:"${file_name}, "path:"${file_item}
            if [ ! -f ${file_name} ];then # 针对pr删掉文件
                continue
            elif [[ ${file_name##*.} == "md" ]] || [[ ${file_name##*.} == "rst" ]] || [[ ${dir1} == "docs" ]];then
                continue
            else
                case_list[${#case_list[*]}]=gpt-3_auto
                case_list[${#case_list[*]}]=llama_auto
                case_list[${#case_list[*]}]=deepseek_auto
                for ((i=0; i<${#target_lists_for_gpt[@]}; i++)); do
                    if [[ ! ${dir3} =~ "benchmarks" ]] && [[ ${file_item} == *${target_lists_for_gpt[i]}* ]];then
                        case_list[${#case_list[*]}]=gpt-3_dygraph
                    fi
                done
            fi
        done
    fi
}
####################################
function contain_case(){
    local e
    for e in "${@:2}";do
        if [[ "$e" == "$1" ]];then
            return 1
        fi
    done
    return 0
}
####################################
function execute_func_list(){
    cd ${log_path} || { echo "Failed to enter log_path: $log_path"; return 1; } 
    total_count=0
    success_count=0
    runtime_fail_count=0
    verification_fail_count=0
    exit_250_count=0
    while IFS= read -r func_name; do
        let total_count++
        let global_total_count++
        execute_num=1
        while true; do
            timeout 10m bash $1 exec_case $func_name $FLAGS_install_deps $FLAGS_download_data  
            result=$?
            if [ $result -eq 0 ]; then
                echo -e "\033[32m test success!"
                let success_count++
                let global_success_count++
            elif [ $result -eq 1 ]; then
                if [ $execute_num -eq 1 ]; then
                    echo -e "\033[31m first time execute failed, try again!"
                    let execute_num++
                    continue
                else
                    echo -e "\033[31m second time execute failed, exit!"
                    mv ${log_path}/$func_name ${log_path}/${func_name}_FAIL.log
                    echo -e "\033[31m ${log_path}/$func_name_FAIL \033"
                    tail -15 ${log_path}/${func_name}_FAIL.log
                    let runtime_fail_count++ 
                    global_runtime_fail_arr+=("$func_name") 
                fi
            elif [ $result -eq 2 ]; then
                echo -e "\033[31m verification failed!"
                let verification_fail_count++
                global_verification_fail_arr+=("$func_name")
            elif [ $result -eq 250 ]; then
                if [ $execute_num -eq 1 ]; then
                    echo -e "\033[31m first time execute failed, try again!"
                    let execute_num++
                    continue
                else
                    echo -e "\033[31m second time execute failed, exit!"
                    mv ${log_path}/$func_name ${log_path}/${func_name}_FAIL.log
                    echo -e "\033[31m ${log_path}/$func_name_FAIL \033"
                    tail -15 ${log_path}/${func_name}_FAIL.log
                    let exit_250_count++
                    global_exit_250_arr+=("$func_name")
                fi
            elif [ $result -eq 124 ]; then
                echo "\033[31m [failed-timeout] Test case execution was terminated after exceeding the 10m limit."
                mv ${log_path}/$func_name ${log_path}/${func_name}_FAIL.log
                echo -e "\033[31m ${log_path}/$func_name_FAIL \033"
                tail -15 ${log_path}/${func_name}_FAIL.log
                let runtime_fail_count++ 
                global_runtime_fail_arr+=("$func_name") 
            else
                echo "test failed!"
                mv ${log_path}/$func_name ${log_path}/${func_name}_FAIL.log
                echo -e "\033[31m ${log_path}/$func_name_FAIL \033"
                tail -15 ${log_path}/${func_name}_FAIL.log
                let runtime_fail_count++ 
                global_runtime_fail_arr+=("$func_name") 
            fi
            break
        done
    done < functions.txt
    echo -e "\033[31m $2 test case has complicated \033"
    echo -e "\033[31m $(printf '\t')  total tests :  $total_count \033"
    echo -e "\033[31m $(printf '\t')  success tests :  $success_count \033"
    echo -e "\033[31m $(printf '\t')  runtime fail tests :  $runtime_fail_count \033"
    echo -e "\033[31m $(printf '\t')  verification fail tests :  $verification_fail_count \033"
    echo -e "\033[31m $(printf '\t')  exit 250 tests(intermittent issue) :  $exit_250_count \033"
}

function clean_file(){
    target_path=$1
    matching_data_dirs=$(find "$target_path" -maxdepth 1 -type d -name "*data*")
    if [ -n "$matching_data_dirs" ]; then
        echo "cleaning data dirs:"
        echo $matching_data_dirs        
        for dir in $matching_data_dirs; do
            rm -rf "$dir"
            echo "deleted $dir"
        done
    else
        echo "$target_path no data dirs found"
    fi

    matching_output_dirs=$(find "$target_path" -maxdepth 1 -type d -name "*output*")
    if [ -n "$matching_output_dirs" ]; then
        echo "cleaning output dirs:"
        echo $matching_output_dirs
        for dir in $matching_output_dirs; do
            rm -rf "$dir"
            echo "deleted $dir"
        done
    else
        echo "$target_path no output dirs found"
    fi
}

####################################
get_diff_TO_case # 获取待执行case列表
case_list=($(awk -v RS=' ' '!a[$1]++' <<< ${case_list[*]}))  # 去重并将结果存储回原列表
if [[ ${#case_list[*]} -ne 0 ]];then
    echo -e "\033[31m =======CI Check case========= \033"
    echo -e "\033[31m ---- case_list length: ${#case_list[*]}, cases: ${case_list[*]} \033"
    echo -e "\033[31m ============================= \033"
    set +e

    # Install paddle
    install_paddle
    # Install paddlenlp
    install_paddlenlp
    # Install external_ops
    install_external_ops   
    case_num=1
    export FLAGS_install_deps=0
    export FLAGS_download_data=""
    if [[ $(contain_case llama_auto ${case_list[@]}; echo $?) -eq 1 ]];then
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: llama_auto \033"
        cmd=/workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh
        bash  $cmd prepare_case llama_case_list_auto $FLAGS_install_deps $FLAGS_download_data
        execute_func_list $cmd llama_auto
        export FLAGS_download_data="llama ""$FLAGS_download_data"
        let case_num++
        clean_file $nlp_dir/llm/auto_parallel/llama
    fi
    if [[ $(contain_case gpt-3_auto ${case_list[@]}; echo $?) -eq 1 ]];then
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: gpt-3_auto \033"
        cmd=/workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh 
        bash $cmd prepare_case llm_gpt_case_list_auto $FLAGS_install_deps $FLAGS_download_data
        execute_func_list $cmd gpt-3_auto
        export FLAGS_install_deps=1
        export FLAGS_download_data="gpt ""$FLAGS_download_data"
        let case_num++        
        clean_file $nlp_dir/llm/auto_parallel/gpt-3
    fi
    if [[ $(contain_case deepseek_auto ${case_list[@]}; echo $?) -eq 1 ]];then
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: deepseek_auto \033"
        cmd=/workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh 
        bash $cmd prepare_case deepseek_case_list_auto $FLAGS_install_deps $FLAGS_download_data
        execute_func_list $cmd deepseek_auto
        export FLAGS_install_deps=1
        export FLAGS_download_data="deepseek ""$FLAGS_download_data"
        let case_num++        
        clean_file $nlp_dir/llm/auto_parallel/deepseek-v3
    fi
    
    if [[ $(contain_case gpt-3_dygraph ${case_list[@]}; echo $?) -eq 1 ]];then
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: gpt-3_dygraph \033"
        cmd=/workspace/PaddleNLP/scripts/distribute/ci_case_dy.sh
        bash $cmd prepare_case gpt_case_list_dygraph $FLAGS_install_deps $FLAGS_download_data
        execute_func_list $cmd gpt-3_dygraph
        export FLAGS_install_deps=1
        export FLAGS_download_data="gpt ""$FLAGS_download_data"
        let case_num++
        clean_file $nlp_dir/slm/model_zoo/gpt-3
    fi
    echo -e "\033[31m ---- end run case  \033"

    echo -e "\033[31m ---- total tests :  $global_total_count \033"
    if [ ${#global_exit_250_arr[@]} -ne 0 ]; then
        echo -e "\033[32m ---- exit 250 test  :  ${#global_exit_250_arr[@]} \033"
        for case in "${global_exit_250_arr[@]}"; do
            echo -e "\t$case(exit 250)"
        done
    fi

    if [ ${#global_runtime_fail_arr[@]} -eq 0 ] && [ ${#global_verification_fail_arr[@]} -eq 0 ]; then
        echo -e "\033[32m ---- all cases Success  \033"
        EXCODE=0
    else 
        echo -e "\033[32m ---- runtime failed test  :  ${#global_runtime_fail_arr[@]} \033"
        for case in "${global_runtime_fail_arr[@]}"; do
            echo -e "\t$case(failed)"
        done
        echo -e "\033[32m ---- verification failed test  :  ${#global_verification_fail_arr[@]} \033"
        for case in "${global_verification_fail_arr[@]}"; do
            echo -e "\t$case(failed)"
        done
        EXCODE=1
    fi
else
    echo -e "\033[32m Changed Not CI case, Skips \033"
    EXCODE=0
fi
exit $EXCODE
