#!/usr/bin/env bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

export log_path=/workspace/case_logs
export root_path=/workspace/PaddleNLP

export gpt_case_path=$root_path/slm/model_zoo/gpt-3
export gpt_data_path=/fleetx_data

export llm_gpt_case_path=$root_path/llm
export llm_gpt_data_path=/llm_gpt_data

unset CUDA_VISIBLE_DEVICES

function track_case_status() {  
    local case_name="$1"  
    local prefix="$2"  
    local original_path  
  
    original_path=$(pwd)  
    cd ${log_path} || { echo "Failed to enter log_path: $log_path"; return 1; }  
  
    total_count=$(ls -1 "$prefix"* 2>/dev/null | grep -Ev 'result\.log|functions\.txt' | wc -l)
    run_fail_count=$(ls -1 "$prefix"*_FAIL* 2>/dev/null | wc -l)  
    loss_fail_count=$(grep 'check failed! ' result.log | awk -v prefix="$prefix" '{if ($2 ~ "^" prefix) print $2}'| wc -l)
    
    echo -e "\033[31m ---- $case_name total tests :  $total_count \033"
    if [ $run_fail_count -eq 0 ] && [ $loss_fail_count  -eq 0 ]; then
        echo -e "\033[32m ---- all cases Success  \033"
    else
        if [[ $run_fail_count -ne 0 ]] ; then
            echo -e "\033[31m ---- $case_name runtime failed test  :  $run_fail_count \033"
            ls -1 "$prefix"*_FAIL* 2>/dev/null | awk -v OFS="\t" '{print "\t" $0 "(failed)"}'
        fi
        if [[ $loss_fail_count -ne 0 ]] ; then
            echo -e "\033[31m ---- $case_name verification failed test  :  $loss_fail_count \033"
            grep 'check failed! ' result.log | awk -v prefix="$prefix" 'BEGIN {OFS="\t"} {if ($2 ~ "^" prefix) print "\t" $2 "(failed)"}'
        fi
        return 2
    fi
    cd "$original_path" || { echo "Failed to return to original path: $original_path"; return 1; }  
    return 0
}

function restore_func() {
    fun_list=$1
    cd ${log_path} || { echo "Failed to enter log_path: $log_path"; return 1; } 
    if [ -e "functions.txt" ]; then
        rm "functions.txt"
        echo "Deleted existing functions.txt"
    fi
    if [ ! -f "${log_path}/blacklist.csv" ]; then
        wget -q -P ${log_path}/ https://paddle-qa.bj.bcebos.com/Auto-Parallel/blacklist.csv --no-proxy || exit 101
        echo "\033 ---- wget blacklist.csv \033"
    fi
    blacklist_file=${log_path}/blacklist.csv
    mapfile -t blacklist < "$blacklist_file"
    for function in ${fun_list[@]};do
        if [[ " ${blacklist[@]} " == *" ${function} "* ]]; then
            echo "\033 ---- Function '$function' is blacklisted and will be skipped. \033"
        else
            echo "$function" >> functions.txt
        fi
    done
}

function gpt_case_list_dygraph() {
    fun_list=(
        # The test name must have "gpt_" as a prefix, which will 
        # be used for tracking the execution status of the case.
        gpt_preprocess_data
        gpt_345M_single
        gpt_1.3B_dp
        gpt_6.7B_stage2_dp2_sharding4
        gpt_6.7B_stage3_dp2_sharding4
        gpt_6.7B_stage2_sharding8
        gpt_175B_DP1_MP4_PP2
        gpt_175B_DP1_MP4_PP2_sp
        gpt_175B_DP1_MP8_PP1
        gpt_175B_DP1_MP8_PP1_sp
        gpt_175B_DP1_MP1_PP8
        gpt_generation_345M_single
        gpt_generation_345M_hybrid
        gpt_345M_mp8_qat
        # gpt_export_345M_mp1
        # gpt_export_345M_mp2
        # gpt_export_qat_345M
        # gpt_inference_345M_single
        # gpt_inference_345M_dp8
        gpt_345M_single_finetune
        gpt_eval_WikiText
        gpt_eval_LAMBADA
    )
    if [ $1 = "prepare_case" ]; then
        restore_func $fun_list  
    elif [ $1 = "exec_case" ]; then
        for fun in "${fun_list[@]}"; do
            eval "$fun"
        done
        track_case_status $FUNCNAME "gpt_"
    else 
        echo -e "\033[31m ---- Invalid status $1 \033[0m"
        return 1
    fi
}

function llm_gpt_case_list_dygraph() {
    fun_list=(
        # The test name must have "llm_gpt_" as a prefix, which will 
        # be used for tracking the execution status of the case.
        llm_gpt_recompute_bs32_bf16_MP2-SD4-stage1
    )
    if [ $1 = "prepare_case" ]; then
        restore_func $fun_list  
    elif [ $1 = "exec_case" ]; then
        for fun in "${fun_list[@]}"; do
            eval "$fun"
        done
        track_case_status $FUNCNAME "llm_gpt_"
    else 
        echo -e "\033[31m ---- Invalid status $1 \033[0m"
        return 1
    fi
}

############ case start ############
function gpt_preprocess_data() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python ppfleetx/data/data_tools/gpt/raw_trans_to_json.py  \
        --input_path ./dataset/wikitext_103_en \
        --output_path ./dataset/wikitext_103_en/wikitext_103_en \
        >>${log_path}/$FUNCNAME 2>&1
    python ppfleetx/data/data_tools/gpt/preprocess_data.py \
        --model_name gpt2 \
        --tokenizer_name GPTTokenizer \
        --data_format JSON \
        --input_path ./dataset/wikitext_103_en/wikitext_103_en.jsonl \
        --append_eos \
        --output_prefix ./dataset/wikitext_103_en/wikitext_103_en  \
        --workers 40 \
        --log_interval 1000 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_345M_single() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python tools/train.py \
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_1.3B_dp() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_6.7B_stage2_dp2_sharding4() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Distributed.sharding.sharding_degree=4 -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.sharding.reduce_overlap=False -o Distributed.sharding.broadcast_overlap=False \
        -o Engine.logging_freq=5 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_6.7B_stage3_dp2_sharding4() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Distributed.sharding.sharding_degree=4 -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.sharding.reduce_overlap=False -o Distributed.sharding.broadcast_overlap=False \
        -o Engine.logging_freq=5 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_6.7B_stage2_sharding8() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=20 -o Engine.eval_freq=20 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Distributed.sharding.sharding_degree=8 -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.sharding.reduce_overlap=True -o Distributed.sharding.broadcast_overlap=True \
        -o Engine.logging_freq=5 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP4_PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=4 -o Distributed.pp_degree=2 \
        -o Model.sequence_parallel=False \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP4_PP2_sp() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=4 -o Distributed.pp_degree=2 -o Model.sequence_parallel=True \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP8_PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=16 -o Model.num_attention_heads=16 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=8 -o Distributed.pp_degree=1 \
        -o Model.sequence_parallel=False \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP8_PP1_sp() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=16 -o Model.num_attention_heads=16 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=8 -o Distributed.pp_degree=1 -o Model.sequence_parallel=True \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP1_PP8() {
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=32 -o Model.num_attention_heads=16 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=1 \
        -o Distributed.mp_degree=1 -o Distributed.pp_degree=8 \
        -o Model.virtual_pp_degree=2 -o Distributed.pp_recompute_interval=2 \
        -o Model.fused_linear=True -o Model.use_recompute=True \
        -o Model.sequence_parallel=False \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_345M_mp8_qat() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/qat_gpt_345M_mp8.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=8 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_generation_345M_single() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python tasks/gpt/generation.py \
        -c ppfleetx/configs/nlp/gpt/generation_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/ \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_generation_345M_hybrid() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0" tasks/gpt/generation.py \
        -c ppfleetx/configs/nlp/gpt/generation_gpt_345M_dp8.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/ \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_export_345M_mp1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=log_export
    rm -rf $log_dir
    rm -rf output

    export PYTHONPATH=$root_path/slm/model_zoo/gpt-3:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=1
    python -m paddle.distributed.launch --log_dir $log_dir --devices "1" \
        ./tools/auto_export.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/generation_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./pretrained/inference_model \
        >>${log_path}/$FUNCNAME 2>&1
    python -m paddle.distributed.launch --devices "1" \
        projects/gpt/inference.py --mp_degree 1 --model_dir output \
        >>${log_path}/$FUNCNAME 2>&1
    unset CUDA_VISIBLE_DEVICES
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_export_345M_mp2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=log_export
    rm -rf $log_dir
    rm -rf output

    export PYTHONPATH=$root_path/slm/model_zoo/gpt-3:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=0,1
    python -m paddle.distributed.launch --devices "0,1" \
        ./tools/auto_export.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/generation_gpt_345M_mp2.yaml \
        -o Generation.use_topp_sampling=False \
        -o Engine.save_load.ckpt_dir=./pretrained/inference_model \
        >>${log_path}/$FUNCNAME 2>&1
    python -m paddle.distributed.launch --devices "0,1" \
        projects/gpt/inference.py --mp_degree 2 --model_dir output \
        >>${log_path}/$FUNCNAME 2>&1
    unset CUDA_VISIBLE_DEVICES
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_export_qat_345M() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=log_export
    rm -rf $log_dir
    rm -rf output

    python ./tools/export.py \
        -c ./ppfleetx/configs/nlp/gpt/generation_qat_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0.0 \
        -o Model.attention_probs_dropout_prob=0.0 \
        -o Engine.save_load.ckpt_dir='./GPT_345M_QAT_wo_analysis/' \
        >>${log_path}/$FUNCNAME 2>&1
    python -m paddle.distributed.launch --devices "0" \
        projects/gpt/inference.py --mp_degree 1 --model_dir output \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_inference_345M_single() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    rm -rf output
    python tools/export.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/ \
        >>${log_path}/$FUNCNAME 2>&1
    python tasks/gpt/inference.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_inference_345M_dp8() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    rm -rf output
    python -m paddle.distributed.launch --devices "0" tools/export.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/ \
        >>${log_path}/$FUNCNAME 2>&1
    python -m paddle.distributed.launch --devices "0" \
        tasks/gpt/inference.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_345M_single_finetune() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python ./tools/train.py \
        -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
        -o Engine.num_train_epochs=1 \
        -o Data.Train.dataset.name=WNLI \
        -o Data.Train.dataset.root=./dataset/WNLI/ \
        -o Data.Eval.dataset.name=WNLI \
        -o Data.Eval.dataset.root=./dataset/WNLI/ \
        -o Data.Eval.dataset.split=dev \
        -o Model.num_classes=2 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_eval_WikiText() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python ./tools/eval.py \
        -c ./ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826 \
        -o Offline_Eval.eval_path=./wikitext-103/wiki.valid.tokens \
        -o Offline_Eval.overlapping_eval=32 \
        -o Offline_Eval.batch_size=16 \
        -o Engine.max_steps=20 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_eval_LAMBADA() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python ./tools/eval.py \
        -c ./ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826 \
        -o Offline_Eval.eval_path=./lambada_test.jsonl \
        -o Offline_Eval.cloze_eval=True \
        -o Offline_Eval.batch_size=16 \
        -o Engine.max_steps=20 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_gpt_recompute_bs32_bf16_MP2-SD4-stage1() {
    echo "=========== $FUNCNAME run begin ==========="
    export FLAGS_cudnn_deterministic=1
    export FLAGS_embedding_deterministic=1
    export PYTHONPATH=$root_path/:$PYTHONPATH
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 run_pretrain.py \
        --model_name_or_path gpt2-medium-en \
        --tokenizer_name_or_path gpt2-medium-en \
        --input_dir ./data \
        --output_dir output \
        --sharding stage1 \
        --sharding_parallel_degree 4 \
        --tensor_parallel_degree 2 \
        --split 949,50,1 \
        --max_seq_length 1024 \
        --seed 1234 \
        --fuse_attention_qkv True \
        --use_flash_attention False \
        --bf16 False \
        --fp16 True \
        --fp16_opt_level O2 \
        --amp_master_grad True \
        --learning_rate 0.00001 \
        --min_learning_rate 0.000005 \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --continue_training 0 \
        --dataloader_num_workers 1 \
        --eval_steps 1000 \
        --disable_tqdm True \
        --gradient_accumulation_steps 2 \
        --weight_decay 0.01 \
        --max_steps 30 \
        --save_steps 5000 \
        --device gpu \
        --skip_memory_metrics 0 \
        --warmup_ratio 0.01 \
        --scale_loss 32768 \
        --per_device_train_batch_size 4 \
        --do_train \
        --recompute True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep 'global_step: 30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep 'global_step: 30' | awk -F 'interval_samples_per_second: ' '{print $2}' | awk -F ',' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep 'global_step: 30' | awk -F 'gpu_max_memory_reserved: ' '{print $2}' | awk -F ',' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=8.93362617
    ips_base=64.75564390065037
    mem_base=8904
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}
############ case end ############

function check_result() {
    echo -e "$1" >> ${log_path}/result.log
    if [ $? -ne 0 ];then
        echo -e "\033[31m $1 run failed! \033[0m" | tee -a ${log_path}/result.log
        return 0
    fi

    if [[ ! $1 =~ "llm" ]]; then
        echo -e "\033 $1 run successfully! \033" | tee -a ${log_path}/result.log
    elif [ $# -ne 7 ]; then
        echo -e "\033[31m $1 parameter transfer failed: $@ \033[0m" | tee -a ${log_path}/result.log
        return 0
    else
        diff_loss=$(echo $2 $3|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
        echo -e "loss_base: $2 loss_test: $3 loss_diff: $diff_loss%" | tee -a ${log_path}/result.log
        if [ $2 != $3 ];then
            echo -e "\033[31m $1 loss diff check failed! \033[0m" | tee -a ${log_path}/result.log
            return 0
        fi
        
        diff_ips=$(echo $4 $5|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
        echo -e "ips_base: $4 ips_test: $5 ips_diff: $diff_ips% " | tee -a $log_path/result.log
        v1=$(echo $diff_ips 5.0|awk '{print($1>=$2)?"0":"1"}')
        v2=$(echo $diff_ips -5.0|awk '{print($1<=$2)?"0":"1"}')
        if [[ $v1 == 0 ]];then
            echo -e "$1 IPS increase greater than 5%, not exit " | tee -a $log_path/result.log
        fi
        if [[ $v2 == 0 ]];then
            echo -e "\033[31m $1 IPS diff check failed! \033[0m" | tee -a $log_path/result.log
            return 0
        fi

        diff_mem=$(echo $6 $7|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
        echo -e "mem_base: $6 mem_test: $7 mem_diff: $diff_mem% " | tee -a $log_path/result.log
        w1=$(echo $diff_mem 5.0|awk '{print($1>=$2)?"0":"1"}')
        w2=$(echo $diff_mem -5.0|awk '{print($1<=$2)?"0":"1"}')
        if [[ $w1 == 0 ]];then
            echo -e "\033[31m $1 MEM diff check failed! \033[0m" | tee -a $log_path/result.log
            return 0
        fi
        if [[ $w2 == 0 ]];then
            echo -e "$1 MEM decreases greater than 5%, not exit " | tee -a $log_path/result.log
        fi
    fi
}

function before_hook_for_gpt() {
    echo -e "\033[31m ---- Set FLAGS for GPT dygraph cases  \033[0m"
    cd ${gpt_case_path}
    env | grep FLAGS
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    export no_proxy=bcebos.com
    if [[ $FLAGS_install_deps == 0 ]];then
        echo -e "\033[31m ---- Install requirements for GPT dygraph cases  \033[0m"
        cp requirements.txt requirements_nlp.txt
        sed -i '/paddlenlp/d' requirements.txt
        python -m pip install -r requirements.txt --force-reinstall
        sed -i '/paddlenlp/!d' requirements_nlp.txt
        python -m pip install -r requirements_nlp.txt
        python -m pip install -r $root_path/requirements.txt
        python -m pip install -r $root_path/requirements-dev.txt
        python -m pip install --no-cache-dir https://paddlenlp.bj.bcebos.com/wheels/paddlenlp-ci-py3-none-any.whl --force-reinstall --no-dependencies
        python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)";
    else
        echo -e "\033[31m ---- Skip install requirements for GPT dygraph cases  \033[0m"
    fi
    echo -e "\033[31m ---- Install ppfleetx/ops  \033[0m"
    cd ppfleetx/ops && python setup_cuda.py install && cd ../..

    unset http_proxy && unset https_proxy
    if [[ ! $FLAGS_download_data =~ "gpt" ]];then
        echo -e "\033[31m ---- download data for GPT dygraph cases  \033[0m"
        rm -rf data
        if [[ -e ${gpt_data_path}/data ]]; then
            echo "data downloaded"
        else
            # download data for gpt
            mkdir ${gpt_data_path}/data;
            wget -q -O ${gpt_data_path}/data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy;
            wget -q -O ${gpt_data_path}/data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz;
        fi
        cp -r ${gpt_data_path}/data ${gpt_case_path}/
    else
        echo -e "\033[31m ---- Skip download data for GPT dygraph cases \033[0m"
    fi

    echo -e "\033[31m ---- download other data  \033[0m"
    rm -rf ckpt
    if [[ -e ${gpt_data_path}/ckpt/PaddleFleetX_GPT_345M_220826 ]]; then
        echo "ckpt/PaddleFleetX_GPT_345M_220826 downloaded"
    else
        # download ckpt for gpt
        mkdir -p ${gpt_data_path}/ckpt
        wget -q -O ${gpt_data_path}/ckpt/GPT_345M.tar.gz \
            https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
        tar -xzf ${gpt_data_path}/ckpt/GPT_345M.tar.gz -C ${gpt_data_path}/ckpt
        rm -rf ${gpt_data_path}/ckpt/GPT_345M.tar.gz
    fi

    rm -rf dataset
    if [[ -e ${gpt_data_path}/dataset/wikitext_103_en ]]; then
        echo "dataset/wikitext_103_en downloaded"
    else
        # download dataset/wikitext_103_en
        mkdir -p ${gpt_data_path}/dataset/wikitext_103_en;
        wget -q -O ${gpt_data_path}/dataset/wikitext_103_en/wikitext-103-en.txt http://fleet.bj.bcebos.com/datasets/gpt/wikitext-103-en.txt
    fi

    rm -rf wikitext-103
    if [[ -e ${gpt_data_path}/wikitext-103 ]]; then
        echo "wikitext-103 downloaded"
    else
        # download wikitext-103 for gpt eval
        wget -q -O ${gpt_data_path}/wikitext-103.zip https://paddlefleetx.bj.bcebos.com/data/wikitext-103.zip
        unzip -q ${gpt_data_path}/wikitext-103.zip -d ${gpt_data_path}/
        rm -rf ${gpt_data_path}/wikitext-103.zip
    fi

    rm -rf lambada_test.jsonl
    if [[ -e ${gpt_data_path}/lambada_test.jsonl ]]; then
        echo "lambada_test.jsonl downloaded"
    else
        # download lambada_test.jsonl for gpt eval
        wget -q -O ${gpt_data_path}/lambada_test.jsonl https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl
    fi

    rm -rf pretrained
    if [[ -e ${gpt_data_path}/pretrained ]]; then
        echo "GPT_345M_FP16 downloaded"
    else
        # download GPT_345M_FP16 for gpt export
        wget -q -O ${gpt_data_path}/GPT_345M_FP16.tar.gz https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_FP16.tar.gz
        tar -zxvf ${gpt_data_path}/GPT_345M_FP16.tar.gz -C ${gpt_data_path}/
        rm -rf ${gpt_data_path}/GPT_345M_FP16.tar.gz
    fi

    rm -rf GPT_345M_QAT_wo_analysis
    if [[ -e ${gpt_data_path}/GPT_345M_QAT_wo_analysis ]]; then
        echo "GPT_345M_QAT_wo_analysis downloaded"
    else
        # download GPT_345M_QAT_wo_analysis for gpt qat
        wget -q -O ${gpt_data_path}/GPT_345M_QAT_wo_analysis.tar https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_QAT_wo_analysis.tar
        tar xf ${gpt_data_path}/GPT_345M_QAT_wo_analysis.tar -C ${gpt_data_path}/
        rm -rf ${gpt_data_path}/GPT_345M_QAT_wo_analysis.tar
    fi

    ln -s ${gpt_data_path}/ckpt ${gpt_case_path}/ckpt
    cp -r ${gpt_data_path}/dataset ${gpt_case_path}/
    ln -s ${gpt_data_path}/wikitext-103 ${gpt_case_path}/wikitext-103
    cp ${gpt_data_path}/lambada_test.jsonl ${gpt_case_path}/
    ln -s ${gpt_data_path}/pretrained ${gpt_case_path}/pretrained
    ln -s ${gpt_data_path}/GPT_345M_QAT_wo_analysis ${gpt_case_path}/GPT_345M_QAT_wo_analysis
}

function before_hook_for_llm_gpt() {
    echo -e "\033[31m ---- Set FLAGS for llm GPT cases  \033[0m"
    cd ${llm_gpt_case_path}
    export FLAGS_cudnn_deterministic=1
    export FLAGS_embedding_deterministic=1
    env | grep FLAGS
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    export no_proxy=bcebos.com
    python -m pip install -r $root_path/requirements.txt
    python -m pip install -r $root_path/requirements-dev.txt
    unset http_proxy && unset https_proxy
    if [[ ! $FLAGS_download_data =~ "llm_gpt" ]];then
        echo -e "\033[31m ---- Download llm GPT data  \033[0m"
        rm -rf data
        if [[ -e ${llm_gpt_data_path}/data ]]; then
            echo "llm GPT data downloaded"
        else
            # download data for llm GPT
            mkdir ${llm_gpt_data_path}/data;
            wget -q -O ${llm_gpt_data_path}/data/gpt2-en-mmap.bin https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/gpt/mmap/gpt2-en-mmap.bin
            wget -q -O ${llm_gpt_data_path}/data/gpt2-en-mmap.idx https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/gpt/mmap/gpt2-en-mmap.idx
        fi
        cp -r ${llm_gpt_data_path}/data ${llm_gpt_case_path}/
    else
        echo -e "\033[31m ---- Skip download llm GPT data \033[0m"
    fi
}

export status=$1

if [[ $status = "prepare_case" ]];then
    export FLAGS_install_deps=$3
    export FLAGS_download_data=$4
    if [[ $2 = "gpt_case_list_dygraph" ]];then
        before_hook_for_gpt
        gpt_case_list_dygraph prepare_case
    elif [[ $2 = "llm_gpt_case_list_dygraph" ]];then
        before_hook_for_llm_gpt
        llm_gpt_case_list_dygraph prepare_case
    else
        echo -e "\033[31m ---- Invalid exec_case $2 \033[0m"
    fi
elif [[ $status = "exec_case" ]];then
    export FLAGS_install_deps=$3
    export FLAGS_download_data=$4
    if [[ $2 =~ "llm_gpt" ]];then
        cd ${llm_gpt_case_path}
    elif [[ $2 =~ "gpt" ]];then
        cd ${gpt_case_path}
    fi
    $2
else
    echo -e "\033[31m ---- Start executing $1 \033[0m"
    export exec_case=$1
    export FLAGS_install_deps=$2
    export FLAGS_download_data=$3

    if [[ $exec_case =~ "llm_gpt" ]];then
        cd ${llm_gpt_case_path}
        before_hook_for_llm_gpt
    elif [[ $exec_case =~ "gpt" ]];then
        cd ${gpt_case_path}
        before_hook_for_gpt
    else
        echo -e "\033[31m ---- Invalid exec_case $exec_case \033[0m"
    fi

    $1 exec_case
fi

