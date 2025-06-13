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

export nlp_dir=${PWD}
export log_path=${nlp_dir}/model_logs
export cudaid1=$2
export cudaid2=$3
export C_COMPILER_PATH=$(which gcc)
export CXX_COMPILER_PATH=$(which g++)
export CC=$(which gcc)
export CXX=$(which g++)

export PADDLE_INFERENCE_MODEL_SUFFIX=$(python -c "
import paddle
try:
    from paddle.base.framework import use_pir_api
    pir_enabled = use_pir_api()
except ImportError:
    pir_enabled = False
model_suffix = '.json' if pir_enabled else '.pdmodel'
print(model_suffix)
")

if [ ! -d "model_logs" ]; then
    mkdir model_logs
fi

print_info() {
    if [ $1 -ne 0 ]; then
        cat ${log_path}/$2.log | grep -v "SKIPPED" | grep -v "PASSED" > ${log_path}/$2_FAIL.log
        echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
        cat ${log_path}/$2_FAIL.log
        cp ${log_path}/$2_FAIL.log ${PPNLP_HOME}/upload/$2_FAIL.log.${AGILE_PIPELINE_BUILD_ID}.${AGILE_JOB_BUILD_ID}
        cd ${PPNLP_HOME} && python upload.py ${PPNLP_HOME}/upload 'paddlenlp/PaddleNLP_CI/PaddleNLP_CI'
        rm -rf upload/* && cd -
        if [ $1 -eq 124 ]; then
            echo -e "\033[31m [failed-timeout] Test case execution was terminated after exceeding the time limit. \033[0m"
        fi
    else
        tail -n 1 ${log_path}/$2.log
        echo -e "\033[32m ${log_path}/$2_SUCCESS \033[0m"
    fi
}
# case list
# 2 msra_ner （不可控，内置）
msra_ner() {
    cd ${nlp_dir}/slm/examples/information_extraction/msra_ner/
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    ## train
    time (python -m paddle.distributed.launch ./train.py \
        --model_type bert \
        --model_name_or_path bert-base-multilingual-uncased \
        --dataset msra_ner \
        --max_seq_length 128 \
        --batch_size 16 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --max_steps 2 \
        --save_steps 2 \
        --output_dir ./tmp/msra_ner/ \
        --device gpu >${log_path}/msra_ner_train.log) >>${log_path}/msra_ner_train.log 2>&1
    print_info $? msra_ner_train
    ## eval
    time (python -u ./eval.py \
        --model_name_or_path bert-base-multilingual-uncased \
        --max_seq_length 128 \
        --batch_size 16 \
        --device gpu \
        --init_checkpoint_path ./tmp/msra_ner/model_2.pdparams >${log_path}/msra_ner_eval.log) >>${log_path}/msra_ner_eval.log 2>&1
    print_info $? msra_ner_eval
    ## predict
    time (python -u ./predict.py \
        --model_name_or_path bert-base-multilingual-uncased \
        --max_seq_length 128 \
        --batch_size 16 \
        --device gpu \
        --init_checkpoint_path ./tmp/msra_ner/model_2.pdparams >${log_path}/msra_ner_predict.log) >>${log_path}/msra_ner_predict.log 2>&1
    print_info $? msra_ner_predict
}
# 3 glue
glue() {
    cd ${nlp_dir}/slm/examples/benchmark/glue/
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    ##  TASK_SST-2
    export TASK_NAME=SST-2
    time (python -u run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --max_seq_length 128 \
        --batch_size 128 \
        --learning_rate 3e-5 \
        --max_steps 1 \
        --logging_steps 1 \
        --save_steps 1 \
        --output_dir ./$TASK_NAME/ \
        --device gpu >${log_path}/glue_${TASK_NAME}_train.log) >>${log_path}/glue_${TASK_NAME}_train.log 2>&1
    print_info $? glue_${TASK_NAME}_train
}
# 4 bert
bert() {
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    # cd ${nlp_dir}/slm/model_zoo/bert/
    # wget -q https://paddle-qa.bj.bcebos.com/paddlenlp/bert.tar.gz
    # tar -xzvf bert.tar.gz
    python -c "import datasets;from datasets import load_dataset; train_dataset=load_dataset('glue', 'sst2', split='train')"
    cd ${nlp_dir}/slm/model_zoo/bert/data/
    wget -q https://bj.bcebos.com/paddlenlp/models/transformers/bert/data/training_data.hdf5
    cd ../
    # pretrain
    time (python -m paddle.distributed.launch run_pretrain.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_predictions_per_seq 20 \
        --batch_size 16  \
        --learning_rate 1e-4 \
        --weight_decay 1e-2 \
        --adam_epsilon 1e-6 \
        --warmup_steps 10000 \
        --input_dir data/ \
        --output_dir pretrained_models/ \
        --logging_steps 1 \
        --save_steps 1 \
        --max_steps 1 \
        --device gpu \
        --use_amp False >${log_path}/bert_pretrain.log) >>${log_path}/bert_pretrain.log 2>&1
    print_info $? bert_pretrain
    time (python -m paddle.distributed.launch run_glue_trainer.py \
        --model_name_or_path bert-base-uncased \
        --task_name SST2 \
        --max_seq_length 128 \
        --per_device_train_batch_size 32   \
        --per_device_eval_batch_size 32   \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --logging_steps 1 \
        --save_steps 1 \
        --max_steps 1 \
        --output_dir ./tmp/ \
        --device gpu \
        --fp16 False\
        --do_train \
        --do_eval >${log_path}/bert_fintune.log) >>${log_path}/bert_fintune.log 2>&1
    print_info $? bert_fintune
    time (python -u ./export_model.py \
        --model_type bert \
        --model_path bert-base-uncased \
        --output_path ./infer_model/model >${log_path}/bert_export.log) >>${log_path}/bert_export.log 2>&1
    print_info $? bert_export
}
# 5 skep (max save 不可控 内置)
skep() {
    cd ${nlp_dir}/slm/examples/sentiment_analysis/skep/
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    ## train_sentence
    time (python -m paddle.distributed.launch train_sentence.py \
        --batch_size 16 \
        --epochs 1 \
        --model_name "skep_ernie_1.0_large_ch" \
        --device gpu --save_dir ./checkpoints >${log_path}/skep_train_sentence.log) >>${log_path}/skep_train_sentence.log 2>&1
    print_info $? skep_train_sentence
    ## train_aspect
    time (python -m paddle.distributed.launch train_aspect.py \
        --batch_size 4 \
        --epochs 1 --device gpu \
        --save_dir ./aspect_checkpoints >${log_path}/skep_train_aspect.log) >>${log_path}/skep_train_aspect.log 2>&1
    print_info $? skep_train_aspect
    # # train_opinion
    time (python -m paddle.distributed.launch train_opinion.py \
        --batch_size 4 \
        --epochs 1 \
        --device gpu \
        --save_dir ./opinion_checkpoints >${log_path}/skep_train_opinion.log) >>${log_path}/skep_train_opinion.log 2>&1
    print_info $? skep_train_opinion
    # predict_sentence
    time (python predict_sentence.py \
        --model_name "skep_ernie_1.0_large_ch" \
        --ckpt_dir checkpoints/model_100 >${log_path}/skep_predict_sentence.log) >>${log_path}/skep_predict_sentence.log 2>&1
    print_info $? skep_predict_sentence
    ## predict_aspect
    time (python predict_aspect.py \
        --device 'gpu' \
        --ckpt_dir ./aspect_checkpoints/model_100 >${log_path}/skep_predict_aspect.log) >>${log_path}/skep_predict_aspect.log 2>&1
    print_info $? skep_predict_aspect
    # # predict_opinion
    time (python predict_opinion.py \
        --device 'gpu' \
        --ckpt_dir ./opinion_checkpoints/model_100 >${log_path}/skep_predict_opinion.log) >>${log_path}/skep_predict_opinion.log 2>&1
    print_info $? skep_predict_opinion
}
# 6 bigbird
bigbird(){
    cd ${nlp_dir}/slm/model_zoo/bigbird/
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    time (python -m paddle.distributed.launch  --log_dir log  run_pretrain.py \
        --model_name_or_path bigbird-base-uncased \
        --input_dir "./data" \
        --output_dir "output" \
        --batch_size 4 \
        --weight_decay 0.01 \
        --learning_rate 1e-5 \
        --max_steps 1 \
        --save_steps 1 \
        --logging_steps 1 \
        --max_encoder_length 512 \
        --max_pred_length 75 >${log_path}/bigbird_pretrain.log) >>${log_path}/bigbird_pretrain.log 2>&1
    print_info $? bigbird_pretrain
}
# 9 ernie
ernie(){
    #data process
    cd ${nlp_dir}/slm/model_zoo/ernie-1.0/

    if [ -d "data_ernie_3.0" ];then
        rm -rf data_ernie_3.0
    fi

    mkdir data_ernie_3.0 
    cd data_ernie_3.0
    wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/wudao_200g_sample_ernie-3.0-base-zh_ids.npy
    wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/wudao_200g_sample_ernie-3.0-base-zh_idx.npz
    cd ../

    # pretrain_trainer
    time (python -u -m paddle.distributed.launch \
        --log_dir "output/trainer_log" \
        run_pretrain_trainer.py \
        --model_type "ernie" \
        --model_name_or_path "ernie-3.0-base-zh" \
        --tokenizer_name_or_path "ernie-3.0-base-zh" \
        --input_dir "./data_ernie_3.0" \
        --output_dir "output/trainer_log" \
        --split 949,50,1 \
        --max_seq_length 512 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --fp16  \
        --fp16_opt_level "O2"  \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 2 \
        --save_steps 2 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1\
        --dataloader_num_workers 4 \
        --eval_steps 1000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --do_train \
        --device "gpu" >${log_path}/ernie_1.0_pretrain_trainer.log) >>${log_path}/ernie_1.0_pretrain_trainer.log 2>&1
    print_info $? ernie_1.0_pretrain_trainer
}
# 11 ofa
ofa(){
    cd ${nlp_dir}/slm/examples/model_compression/ofa/
    cd ../../benchmark/glue/
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    # finetuing
    time (python -u run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name SST-2 \
        --max_seq_length 128 \
        --batch_size 32   \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --max_steps 1 \
        --logging_steps 1 \
        --save_steps 1 \
        --output_dir ./ \
        --device gpu  >${log_path}/ofa_pretrain.log) >>${log_path}/ofa_pretrain.log 2>&1
    print_info $? ofa_pretrain
    mv sst-2_ft_model_1.pdparams/  ${nlp_dir}/slm/examples/model_compression/ofa/
    cd -
    #model slim
    # export CUDA_VISIBLE_DEVICES=${cudaid2}
    # time (python -m paddle.distributed.launch run_glue_ofa.py  \
    #     --model_type bert \
    #     --model_name_or_path ./sst-2_ft_model_1.pdparams/ \
    #     --task_name SST-2 --max_seq_length 128     \
    #     --batch_size 32       \
    #     --learning_rate 2e-5     \
    #     --num_train_epochs 1     \
    #     --max_steps 1 \
    #     --logging_steps 1    \
    #     --save_steps 1     \
    #     --output_dir ./ofa/SST-2 \
    #     --device gpu  \
    #     --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5 >${log_path}/ofa_slim) >>${log_path}/ofa_slim 2>&1
    # print_info $? ofa_slim
}
# 12 albert
albert() {
    cd ${nlp_dir}/slm/examples/benchmark/glue/
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    time (python -m paddle.distributed.launch run_glue.py \
        --model_type albert \
        --model_name_or_path albert-base-v2 \
        --task_name SST-2 \
        --max_seq_length 128 \
        --batch_size 32 \
        --learning_rate 1e-5 \
        --max_steps 1 \
        --warmup_steps 1256 \
        --logging_steps 1 \
        --save_steps 1 \
        --output_dir ./albert/SST-2/ \
        --device gpu >${log_path}/albert_sst-2_train.log) >>${log_path}/albert_sst-2_train.log 2>&1
    print_info $? albert_sst-2_train
}
# 13 squad
# squad() {
#     cd ${nlp_dir}/slm/examples/machine_reading_comprehension/SQuAD/
#     export CUDA_VISIBLE_DEVICES=${cudaid1}
#     # finetune
#     time (python -m paddle.distributed.launch run_squad.py \
#         --model_type bert \
#         --model_name_or_path bert-base-uncased \
#         --max_seq_length 384 \
#         --batch_size 12 \
#         --learning_rate 3e-5 \
#         --num_train_epochs 1 \
#         --max_steps 1 \
#         --logging_steps 1 \
#         --save_steps 1 \
#         --warmup_proportion 0.1 \
#         --weight_decay 0.01 \
#         --output_dir ./tmp/squad/ \
#         --device gpu \
#         --do_train \
#         --do_predict >${log_path}/squad_train) >>${log_path}/squad_train 2>&1
#     print_info $? squad_train
#     # export model
#     time (python -u ./export_model.py \
#         --model_type bert \
#         --model_path ./tmp/squad/model_1/ \
#         --output_path ./infer_model/model >${log_path}/squad_export) >>${log_path}/squad_export 2>&1
#     print_info $? squad_export
#     predict
#     time (python -u deploy/python/predict.py \
#         --model_type bert \
#         --model_name_or_path ./infer_model/model \
#         --batch_size 2 \
#         --max_seq_length 384 >${log_path}/squad_predict) >>${log_path}/squad_predict 2>&1
#     print_info $? squad_predict
# }
# 15 lexical_analysis
lexical_analysis(){
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    cd ${nlp_dir}/slm/examples/lexical_analysis/
    #train
    time (python download.py --data_dir ./ )
    time (python -m paddle.distributed.launch train.py \
        --data_dir ./lexical_analysis_dataset_tiny \
        --model_save_dir ./save_dir \
        --epochs 1 \
        --save_steps 15 \
        --logging_steps 1\
        --batch_size 32 \
        --device gpu >${log_path}/lexical_analysis_train.log) >>${log_path}/lexical_analysis_train.log 2>&1
    print_info $? lexical_analysis_train
    #export
    time (python export_model.py \
        --data_dir=./lexical_analysis_dataset_tiny \
        --params_path=./save_dir/model_15.pdparams \
        --output_path=./infer_model/static_graph_params >${log_path}/lexical_analysis_export.log) >>${log_path}/lexical_analysis_export.log 2>&1
    print_info $? lexical_analysis_export
    # predict
    time (python predict.py --data_dir ./lexical_analysis_dataset_tiny \
        --init_checkpoint ./save_dir/model_15.pdparams \
        --batch_size 32 \
        --device gpu >${log_path}/lexical_analysis_predict.log) >>${log_path}/lexical_analysis_predict.log 2>&1
    print_info $? lexical_analysis_predict
    # deploy
    time (python deploy/predict.py \
        --model_file=infer_model/static_graph_params${PADDLE_INFERENCE_MODEL_SUFFIX} \
        --params_file=infer_model/static_graph_params.pdiparams \
        --data_dir lexical_analysis_dataset_tiny >${log_path}/lexical_analysis_deploy.log) >>${log_path}/lexical_analysis_deploy.log 2>&1
    print_info $? lexical_analysis_deploy
}

# 22 transformer
transformer() {
    cd ${nlp_dir}/slm/examples/machine_translation/transformer/
    wget -q https://paddle-qa.bj.bcebos.com/paddlenlp/WMT14.en-de.partial.tar.gz
    tar -xzvf WMT14.en-de.partial.tar.gz
    time (
        sed -i "s/save_step: 10000/save_step: 1/g" configs/transformer.base.yaml
        sed -i "s/print_step: 100/print_step: 1/g" configs/transformer.base.yaml
        sed -i "s/epoch: 30/epoch: 1/g" configs/transformer.base.yaml
        sed -i "s/max_iter: None/max_iter: 2/g" configs/transformer.base.yaml
        sed -i "s/batch_size: 4096/batch_size: 1000/g" configs/transformer.base.yaml

        python train.py --config ./configs/transformer.base.yaml \
            --train_file ${PWD}/WMT14.en-de.partial/train.tok.clean.bpe.en ${PWD}/WMT14.en-de.partial/train.tok.clean.bpe.de \
            --dev_file ${PWD}/WMT14.en-de.partial/dev.tok.bpe.en ${PWD}/WMT14.en-de.partial/dev.tok.bpe.de \
            --vocab_file ${PWD}/WMT14.en-de.partial/vocab_all.bpe.33708 \
            --unk_token "<unk>" --bos_token "<s>" --eos_token "<e>" >${log_path}/transformer_train.log
    ) >>${log_path}/transformer_train.log 2>&1
    print_info $? transformer_train
    #predict
    time (
        sed -i 's#init_from_params: "./trained_models/step/"#init_from_params: "./trained_models/step_final/"#g' configs/transformer.base.yaml
        python predict.py --config ./configs/transformer.base.yaml \
            --test_file ${PWD}/WMT14.en-de.partial/test.tok.bpe.en ${PWD}/WMT14.en-de.partial/test.tok.bpe.de \
            --without_ft \
            --vocab_file ${PWD}/WMT14.en-de.partial/vocab_all.bpe.33708 \
            --unk_token "<unk>" --bos_token "<s>" --eos_token "<e>" >${log_path}/transformer_predict.log
    ) >>${log_path}/transformer_predict.log 2>&1
    print_info $? transformer_predict
    #export
    time (
        python export_model.py --config ./configs/transformer.base.yaml \
            --vocab_file ${PWD}/WMT14.en-de.partial/vocab_all.bpe.33708 \
            --bos_token "<s>" --eos_token "<e>" >${log_path}/transformer_export.log
    ) >>${log_path}/transformer_export.log 2>&1
    print_info $? transformer_export
    #infer
    time (
        python ./deploy/python/inference.py --config ./configs/transformer.base.yaml \
            --profile \
            --test_file ${PWD}/WMT14.en-de.partial/test.tok.bpe.en ${PWD}/WMT14.en-de.partial/test.tok.bpe.de \
            --vocab_file ${PWD}/WMT14.en-de.partial/vocab_all.bpe.33708 \
            --unk_token "<unk>" --bos_token "<s>" --eos_token "<e>" >${log_path}/transformer_infer.log
    ) >>${log_path}/transformer_infer.log 2>&1
    print_info $? transformer_infer

    # fast_transformer
}
#28 question_matching
question_matching() {
    cd ${nlp_dir}/slm/examples/text_matching/question_matching/
    wget -q https://paddle-qa.bj.bcebos.com/paddlenlp/data_v4.tar.gz
    tar -xvzf data_v4.tar.gz
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    #train
    time (
        python -u -m paddle.distributed.launch train.py \
            --train_set ./data_v4/train/ALL/train \
            --dev_set ./data_v4/train/ALL/dev \
            --device gpu \
            --eval_step 10 \
            --max_steps 10 \
            --save_dir ./checkpoints \
            --train_batch_size 32 \
            --learning_rate 2E-5 \
            --epochs 1 \
            --rdrop_coef 0.0 >${log_path}/question_matching_train.log) >>${log_path}/question_matching_train.log 2>&1
    print_info $? question_matching_train
    #predict
    time (
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        python -u \
            predict.py \
            --device gpu \
            --params_path "./checkpoints/model_10/model_state.pdparams" \
            --batch_size 128 \
            --input_file ./data_v4/test/public_test_A \
            --result_file 0.0_predict_public_result_test_A_re >${log_path}/question_matching_predict.log) >>${log_path}/question_matching_predict.log 2>&1
    print_info $? question_matching_predict
}
# 29 ernie-csc
ernie-csc() {
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    cd ${nlp_dir}/slm/examples/text_correction/ernie-csc
    #dowdnload data
    python download.py --data_dir ./extra_train_ds/ --url https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml
    #trans xml txt
    python change_sgml_to_txt.py -i extra_train_ds/train.sgml -o extra_train_ds/train.txt
    #2卡训练
    python -m paddle.distributed.launch  train.py --batch_size 32 --logging_steps 100 --epochs 1 --learning_rate 5e-5 --model_name_or_path ernie-1.0-base-zh --output_dir ./checkpoints/ --extra_train_ds_dir ./extra_train_ds/  >${log_path}/ernie-csc_train.log  2>&1
    print_info $? ernie-csc_train
    #predict
    sh run_sighan_predict.sh >${log_path}/ernie-csc_predict.log 2>&1
    print_info $? ernie-csc_predict
    #export model
    python export_model.py --params_path ./checkpoints/best_model.pdparams --output_path ./infer_model/static_graph_params >${log_path}/ernie-csc_export.log 2>&1
    print_info $? ernie-csc_export
    #python deploy
    python predict.py --model_file infer_model/static_graph_params${PADDLE_INFERENCE_MODEL_SUFFIX} --params_file infer_model/static_graph_params.pdiparams >${log_path}/ernie-csc_deploy.log 2>&1
    print_info $? ernie-csc_deploy
}

clue() {
    cd ${nlp_dir}/slm/examples/benchmark/clue/classification
    time (python -u ./run_clue_classifier_trainer.py \
        --model_name_or_path ernie-3.0-base-zh \
        --dataset "clue afqmc" \
        --max_seq_length 128 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --logging_steps 1 \
        --seed 42 \
        --save_steps 3 \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --adam_epsilon 1e-8 \
        --output_dir ./tmp \
        --device gpu \
        --do_train \
        --do_eval \
        --metric_for_best_model "eval_accuracy" \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --max_steps 1 >${log_path}/clue-trainer_api.log) >>${log_path}/clue-trainer_api.log 2>&1
    print_info $? clue-tranier_api
    time (python -u run_clue_classifier.py \
        --model_name_or_path ernie-3.0-base-zh \
        --task_name afqmc \
        --max_seq_length 128 \
        --batch_size 16 \
        --learning_rate 3e-5 \
        --num_train_epochs 3 \
        --logging_steps 100 \
        --seed 42 \
        --save_steps 1 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --adam_epsilon 1e-8 \
        --output_dir ./output/afqmc \
        --device gpu \
        --max_steps 1 \
        --do_train >${log_path}/clue-class.log) >>${log_path}/clue-class.log 2>&1
    print_info $? clue-class
    # cd ${nlp_dir}/slm/examples/benchmark/clue/mrc
    # export CUDA_VISIBLE_DEVICES=${cudaid1}
    # python -m paddle.distributed.launch run_cmrc2018.py \
    #     --model_name_or_path ernie-3.0-base-zh \
    #     --batch_size 16 \
    #     --learning_rate 3e-5 \
    #     --max_seq_length 512 \
    #     --num_train_epochs 2 \
    #     --do_train \
    #     --do_predict \
    #     --warmup_proportion 0.1 \
    #     --weight_decay 0.01 \
    #     --gradient_accumulation_steps 2 \
    #     --max_steps 1 \
    #     --output_dir ./tmp >${log_path}/clue-mrc >>${log_path}/clue-mrc 2>&1
    # print_info $? clue-mrc
}
#33 taskflow
taskflow (){
    cd ${nlp_dir}
    timeout 10m python -m pytest scripts/regression/test_taskflow.py >${log_path}/taskflow.log 2>&1
    print_info $? taskflow
}
ernie-3.0(){
    cd ${nlp_dir}/slm/model_zoo/ernie-3.0/
    #训练
    python run_seq_cls.py  --model_name_or_path ernie-3.0-medium-zh  --dataset afqmc --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=configs/default.yml --max_steps=2 --save_step=2 >${log_path}/ernie-3.0_train_seq_cls.log 2>&1
    print_info $? ernie-3.0_train_seq_cls
    python run_token_cls.py --model_name_or_path ernie-3.0-medium-zh --dataset msra_ner --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=configs/default.yml --max_steps=2 --save_step=2 >${log_path}/ernie-3.0_train_token_cls.log 2>&1
    print_info $? ernie-3.0_train_token_cls
    python run_qa.py --model_name_or_path ernie-3.0-medium-zh --dataset cmrc2018  --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=configs/default.yml --max_steps=2 --save_step=2 >${log_path}/ernie-3.0_train_qa.log 2>&1
    print_info $? ernie-3.0_train_qa
    # 预测
    python run_seq_cls.py  --model_name_or_path best_models/afqmc/  --dataset afqmc --output_dir ./best_models --do_predict --config=configs/default.yml >${log_path}/ernie-3.0_predict_seq_cls.log 2>&1
    print_info $? ernie-3.0_predict_seq_cls
    python run_token_cls.py  --model_name_or_path best_models/msra_ner/  --dataset msra_ner --output_dir ./best_models --do_predict --config=configs/default.yml >${log_path}/ernie-3.0_predict_token_cls.log 2>&1
    print_info $? ernie-3.0_predict_token_cls
    python run_qa.py --model_name_or_path best_models/cmrc2018/ --dataset cmrc2018  --output_dir ./best_models --do_predict --config=configs/default.yml >${log_path}/ernie-3.0_predict_qa.log 2>&1
    print_info $? ernie-3.0_predict_qa
    #压缩 skip for paddleslim api error https://github.com/PaddlePaddle/PaddleSlim/blob/9f3e9b2f0f9948b780900d1299f2c3fe47322deb/paddleslim/nas/ofa/layers.py#L1301C32-L1302 
    # python compress_seq_cls.py  --model_name_or_path best_models/afqmc/  --dataset afqmc --output_dir ./best_models/afqmc --config=configs/default.yml --max_steps 10 --eval_steps 5 --save_steps 5 --save_steps 5 --algo_list mse --batch_size_list 4 >${log_path}/ernie-3.0_compress_seq_cls >>${log_path}/ernie-3.0_compress_seq_cls 2>&1
    # print_info $? ernie-3.0_compress_seq_cls
    # python compress_token_cls.py  --model_name_or_path best_models/msra_ner/  --dataset msra_ner --output_dir ./best_models/msra_ner --config=configs/default.yml --max_steps 10 --eval_steps 5 --save_steps 5  --algo_list mse --batch_size_list 4 >${log_path}/ernie-3.0_compress_token_cls >>${log_path}/ernie-3.0_compress_token_cls 2>&1
    # print_info $? ernie-3.0_compress_token_cls
    # python compress_qa.py --model_name_or_path best_models/cmrc2018/ --dataset cmrc2018  --output_dir ./best_models/cmrc2018 --config=configs/default.yml --max_steps 10 --eval_steps 5 --save_steps 5  --algo_list mse --batch_size_list 4 >${log_path}/ernie-3.0_compress_qa >>${log_path}/ernie-3.0_compress_qa 2>&1
    # print_info $? ernie-3.0_compress_qa
}
uie(){
    cd ${nlp_dir}/slm/model_zoo/uie/
    mkdir data && cd data && wget https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json && cd ../
    python doccano.py --doccano_file ./data/doccano_ext.json --task_type ext --save_dir ./data --splits 0.8 0.2 0 --schema_lang ch >${log_path}/uie_doccano.log 2>&1
    print_info $? uie_doccano
    python -u -m paddle.distributed.launch finetune.py --device gpu --logging_steps 2 --save_steps 2 --eval_steps 2 --seed 42 \
        --model_name_or_path uie-base --output_dir ./checkpoint/model_best --train_path data/train.txt --dev_path data/dev.txt \
        --max_seq_length 512 --per_device_eval_batch_size 16 --per_device_train_batch_size 16 --num_train_epochs 100 --learning_rate 1e-5 \
        --do_train --do_eval --do_export --export_model_dir ./checkpoint/model_best --label_names start_positions end_positions \
        --overwrite_output_dir --disable_tqdm True --metric_for_best_model eval_f1 --load_best_model_at_end True \
        --save_total_limit 1 --max_steps 2  >${log_path}/uie_train.log 2>&1
    print_info $? uie_train
    python evaluate.py --model_path ./checkpoint/model_best --test_path ./data/dev.txt --batch_size 16 --max_seq_len 512 >${log_path}/uie_eval.log 2>&1
    print_info $? uie_eval
}
ernie-layout(){
    cd ${nlp_dir}/slm/model_zoo/ernie-layout/
    # train ner
    python -u run_ner.py --model_name_or_path ernie-layoutx-base-uncased --output_dir ./ernie-layoutx-base-uncased/models/funsd/ \
        --dataset_name funsd --do_train --do_eval --max_steps 2 --eval_steps 2 --save_steps 2 --save_total_limit 1 --seed 1000 --overwrite_output_dir \
        --load_best_model_at_end --pattern ner-bio --preprocessing_num_workers 4 --overwrite_cache false --doc_stride 128 --target_size 1000 \
        --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --learning_rate 2e-5 --lr_scheduler_type constant --gradient_accumulation_steps 1 \
        --metric_for_best_model eval_f1 --greater_is_better true >${log_path}/ernie-layout_train.log 2>&1
    print_info $? ernie-layout_train
    # export ner
    python export_model.py --task_type ner --model_path ./ernie-layoutx-base-uncased/models/funsd/ --output_path ./ner_export >${log_path}/ernie-layout_export.log 2>&1
    print_info $? ernie-layout_export
    # deploy ner
    cd ${nlp_dir}/slm/model_zoo/ernie-layout/deploy/python
    wget https://bj.bcebos.com/paddlenlp/datasets/document_intelligence/images.zip && unzip images.zip
    python infer.py --model_path_prefix ../../ner_export/inference --task_type ner --lang "en" --batch_size 8 >${log_path}/ernie-layout_deploy.log 2>&1
    print_info $? ernie-layout_deploy
}
ernie-1.0(){
    ernie
}

ernie_layout(){
    ernie-layout
}

ernie_csc(){
    ernie-csc
}

segment_parallel_utils(){
    cd ${nlp_dir}
    echo "test segment_parallel_utils, cudaid1:${cudaid1}, cudaid2:${cudaid2}"
    if [[ ${cudaid1} != ${cudaid2} ]]; then
        time (python -m paddle.distributed.launch tests/transformers/test_segment_parallel_utils.py >${log_path}/segment_parallel_utils.log) >>${log_path}/segment_parallel_utils.log 2>&1
        print_info $? segment_parallel_utils
    else
        echo "only one gpu:${cudaid1} is set, skip test"
    fi

}
ring_flash_attention(){
    cd ${nlp_dir}
    echo "test ring_flash_attention, cudaid1:${cudaid1}, cudaid2:${cudaid2}"
    if [[ ${cudaid1} != ${cudaid2} ]]; then
        time (python -m paddle.distributed.launch tests/transformers/test_ring_flash_attention.py >${log_path}/ring_flash_attention.log) >>${log_path}/ring_flash_attention.log 2>&1
        print_info $? ring_flash_attention
    else
        echo "only one gpu:${cudaid1} is set, skip test"
    fi

}

llm(){
    export http_proxy=${proxy} && export https_proxy=${proxy}
    echo ' Testing all LLMs '
    cd ${nlp_dir}
    timeout 50m python -m pytest tests/llm/test_*.py -vv --timeout=300 --alluredir=result >${log_path}/llm.log 2>&1
    print_info $? llm
}

$1
