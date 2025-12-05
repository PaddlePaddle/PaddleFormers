unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM

export DISTRIBUTED_TRAINER_ENDPOINTS=`hostname -i`
export PYTHONPATH="/root/paddlejob/workspace/env_run/output/wangna11/PaddleFormers/third_party/PaddleNLP/":$PYTHONPATH
export PYTHONPATH="/root/paddlejob/workspace/env_run/output/wangna11/PaddleFormers/third_party/PaddleSlim/":$PYTHONPATH
export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/wangna11/miniconda3/envs/qwen/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
export LOAD_STATE_DICT_THREAD_NUM=128
export DISABLE_FASTER_SET_STATE_DICT=1 #---#

export CUDA_VISIBLE_DEVICES=4,5,6,7

model_name_or_path=Qwen/qwen30b_a3b_model_1119/
data_path=/root/paddlejob/workspace/env_run/output/wangna11/wwb/sft_1119.jsonl

save_name=qwen30b_a3b_model_1_tmp
log_dir=/root/paddlejob/workspace/env_run/output/wangna11/PaddleFormers/log_${save_name}
save_path=/root/paddlejob/workspace/env_run/output/wangna11/PaddleFormers/output/${save_name}

rm -rf ${log_dir}
rm -rf ${save_path}
mkdir -p ${save_path}

python -u -m paddle.distributed.launch \
    --log_dir ${log_dir} \
    qwen_quant.py \
    --model_name_or_path ${model_name_or_path} \
    --dtype bfloat16 \
    --mode dynamic \
    --total_max_length 8192 \
    --data_file ${data_path} \
    --save_path ${save_path} \
    --quant_type W4A8C8 \
    --gptq True \
    --output_file ./output.txt \
