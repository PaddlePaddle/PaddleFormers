export PYTHONPATH=../../../../:$PYTHONPATH

set -x
unset CUDA_VISIBLE_DEVICES
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
export NNODES=1
export PADDLE_TRAINERS_NUM=1

# export FLAGS_cudnn_deterministic=1
# export FLAGS_embedding_deterministic=1
# export FLAGS_enable_auto_parallel_align_mode=1
# export FLAGS_log_memory_stats=1

python -u -m paddle.distributed.launch \
    --log_dir "output" \
    --run_mode=collective \
    ./run_pretrain_auto.py \
    --model_name_or_path "/root/paddlejob/gpfs/huggingface/meta-llama/Llama-3.1-8B-Instruct" \
    --tokenizer_name_or_path "/root/paddlejob/gpfs/huggingface/meta-llama/Llama-3.1-8B-Instruct" \
    --input_dir "./data" \
    --output_dir "./output" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 1 \
    --hybrid_parallel_topo_order "pp_first" \
    --sharding "stage1" \
    --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate" \
    --sharding_parallel_config "enable_overlap enable_tensor_fusion" \
    --tensor_parallel_config "" \
    --pipeline_parallel_config "enable_send_recv_overlap enable_split_backward" \
    --pipeline_schedule_mode "VPP" \
    --virtual_pipeline_seg_method "LlamaDecoderLayerNet" \
    --virtual_pp_degree 1 \
    --sequence_parallel 0 \
    --use_flash_attention true \
    --use_fused_rms_norm false \
    --fuse_attention_ffn true \
    --fuse_attention_qkv true \
    --use_fused_rope true \
    --fused_linear_param_grad_add true \
    --max_seq_length 4096 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --warmup_steps 30 \
    --logging_steps 2 \
    --max_steps 10 \
    --save_steps 5000 \
    --eval_steps 1000 \
    --weight_decay 0.01 \
    --bf16 true \
    --fp16_opt_level "O2" \
    --amp_custom_black_list "reduce_sum c_softmax_with_cross_entropy" \
    --amp_custom_white_list "lookup_table lookup_table_v2" \
    --amp_master_grad true \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer true \
    --recompute false \
    --recompute_use_reentrant true \
    --distributed_dataloader 0 \
    --recompute_granularity "full" \
    --save_total_limit 2 \
    --device "gpu" \
    --enable_auto_parallel true \
    --model_type "llama_network" \
    --use_intermediate_api true
