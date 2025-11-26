# CUDA_VISIBLE_DEVICES=0 paddleformers-cli train \
#     examples/config/sft/full.yaml \
#     model_name_or_path=./models/Qwen2.5-VL-3B-Instruct/ \
#     train_dataset_path=./data/vl/experiment.jsonl \
#     eval_dataset_path=./data/vl/experiment.jsonl

CUDA_VISIBLE_DEVICES=0,1 paddleformers-cli train \
    examples/config/sft/full.yaml \
    model_name_or_path=./models/Qwen2.5-VL-3B-Instruct/ \
    train_dataset_path=./data/vl/experiment.jsonl \
    eval_dataset_path=./data/vl/experiment.jsonl \
    stage="VL-SFT" \
    tensor_parallel_degree=2 \
    max_steps=50 random_shuffle=true recompute=false
