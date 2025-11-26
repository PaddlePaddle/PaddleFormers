# CUDA_VISIBLE_DEVICES=0 paddleformers-cli train \
#     examples/config/sft/full.yaml \
#     model_name_or_path=./models/Qwen2.5-VL-3B-Instruct/ \
#     train_dataset_path=./data/vl/experiment.jsonl \
#     eval_dataset_path=./data/vl/experiment.jsonl

CUDA_VISIBLE_DEVICES=0 paddleformers-cli train \
    examples/config/sft/full.yaml \
    model_name_or_path=./models/Qwen3-0.6B-base/ \
    train_dataset_path=./data/sft/train.jsonl \
    eval_dataset_path=./data/sft/train.jsonl \
    max_steps=50 random_shuffle=true
