rm -rf paddleformers_dist_log checkpoints/ernie-0.3B-sft-full/ vdl_log/
paddleformers-cli train examples/config/iluvatar/ERNIE-4.5-0.3B-PT/sft/full_8k.yaml
