unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

export NNODES=1
export PADDLE_TRAINERS_NUM=1
export FLAGS_selected_gpus=1
export Align_Fleet=0

export PYTHONPATH=$PYTHONPATH:../..:../../..:/root/paddlejob/gpfs/zhangweilong/PaddleFleet/src/
source /root/paddlejob/gpfs/zhangweilong/py310_zwl/bin/activate
export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1 

# python test.py
python run_qwen_vl.py