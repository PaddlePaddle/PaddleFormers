#!/bin/bash
# KimiK2 PaddleFleet 多机训练脚本
# 参考: run_glm45_4nodes_128k.sh

set -e

# 清理分布式环境变量
for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
    unset ${name}
done
unset http_proxy https_proxy

# 激活 PaddleFleet 环境
if [ -f '/root/paddlejob/share-storage/gpfs/system-public/liyamei/PaddleFleet/.venv/bin/activate' ]; then
    source /root/paddlejob/share-storage/gpfs/system-public/liyamei/PaddleFleet/.venv/bin/activate
    echo "activate PaddleFleet env"
fi

export root_dir=/root/paddlejob/share-storage/gpfs/system-public/liyamei/PaddleFormers
cd $root_dir

# 多机配置
# START_RANK/END_RANK: 根据实际机器分配调整
START_RANK=${START_RANK:-0}
END_RANK=${END_RANK:-4}
nnodes=$(($END_RANK-$START_RANK))

# 从 hostfile 获取 master 地址（根据 START_RANK 确定主节点）
hostfile_path="/root/paddlejob/share-storage/gpfs/system-public/liyamei/hostfile"
if [ -f "$hostfile_path" ]; then
    master=$(cat $hostfile_path | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}')
else
    master=$(hostname -i)
fi
port=${MASTER_PORT:-36677}

# 获取当前 rank
current_rank=${PADDLE_TRAINER_ID:-0}
rank=$(($current_rank-$START_RANK))

# 环境变量
export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1
export PADDLEFORMERS_DIST_LOG=./kimik2_dist_log
export PADDLE_DUMP_DIR=/root/paddlejob/share-storage/gpfs/system-public/liyamei/kimik2_dump

# 清理残留进程
ps aux | grep paddleformers-cli | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
rm -rf core.* 2>/dev/null || true

# 配置文件
config_yaml=$root_dir/tests/config/benchmark/config/pt/Kimi-K2.yaml
log_dir=$root_dir/kimik2_output
mkdir -p $log_dir
log_file=$log_dir/train.log

echo "=========================================="
echo "KimiK2 PT Benchmark Training"
echo "Config: $config_yaml"
echo "NNODES: $nnodes"
echo "MASTER_ADDR: $master"
echo "MASTER_PORT: $port"
echo "RANK: $rank"
echo "Log: $log_file"
echo "=========================================="

# 启动训练
NNODES=$nnodes MASTER_ADDR=$master MASTER_PORT=$port RANK=$rank \
    paddleformers-cli train $config_yaml 2>&1 | tee $log_file

echo "Training completed. Log saved to $log_file"
