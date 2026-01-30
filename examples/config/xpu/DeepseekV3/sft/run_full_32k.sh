#!/bin/bash
export XPU_BLACK_LIST="index_elementwise_put,index_elementwise_put_with_tensor,index_elementwise_put_with_tensor_grad,index_elementwise_put_grad"

export BKCL_TIMEOUT=1000
export BKCL_SOCKET_IFNAME=eth0
export BKCL_ENABLE_XDR=1
export BKCL_FORCE_RDMA_NICS_ORDER=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export BKCL_DEEPEP_NORMAL_CLUSTER_NUM=8

# ============= xshmem =============
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=24
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_USE_RDMA=1
export BKCL_FORCE_L3_RDMA=0   # 开1空间不够会报OOM错误
export BKCL_RDMA_VERBS=1

for id in $(ipcs -m | awk '/0x/ {print $2}'); do ipcrm -m $id; done

source /root/formers_venv/bin/activate #python环境存放位置

# 修改num_nodes， your_master_addr， your_master_port， 为实际配置
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli train /root/PaddleFormers/examples/config/xpu/DeepseekV3/sft/full_32k_config.yaml