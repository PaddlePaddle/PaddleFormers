# 1. 背景说明

PaddleFormers 提供了 DeepSeek-V3 的预训练加速版本模型。基于 PaddlePaddle 框架在 DeepSeek-V3 上的实战经验，我们将整套高效训练优化能力做成了 **“开箱即用”** 的配置，包括混合并行、Dualpipe 及 Overlap 策略、FP8 混合精度训练、精细化重计算以及算子融合等。对 671B 全参规模模型，在 256 张 NVIDIA H800 GPU、sequence_len=4K 采用 PP8-EP32-Sharding_stage1 的并行策略，热启 huggingface ckpt 性能可达到 1400+ tokens/s/card，折算得到 MFU=41%。

# 2. 硬件配置要求

## 2.1. 最低配置

GPU: NVIDIA H100 80GB (推荐) 或 H800、H20等

数量: 如需完整跑通 671B 规模模型，需要至少256卡，您可灵活调节模型配置，并同步调整 GPU 数量，一般需8卡以上, 多机多卡训练可获得更好性能

网络要求：支持 NCCL 通信

## 2.2. 环境要求

操作系统: Ubuntu 20.04/22.04 LTS

CUDA: 12.9

cuDNN: 8.9.7+

NCCL: 2.18.3+

Python: 3.10

推荐使用官方镜像。

# 3. 启动训练

## 3.1. 数据准备
为了方便用户运行测试本模型，本项目提供了处理好的100k 条 doc 的训练样本。将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```shell
# Download dsv3 model data
mkdir -p data
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/ds3_industrycorpus2_94k.bin
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/ds3_industrycorpus2_94k.idx
```
你也可以从文本自行制作数据集，参考[数据集格式说明](../../../docs/zh/dataset_format.md)

## 3.2 启动训练
PaddleFormers repo 的最佳实践示例中为您提供了一个层数缩减为29层、专家数为8的示例，8卡即可训练：

```shell
# PaddleFormers/examples/best_practices/DeepseekV3_Pretrain/
# 可参考 run.sh、train_gpu.sh
paddleformers-cli train ./config/pretrain_argument.yaml
```
若您需要体验671B 完整规模的训练，需要至少 32 机 256 卡的分布式训练：

```shell
NNODES={num_nodes} \
MASTER_ADDR={your_master_addr} \
MASTER_PORT={your_master_port} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli train ./config/pretrain_argument.yaml # 注意：仓库中为您提供的示例为缩规模型
```
注意事项：

* 以上单机多机配置需每卡至少 80G 显存，配置中默认开启`offload_optim`，会对性能造成影
* 如您需要热启 DeepSeek 官方在 huggingface 开源的 checkpoint，您可在 pretrain_argument.yaml 中通过配置 resume_from_huggingface_ckpt 参数，指定您的 checkpoint 路径即可
* 更详细的分布式启动命令请参考[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/3.3/api/paddle/distributed/launch_cn.html)。
* 运行命令前请参考下面环境变量进行设置：

```shell
export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=162
export NVSHMEM_BOOTSTRAP=UID
unset NVSHMEM_HCA_LIST
unset NVSHMEM_ENABLE_NIC_PE_MAPPING
# Flags for allocator
export FLAGS_large_pool_auto_growth_chunk_size_in_mb=128
export FLAGS_small_pool_auto_growth_chunk_size_in_mb=10
export FLAGS_small_pool_size_in_mb=1
export FLAGS_share_tensor_for_grad_tensor_holder=1
export FLAGS_use_default_stream=false
export USE_DS_GEMM=false
```

# 4. 注意事项

## 4.1. 部分参数释义

|名称|影响范围|算子层面|定义位置|
|-|-|-|-|
|dsv3_use_fp8_gemm|moe_layer.py: 决定是否在 forward_flex_token 中进入 FP8的 FusionMoe 模块 modeling_pp.py:  在 build_overlapped_nodes 中决定 overlap_element_class 是否采用 FusionFp8DecoderLayerNode; 在 OverlapedScheduleChunk 中决定是否开启 use_fusion; 在 build_schedule_node 中决定开启 fp8独特的 moe、post_process、decoder node modeling.py:  决定 Linear 是普通线性层还是 FP8Linear。决定 DeepseekV2MLPClass 是 FP8Mlp 还是普通的 DeepseekV2MLP|影响较广，非单算子|[config.json](./pretrain/config/config.json)|
|dsv3_use_fp8_dispatch|moe_layer.py:  在 forward_flex_token 中决定是否进行 pre_dispatch; 在 Fp8DispatchQuantNode::forward 中决定是否在 pre_dispatch 前先进行1x128 quant; 在 Fp8CombineQuantNode 的 backward 中，增加额外多的多流等待机制，用于接收 fp8 combine 的 grad 和其 scale; 在 FusionMlpNode 的 forward 中决定 subbatch 策略的开启; 在 FusionMoeNode 的前反向中，mlp 前后决定是否进行 dispatch_quant modeling_pp.py:  决定 combine_backward_wait_event 是 quant_event 还是 previous_event|影响较广，非单算子|[config.json](./pretrain/config/config.json)|
