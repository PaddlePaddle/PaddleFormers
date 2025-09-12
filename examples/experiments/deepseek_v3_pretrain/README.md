# DeepSeek-V3 预训练模型使用指南

## 1. 硬件资源要求

### 最低配置

GPU: NVIDIA H100 80GB (推荐) 或 H800、H20等

数量: 可根据配置调整 GPU 数量，一般需8卡以上, 多机多卡训练可获得更好性能

网络要求：支持 NCCL 通信

### 环境要求

操作系统: Ubuntu 20.04/22.04 LTS

CUDA: 12.9

cuDNN: 8.9.7+

NCCL: 2.18.3+

Python: 3.10

推荐使用虚拟环境:

```shell
python3.10 -m venv deepseek_env
source deepseek_env/bin/activate
```

### 安装 PaddlePaddle
 python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

 （其他版本安装方式可参考[官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)）

### 安装其他依赖
pip install -r requirements.txt

## 2. 数据准备

PaddleFormers 将 DeepSeek-V3 的预训练加速版本模型加入到 examples 中， 用户只需修改一系列配置即可使用不同的单点策略，达到 DeepSeek-V3 高性能训练目的。更多模型支持持续更新中。


为了方便用户运行测试本模型，本项目提供了处理好的100k 条 doc 的训练样本。将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：


```shell
# Download llama model data
mkdir -p data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
```

单机8卡训练:

```shell
# 单机 8 卡可训练、29 层实验版本
# 可参考 run.sh、train_gpu.sh
python -u run_pretrain.py ./config/pretrain_argument.json
```

高性能、多卡、多机训练:

```shell
# 多卡模型预训练参考:
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_pretrain.py ./config/pretrain_argument.json
# 多机训练参考: 需 32 机 256 卡进行训练，完整规模版本
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7"  --master=<master_ip>:<port> --nnodes=256  run_pretrain.py ./config/pretrain_argument.json
```

- 注：以上单机多机配置需每卡至少 80G 显存，配置中默认开启`offload_optim`，会对性能造成影
- 更详细的分布式启动命令请参考[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/paddle/distributed/launch_cn.html#launch)。

## 3. 注意事项

### 部分参数释义

|名称|影响范围|算子层面|定义位置|
|-|-|-|-|
|dsv3_use_fp8_gemm|moe_layer.py: 决定是否在 forward_flex_token 中进入 FP8的 FusionMoe 模块 <br>modeling_pp.py:  在 build_overlapped_nodes 中决定 overlap_element_class 是否采用 FusionFp8DecoderLayerNode; 在 OverlapedScheduleChunk 中决定是否开启 use_fusion; 在 build_schedule_node 中决定开启 fp8独特的 moe、post_process、decoder node <br>modeling.py:  决定 Linear 是普通线性层还是 FP8Linear。决定 DeepseekV2MLPClass 是 FP8Mlp 还是普通的 DeepseekV2MLP|影响较广，非单算子|examples/experiments/deepseek_v3_pretrain/config/config.json|
|dsv3_use_fp8_dispatch|moe_layer.py:  在 forward_flex_token 中决定是否进行 pre_dispatch; 在 Fp8DispatchQuantNode::forward 中决定是否在 pre_dispatch 前先进行1x128 quant; 在 Fp8CombineQuantNode 的 backward 中，增加额外多的多流等待机制，用于接收 fp8 combine 的 grad 和其 scale; 在 FusionMlpNode 的 forward 中决定 subbatch 策略的开启; 在 FusionMoeNode 的前反向中，mlp 前后决定是否进行 dispatch_quant <br>modeling_pp.py:  决定 combine_backward_wait_event 是 quant_event 还是 previous_event|影响较广，非单算子|examples/experiments/deepseek_v3_pretrain/config/config.json|
|use_ds_gemm|fp8_utils.py:  决定在环境中 import deep_gemm（use_ds_gemm=true）还是使用框架内算子|deep_gemm|examples/experiments/deepseek_v3_pretrain/config/config.json|
|reorder_pipeline_priority|training_args.py:  在 order 顺序里把 sharding 和 pp 的优先级提前|无|examples/experiments/deepseek_v3_pretrain/config/pretrain_argument.json|

