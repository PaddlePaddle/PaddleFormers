# 1. 背景说明

PaddleFormers 提供了 ERNIE-4.5 的预训练加速版本模型，当前支持 [ERNIE-4.5-21B-A3B](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Base-PT) 和 [ERNIE-4.5-300B-A47B](https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-Base-PT) 两个文本模型。基于 PaddlePaddle 框架在 ERNIE‑4.5 上的实战经验，我们把[《ERNIE 4.5 Technical Report》](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)中的一整套高效训练优化能力做成了 **“开箱即用”** 的配置，包括异构混合并行、FP8 混合精度训练、精细化重计算以及算子融合等。这套优化方案在 2016 张 NVIDIA H800 GPU 上预训练 ERNIE-4.5-300B-A47B 模型的 Model FLOPs Utilization（MFU）可达 47%，在同等算力下显著提升训练吞吐。现在，用户只需在 PaddleFormers 中调整少量配置，就可以在从 8 卡到上千卡的不同集群规模上复用这些高效策略，低门槛获得接近论文同级别的高性能 ERNIE‑4.5 预训练能力。更多模型与策略支持持续更新中。

# 2. 硬件资源要求

## 2.1. 最低配置

GPU：需要 NVIDIA Hopper 架构的 GPU，如 H100 80GB (推荐) 或 H800、H20 等

数量：可根据配置调整 GPU 数量，一般 21B 模型需要至少 8 卡，300B 模型需要至少 96 卡，更多卡数训练可以获得更好性能

网络要求：支持 NCCL 通信

## 2.2. 环境要求

操作系统：Ubuntu 20.04/22.04 LTS

CUDA: 12.9

cuDNN: 8.9.7+

NCCL: 2.18.3+

Python: 3.10

推荐使用官方镜像。

# 3. 运行训练

## 3.1. 权重下载

我们提供了 21B 和 300B 模型的预训练权重，如果你需要基于权重进行后预训练，可以从以下地址下载权重；如果你是从头开始预训练，则不需要下载权重

* 21B：[https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Base-PT](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Base-PT)
* 300B：[https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-Base-PT](https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-Base-PT)

下载权重后，由于权重自带的`config.json`是针对推理配置的，需要替换为 PaddleFormers 提供的训练用配置，参考命令如下：

```shell
# 下载训用配置
wget https://paddleformers.bj.bcebos.com/models/eb45_model_configs.tar.gz
tar xf eb45_model_configs.tar.gz

# 21B
mv ERNIE-4.5-21B-A3B-Base-PT/config.json ERNIE-4.5-21B-A3B-Base-PT/config.json.bak
cp model_configs/ERNIE-4p5-21B-A3B/model_config.json ERNIE-4.5-21B-A3B-Base-PT/config.json

# 300B
mv ERNIE-4.5-300B-A47B-Base-PT/config.json ERNIE-4.5-300B-A47B-Base-PT/config.json.bak
cp model_configs/ERNIE-4p5-300B-A47B/model_config.json ERNIE-4.5-300B-A47B-Base-PT/config.json
```

## 3.2. 数据准备

为了方便用户运行测试本模型，本项目提供了处理好的 94k 条 doc 的训练样本。将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```shell
mkdir -p data
wget -P data https://paddleformers.bj.bcebos.com/datasets/release/v1.0/eb45_industrycorpus2_94k.bin
wget -P data https://paddleformers.bj.bcebos.com/datasets/release/v1.0/eb45_industrycorpus2_94k.idx
```

你也可以从文本自行制作数据集，参考[数据集格式说明](../../../docs/zh/dataset_format.md)

另外，训练时需要指定 tokenizer 文件的路径，您可以指定 PaddleFormers 目录下的：

```yaml
tokenizer_name_or_path: examples/experiments/ernie_pretrain/ernie/src/tokenizers/tokenizer_model
```

或从源上下载：

```shell
wget https://paddleformers.bj.bcebos.com/models/eb45_tokenizer_model.tar.gz
tar xf eb45_tokenizer_model.tar.gz
```
然后指定配置为：

```yaml
tokenizer_name_or_path: tokenizer_model
```

## 3.3. 配置修改

如果你需要从头开始预训练，请对相应模型的 yaml 配置中的如下参数进行修改：

```yaml
model_name_or_path: model_configs/ERNIE-4p5-21B-A3B/
input_dir: "1.0 ./data/eb45_industrycorpus2_94k"  # 指定上述下载的数据集的路径的前缀（省略.bin和.idx）
from_scratch: 1  # 0表示从下载的权重加载参数，1则是随机初始化参数；后预训练应当设为0，从头训练则设为1
continue_training: False
```

如果你需要基于上述下载的权重进行后预训练，请对相应模型的 yaml 配置中的如下参数进行修改：

```yaml
model_name_or_path: /path/to/your/ERNIE-4.5-21B-A3B-Base-PT  # 指定上述下载的模型的路径
input_dir: "1.0 ./data/eb45_industrycorpus2_94k"  # 指定上述下载的数据集的路径的前缀（省略.bin和.idx）
from_scratch: 0  # 0表示从下载的权重加载参数，1则是随机初始化参数；后预训练应当设为0，从头训练则设为1
```

不同模型对应的 yaml 配置分别为：

* 21B：`examples/config/pt/eb45_pretrain/21b_8_gpus.yaml`
* 300B：`examples/config/pt/eb45_pretrain/300b_96gpus.yaml`

## 3.4. 启动训练

单机 8 卡训练 21B 模型：

```shell
paddleformers-cli train examples/config/pt/eb45_pretrain/21b_8_gpus.yaml
```
多机、多卡训练 300B 模型：

```shell
NNODES={num_nodes} \
MASTER_ADDR={your_master_addr} \
MASTER_PORT={your_master_port} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli train examples/config/pt/eb45_pretrain/300b_96gpus.yaml
```
注意事项：

* 运行命令前请参考下面环境变量进行设置：

```shell
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
unset GLOG_vmodule GLOG_v
export PADDLE_DISABLE_CUDNN_FA=1
export FLAGS_use_auto_growth_pinned_allocator=True
export FLAGS_pipeline_nccl_comm_init_option=1
export FLAGS_sharding_v2_check_zero_padding=1
export FLAGS_use_paddle_recall_error=0
export FLAGS_tcp_max_syn_backlog=16384
export FLAGS_call_stack_level=2

SM=`nvidia-smi --query-gpu=compute_cap --format=csv | tail -n 1 | sed 's/\.//g'`
if [ $SM -eq 90 ]
then
    export FLAGS_flash_attn_version=3
else
    export FLAGS_flash_attn_version=2
fi

export PYTHONPATH=$PYTHONPATH:./ernie
```

* 以上单机多机配置需每卡至少 80G 显存，配置中默认开启 tensorwise_offload_optimizer，会对性能造成影响
* 更详细的分布式启动命令请参考[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/3.3/api/paddle/distributed/launch_cn.html)

## 3.5. 权重存取

**说明：** ERNIE-4.5 的权重有两种格式，ShardingIO 和 UnifiedCheckpoint。ShardingIO 格式会同时保存模型参数、优化器状态和 Dataloader 顺序等，可以在断点重训时实现 loss 的接续，但仅可用于训练程序，无法用于推理或其他框架。UnifiedCheckpoint 格式即通用格式，只记录了模型参数，可用于推理和其他框架。

默认情况下，训练过程中每`save_steps`步会在`./output/`文件夹（可由 yaml 配置）下保存一个 checkpoint 文件夹，例如`./output/checkpoint-100`、`./output/checkpoint-200`，以此类推；此类文件夹中存储的是 ShardingIO 格式的权重，仅可用于进行断点重训。完成全部`max_steps`步训练后，会在`./output/`根目录下保存一份 UnifiedCheckpoint 格式的最终权重，你可以将整个 output 文件夹打包，用于推理等其他用途。

#### 断点重训

假设你在训练 100 步后中断了训练，现在需要从 100 步开始继续训练，则在 yaml 配置中指定如下参数，然后启动训练即可：

```yaml
resume_from_checkpoint: ./output/checkpoint-100  # 指定要断点重训的checkpoint的路径
```

**注意：** 断点重训时`from_scratch`参数必须与启动训练时一致，否则 loss 不接续。

#### 权重导出

默认情况下，程序只会导出训练`max_steps`步的模型参数，如果你想导出中间某个 checkpoint 的模型参数，则在 yaml 配置中指定如下参数：

```yaml
resume_from_checkpoint: ./output/checkpoint-500  # 指定要导出的checkpoint的路径
max_steps: 1  # 设为1（或任何小于已训练步数的值）表示不进行训练，仅加载参数，然后立即导出，不会对参数进行任何更新
```

然后按正常训练的方式启动，程序将在加载 checkpoint 后立即导出权重，不会发生实际的训练。

**注意：** output 下原有的导出结果会被覆盖，请提前保存，或在 yaml 中指定一个新的`output_dir`位置。

# 4. 注意事项

你可以在 yaml 中修改以下参数以适配不同的环境和获得不同的性能

|名称|简介|影响效果|
|-|-|-|
|use_fp8_mlpuse_fp8_fuse_node|是否使用 FP8 算子（这两个参数需要同时设为 true 或 false）|FP8 算子通过使用低精度运算有效提高性能，默认为 true；但部分环境可能存在 FP8 兼容性问题，如果遇到 FP8 算子报错，可以将这两个参数设置为 false|
|tensorwise_offload_optimizer|是否将优化器参数卸载到 CPU 上|设为 true 会将优化器状态逐张量卸载到 CPU 上，仅在每个参数更新时装载回 GPU，可以节约显存，但会产生拷贝开销，默认为 false；当你使用足够多卡数或足够大的显存时，可以设为 false 以提高性能|
|use_recompute|是否使用重计算|重计算可以释放前向的部分中间结果并在反向时重新计算，从而节省显存，但会增加反向开销，默认为 false；如果你的显存预算较紧张，可以设为 true|
