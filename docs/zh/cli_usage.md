# 1. 命令行界面

## 1.1. 概述

PaddleFormers CLI（Command Line Interface）提供了基于终端的程序交互，通过配置文件来管理各类参数，高效灵活地执行模型训练、推理和评估任务。

## 1.2. 快速入门

**安装**

参考[README 文档](../../README.md)进行 paddleformers 安装

验证安装：

```shell
paddleformers-cli help
```

预期输出：

```shell
------------------------------------------------------------
| Usage:                                                    |
|   paddleformers-cli train : model finetuning              |
|   paddleformers-cli export : model export                 |
|   paddleformers-cli help: show helping info               |
------------------------------------------------------------
```

**AI 计算卡配置**

默认情况下，CLI 中使用所有可用的 AI 计算卡。
如果您想指定特定的计算卡，请在运行 CLI 之前设置对应的环境变量。

对于英伟达 GPU 或者天数智芯计算卡，通过以下环境变量进行设置：

```shell
# Single GPU / Iluvatar GPU
export CUDA_VISIBLE_DEVICES=0
# Multi GPUs / Iluvatar GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

对于昆仑芯计算卡，通过以下环境变量进行设置：

```shell
# Single XPU
export XPU_VISIBLE_DEVICES=0
# Multi XPUs
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 1.3. CLI 具体用法

以下环节使用 **Qwen/Qwen3-0.6B-Base** 模型进行演示

### 1.3.1. 模型预训练

```shell
# Example 1: PT-Full using online dataset
paddleformers-cli train examples/config/pt/full.yaml
# Example 2: PT-Full using offline dataset
paddleformers-cli train examples/config/pt/full_offline_data.yaml
```

### 1.3.2. SFT 和 LoRA 微调

```shell
# Example 1: SFT
paddleformers-cli train examples/config/sft/lora.yaml
# Example 2: SFT-Full
paddleformers-cli train examples/config/sft/full.yaml
```

### 1.3.3. DPO 和 LoRA 微调

```shell
# Example 1: 8K seq length, DPO
paddleformers-cli train examples/config/dpo/full.yaml
# Example 2: 8K seq length, DPO-LoRA
paddleformers-cli train examples/config/dpo/lora.yaml
```

### 1.3.4. 模型导出

```shell
paddleformers-cli export examples/config/run_export.yaml
```

### 1.3.5. 多节点训练

#### 方式一

```shell
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train examples/config/sft_full.yaml
```

#### 方式二 (mpirun)

先写一个脚本，例如`scripts/train_96_gpus.sh`，内容为：

```shell
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train examples/config/sft_full.yaml
```

然后：

```shell
mpirun bash scripts/train_96_gpus.sh
```

## 1.4. 参数传递

paddleformers-cli 支持在输入命令中传入参数用于覆盖配置文件中的内容，具体的用法如下：

```shell
paddleformers-cli train examples/config/sft/lora.yaml key1=value key2=value2

# 示例 修改模型名称和LoRA配置
paddleformers-cli train examples/config/sft/lora.yaml model_name_or_path=./models/Qwen3-0.6B lora_rank=8
```
