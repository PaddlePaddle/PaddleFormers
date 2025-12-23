# 命令行界面

## 概述

CLI（Command Line Interface）提供基于终端的程序交互，通过参数化配置，高效灵活地执行模型训练、推理和评估任务。

## 快速入门

**安装**

参考[readme文档](../../README.md)进行paddleformers安装

验证安装：
```bash
paddleformers-cli help
```

预期输出：
```
------------------------------------------------------------
| Usage:                                                    |
|   paddleformers-cli train : model finetuning              |
|   paddleformers-cli export : model export                 |
|   paddleformers-cli help: show helping info               |
------------------------------------------------------------
```

**GPU 配置**

默认情况下，CLI 中使用所有可用的 GPU。
如果您想指定某些 GPU，请在运行 CLI 之前设置 CUDA_VISIBLE_DEVICES：

```bash
# Single GPU / Iluvatar GPU
export CUDA_VISIBLE_DEVICES=0
# Multi GPUs / Iluvatar GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Single XPU
export XPU_VISIBLE_DEVICES=0
# Multi XPUs
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

**代理配置**

```bash
export HTTPS_PROXY={your_proxy}
export HTTP_PROXY={your_proxy}
```

## CLI 具体用法

使用 **Qwen/Qwen3-0.6B-Base** 模型的示例：

### 1. 模型预训练

```bash
# Example 1: PT-Full using online dataset
paddleformers-cli train examples/config/pt/full.yaml
# Example 2: PT-Full using offline dataset
paddleformers-cli train examples/config/pt/full_offline_data.yaml
```

### 2. 模型微调

#### 2.1. SFT 和 LoRA 微调
```bash
# Example 1: SFT
paddleformers-cli train examples/config/sft/lora.yaml
# Example 2: SFT-Full
paddleformers-cli train examples/config/sft/full.yaml
```

#### 2.2. DPO 和 LoRA 微调
```bash
# Example 1: 8K seq length, DPO
paddleformers-cli train examples/config/dpo/full.yaml
# Example 2: 8K seq length, DPO-LoRA
paddleformers-cli train examples/config/dpo/lora.yaml
```

### 3. 模型导出
```bash
paddleformers-cli export examples/config/run_export.yaml
```

### 4. 多节点训练

#### 4.1. 方式一

```bash
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train examples/config/sft_full.yaml
```

#### 4.2. 方式二 (mpirun)

先写一个脚本，例如`scripts/train_96_gpus.sh`，内容为：
```bash
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train examples/config/sft_full.yaml
```

然后：
```bash
mpirun bash scripts/train_96_gpus.sh
```

**Note**:
* cli支持命令行传参，如：
```bash
paddleformers-cli train examples/config/sft/lora.yaml model_name_or_path=./models/Qwen3-0.6B lora_rank=8
```
