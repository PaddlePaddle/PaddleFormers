# 命令行界面

## 概述

CLI（Command Line Interface）提供基于终端的程序交互，通过参数化配置，高效灵活地执行模型训练、推理和评估任务。

## 快速入门

**安装**

在PaddleFormers根目录下运行：
```bash
python -m pip install -e .
```

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

**GPU配置**

默认情况下，CLI 中使用所有可用的 GPU。
如果您想指定某些 GPU，请在运行 CLI 之前设置 CUDA_VISIBLE_DEVICES：

```bash
# Single GPU
export CUDA_VISIBLE_DEVICES=0
# Multi GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Single XPU
export XPU_VISIBLE_DEVICES=0
# Multi XPUs
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Single NPU
export ASCEND_RT_VISIBLE_DEVICES=0
# Multi NPUs
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

* 注：在`Chat`模块，CUDA_VISIBLE_DEVICES配置的GPU数量应该等于`tensor_parallel_degree`在配置中。
或者，您也可以取消设置 CUDA_VISIBLE_DEVICES。

# 1. CLI 用法

使用 **Qwen/Qwen3-0.6B-Base** 模型的示例：

## 1.1.聊天
待补充

## 1.2.模型预训练

```bash
# Example 1: SFT-Full using online dataset
paddleformers-cli train examples/config/pt/full.yaml
# Example 2: SFT-Full using offline dataset
paddleformers-cli train examples/config/pt/full_offline_data.yaml
```

## 1.3.模型微调

### 1.3.1. SFT 和 LoRA 微调
```bash
# Example 1: SFT
paddleformers-cli train examples/config/sft/lora.yaml
# Example 2: SFT-Full
paddleformers-cli train examples/config/sft/full.yaml
```

### 1.3.2. DPO 和 LoRA 微调
```bash
# Example 1: 8K seq length, DPO
paddleformers-cli train examples/config/dpo/full.yaml
# Example 2: 8K seq length, DPO-LoRA
paddleformers-cli train examples/config/dpo/lora.yaml
```

## 1.4.模型评估
待补充

## 1.5.模型导出
```bash
paddleformers-cli export examples/config/run_export.yaml
```

## 1.6.多节点训练
```bash
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train examples/config/sft_full.yaml
```
