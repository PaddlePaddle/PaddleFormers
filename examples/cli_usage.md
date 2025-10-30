# CLI

## Overview

CLI (Command Line Interface) provides terminal-based interaction with the program, enabling efficient and flexible execution of model training, inference, and evaluation tasks through parameterized configurations.

## Quick Start

**Installation**

Run in the PaddleFormers root directory:
```bash
python -m pip install -e .
```

Verify installation:
```bash
paddleformers-cli help
```

Expected output:
```
------------------------------------------------------------
| Usage:                                                    |
|   paddleformers-cli train : model finetuning              |
|   paddleformers-cli export : model export                 |
|   paddleformers-cli help: show helping info               |
------------------------------------------------------------
```

**GPU Configuration**

By default, all available gpus are used in CLI.
If you wan to specify certain gpus, please set CUDA_VISIBLE_DEVICES before running CLI:

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

* Note: In `Chat` module, the number of gpus configured by CUDA_VISIBLE_DEVICES should be equal to `tensor_parallel_degree` in the config.
Alternatively, you can also unset CUDA_VISIBLE_DEVICES.

# 1. CLI Usage

Examples using **Qwen/Qwen3-0.6B-Base** model:

## 1.1. Chat
待补充

## 1.2. Model Pre-training

```bash
# Example 1: SFT-Full using online dataset
paddleformers-cli train examples/config/pt/full.yaml
# Example 2: SFT-Full using offline dataset
paddleformers-cli train examples/config/pt/full_offline_data.yaml
```

## 1.3. Model Fine-tuning

### 1.3.1. SFT & LoRA Fine-tuning
```bash
# Example 1: SFT
paddleformers-cli train examples/config/sft/lora.yaml
# Example 2: SFT-Full
paddleformers-cli train examples/config/sft/full.yaml
```

### 1.3.2. DPO & LoRA Fine-tuning
```bash
# Example 1: 8K seq length, DPO
paddleformers-cli train examples/config/dpo/full.yaml
# Example 2: 8K seq length, DPO-LoRA
paddleformers-cli train examples/config/dpo/lora.yaml
```

## 1.4. Model Eval
待补充

## 1.5. Model Export
```bash
paddleformers-cli export examples/config/run_export.yaml
```

## 1.6. Multi-Node Training
```bash
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train examples/config/sft_full.yaml
```
