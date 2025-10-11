# CLI

## Overview

CLI (Command Line Interface) provides terminal-based interaction with the program, enabling efficient and flexible execution of model training, inference, and evaluation tasks through parameterized configurations.

## Quick Start

**Installation**

Run in the erniekit root directory:
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
| Usage:                                                     |
|   erniekit train -h: model finetuning                      |
|   erniekit export -h: model export                         |
|   erniekit split -h: model split                           |
|   erniekit eval -h: model evaluation                       |
|   erniekit server -h: model deployment                     |
|   erniekit chat -h: launch a chat interface in CLI         |
|   erniekit webui -h: launch webui                          |
|   erniekit version: show version info                      |
|   erniekit help: show helping info                         |
------------------------------------------------------------
```

**GPU Configuration**

By default, all available gpus are used in CLI/WebUI.
If you wan to specify certain gpus, please set CUDA_VISIBLE_DEVICES before running CLI/WebUI:

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

Examples using **ERNIE-4.5-0.3B** model:

## 1.1. Chat
```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# Load model and start service
erniekit server examples/configs/ERNIE-4.5-0.3B/run_chat.yaml
# Launch CLI chat interface
erniekit chat examples/configs/ERNIE-4.5-0.3B/run_chat.yaml
```

* Note: the command-line dialogue for VL-model only supports pure text input.

## 1.2. Model Fine-tuning

### 1.2.1. SFT & LoRA Fine-tuning
```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# Example 1: 8K seq length, SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_8k.yaml
# Example 2: 32K seq length, SFT
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_32k.yaml
# Example 3: 8K seq length, SFT-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_lora_8k.yaml
# Example 4: 32K seq length, SFT-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/sft/run_sft_lora_32k.yaml
```

### 1.2.2. DPO & LoRA Fine-tuning
```bash
# download model from huggingface
huggingface-cli download baidu/ERNIE-4.5-0.3B-Paddle --local-dir baidu/ERNIE-4.5-0.3B-Paddle
# Example 1: 8K seq length, DPO
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_8k.yaml
# Example 2: 32K seq length, DPO
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_32k.yaml
# Example 3: 8K seq length, DPO-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_lora_8k.yaml
# Example 4: 32K seq length, DPO-LoRA
erniekit train examples/configs/ERNIE-4.5-0.3B/dpo/run_dpo_lora_32k.yaml
```

## 1.3. Model Evaluation
```bash
erniekit eval examples/configs/ERNIE-4.5-0.3B/run_eval.yaml
```

## 1.4. Model Export
```bash
erniekit export examples/configs/ERNIE-4.5-0.3B/run_export.yaml
```

## 1.5. Multi-Node Training
```bash
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 erniekit train examples/configs/ERNIE-4.5-300B-A47B/sft/run_sft_lora_8k.yaml
```
