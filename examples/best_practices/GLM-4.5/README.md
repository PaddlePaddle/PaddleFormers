# 1. 背景说明

### 1.1. 单步 MTP（Single-Step MTP）
单步 MTP 是模型在单次前向传播中，仅预测**未来1个 token**的多 token 预测模式。
- 本质：在传统单 token 预测基础上，新增独立 MTP 输出头，仅学习单步预测能力。

<div align="center">
   <img width="1634" height="880" alt="image" src="https://github.com/user-attachments/assets/9ec83d65-5423-4ecc-9b6e-5a8ef1b0cdc3" />
</div>
>图片引自：[DeepSeek V3技术报告](https://arxiv.org/pdf/2412.19437)

### 1.2. 多步 MTP（Multi-Step MTP）
多步 MTP 是模型在单次前向传播中，**只使用一层 MTP 权重, 递归预测未来 N 个 token**（如3步）的模式，通过级联 MTP 模块实现因果预测。
- 训练流程：训练基于单步 MTP 或无 MTP 模型热启，快速拓展多步 MTP 能力。
- 核心机制：第1步 MTP 预测 t+1，第2步基于第1步输出预测 t+2，第3步基于第2步输出预测 t+3，保持序列逻辑连贯。
- 推理适配：直接支持**N 步投机解码**，提升接受率与生成速度。
- 训练收益：投机解码从0/1步预测变为 N 步预测，平均接受长度显著提升。

### 1.3. 单步 vs 多步 MTP 核心对比
| 维度 | 单步 MTP | 多步 MTP |
|------|----------|----------|
| 预测步数 | 1步 | N 步（如3步） |
| 模块结构 | 单个 MTP 输出头 | 单个 MTP 输出头 |
| 推理支持 | 仅1步投机 | N 步投机解码 |

PaddleFormers 提供冻结主干的 MTP 权重训练，可以基于无 MTP（或只有一层 MTP 权重）的模型进行进 N 步 MTP 能力训练，让模型具有多步 MTP 能力。


# 2. 硬件配置要求

## 2.1. 最低配置

GPU: NVIDIA H100/A100 80GB (推荐)

数量: 如果基于 GLM-4.5-Air 模型(103B)进行训练，需要最少32卡（4机），如果需要进行128k 长文训练则需要8机（64卡）

网络要求：支持 NCCL 通信

## 2.2. 环境要求

操作系统: Ubuntu 20.04/22.04 LTS

CUDA: 12.9

cuDNN: 8.9.7+

NCCL: 2.18.3+

Python: 3.10

推荐使用官方镜像。

# 3. 相关参数

- 是否冻结主干

在需要拓展模型多步 MTP 能力，同时保持主干模型原有能力不变的场景下，建议开启此开关。

```yaml
# 是否冻结主干权重（MTP-only训练必开）
train_mtp_only: true
```

- 单步/多步训练指定

```yaml
# MTP训练步数，填1则为单步，填3则为3步
mtp_num_layers: 3
```

# 4. 实例训练
以下为 8 机环境下，冻结主干网络、训练 3 步 MTP 的完整参数配置，可直接参考使用：

```yaml
### data
train_dataset_type: erniekit
eval_dataset_type: erniekit
train_dataset_path: ./tests/fixtures/dummy/sft/train.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./tests/fixtures/dummy/sft/eval.jsonl
eval_dataset_prob: "1.0"

max_seq_len: 65536
packing: true
use_template: false
random_shuffle: true
mix_strategy: concat
truncate_packing: false
padding_free: true

### model
model_name_or_path: GLM-4.5-Air
_attn_implementation: flashmask

### finetuning
# base
stage: SFT
fine_tuning: full
seed: 23
do_train: true
do_eval: false
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 1
max_steps: 1100
eval_iters: 1000
eval_steps: 1000000
evaluation_strategy: steps
save_steps: 50000
save_hf_steps: 100
save_strategy: steps
save_total_limit: 3
logging_steps: 1
gradient_accumulation_steps: 32
logging_dir: ./vdl_log
output_dir: ./output
disable_tqdm: true
eval_accumulation_steps: 16

# train
warmup_steps: 110
learning_rate: 1.0e-5
weight decay: 0.1


# performance
context_parallel_size: 1
tensor_model_parallel_size: 8
expert_model_parallel_size: 16
sequence_parallel: true
use_expert_parallel: true
pipeline_model_parallel_size: 4
num_empty_layers_add_in_head: 0
num_empty_layers_add_in_tail: 14

# performance
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1


moe_expert_fusion: false
moe_deep_gemm: false
moe_shared_expert_overlap: true

apply_rope_fusion: true
fuse_rms_norm: true
moe_router_force_load_balancing: false
router_aux_loss_coef: 0.0001

split_param: true
sharding: stage1
stage1_overlap: true

# pp
pp_delay_scale_loss: true
overlap_p2p_comm: true
variable_seq_lengths: true
best_unbalanced_scheduler: true
pp_release_grads: true

tp_delay_scale_loss: True

amp_master_grad: true
bf16: true
fp16_opt_level: O2
fa_version: 3

save_checkpoint_format: flex_checkpoint
load_checkpoint_format: flex_checkpoint
dataloader_shuffle: false

dataloader_num_workers: 8
prefetch_factor: 2

fp32_residual_connection: false
tensorwise_offload_optimizer: true

mtp_loss_scaling_factor: 0.1
num_nextn_predict_layers: 1
mtp_distillation_loss: false
mtp_num_layers: 1
train_mtp_only: true
```

配套启动脚本如下

```shell
NNODES={num_nodes} MASTER_ADDR={your_master_addr} MASTER_PORT={your_master_port} RANK={rank} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train mtp_depth_3.yaml
```


# 5. 多步 MTP 推理部署（FastDeploy）
```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model checkpoint/ \  # 多步MTP训练后模型
  --port 8390 \
  --engine-worker-queue-port 8392 \
  --cache-queue-port 8393 \
  --metrics-port 8394 \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  # 多步MTP核心推理配置
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "num_model_steps": 3,"model": "checkpoint/"}' \
  --max-num-seqs 32
```
- 关键参数：
  - `num_speculative_tokens: 3`：投机生成3个 token
  - `num_model_steps: 3`：匹配训练时的3步 MTP，实现训推一致
