# 1. 背景说明

大模型的训练流程通常包含四个关键阶段：

1. 预训练（Pre-training）：通过海量无标注数据进行预训练，学习语言能力和世界知识，构建通用基座；
2. 后预训练（Post-Pre-training）：通过注入特定领域知识以增强专业性，构建领域基座模型；
3. 有监督微调（SFT）：利用高质量指令数据进行微调，模型具备对话与任务处理能力；
4. **偏好对齐（DPO/RLHF）：采用偏好优化（DPO/RLHF）技术，基于成对的优劣反馈数据对模型进行价值观对齐，从而输出符合人类预期的最终版本。（本文讲解 DPO）**

本文旨在说明如何基于 PaddleFormers 进行模型的偏好对齐。

# 2. 数据准备

为了方便演示，我们提供一个 demo 数据，执行下载并解压。如果想要使用自己的数据进行训练，请参考[数据集格式说明](./dataset_format.md)进行数据的准备。

```shell
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/dpo_online_data_messages.tar.gz
mkdir -p data/dpo && tar -xf dpo_online_data_messages.tar.gz -C data/dpo/
```

**demo 数据格式：** messages 格式，每条数据都是一个字典。如下字段：

* `messages`：对话内容，最后一轮为用户指令。
* `chosen_response`：对应用户指令的回复的 **优势答案** 。
* `rejected_response`：对应用户指令的回复的 **劣势答案** 。
    * `role`：角色，user 为用户 Query，assistant 为模型的回复。
    * `content`：文本内容。

```json
{
    "messages": [
        {
            "role": "user",
            "content": "What kind of blankets does a elepahant like to eat?"
        }
    ],
    "chosen_response": [
        {
            "role": "assistant",
            "content": "Elephants are herbivores and their diet consists mainly of leaves, fruits, bark, and roots. They do not eat blankets."
        }
    ],
    "rejected_response": [
        {
            "role": "assistant",
            "content": "Elephants are herbivores and prefer to eat leaves, grass, bark and fruits of woody plants and tree species. They also eat small animals, roots, stems and some artificial fodders like hay. They often prefer to eat three or four times a day, and generally take only a few bites from an area before moving onto another part of the food source."
        }
    ]
}
```

# 3. 训练

DPO 支持全量训练和 LoRA 训练：

* 全量 DPO 更新全部参数，适合数据算力充足、追求极致效果的场景。

* LoRA 仅训练低秩矩阵，适配资源受限、需快速迁移的任务。

## 3.1. 超参数配置

训练配置推荐使用 yaml 格式文件，如下示例：

#### 全量训练配置

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./data/dpo/train_messages.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./data/dpo/eval_messages.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 8192
packing: true
mix_strategy: concat

### model
model_name_or_path: baidu/ERNIE-4.5-0.3B-PT
_attn_implementation: flashmask

### finetuning
# base
stage: DPO
fine_tuning: full
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 1
max_steps: -1
eval_steps: 100
evaluation_strategy: steps
save_steps: 100
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 4
logging_dir: ./vdl_log
output_dir: ./checkpoints/ernie45-dpo-full
disable_tqdm: true
eval_accumulation_steps: 16
model_with_dpo_criterion: true

# train
warmup_steps: 20
learning_rate: 1.0e-6

# performance
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage2
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2
save_checkpoint_format: flex_checkpoint
load_checkpoint_format: flex_checkpoint
```

#### lora 训练配置

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./data/dpo/train_messages.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./data/dpo/eval_messages.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 8192
packing: true
mix_strategy: concat

### model
model_name_or_path: baidu/ERNIE-4.5-0.3B-PT
_attn_implementation: flashmask
lora: true
lora_rank: 8

### finetuning
# base
stage: DPO
fine_tuning: lora
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 1
max_steps: -1
eval_steps: 100
evaluation_strategy: steps
save_steps: 100
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 4
logging_dir: ./vdl_log
output_dir: ./checkpoints/ernie45-dpo-lora
disable_tqdm: true
eval_accumulation_steps: 16
model_with_dpo_criterion: true

# train
warmup_steps: 20
learning_rate: 1.0e-5

# performance
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage2
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2
save_checkpoint_format: flex_checkpoint
load_checkpoint_format: flex_checkpoint
```

#### 配置说明（详见[训练参数配置说明](./training_arguments.md)）

`train(eval)_dataset_type` ：由于 demo 数据类型是 messages 格式，配置为`messages`

`packing & mix_strategy`：数据处理策略，详见[数据流参数说明](./data_processing_guide.md)

`model_name_or_path`：模型本地路径或 HuggingFace 仓库对应的名称，如`baidu/ERNIE-4.5-0.3B-PT`，推荐使用 SFT 后的模型

`_attn_implementation`：模型 Attention Mask 实现方式，推荐使用 `flashmask`，是一种针对 FlashAttention 的一种核心优化技术。

`lora`：Bool 类型，是否 lora 训练，默认`False`。

`lora_rank`：开启 lora 训练时设置，一般设置`8`即可，如果需要适配更复杂训练任务，可适当提高（如16、32等）。

`stage`：与训练类型相关，设置`DPO`

`fine_tuning`：`full`表示全量训练，`lora`仅训练 LoRA 参数

`max_steps`：设置`-1`表示训练到最大 step 停止，也可指定设置最大步数

`load_checkpoint_format`：加载模型格式方式，推荐`flex_checkpoint`

`save_checkpoint_format`：保存模型格式方式，推荐`flex_checkpoint`

`model_with_dpo_criterion`：默认关闭，组网中包含 dpo criterion 时需要开启，否则关闭，如 ernie45需要开启。

## 3.2. 启动训练

#### 全量训练

```shell
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train dpo_full.yaml
```

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train dpo_full.yaml
```

#### lora 训练

```shell
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train dpo_lora.yaml
```

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train dpo_lora.yaml
```

# 4. 参数合并

针对 lora 训练后模型，需要将 LoRA 参数合并到主模型才能用于推理部署使用，为此我们提供了 LoRA 参数合并功能。

#### 配置

合参配置使用 yaml 格式文件，如下示例：

```
fine_tuning: LoRA
model_name_or_path: baidu/ERNIE-4.5-0.3B-PT
download_hub: huggingface
output_dir: ./checkpoints/ernie45-dpo-lora
```
`model_name_or_path`：主模型的本地路径或 HuggingFace 仓库对应的名称，如`baidu/ERNIE-4.5-0.3B-PT`

`download_hub`：仓库源，本地路径无需设置；如果`model_name_or_path`是远程仓库模型，需要同时指定对应的仓库源，可设置`huggingface`、`aistudio`、`modelscope`。

`output_dir`：lora 训练后参数的保存路径，与上述 lora 训练配置的`output_dir`保持一致即可。

# 5. 推理部署

同[基于 PaddleFormers 进行指令微调（SFT & LoRA）](./sft_and_lora_guide.md)
