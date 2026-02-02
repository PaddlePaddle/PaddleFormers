# 1. 背景说明

大模型的训练流程通常包含四个关键阶段：

1. **预训练（Pre-training）：通过海量无标注数据进行预训练，学习语言能力和世界知识，构建通用基座；（本文讲解）**
2. 后预训练（Post-Pre-training）：通过注入特定领域知识以增强专业性，构建领域基座模型；
3. 有监督微调（SFT）：利用高质量指令数据进行微调，模型具备对话与任务处理能力；
4. 偏好对齐（DPO/RLHF）：采用偏好优化（DPO/RLHF）技术，基于成对的优劣反馈数据对模型进行价值观对齐，从而输出符合人类预期的最终版本。

本文旨在说明如何基于 PaddleFormers 进行模型的预训练（后预训练流程可以直接参考本文档）。

# 2. 数据准备

预训练数据规模非常庞大，通常以 TB 为单位。为了方便演示，我们提供一个 demo 数据集，执行下载并解压。如果想要使用自己的数据进行训练，请参考[数据集格式说明](./dataset_format.md)进行数据的准备。

```shell
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/pt_online_data_messages.tar.gz
mkdir -p data/pt && tar -xf pt_online_data_messages.tar.gz -C data/pt/
```

**demo 数据格式：** messages 格式，每条数据都是一个字典，包含以下字段：

* `messages` : `List(Dict）`，包含`role`和`content`两个字段。
    * `role`：固定为 assistant 即可。
    * `content`：文本内容。

```json
{
    "messages": [
        {
            "role": "assistant",
            "content": "农业属于第一级产业，包括作物种植、畜牧、渔业养殖、林业等活动，负责主副食和经济作物供应。农业的主要产品是食物、纤维、能源和原材料（例如橡胶），其中食物包括谷物、蔬菜、水果、食用油、肉类、奶制品、蛋和菌类。全球农业年产出约 110 亿吨食物，3200 万吨自然纤维和 40 亿立方米木材。不过，其中有 14 的食物在到达零售环节之前被浪费。自 20 世纪开始，基于单一作物种植的工业化农业开始成为世界农业产出的主要来源。农业的出现是人类文明转向定居形式的里程碑，借由野生动植物的驯化、培育与繁殖，人们获得了充足的食物与资源，并促进早期城市的发展与成型。人类在 10.5 万年前开始从野外采集谷物，但直到 1.15 万年前才开始种植，并在大约 1 万年前驯化了绵羊、山羊、猪、牛等家畜。世界上至少有 11 个地区独立发展出了作物种植。现代农业技术、植物育种、农业化学产品（例如杀虫剂和化肥）的发展显着增加了作物产量，但也引发了诸多生态与环境问题。选择育种和现代畜牧业技术发展增加了肉类制品产量，但也引发了动物福利和环境忧虑。上述环境问题包括气候变化、地下含水层枯竭、森林砍伐、抗生素耐药性和农业相关污染。农业既是环境退化的原因，也深受其影响，生物多样性丧失、荒漠化、土壤退化、气候变化等因素都会降低作物产量。进入 21 世纪后，可持续农业的比例渐渐提高，包括朴门和有机农业，着重在生态平衡与就近百里饮食。转基因作物被广泛使用，但也有部分国家禁止此类作物种植。"
        }
    ]
}
```

# 3. 训练

预训练仅推荐使用全量参数训练，不适合 LoRA 等少量参数训练。

## 3.1. 超参数配置

训练配置推荐使用 yaml 格式文件，如下示例：

#### 全量训练配置

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./data/pt/train_messages.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./data/pt/eval_messages.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 8192
packing: true
mix_strategy: concat

### model
model_name_or_path: baidu/ERNIE-4.5-0.3B-Base-PT
_attn_implementation: flashmask

### finetuning
# base
stage: PT
fine_tuning: full
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 3
max_steps: 100
eval_steps: 100
evaluation_strategy: steps
save_steps: 100
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 4
logging_dir: ./vdl_log
output_dir: ./checkpoints/ernie45-pt-full
disable_tqdm: true
eval_accumulation_steps: 16

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

`model_name_or_path`：模型本地路径或 HuggingFace 仓库对应的名称，如`baidu/ERNIE-4.5-0.3B-Base-PT`

`_attn_implementation`：模型 Attention Mask 实现方式，推荐使用 `flashmask`，是一种针对 FlashAttention 的一种核心优化技术。

`stage`：与训练类型相关，预训练设置`PT`

`fine_tuning`：预训练参数量相关，`full`表示全量训练，`lora`训练少量参数（预训练不推荐）

`max_steps`：设置`-1`表示训练到最大 step 停止，也可指定设置最大步数

`load_checkpoint_format`：加载模型格式方式，推荐`flex_checkpoint`

`save_checkpoint_format`：保存模型格式方式，推荐`flex_checkpoint`

## 3.2. 启动训练

PaddleFormers 支持 cli 直接启动训练，如下示例：

#### 全量训练

```shell
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train pt_full.yaml
```

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train pt_full.yaml
```

# 4. 推理部署

## 4.1. 本地离线推理
本地离线推理使用 generation 接口，如下示例

```python
import paddle
from paddleformers.transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

model_name_or_path = './checkpoints/ernie45-pt-full'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    load_checkpoint_format="flex_checkpoint"
)

prompt = "农业的主要产品是"
model_inputs = {
    "input_ids": paddle.to_tensor([tokenizer.encode(prompt)]),
}
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128,
)
output_ids = generated_ids[0].tolist()[0]

# decode the generated ids
generate_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print("output_text: \n", prompt + generate_text)
```

```text
output_text:
 农业的主要产品是食用菌，主要分布在台湾的台北、高雄、马祖、NASL等城市。台湾的食用菌主要有梭子蟹、香菇、滑盖蟹、火腿、笋干等，其中香菇主要分布在台湾的台北、高雄、马祖三地，目前已大量出口。
```

## 4.2. 部署推理服务

[基于 FastDeploy / vLLM 部署模型](./deployment_guide.md)
