# 1. 任务简介

在大模型训练过程中，**监督微调（Supervised Fine-Tuning，SFT）** 与 **偏好对齐优化（Direct Preference Optimization，DPO）** 是提升模型指令遵循能力与输出质量的两种关键方法。SFT 通过高质量的指令—响应数据，使模型学习“如何回答问题”；而 DPO 则基于人类或模型偏好数据，引导模型在多个候选回答中更倾向于生成更优、更符合人类偏好的结果，从而进一步优化模型行为。

SFT 通常用于让基础模型快速具备特定任务能力和基本指令对齐能力，是模型从“能生成”走向“会回答”的关键步骤；DPO 则在此基础上，通过偏好学习强化模型对答案质量、风格和安全性的理解，使模型输出在一致性、可控性和用户体验方面进一步提升。二者结合，能够实现 **SFT 效果优于 Base，DPO 效果优于 SFT** 的逐级优化目标。

本文以 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 模型为例，演示如何基于 [PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers) 先进行 SFT 微调，再进行 DPO 训练，构建一个指令遵循能力更强、回答质量更高的对齐模型。

本次实验旨在说明如何在 PaddleFormers 框架下完成 SFT 与 DPO 的串联训练，并验证模型在指令理解与偏好对齐方面的效果提升。整个实验使用1张40G A100显卡进行训练，耗时约10分钟。

# 2. 任务准备

## 2.1. 模型准备

PaddleFormers 通过在训练配置文件中指定字段`model_name_or_path`来设置所用的模型。启动训练时如果本地没有该模型的缓存，那么 PaddleFormers 会自动下载模型并加载使用。

您也可以将对应的字段指定成您的本地路径，来加载已经下载好的模型。

## 2.2. 数据准备

我们这里使用 emoji 数据集（[alpaca_emoji.json](https://huggingface.co/datasets/Orion-zhen/dpo-emoji-zh/blob/main/alpaca_emoji.json)） 来演示。这是一个 DPO 数据集，它会教模型在问答中更倾向于生成人类偏好（带有 emoji）的结果，使得模型的问答结果更显生动、活泼、友好。

为了快速演示并获得更直观的效果，我们使用如下脚本处理数据集，通过该 DPO 数据集生成以下两个数据集：

* SFT 数据集：取自数据集的前200条数据，其中模型回答有30%取自 DPO 数据集中的 chosen 回答（带有 emoji），其余来自于 rejected 回答（不带 emoji）。用以 SFT 训练使模型在回答中有输出 emoji 的基础能力。
* DPO 数据集：取自数据集的200-400条数据。用以 DPO 训练引导模型输出更符合人类偏好的、带有 emoji 的结果。

（如果想要构建自己的数据集，请参考以下文档：[数据集格式说明](../../../docs/zh/dataset_format.md)）：

```python
import os
import json
from datasets import load_dataset

dataset_dir = "./datasets"

input_dataset_file = os.path.join(dataset_dir, f"Orion-zhen_dpo-emoji-zh.jsonl")

output_trainset_sft = os.path.join(dataset_dir, f"emoji_trainset_sft_demo.jsonl")
output_trainset_dpo = os.path.join(dataset_dir, f"emoji_trainset_dpo_demo.jsonl")

SFT_SIZE = 200
DPO_SIZE = 200

def parse_sft_data_to_message(data, index):
    user_content = data['prompt']
    assistant_content = data['chosen'] if index % 10 < 3 else data['rejected']
    return {
        "messages": [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }


def parse_dpo_data_to_message(data):
    return {
        "messages": [
            {
                "role": "user",
                "content": data["prompt"]
            }
        ],
        "chosen_response": [
            {
                "role": "assistant",
                "content": data["chosen"]
            }
        ],
        "rejected_response": [
            {
                "role": "assistant",
                "content": data["rejected"]
            }
        ]
    }


if not os.path.exists(input_dataset_file):
    ds = load_dataset("Orion-zhen/dpo-emoji-zh", "default")
    dataset_raw = []
    with open(input_dataset_file, "w") as f:
        for data in ds["train"]:
            dataset_raw.append(data)
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
else:
    with open(input_dataset_file, "r") as f:
        dataset_raw = [json.loads(line.strip()) for line in f.readlines() if line.strip()]

dataset_sft = []
for index, data in enumerate(dataset_raw[:SFT_SIZE]):
    parsed_data = parse_sft_data_to_message(data, index)
    dataset_sft.append(parsed_data)

dataset_dpo = []
for data in dataset_raw[SFT_SIZE : SFT_SIZE + DPO_SIZE]:
    parsed_data = parse_dpo_data_to_message(data)
    dataset_dpo.append(parsed_data)

print(f"dataset_raw len:{len(dataset_raw)}")
print(f"dataset_sft len:{len(dataset_sft)}")
print(f"dataset_dpo len:{len(dataset_dpo)}")

with open(output_trainset_sft, "w", encoding="utf-8") as f:
    for data in dataset_sft:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

with open(output_trainset_dpo, "w", encoding="utf-8") as f:
    for data in dataset_dpo:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
```

处理后的数据中每一行为一条 messages 格式的数据，数据示例如下：

```json
{"messages": [{"role": "user", "content": "保持健康的三个提示。"}, {"role": "assistant", "content": "🙂\n以下是保持健康的三个提示：\n1. 保持身体活动🏃‍♀️。每天做适当的身体运动，如散步、跑步或游泳🏊，能促进心血管健康❤️，增强肌肉力量💪，并有助于减少体重📉。\n2. 均衡饮食🥗。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品🚫，以保持健康的饮食习惯🍽️。\n3. 睡眠充足😴。睡眠对人体健康至关重要🔑，成年人每天应保证 7-8 小时的睡眠⏰。良好的睡眠有助于减轻压力😌，促进身体恢复🌟，并提高注意力和记忆力🧠。\n\n希望这些建议能帮助你保持健康！💪"}]}
```

```json
{"messages": [{"role": "user", "content": "找到给定梯形的面积。输入：底边：13厘米和16厘米；高度：12厘米。"}], "chosen_response": [{"role": "assistant", "content": "🎉\n根据给定的数据，我们可以使用梯形面积公式：面积 = （下底 + 上底） x 高 / 2 😊。\n带入数据得到：\n面积 = (13厘米 + 16厘米) x 12厘米 / 2\n面积 = 174厘米²\n因此，给定梯形的面积为174平方厘米 📏✨。"}], "rejected_response": [{"role": "assistant", "content": "根据给定的数据，我们可以使用梯形面积公式： 面积 = （下底 + 上底） x 高 / 2。\n\n带入数据得到：\n面积 = (13厘米 + 16厘米) x 12厘米 / 2\n面积 = 174厘米²\n\n因此，给定梯形的面积为174平方厘米。"}]}
```

* 在 SFT 数据中，仅包含一个名为`messages`的字段。该字段`messages`是一个由一系列对话内容构成的列表，列表中的每条对话内容均包含两个子字段，分别为`role`和`content`。在`role`字段中，可以填入`system`、`user`或`assistant`，分别表示“系统设定”、“用户输入”和“模型回复”，需特别说明的是，`system`这一角色标识仅能出现在对话的首轮。而`content`字段则用于填入具体的对话内容。

* 在 DPO 数据中，包含一个名为`messages`的字段，结构同上；此外，还包含两个名为`chosen_response`和`rejected_response`的字段，其分别表示偏好的模型输出和拒绝的模型输出，结构和`messages`的字段一致，但`role`字段仅为`assistant`，表示“模型回复”。

## 2.3. 训练前效果测试

在正式开始模型训练前，我们从数据集（[alpaca_emoji.json](https://huggingface.co/datasets/Orion-zhen/dpo-emoji-zh/blob/main/alpaca_emoji.json)）中未使用的数据中随机挑选两条负样本，测试模型的回复。

使用如下脚本进行模型的预测：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_name = "Qwen/Qwen3-0.6B"

# prepare the model input
prompts = [
    "写一篇关于La Taqueria餐厅的评论。",
    "比较美国和加拿大的学生债务危机。",
]

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    fix_mistral_regex=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map="auto"
)

for prompt in prompts:
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        do_sample=False,
        max_new_tokens=4096
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    print("prompt: ", prompt)
    print("content: ", content)
```

以这两条数据为例：

```python
prompt="写一篇关于La Taqueria餐厅的评论。"
prompt="比较美国和加拿大的学生债务危机。"
```

在训练前，模型回复如下：

```text
prompt:  写一篇关于La Taqueria餐厅的评论。
content:  当然可以！以下是一篇关于 **La Taqueria** 餐厅的评论，供你参考：

---

**La Taqueria：一家充满热情与传统魅力的餐厅**

在巴塞罗那的街头，有一家名为 **La Taqueria** 的餐厅，以其独特的风格和热情的氛围吸引了众多美食爱好者。这家位于市中心的餐厅，不仅提供正宗的西班牙美食，更以一种轻松、亲切的方式，将传统的西班牙风味与现代的用餐体验完美融合。

餐厅的装修风格融合了复古与现代元素，墙壁上挂着一些老照片和装饰画，营造出浓厚的历史感。主厨是一位经验丰富的西班牙人，他总是能用地道的西班牙语与客人交流，展现出对美食的热爱和对文化的尊重。

La Taqueria 的菜单以传统为主，但每一道菜都充满了创意和惊喜。例如，他们以 **西班牙烤肉** 为主角，搭配香料和新鲜的蔬菜，让人感受到地道的风味。此外，餐厅还提供多种甜点和饮品，从传统的 **西班牙咖啡** 到创意甜点，每一款都让人流连忘返。

除了美食，La Taqueria 也注重环境和氛围的营造。餐厅的灯光柔和，音乐轻快，让人在享受美食的同时，也能感受到一种轻松愉快的氛围。无论是朋友聚餐，还是家庭聚会，这里都是一个理想的场所。

总的来说，La Taqueria 不仅是一家餐厅，更是一个充满热情和文化气息的地方。它用美食和热情，为每一位顾客带来难忘的用餐体验。

---

如果你有特定的风格或方向（如美食、文化、个人体验等），我也可以进一步调整内容。
```

```text
prompt:  比较美国和加拿大的学生债务危机。
content:  美国和加拿大在学生债务危机方面的情况各有不同，主要体现在以下几个方面：

### 1. **背景与政策差异**
   - **美国**：美国学生债务危机主要源于高等教育机构（如大学、学院）的债务问题，尤其是公立大学和私立大学。美国的高等教育体系相对开放，学生可以自由选择贷款机构，但政策上对债务的管理较为宽松。此外，美国的高等教育体系中，学生通常需要承担学费、教材费、住宿费等，而这些费用的负担可能因地区、专业和学生家庭经济状况而有所不同。
   - **加拿大**：加拿大学生债务危机主要集中在高等教育机构（如大学、学院）的债务问题，但政策上对债务的管理相对严格。加拿大政府通过税收和财政政策来应对债务问题，例如通过提高学费或增加政府补贴来缓解债务压力。

### 2. **债务规模与影响**
   - **美国**：美国学生债务危机的规模较大，尤其是在公立大学和私立大学中。2010年代以来，美国高等教育机构的债务水平显著上升，部分大学甚至出现债务危机。学生贷款的规模也较大，部分学生可能面临高额利息和还款压力。
   - **加拿大**：加拿大学生债务危机的规模相对较小，主要集中在高等教育机构，但同样存在债务问题。加拿大政府通过提高学费和增加政府补贴来应对债务压力，但具体规模和影响因地区和政策而异。

### 3. **经济与社会影响**
   - **美国**：美国学生债务危机对经济和社会的影响较为广泛，包括对就业市场的影响、对家庭财务状况的冲击以及对社会整体经济的潜在影响。由于学生贷款的高利率，部分学生可能面临较高的财务负担。
   - **加拿大**：加拿大学生债务危机的影响相对较小，主要集中在高等教育机构，但同样可能对社会经济结构产生一定影响，尤其是在教育和就业领域。

### 4. **应对措施**
   - **美国**：美国政府和高等教育机构采取了多种措施来应对债务危机，包括提高学费、增加政府补贴、优化贷款结构等。
   - **加拿大**：加拿大政府也采取了类似的措施，通过提高学费和增加政府补贴来缓解债务压力。

### 总结
美国和加拿大的学生债务危机各有特点，美国的规模较大，影响更广泛；加拿大则相对较小，影响有限。两者在应对措施和政策上都有其独特之处，具体影响和规模因地区、政策和经济环境的不同而有所差异。
```

模型的回答为纯文本描述，这并不符合我们设定的人类偏好，我们希望模型的回答能够更生动活泼一些，最好带有一些 emoji。

我们的目标是通过微调，使得模型可以在输出中带有明显、贴切的 emoji。

# 3. 模型训练

## 3.1 SFT

### 3.1.1 训练超参数配置

训练超参数可参考[full.yaml](../../config/sft/full.yaml)配置，以下是本次实验使用的 yaml：

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./datasets/emoji_trainset_sft_demo.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./datasets/emoji_trainset_sft_demo.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 8192
packing: false
mix_strategy: concat

### model
model_name_or_path: Qwen/Qwen3-0.6B
attn_impl: flashmask

### finetuning
# base
stage: SFT
fine_tuning: full
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 5
max_steps: -1
eval_steps: 100
evaluation_strategy: steps
save_steps: 100
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 16
logging_dir: ./vdl/paddleformers_qwen3_0p6b_sft_emoji/
output_dir: ./checkpoints/paddleformers_qwen3_0p6b_sft_ckpts_emoji/
disable_tqdm: true
eval_accumulation_steps: 16

# train
warmup_ratio: 0.05
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
unified_checkpoint: true
```

其中有几个比较重要的参数，在此简单说明：

* max_seq_len：最大数据长度，我们需要保证数据长度小于此值，以防止数据被截断。建议在训练前估计数据集中数据长度的范围，防止大部分数据被截断从而影响训练效果。
* num_train_epochs：训练的 epoch 数。推荐设置为2-3，本次实验由于训练集数据较少，选择训练5个 epoch 以确保训练效果。
* per_device_train_batch_size：每张卡的 batch size 大小。推荐设置为1，可根据显存占用情况增加。
* gradient_accumulation_steps：梯度累积的步数。推荐参考以下算式设置：`每次梯度更新时的批大小 = 训练总卡数 * per_device_train_batch_size * gradient_accumulation_steps = 16 或 32`
* warmup_ratio：学习率的预热步数占总步数的占比，这里设置为总步数的5%。

### 3.1.2 通过命令行训练

仅需一行代码即可完成模型训练：

```shell
paddleformers-cli train sft_full_emoji.yaml
```

### 3.1.3 训练曲线观察

在训练过程中，我们可以利用 visualdl 工具来查看模型的 loss 曲线：

```shell
# 指定端口和地址，以及日志目录（需要和前面训练配置文件中的目录对应上）
visualdl --logdir ./vdl_log/ --port 8080 --host 0.0.0.0
```

打开 VisualDL 启动的 web 链接，可以看到训练的 loss 曲线如下，模型的训练 loss 在不断稳定下降：

<div align="center">
  <img width="422" height="328" alt="emoji_sft_vdl_log" src="https://github.com/user-attachments/assets/fc9430d1-2375-483d-a18a-70d0aef512a1" />
</div>

### 3.1.4 效果评估

完成模型训练后，我们再次以这两条数据为例，测试模型的回复：

```python
prompt="写一篇关于La Taqueria餐厅的评论。"
prompt="比较美国和加拿大的学生债务危机。"
```

模型回复如下：

```text
prompt:  写一篇关于La Taqueria餐厅的评论。
content:  🌟
在巴塞罗那的美食之旅中，La Taqueria餐厅无疑是一个不容错过的美味天堂。这家位于市中心的餐厅以其独特的西班牙风味和精致的菜品而闻名，是游客和美食爱好者的好去处。

🌈
La Taqueria的招牌菜是香煎三文鱼，搭配着浓郁的香料酱汁和新鲜的蔬菜，口感鲜美，令人回味无穷。此外，餐厅还提供多种主食和甜点选择，从香烤土豆到焦糖化的马卡龙，应有尽有。

🎉
走进La Taqueria，你会被其优雅的装饰和舒适的环境所吸引。餐厅的灯光柔和，音乐轻柔，为每一位顾客营造了一个放松愉悦的用餐氛围。此外，餐厅还提供免费Wi-Fi和充电宝，方便您的行程。

📍
La Taqueria的地址是市中心的X号，步行5分钟即可到达。如果您正在寻找一家提供地道西班牙美食的餐厅，La Taqueria绝对值得您的访问。

🌈
总之，La Taqueria餐厅是一个值得一试的美食天堂，它以其独特的西班牙风味和美味的菜品吸引了无数食客。无论是首次来到巴塞罗那的游客，还是寻求美食之旅的美食爱好者，La Taqueria都将是您不容错过的美味之地。享受您的美食之旅吧！🎉
```

```text
prompt:  比较美国和加拿大的学生债务危机。
content:  美国和加拿大都是全球学生债务危机的发源地之一，但两者的经济结构、政策环境和债务水平存在显著差异。以下是两者的比较：

1. 经济结构：美国的学生债务危机主要源于高等教育投资的增加，而加拿大的学生债务危机则与高等教育投资的增加和政府财政政策有关。

2. 政策环境：美国政府对高等教育的资助政策较为宽松，而加拿大的政府对高等教育的资助政策较为严格。

3. 债务水平：美国的学生债务水平较高，据2022年的数据，美国大学学生平均债务水平约为12.5万美元，而加拿大的学生债务水平约为8.5万美元。

4. 债务来源：美国的学生债务主要来自学费和贷款，而加拿大的学生债务主要来自学费和贷款。

5. 债务影响：美国的学生债务危机对经济和社会造成了重大影响，而加拿大的学生债务危机也对经济和社会造成了重大影响。

总之，美国和加拿大的学生债务危机各有特点，但都反映了全球高等教育投资增加和政府财政政策对高等教育资助的影响。
```

从模型回复可以看出，模型已经有了一定的在输出中包含 emoji 的能力，且所使用的 emoji 都贴合语境，但仍然有部分问题的回答是不带有 emoji 的。

至此，我们的目标是，在上述通过 SFT 微调的模型的基础上，进一步通过 DPO 微调，来使模型学习到我们设定的人类偏好，使得模型可以在输出中带有明显、贴切的 emoji。

## 3.2 DPO

### 3.2.1 训练超参数配置

训练超参数可参考[full.yaml](../../config/dpo/full.yaml)配置，以下是本次实验使用的 yaml，需要注意的是，我们需要将 `model_name_or_path` 设置为 SFT 后保存的模型路径，用于加载 SFT 模型进行 DPO 训练：

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./datasets/emoji_trainset_dpo_demo.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./datasets/emoji_trainset_dpo_demo.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 8192
packing: false
mix_strategy: concat

### model
model_name_or_path: ./checkpoints/paddleformers_qwen3_0p6b_sft_ckpts_emoji/
attn_impl: flashmask

### finetuning
# base
stage: DPO
fine_tuning: full
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 1
per_device_train_batch_size: 1
num_train_epochs: 5
max_steps: -1
eval_steps: 100
evaluation_strategy: steps
save_steps: 100
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 16
logging_dir: ./vdl/paddleformers_qwen3_0p6b_dpo_emoji/
output_dir: ./checkpoints/paddleformers_qwen3_0p6b_dpo_ckpts_emoji/
disable_tqdm: true
eval_accumulation_steps: 16

# train
# warmup_steps: 20
warmup_ratio: 0.05
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
unified_checkpoint: true
```

### 3.2.2 通过命令行训练

仅需一行代码即可完成模型训练：

```shell
paddleformers-cli train dpo_full_emoji.yaml
```

### 3.2.3 训练曲线观察

在训练过程中，我们可以利用 visualdl 工具来查看模型的 loss 曲线：

```shell
# 指定端口和地址，以及日志目录（需要和前面训练配置文件中的目录对应上）
visualdl --logdir ./vdl_log/ --port 8080 --host 0.0.0.0
```

打开 VisualDL 启动的 web 链接，可以看到训练的 loss 曲线如下，模型的训练 loss 很快下降，在30-40steps 时基本收敛：

<div align="center">
  <img width="426" height="331" alt="emoji_dpo_vdl_log" src="https://github.com/user-attachments/assets/8ef706d4-30b7-4daa-aed7-3cd4c4a77f9b" />
</div>

# 4. 效果评估

完成模型训练后，我们再次以这两条数据为例，测试模型的回复：

```python
prompt="写一篇关于La Taqueria餐厅的评论。"
prompt="比较美国和加拿大的学生债务危机。"
```

模型回复如下：

```text
prompt:  写一篇关于La Taqueria餐厅的评论。
content:  🌟
在巴塞罗那的美食之旅中，La Taqueria餐厅无疑是一个不容错过的美味天堂！这家位于市中心的餐厅以其独特的西班牙风味和精致的菜品而闻名，绝对值得你的一试！🌈

📍
La Taqueria餐厅的地址是市中心的X号街，步行仅需10分钟即可到达。餐厅的外观设计融合了传统西班牙风格与现代简约风格，让人一眼就被吸引。无论是从外观还是内部装修，都给人一种轻松愉快的氛围，让人忍不住想要进去品尝美食。

🍽️
餐厅的菜单种类繁多，从传统的西班牙菜到创新的现代料理，应有尽有。我们推荐的主菜是香煎三文鱼配香草酱汁，鲜嫩多汁，味道鲜美，令人回味无穷。此外，我们还提供多种主食和甜点选择，如烤蔬菜拼盘、香煎多佛鱼和焦糖巧克力蛋糕，每一口都是美味的享受。

☕️
La Taqueria餐厅的特色饮品也是不容错过的。我们推荐的特色饮品是香草拿铁，用新鲜香草和浓郁的咖啡香调制而成，口感丰富，香气扑鼻。此外，餐厅还提供多种甜品，如焦糖巧克力蛋糕和香草冰淇淋，都是非常美味的选择。

🌈
La Taqueria餐厅的环境优雅，服务人员热情周到，让人倍感宾至如归。我们推荐的主厨是一位经验丰富的西班牙厨师，他总是能为我们带来令人惊艳的菜品。无论是第一次来巴塞罗那的美食之旅，还是再次光临La Taqueria，这里都是一个绝佳的美食体验。

🎉
总之，La Taqueria餐厅是一个值得一试的美食天堂，它以其独特的西班牙风味和精致的菜品，为每一位美食爱好者提供了完美的用餐体验。如果你正在寻找一个美食之旅的起点，那么La Taqueria餐厅绝对值得你的一试！🌈
希望这些建议能帮助你更好地规划你的巴塞罗那美食之旅！✨
```

```text
prompt:  比较美国和加拿大的学生债务危机。
content:  🤔
🤔
美国和加拿大都是全球重要的经济体，但它们的学生债务危机（student debt crisis）却各具特色。🤔
🤔
首先，美国的学生债务危机主要由联邦政府和私人机构共同承担。美国联邦政府通过《高等教育法案》（Higher Education Act）向大学和私立机构提供大量资金，而私人机构则通过贷款和投资吸引大量学生。这些资金被用于建设大学和提供奖学金，但同时也助长了学生贷款的增加。此外，美国联邦政府还通过《高等教育法案》向大学提供大量资金，而私人机构则通过贷款和投资吸引大量学生。这些资金被用于建设大学和提供奖学金，但同时也助长了学生贷款的增加。此外，美国联邦政府还通过《高等教育法案》向大学提供大量资金，而私人机构则通过贷款和投资吸引大量学生。这些资金被用于建设大学和提供奖学金，但同时也助长了学生贷款的增加。💰
💰
相比之下，加拿大的学生债务危机主要由私人机构和政府共同承担。加拿大政府通过《高等教育法案》向大学和私立机构提供大量资金，而私人机构则通过贷款和投资吸引大量学生。这些资金被用于建设大学和提供奖学金，但同时也助长了学生贷款的增加。此外，加拿大政府还通过《高等教育法案》向大学提供大量资金，而私人机构则通过贷款和投资吸引大量学生。这些资金被用于建设大学和提供奖学金，但同时也助长了学生贷款的增加。💰
💰
总结来说，美国和加拿大的学生债务危机都源于政府和私人机构对高等教育的资助，但美国的债务危机主要由联邦政府和私人机构共同承担，而加拿大的债务危机主要由私人机构和政府共同承担。两者都存在学生贷款增加的问题，但美国的债务危机规模更大，影响范围更广。📚
📚
希望这能帮到你！😉
```

从模型的回复可以看出，相比于训练前的模型和 SFT 微调后的模型，经过 DPO 微调的模型输出都带有贴切的 emoji 表情，问答结果显得更加生动、活泼、友好，更贴近了我们设定的人类偏好。

# 5. 模型部署

您可以参考[基于 FastDeploy / vLLM 部署模型](../../../docs/zh/deployment_guide.md)，将训练完成的模型使用 vLLM / FastDeploy 等工具进行部署使用
