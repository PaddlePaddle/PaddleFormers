# 1. 任务简介

PaddleOCR-VL 是一款为文档解析任务量身打造的、性能顶尖 (SOTA) 且轻量高效的模型。它的核心是 PaddleOCR-VL-0.9B——一个紧凑而强大的视觉语言模型 (VLM)。该模型创新地集成了 NaViT 风格的动态分辨率视觉编码器与 ERNIE-4.5-0.3B 语言模型，从而能够精准地识别各类文档元素。

这款模型不仅能高效支持 109 种语言，还擅长识别文本、表格、公式、图表等复杂元素，并始终保持极低的资源占用。在多个权威的公开及内部基准测试中，PaddleOCR-VL 的页面级文档解析与元素级识别性能均达到了业界顶尖水平。其性能远超现有方案，面对顶级视觉语言模型也极具竞争力，且推理速度飞快。这些杰出特性使其成为在真实场景中落地部署的理想选择。

虽然 PaddleOCR-VL-0.9B 在常见场景下表现出色，但在许多特定或复杂的业务场景中，其性能会遇到瓶颈。例如：

* 特定行业与专业领域
    * 金融与财会领域：识别发票、收据、银行对账单、财务报表等
    * 医疗领域：识别病历、化验单、医生手写处方、药品说明书等
    * 法律领域：识别合同、法律文书、法庭文件、证书等

* 非标准化的文本与字体
    * 手写体识别：识别手写的表单、笔记、信件、问卷调查等
    * 艺术字体与设计字体：识别海报、广告牌、产品包装、菜单上的艺术字体等
    * 古籍与历史文献：识别古代手稿、旧报纸、历史档案等

* 特定任务与输出格式
    * 表格识别与结构化输出：将图像中的表格转换为 Excel、CSV 或 JSON 格式
    * 数学公式识别：识别教科书、论文中的数学公式，并输出为 LaTeX 等格式


这时，就需要通过 SFT (Supervised Fine-Tuning) 来提升模型的准确性和鲁棒性。

本教程旨在提供基于 PaddleFormers 微调 PaddleOCR-VL 模型适配孟加拉语的微调教程，资源需求和运行耗时见下方表格：

|SFT|显存|用时|
|-|-|-|
|全参|30G|1h 36min|
|LoRA|20G|1h 46min|

# 2. 任务准备

## 2.1 模型准备

PaddleFormers 通过在训练配置文件中指定字段`model_name_or_path`来设置所用的模型。启动训练时如果本地没有该模型的缓存，那么 PaddleFormers 会自动下载模型并加载使用。

您也可以将对应的字段指定成您的本地路径，来加载已经下载好的模型。

## 2.2 数据集准备

**Demo 数据**

为了方便起见，我们也提供了一个快速上手的孟加拉语数据集（训练集和测试集），可用于微调 PaddleOCR-VL-0.9B 对孟加拉语进行识别，使用以下命令下载：

```shell
wget https://paddleformers.bj.bcebos.com/datasets/ocr-vl/ocr_vl_sft-train_Bengali.jsonl
wget https://paddleformers.bj.bcebos.com/datasets/ocr-vl/ocr_vl_sft-test_Bengali.jsonl
```

孟加拉语训练数据示例：

<div align="center">
  <img width="236" height="112" alt="bengali_train_demo" src="https://github.com/user-attachments/assets/b65e899f-9308-4adf-b3a4-d7e86587fcc5" />
</div>

```json
{
    "messages": [
        {"role": "user", "content": "<image>OCR:"},
        {"role": "assistant", "content": "দডর মথ বধ বকসট একনজর দখই চনত পরল তর অনমন\nঠক পনতই লকয রখছ\nর নচ থকই চচয বলল কশর, “এইই; পযছ! পযছ!'\nওপর"}
    ],
    "images": ["./assets/train_example.jpg"]
}
```

一个 OCR SFT 数据样本中需包含以下字段：

* `messages`：文本数据列表，记录了用户与模型之间的交互过程，其中每个元素包含一个 `role` 和一个 `content`。
    * `role`：代表消息发送者的身份。
        * `"user"`：用户，代表输入端。
        * `"assistant"`：助手/模型，代表输出端。

    * `content`：消息的具体内容。
        * 输入端包含指令和图片占位符。
            * 提示指令 `Prompt`：根据识别任务设置
                * 文字识别 `"OCR:"`（最通用）
                * 表格识别 `"Table Recognition"`
                * 公式识别 `"Formula Recognition"`
                * 图表识别 `"Chart Recognition"`
                * 或者根据微调任务自定义提示

            * 图片占位符 `<image>`：在文本数据中标记图片插入的位置。

        * 输出端包含模型预期生成的正确答案，即图片中需要识别的字符。

* `images`：图像数据列表，存储了对话中涉及到的图片路径（本地路径或 URL）。

**自行准备数据**

如果您想要基于自己的数据集进行训练，请参考 [数据集格式说明](../../../docs/zh/dataset_format.md)准备数据。

**其他任务格式**

表格/公式/图表数据会使用特殊的识别格式，细节请参考 [表格/公式/图表数据格式](#81-表格公式图表数据格式)。

# 3. 训练配置

我们针对孟加拉语示例数据集提供了配置文件，其中的关键训练超参数如下：

* `num_train_epochs=2`：训练的 epoch 数。
* `warmup_ratio=0.01`：线性预热步数, 建议设置成训练步数的 1%。
* `per_device_train_batch_size=8`：每张卡的 batch size 大小，建议根据显存占用情况调整。
* `max_seq_len=16384`：最大序列长度，超出该长度的数据将被截断或者丢弃。建议在训练前估计数据集中数据长度的范围，防止大部分数据被截断从而影响训练效果。
* `gradient_accumulation_steps=8`：梯度累积步数。
    * 每达到该步数整数倍更新一次模型参数。
    * 当显存不足时，可以减小 `per_device_train_batch_size` 并增大 `gradient_accumulation_steps`。
    * 用时间换空间策略，可以减少显存占用，但会延长训练时间。

* `learning_rate`：学习率，即每次参数更新的幅度。
    * 全参训练 `learning_rate=5e-6`
    * LoRA 训练 `learning_rate=5e-4`

更多相关参数可在配置文件中查看。

**全参配置**

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./ocr_vl_sft-train_Bengali.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./ocr_vl_sft-test_Bengali.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 16384
padding_free: True
truncate_packing: False
dataloader_num_workers: 8
mix_strategy: concat
template_backend: custom
template: paddleocr_vl

### model
model_name_or_path: PaddlePaddle/PaddleOCR-VL
attn_impl: flashmask

### finetuning
# base
stage: VL-SFT
fine_tuning: full
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 8
per_device_train_batch_size: 8
num_train_epochs: 2
max_steps: -1
max_estimate_samples: 500
eval_steps: 400
evaluation_strategy: steps
save_steps: 400
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 8
logging_dir: ./PaddleOCR-VL-SFT-Bengali/visualdl_logs/
output_dir: ./PaddleOCR-VL-SFT-Bengali
disable_tqdm: true
eval_accumulation_steps: 16

# train
lr_scheduler_type: cosine
warmup_ratio: 0.01
learning_rate: 5.0e-6
min_lr: 5.0e-7

# optimizer
weight_decay: 0.1
adam_epsilon: 1.0e-8
adam_beta1: 0.9
adam_beta2: 0.95

# performance
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage2
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2

# save
unified_checkpoint: False
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
```

**LoRA 配置**

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./ocr_vl_sft-train_Bengali.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./ocr_vl_sft-test_Bengali.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 16384
padding_free: True
truncate_packing: False
dataloader_num_workers: 8
mix_strategy: concat
template_backend: custom
template: paddleocr_vl

### model
model_name_or_path: PaddlePaddle/PaddleOCR-VL
attn_impl: flashmask
lora: true
lora_rank: 8

### finetuning
# base
stage: VL-SFT
fine_tuning: lora
seed: 23
do_train: true
do_eval: true
per_device_eval_batch_size: 8
per_device_train_batch_size: 8
num_train_epochs: 2
max_steps: -1
max_estimate_samples: 500
eval_steps: 400
evaluation_strategy: steps
save_steps: 400
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 8
logging_dir: ./PaddleOCR-VL-SFT-Bengali-lora/visualdl_logs/
output_dir: ./PaddleOCR-VL-SFT-Bengali-lora
disable_tqdm: true
eval_accumulation_steps: 16

# train
lr_scheduler_type: cosine
warmup_ratio: 0.01
learning_rate: 5.0e-4
min_lr: 5.0e-5

# optimizer
weight_decay: 0.1
adam_epsilon: 1.0e-8
adam_beta1: 0.9
adam_beta2: 0.95

# performance
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage2
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2

# save
unified_checkpoint: false
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
```

# 4. SFT 训练

## 4.1 SFT 全参训练

使用以下命令行即可启动全参训练：

```shell
CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli train examples/best_practices/PaddleOCR-VL/paddleocr-vl_full_16k_config.yaml \
                        model_name_or_path=./PaddlePaddle/PaddleOCR-VL \
                        train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl \
                        eval_dataset_path=./ocr_vl_sft-test_Bengali.jsonl \
                        pre_alloc_memory=24
```

设置 `pre_alloc_memory` 预分配显存从而减少显存碎片，根据序列长度、批大小和硬件显存调整。

PaddleFormers 默认使用机器上的全部 GPU，可以通过环境变量 `CUDA_VISIBLE_DEVICES` 设置 PaddleFormers 能够使用的 GPU。

GPU 的数目 `GPU_num` 会影响训练超参数 `learning_rate & per_device_train_batch_size & gradient_accumulation_steps` 配置。理论上，每个更新步使用的样本数目 `sample_num = G*B*A`，近似与学习率 `learning_rate` 成正线形关系，因此，当 GPU 数目增加 `N` 倍变为 `N*GPU` 时，有两种调整方式：

1. 保持 `sample_num` 不变

    * 将 `packing_size` 减少 `x` 倍，变成 `packing_size/x`
    * 将 `gradient_accumulation_steps` 减少 `y` 倍，变成 `gradient_accumulation_steps/y`
    * 满足 `x*y = N` 即可

2. 将 `learning_rate` 增加 `N` 倍，变成 `N*learning_rate`

可以通过 `visualdl` 对训练过程可视化，使用以下命令行即可启动（下方命令将端口 port 设置为 `8084`，需要根据实际情况设置可用端口）：

```shell
visualdl --logdir ./PaddleOCR-VL-SFT-Bengali/visualdl_logs/ --port 8084
```

成功启动后该服务后，在浏览器输入 `ip:port` ，则可以看到训练日志（通过 `hostname -i` 命令可以查看机器的 ip 地址）。

损失曲线如下：

<div align="center">
  <img width="856" height="602" alt="bengali_sft_vdl_log" src="https://github.com/user-attachments/assets/76f8dd07-8632-469c-a2d8-ecb26d35be57" />
</div>

## 4.2 SFT LoRA 训练

使用以下命令行即可启动 LoRA 训练：

```shell
CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli train examples/best_practices/PaddleOCR-VL/paddleocr-vl_lora_16k_config.yaml \
                        model_name_or_path=./PaddlePaddle/PaddleOCR-VL \
                        train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl \
                        eval_dataset_path=./ocr_vl_sft-test_Bengali.jsonl\
                        pre_alloc_memory=16
```

# 5. 模型结构说明

## 5.1 SFT 全参

全参训练结束后，模型会保存在 `output_dir=./PaddleOCR-VL-SFT-Bengali` 指定路径下，其中包含：

* config.json：模型配置文件
* model-00002-of-00001.safetensors：模型权重文件
* model.safetensors.index.json：模型权重索引文件
* tokenizer.model & tokenizer_config.json & special_tokens_map.json & added_tokens.json：分词器文件
* train_args.bin：训练参数文件，记录训练使用的参数等
* train_state.json：训练状态文件，记录训练步数和最优指标等
* train_results.json & all_results.json：训练结果文件，记录训练进度&用时&每步耗时&每样本耗时等
* generation.json：生成配置文件
* checkpoint-[save_steps*n]：检查点文件夹，在 `save_steps` 整数倍保存训练状态，除以上文件外，还会保存 master-weight & optimizer-state & scheduler-state 等，可用于训练中断后恢复训练

## 5.2 SFT LoRA

LoRA 训练结束后，模型会保存在 `output_dir=./PaddleOCR-VL-SFT-Bengali-lora` 指定路径下。相较于 SFT 全参，SFT LoRA 的模型结构会有所不同，其中包含：

* lora_config.json：LoRA 模型配置文件
* peft_model-00001-of-00001.safetensors：LoRA 模型权重文件
* peft_model.safetensors.index.json：LoRA 权重索引文件

使用以下命令行即可合并 LoRA 权重：

```shell
CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli export ./examples/config/run_export.yaml \
    model_name_or_path=./PaddlePaddle/PaddleOCR-VL \
    output_dir=./PaddleOCR-VL-SFT-Bengali-lora
```

合并后的完整模型权重保存在 `output_dir=./PaddleOCR-VL-SFT-Bengali-lora/export` 路径下。

# 6. 推理

## 6.1 单样本推理

孟加拉语测试图像：

<div align="center">
  <img width="439" height="216" alt="bengali_pred_demo" src="https://github.com/user-attachments/assets/71b3e95d-fd9a-4210-a5a3-9601faeeb112" />
</div>

使用以下命令行进行单样本推理：

```shell
python generate.py
```

```python
import requests
from io import BytesIO

import paddle
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig

model_path = "./PaddleOCR-VL-SFT-Bengali-full_single"

model = AutoModelForConditionalGeneration.from_pretrained(
    model_path, convert_from_hf=True,
).eval()

# change the implementation of attention(default is "eager")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

processor = AutoProcessor.from_pretrained(model_path)

image_path = "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/bengali_sft/5b/7a/5b7a5c1c-207a-4924-b5f3-82890dc7b94a.png"
image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
task = "ocr" # Options: 'ocr' | 'table' | 'chart' | 'formula'

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {"type": "text", "text": PROMPTS[task]},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd",
)

generation_config = GenerationConfig(
    do_sample=False, # greedy_search
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    use_cache=True
)

with paddle.no_grad():
    generated_ids = model.generate(**inputs, generation_config=generation_config, max_new_tokens=1024)
    output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True)

print(output_text[0])

# GT = নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল
# Excepted Answer = নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল
```

预期输出为测试图像中的孟加拉语文字：নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল。

## 6.2 测试集评估

使用归一化 Levenshtein 编辑距离作为评估指标：

* Levenshtein 编辑距离：从预测字符串 A 变为真实字符串 B 最少需要的操作次数（插入/删除/修改一个字符）。
* 归一化 Levenshtein 编辑距离：将编辑距离除以 max(A, B) 来进行归一化。

使用以下命令行安装 Levenshtein 库：

```shell
pip install Levenshtein
```

使用以下命令行进行测试集评估：

```shell
model_path="./PaddleOCR-VL-SFT-Bengali"

CUDA_VISIBLE_DEVICES=0 \
python -m paddle.distributed.launch --log_dir ./log \
    eval.py \
    --model_name_or_path "${model_path}" \
    --data_path /PATH/TO/ocr_vl_sft-test_Bengali.jsonl \
    --output_path "${model_name}_eval_result.jsonl"
```

```python
import argparse
import json
import os
import sys
import time
import requests
from io import BytesIO

from PIL import Image
import paddle
import paddle.distributed as dist
from tqdm import tqdm
import Levenshtein  # Requires python-Levenshtein

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor, AutoConfig
from paddleformers.generation import GenerationConfig

def parse_args():
    parser = argparse.ArgumentParser(description="PaddleFormers & PaddleOCR-VL Model Evaluation Script")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model path or name")
    parser.add_argument("--data_path", type=str, required=True, help="Test data path (jsonl format)")
    parser.add_argument("--output_path", type=str, default="eval_results.jsonl", help="Result save path")
    parser.add_argument("--max_length", type=int, default=1024, help="Max generation length")
    parser.add_argument("--device", type=str, default="gpu", help="Device: gpu / cpu / xpu / iluvatar_gpu")
    return parser.parse_args()


def load_model_and_processor(model_path, device):
    print(f"Loading model: {model_path} ...")
    paddle.set_device(device)

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForConditionalGeneration.from_pretrained(model_path, convert_from_hf=True,)
    # change the implementation of attention(default is "eager")
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    model.eval()
    print("Model loaded successfully!")
    return model, processor


def compute_metrics(predictions, references):
    """
    Compute evaluation metrics: Normalized Edit Distance
    """
    total_ned = 0
    num_samples = len(predictions)

    if num_samples == 0:
        return 0.0

    for pred, ref in zip(predictions, references):
        # Compute NED
        dist = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref))
        if max_len > 0:
            total_ned += dist / max_len

    avg_ned = total_ned / num_samples
    return avg_ned


def generate_response(model, processor, messages, max_length=1024):

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd",
    )

    generation_config = GenerationConfig(
        do_sample=False, # greedy_search
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True
    )

    with paddle.no_grad():
        generated_ids = model.generate(**inputs, generation_config=generation_config, max_new_tokens=max_length)
        output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True)

    return output_text


def main():
    start_time = time.time()
    args = parse_args()

    # Initialize distributed environment
    try:
        dist.init_parallel_env()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except Exception:
        rank = 0
        world_size = 1
        print("Distributed environment not detected, using single card mode.")

    # 1. Load Model
    model, processor = load_model_and_processor(args.model_name_or_path, args.device)

    # 2. Read Data
    if rank == 0:
        print(f"Reading data: {args.data_path}")
    samples = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Data Splitting
    total_samples = len(samples)
    samples = samples[rank::world_size]

    if rank == 0:
        print(f"Total test samples loaded: {total_samples}")
    print(f"[Rank {rank}] Assigned {len(samples)} samples")

    # 3. Inference Loop
    results = []
    # Predictions and references will be aggregated by rank 0 at the end

    for sample in tqdm(samples, desc=f"[Rank {rank}] Inferencing", position=rank):

        query = sample["messages"][0]["content"]
        image_path = sample["images"][0]
        image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": query.replace('<image>', '')},
                ],
            }
        ]
        output = generate_response(model, processor, messages, args.max_length)
        sample["answer"] = output[0]
        sample["label"] = sample["messages"][1]["content"]

        results.append(sample)

    # 4. Save partial results
    part_file = f"{args.output_path}.part{rank}"
    with open(part_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"[Rank {rank}] Results saved to temporary file: {part_file}")

    # Wait for all processes to complete
    if world_size > 1:
        dist.barrier()

    # 5. Rank 0 Aggregation and Evaluation
    if rank == 0:
        all_results = []
        print("Aggregating results from all Ranks...")
        # Collect all partial results
        for r in range(world_size):
            part_file_r = f"{args.output_path}.part{r}"
            if os.path.exists(part_file_r):
                with open(part_file_r, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_results.append(json.loads(line))
                # Remove temporary file
                try:
                    os.remove(part_file_r)
                except OSError as e:
                    print(f"Warning: Unable to remove temporary file {part_file_r}: {e}")
            else:
                print(f"Warning: Result file {part_file_r} for Rank {r} not found")

        # Extract predictions and labels
        predictions = [res.get("answer", "") for res in all_results]
        references = [res.get("label", "") for res in all_results]

        # Compute metrics
        print("Computing evaluation metrics...")
        avg_ned = compute_metrics(predictions, references)

        # Output results
        print("\n" + "="*40)
        print("        Evaluation Report")
        print("="*40)
        print(f"Model: {args.model_name_or_path}")
        print(f"Total Samples: {len(all_results)}")
        print("-" * 40)
        print(f"Avg. NED: {avg_ned:.4f} (Lower is better)")
        print("="*40)

        # Save detailed results
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for res in all_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"\nDetailed results saved to: {args.output_path}")

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
```

在 1*A800-80 G 上推理时长约为 53 分钟。评估结果保存在 `output_path=./PaddleOCR-VL-SFT-Bengali_eval_result.jsonl` 文件中。

微调前后的模型测试集评估结果如下：

|Model|Avg. NED|
|-|-|
|PaddleOCR-VL|0.8214|
|PaddleOCR-VL-Bengali-SFT (Full)|0.0065|
|PaddleOCR-VL-Bengali-SFT (LoRA)|0.0064|

## 6.3 部署推理

部署 PaddleOCR-VL 模型，请参考 [基于 FastDeploy / vLLM 部署模型](../../../docs/zh/deployment_guide.md)和 [FastDeploy - PaddleOCR-VL-0.9B Best Practices](https://paddlepaddle.github.io/FastDeploy/zh/best_practices/PaddleOCR-VL-0.9B/)

# 7. 更多硬件上的使用说明

## 7.1 昆仑芯 P800

XPU 环境配置请参考 [XPU 安装说明文档](../../../docs/zh/XPU_installation_guide.md)

使用以下命令行即可启动全参训练：

```shell
export FLAGS_use_stride_kernel=True

XPU_VISIBLE_DEVICES=0 paddleformers-cli train examples/best_practices/PaddleOCR-VL/paddleocr-vl_full_16k_config.yaml \
                        model_name_or_path=./PaddlePaddle/PaddleOCR-VL \
                        train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl \
                        eval_dataset_path=./ocr_vl_sft-test_Bengali.jsonl \
                        pre_alloc_memory=24 \
                        device=xpu
```

目前使用 1*P800 基于 PaddleFormers 微调 PaddleOCR-VL 模型，资源需求和运行耗时：

|SFT|显存|用时|指标|
|-|-|-|-|
|全参|30G|3h 22min|0.0067|
|LoRA|20G|3h 33min|0.0065|

## 7.2 天数智芯 150s

ILUVATAR-GPU 环境配置请参考 [ILUVATAR-GPU 安装说明文档](../../../docs/zh/ILUVATAR-GPU_installation_guide.md)

使用以下命令行即可启动全参训练：

```shell
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train examples/best_practices/PaddleOCR-VL/paddleocr-vl_full_16k_config.yaml \
                        model_name_or_path=./PaddlePaddle/PaddleOCR-VL \
                        train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl \
                        eval_dataset_path=./ocr_vl_sft-test_Bengali.jsonl \
                        per_device_train_batch_size=2 \
                        per_device_eval_batch_size=2 \
                        gradient_accumulation_steps=32 \
                        attn_impl=sdpa \
                        pre_alloc_memory=18 \
                        device=iluvatar_gpu
```

由于天数150s 的 `Conv2d` 算子对长序列性能不友好，我们选择将 `per_device_train_batch_size` 设为 2，并将 `gradient_accumulation_steps` 设为 32。由于天数150s 不支持 FlashMask 算子，我们选择 SDPA 算子用于计算注意力。

目前使用 1*天数智芯150s 基于 PaddleFormers 微调 PaddleOCR-VL 模型，资源需求和运行耗时：

|SFT|显存|用时|指标|
|-|-|-|-|
|全参|24G|14h 24min|0.0066|
|LoRA|13G|13h 50min|0.0062|

## 7.3 Nvidia-4090D

使用以下命令行即可启动全参训练：

```shell
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train examples/best_practices/PaddleOCR-VL/paddleocr-vl_full_16k_config.yaml \
                        model_name_or_path=./PaddlePaddle/PaddleOCR-VL \
                        train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl \
                        eval_dataset_path=./ocr_vl_sft-test_Bengali.jsonl \
                        per_device_train_batch_size=4 \
                        per_device_eval_batch_size=4 \
                        gradient_accumulation_steps=16 \
                        pre_alloc_memory=18
```

由于 4090D 显存为 24G，我们选择将 `per_device_train_batch_size` 设为 4，并将 `gradient_accumulation_steps` 设为 16。

目前使用 1*4090D 基于 PaddleFormers 微调 PaddleOCR-VL 模型，资源需求和运行耗时：

|SFT|显存|用时|指标|
|-|-|-|-|
|全参|24G|3h 17min|0.0068|
|LoRA|13G|2h 01min|0.0062|

# 8. 注意事项

## 8.1. 表格/公式/图表数据格式

特别地，表格/公式/图表数据使用特殊的识别格式：

**表格数据：OTSL 格式**

<div align="center">
  <img width="586" height="432" alt="table_example" src="https://github.com/user-attachments/assets/744d8e04-2cd0-498c-babb-5e1560ce600e" />
</div>

```json
{
    "messages": [
        {"role": "user", "content": "<image>Table Recognition:"},
        {"role": "assistant", "content": "<fcel>分组<fcel>频数<fcel>频率<nl><fcel>[41,51)<fcel>2<fcel>\\( \\frac{2}{30} \\)<nl><fcel>[51,61)<fcel>1<fcel>\\( \\frac{1}{30} \\)<nl><fcel>[61,71)<fcel>4<fcel>\\( \\frac{4}{30} \\)<nl><fcel>[71,81)<fcel>6<fcel>\\( \\frac{6}{30} \\)<nl><fcel>[81,91)<fcel>10<fcel>\\( \\frac{10}{30} \\)<nl><fcel>[91,101)<fcel>5<fcel>\\( \\frac{5}{30} \\)<nl><fcel>[101,111)<fcel>2<fcel>\\( \\frac{2}{30} \\)<nl>"}
    ],
    "images": ["./assets/table_example.png"]
 }
```

更多的 OSTL 格式控制符以及具体意义如下所示：

1. `<ecel>`: 结束当前单元格（End Cell）。用于标记单元格的结束。
2. `<fcel>`: 开始一个新的单元格（First Cell）。通常用于表格中的第一个单元格。
3. `<xcel>`: 开始一个新的单元格（eXtended Cell）。用于表格中除第一个单元格外的其他单元格。
4. `<lcel>`: 结束当前行并开始新行（Last Cell）。用于标记一行的结束。
5. `<ucel>`: 合并单元格（Union Cell）。用于表示跨多行或多列的合并单元格。
6. `<nl>`: 换行（New Line）。用于文本中的换行操作。

表格建议使用 PPOCRLabel 标注（[PaddleX - 表格数据标注教程](https://github.com/PaddlePaddle/PaddleX/blob/release/3.3/docs/data_annotations/ocr_modules/table_recognition.md)），标注结果为 HTML 格式表格，可以使用脚本将 HTML 格式转化为 OTSL 格式。

[附件]
模型识别输出的 OTSL 格式表格，可以参考 [convert_otsl_to_html](https://github.com/PaddlePaddle/PaddleX/blob/release/3.3/paddlex/inference/pipelines/paddleocr_vl/uilts.py#L810) 将 OTSL 转 HTML 来观察识别得到的表格是否符合预期。

**公式数据: Latex 格式**

<div align="center">
  <img alt="formula_example" src="https://github.com/user-attachments/assets/93150e31-fd7b-4799-8139-919b5e838409" />
</div>

```json
{
    "messages": [
        {"role": "user", "content": "<image>Formula Recognition:"},
        {"role": "assistant", "content": "\\[t_{n}\\in[0,\\infty]\\]"}
    ],
    "images": ["./assets/formula_example.jpg"]
 }
```

**图表数据：Markdown 格式**

<div align="center">
  <img width="611" height="288" alt="chart_example" src="https://github.com/user-attachments/assets/575389c8-7691-4ffc-9dc3-2b972bbd8d4f" />
</div>

```json
{
    "messages": [
        {"role": "user", "content": "<image>Chart Recognition:"},
        {"role": "assistant", "content": "  | 22Q3 | 22Q3yoy\n电商 | 85 | 100%\n川渝 | 140 | 8%\n云贵陕 | 95 | 12%\n外围地区 | 45 | 20%"}
    ],
    "images": ["./assets/chart_example.png"]
 }
```
