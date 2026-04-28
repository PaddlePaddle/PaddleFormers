# 1. 任务简介

本教程旨在提供基于 PaddleFormers 微调 DeepSeek-OCR-2 模型识别孟加拉语的微调教程，值得一提的是 DeepSeek-OCR-2 已经具有孟加拉语的识别能力，此教程着重展示微调的流程，资源需求和运行耗时见下方表格：

|硬件|SFT|显存|用时|
|-|-|-|-|
|8*A800|全参|52|22min|
|8*A800|LoRA|36|20min|

# 2. 任务准备

## 2.1 模型准备

PaddleFormers 通过在训练配置文件中指定字段`model_name_or_path`来设置所用的模型。启动训练时如果本地没有该模型的缓存，那么 PaddleFormers 会自动下载模型并加载使用。

您也可以将对应的字段指定成您的本地路径，来加载已经下载好的模型。

## 2.2 数据集准备

**Demo 数据**

为了方便起见，我们也提供了一个快速上手的孟加拉语数据集（训练集和测试集），可用于微调 DeepSeek-OCR-2 对孟加拉语进行识别，使用以下命令下载：

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
                * 文字识别（无布局） `"Free OCR. "`
                * 文档识别（带布局） `"<|grounding|>Convert the document to markdown. "`
                * 或者根据微调任务自定义提示

            * 图片占位符 `<image>`：在文本数据中标记图片插入的位置。

        * 输出端包含模型预期生成的正确答案，即图片中需要识别的字符。

* `images`：图像数据列表，存储了对话中涉及到的图片路径（本地路径或 URL）。

**自行准备数据**

如果您想要基于自己的数据集进行训练，请参考 [数据集格式说明](../../../docs/zh/dataset_format.md)准备数据。



# 3. 训练配置

我们针对孟加拉语示例数据集提供了配置文件，其中的关键训练超参数如下：

* `num_train_epochs=1`：训练的 epoch 数。
* `warmup_ratio=0.01`：线性预热步数, 建议设置成训练步数的 1%。
* `per_device_train_batch_size=8`：每张卡的 batch size 大小，建议根据显存占用情况调整。
* `max_seq_len=8192`：最大序列长度，超出该长度的数据将被截断或者丢弃。建议在训练前估计数据集中数据长度的范围，防止大部分数据被截断从而影响训练效果。
* `gradient_accumulation_steps=1`：梯度累积步数。
    * 每达到该步数整数倍更新一次模型参数。
    * 当显存不足时，可以减小 `per_device_train_batch_size` 并增大 `gradient_accumulation_steps`。
    * 用时间换空间策略，可以减少显存占用，但会延长训练时间。

* `learning_rate`：学习率，即每次参数更新的幅度。
    * 全参训练 `learning_rate=5e-6`
    * LoRA 训练 `learning_rate=5e-4`

更多相关参数可在配置文件中查看。

<details>
  <summary><b> 全参配置（点击展开/收起）</b></summary>

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./ocr_vl_sft-train_Bengali.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./ocr_vl_sft-test_Bengali.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 8192
padding_free: False
packing: False
truncate_packing: False
dataset_type: map
dataloader_num_workers: 8
mix_strategy: concat
template_backend: custom
template: deepseek_ocr2

### model
model_name_or_path: deepseek-ai/DeepSeek-OCR-2
_attn_implementation: flashmask
copy_custom_file_list: "configuration_deepseek_v2.py conversation.py deepencoderv2.py modeling_deepseekocr2.py modeling_deepseekv2.py"

### finetuning
# base
stage: VL-SFT
fine_tuning: full
seed: 42
do_train: true
do_eval: true
per_device_eval_batch_size: 8
per_device_train_batch_size: 8
num_train_epochs: 1
max_steps: -1
max_estimate_samples: 500
eval_steps: 400
evaluation_strategy: steps
save_steps: 400
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 1
logging_dir: ./Deepseek-OCR2-Bengali/visualdl_logs/
output_dir: ./Deepseek-OCR2-SFT-Bengali
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
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2
pre_alloc_memory: 52
freeze_config: freeze_vision | freeze_aligner

# save
unified_checkpoint: False
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
```
</details>


<details>
  <summary><b> LoRA 配置（点击展开/收起）</b></summary>

```yaml
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: ./ocr_vl_sft-train_Bengali.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: ./ocr_vl_sft-test_Bengali.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 8192
padding_free: False
packing: False
truncate_packing: False
dataset_type: map
dataloader_num_workers: 8
mix_strategy: concat
template_backend: custom
template: deepseek_ocr2

### model
model_name_or_path: deepseek-ai/DeepSeek-OCR-2
_attn_implementation: flashmask
lora: true
lora_rank: 8
lora_alpha: 32
copy_custom_file_list: "configuration_deepseek_v2.py conversation.py deepencoderv2.py modeling_deepseekocr2.py modeling_deepseekv2.py"

### finetuning
# base
stage: VL-SFT
fine_tuning: lora
seed: 42
do_train: true
do_eval: true
per_device_eval_batch_size: 8
per_device_train_batch_size: 8
num_train_epochs: 1
max_steps: -1
max_estimate_samples: 500
eval_steps: 400
evaluation_strategy: steps
save_steps: 400
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 1
logging_dir: ./Deepseek-OCR2-Bengali-lora/visualdl_logs/
output_dir: ./Deepseek-OCR2-SFT-Bengali-lora
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
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2
pre_alloc_memory: 36
freeze_config: freeze_vision | freeze_aligner

# save
unified_checkpoint: False
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
```

</details>

# 4. SFT 训练

## 4.1 SFT 全参训练

使用以下命令行即可启动全参训练：

```shell
CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli train examples/best_practices/DeepSeek-OCR-2/deepseek_ocr2_full_8k_config.yaml.yaml \
                        model_name_or_path=deepseek-ai/DeepSeek-OCR-2 \
                        train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl \
                        eval_dataset_path=./ocr_vl_sft-test_Bengali.jsonl \
                        pre_alloc_memory=52
```

设置 `pre_alloc_memory` 预分配显存从而减少显存碎片，根据序列长度、批大小和硬件显存调整。

PaddleFormers 默认使用机器上的全部 GPU，可以通过环境变量 `CUDA_VISIBLE_DEVICES` 设置 PaddleFormers 能够使用的 GPU。

GPU 的数目 `GPU_num` 会影响训练超参数 `learning_rate & per_device_train_batch_size & gradient_accumulation_steps` 配置。理论上，每个更新步使用的样本数目 `sample_num = G*B*A`，近似与学习率 `learning_rate` 成正线形关系，因此，当 GPU 数目增加 `N` 倍变为 `N*GPU` 时，有两种调整方式：

1. 保持 `sample_num` 不变

    * 将 `per_device_train_batch_size` 减少 `x` 倍，变成 `per_device_train_batch_size/x`
    * 将 `gradient_accumulation_steps` 减少 `y` 倍，变成 `gradient_accumulation_steps/y`
    * 满足 `x*y = N` 即可

2. 将 `learning_rate` 增加 `N` 倍，变成 `N*learning_rate`

可以通过 `visualdl` 对训练过程可视化，使用以下命令行即可启动（下方命令将端口 port 设置为 `8084`，需要根据实际情况设置可用端口）：

```shell
visualdl --logdir ./Deepseek-OCR2-SFT-Bengali/visualdl_logs/ --port 8084
```

成功启动后该服务后，在浏览器输入 `ip:port` ，则可以看到训练日志（通过 `hostname -i` 命令可以查看机器的 ip 地址）。

损失曲线如下：

<div align="center">
  <img width="500" alt="table_test_example" src="https://github.com/forBlank/PaddleFormers/blob/paddleocr_vl_v15_doc/examples/best_practices/DeepSeek-OCR-2/assets/deepseek_orc2_train_loss.png" />
</div>


## 4.2 SFT LoRA 训练

使用以下命令行即可启动 LoRA 训练：

```shell
CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli train examples/best_practices/DeepSeek-OCR-2/deepseek_ocr2_lora_8k_config.yaml.yaml \
                        model_name_or_path=deepseek-ai/DeepSeek-OCR-2 \
                        train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl \
                        eval_dataset_path=./ocr_vl_sft-test_Bengali.jsonl\
                        pre_alloc_memory=36
```

# 5. 模型结构说明

## 5.1 SFT 全参

全参训练结束后，模型会保存在 `output_dir=./Deepseek-OCR2-SFT-Bengali` 指定路径下，其中包含：

* config.json：模型配置文件
* model-0000X-of-0000Y.safetensors：模型权重文件
* model.safetensors.index.json：模型权重索引文件
* tokenizer.json & tokenizer_config.json：分词器文件
* train_args.bin：训练参数文件，记录训练使用的参数等
* train_state.json：训练状态文件，记录训练步数和最优指标等
* train_results.json & all_results.json：训练结果文件，记录训练进度&用时&每步耗时&每样本耗时等
* generation.json：生成配置文件
* checkpoint-[save_steps*n]：检查点文件夹，在 `save_steps` 整数倍保存训练状态，除以上文件外，还会保存 master-weight & optimizer-state & scheduler-state 等，可用于训练中断后恢复训练

## 5.2 SFT LoRA

LoRA 训练结束后，模型会保存在 `output_dir=./Deepseek-OCR2-SFT-Bengali-lora` 指定路径下。相较于 SFT 全参，SFT LoRA 的模型结构会有所不同，其中包含：

* lora_config.json：LoRA 模型配置文件
* peft_model-0000X-of-0000Y.safetensors：LoRA 模型权重文件
* peft_model.safetensors.index.json：LoRA 权重索引文件

使用以下命令行即可合并 LoRA 权重：

```shell
CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli export examples/best_practices/DeepSeek-OCR-2/deepseek_ocr2_lora_export.yaml \
    model_name_or_path=deepseek-ai/DeepSeek-OCR-2 \
    output_dir=./Deepseek-OCR2-SFT-Bengali-lora
```

合并后的完整模型权重保存在 `output_dir=./Deepseek-OCR2-SFT-Bengali-lora/export` 路径下。

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

<details>
  <summary><b> 单样本推理脚本（点击展开/收起）</b></summary>

```python
from paddleformers.transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

model_name = './Deepseek-OCR2-SFT-Bengali'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, _attn_implementation='sdpa')
model = model.eval().bfloat16()

prompt = "<image>\nFree OCR. "
image_file = "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/bengali_sft/5b/7a/5b7a5c1c-207a-4924-b5f3-82890dc7b94a.png"
output_path = './output'

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 768, crop_mode=True, save_results = False)
```

</details>

预期输出为测试图像中的孟加拉语文字：

```
নট চলল রফযনর পঠ সওযর
হয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয
ঝগড
দরগর কছ চল এল
```


## 6.3 部署推理

使用 vLLM 部署 DeepSeek-OCR-2 模型，请参考 [DeepSeek 官方文档 - vLLM-Inference](https://github.com/deepseek-ai/DeepSeek-OCR-2/#vllm-inference)。
