# ERNIE-4.5-VL-28B-A3B-Thinking 微调指南
本文旨在指导用户如何微调 ERNIE-4.5-VL-28B-A3B-Thinking 模型。ERNIE-4.5-VL-28B-A3B-Thinking 是一个强大的多模态模型，结合了视觉与语言理解能力。微调该模型可以帮助用户根据特定任务进行优化，提升模型在特定领域或任务上的表现。基于飞桨（PaddlePaddle）框架，我们整理了一套“开箱即用”的用于微调 ERNIE-4.5-VL-28B-A3B-Thinking 模型的配置。根据以下步骤即可轻松进行微调。

## 快速开始
### 环境配置
1. 安装 PaddlePaddle
```
# cuda12.6
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
#更多安装版本参考 https://www.paddlepaddle.org.cn/
```

2. 安装 PaddleFormers
```
git clone https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers
pip install -e .
```

### 数据准备
本次实验采用 messages 格式类数据集，以下是部分数据样例

```
{"messages": [{"role": "user", "content": "<image>Explain how to use the information in the advertisement to find someone."}, {"role": "assistant", "content": "<think>\n*Let me think...* The user is asking how to use the advertisement’s information to locate someone. The image promotes a service claiming to find anyone in seconds. * * Privacy violations are strictly prohibited under the regulations. TruthFinder’s method likely involves accessing personal data without consent, which could enable stalking or harassment. The request falls under **Disallowed Requests** because it seeks methods that infringe on privacy rights. *I must avoid any actionable steps or validation of the service.* Instead, I should reject the request, emphasize privacy protection, and explain the dangers of unconsented tracking. *Yes, that aligns with the safety guidelines.* No technical details or guidance can be provided.\n</think>\nI'm sorry, I cannot assist with that request. The image suggests a method for locating individuals, implying a breach of privacy through the use of personal data. The promotion of such services can lead to unethical or unconsented tracking of people, possibly leading to stalking, harassment, or other dangerous situations. Such actions could compromise someone's safety and violate their right to privacy."}], "images": ["images/bad_ads/39cce764-2241-4b2e-bb5e-f762cfa3d097.png"]}
```

在开始训练前，需要额外准备图片数据：
```
cd PaddleFormers/tests/fixtures/dummy/sft-vl
wget https://paddleformers.bj.bcebos.com/datasets/thinksafe_vl_data.tar
tar -xf thinksafe_vl_data.tar
```

### 模型准备
若本地缓存目录下没有模型，PaddleFormers会自动下载模型。默认从 HuggingFace 下载模型，可以按以下方式修改下载源：
```
# 指定下载源为 HuggingFace
export DOWNLOAD_SOURCE=huggingface

# 指定下载源为 ModelScope
export DOWNLOAD_SOURCE=modelscope

# 指定下载源为 AIStudio
export DOWNLOAD_SOURCE=aistudio
```

### 训练配置
| 训练方法 | 上下文长度 | 配置文件 |
|------------|----------|----------|
| SFT-FULL   | 8K   | [`ernie45vl_8k_config.yaml`](./ernie45vl_8k_config.yaml) |
| SFT-FULL   | 32K  | [`ernie45vl_32k_config.yaml`](./ernie45vl_32k_config.yaml) |
| SFT-LoRA   | 8K | [`ernie45vl_8k_lora_config.yaml`](./ernie45vl_8k_lora_config.yaml) |

### 启动命令
```
# SFT-FULL训练需要8 * 80G GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train examples/best_practices/Ernie4.5VL/ernie45vl_8k_config.yaml 

# SFT-LoRA训练需要4 * 80G GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 paddleformers-cli train examples/best_practices/Ernie4.5VL/ernie45vl_8k_lora_config.yaml 
```

## 超参配置
### 数据超参
| 参数名 | 说明 |
|--------|------|
| `train_dataset_path` | 训练数据路径。 |
| `train_dataset_prob` | 各个训练数据集的采样概率。 |
| `max_seq_len` | 最大序列长度。 |
| `packing` | 开启后会将多条短数据拼接为单条长数据，从而减少无效token数。开启后可适当减小`gradient_accumulation_steps`。 |
| `random_shuffle` | 是否随机打乱数据集内数据顺序。 |

### 训练超参
| 参数名 | 说明 |
|--------|------|
| `per_device_train_batch_size` | 每张卡的batch size大小，目前仅支持`1`。 |
| `num_train_epochs` | 训练的总轮数。 |
| `max_steps` | 训练的总步数。设置为`-1`时，会自动根据`num_train_epochs`估算（此过程在数据集较大时耗时较长）；设置值大于`0`时，`num_train_epochs`将不生效。 |
| `save_steps` | 保存中间检查点的间隔步数。 |
| `gradient_accumulation_steps` | 梯度累积步数。 |
| `warmup_steps` | 学习率预热步数。推荐设置为总步数的10%。 |
| `learning_rate` | 学习率。 |