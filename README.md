**简体中文**🀄 | [English🌎](./README_en.md)

<p align="center">
  <img src="https://github.com/user-attachments/assets/9d1c1937-7fac-48f8-9d61-f7ac67b61b18" align="middle"  width="500" />
</p>

<p align="center">
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFormers/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleFormers?color=ccf"></a>
</p>

<h4 align="center">
  <a href=#最新动态> 最新动态 </a> |
  <a href=#核心特性> 核心特性 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#社区交流> 社区交流 </a>
</h4>


**PaddleFormers** 是一款基于 [飞桨深度学习框架（PaddlePaddle）](https://www.paddlepaddle.org.cn/) 的轻量级大模型开发套件，兼具**简单易用**与**工程高效**的特性。套件围绕大语言模型训练流程提供统一的模型定义接口、模块化训练组件与丰富的分布式训练策略，帮助开发者以简洁高效的方式训练大模型，广泛适用于科研探索与企业级模型开发等多种场景。

## 📣 最新动态

[2025/06/28] 🎉  **PaddleFormers 0.1** 正式发布！作为首个版本，PaddleFormers 支持大语言模型的 SFT、DPO 等训练范式，使用统一 Trainer 接口支持分布式策略配置化训练，并集成 PEFT、MergeKit、量化等能力满足大模型使用多样需求。


## 核心特性

### <a href=#易用分布式策略> ⚙️ 易用分布式策略 </a>

支持4D 分布式训练策略，统一的 Trainer API 实现分布式策略的配置化管理，降低大模型分布式训练使用门槛。

### <a href=#高效后训练优化> 🛠 高效后训练优化 </a>

大模型 SFT 和 DPO 训练支持 Packing 数据流与[FlashMask](https://arxiv.org/abs/2410.01359)高性能算子，降低无效数据填充和计算，显著提升训练吞吐率。


### <a href=#工业级存储方案> 💾  工业级存储方案 </a>
提供 Unified Checkpoint 大模型存储工具，支持训练断点续训和动态资源扩缩容。同时提供异步存储加速最高达 95%，优化器状态压缩节省 78% 存储空间，满足工业级训练对效率与稳定性的双重要求。


## 📦 安装

PaddleFormers 支持 Python 3.8+，安装前请确保已正确安装 [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick) 3.1+ 环境。

```bash
# pip
pip install paddleformers

# 如果需要安装PaddleFormers develop版本
git clone https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers
pip install -e .
```


## 🚀 快速开始

### 大模型文本生成

以下示例展示了如何使用 PaddleFormers `Auto API` 加载 Qwen 模型进行文本生成：


```python
from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype="bfloat16")
input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
outputs = model.generate(**input_features, max_new_tokens=128)
print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
```


### 大模型 SFT 训练

```python
from paddleformers.trl import SFTConfig, SFTTrainer
from datasets import load_dataset
dataset = load_dataset("ZHUI/alpaca_demo", split="train")

training_args = SFTConfig(output_dir="Qwen/Qwen2.5-0.5B-SFT", device="gpu")
trainer = SFTTrainer(
    args=training_args,
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
)
trainer.train()
```

## 🤝 贡献指南

我们欢迎任何形式的贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 [Apache 2.0 许可证](LICENSE)。
