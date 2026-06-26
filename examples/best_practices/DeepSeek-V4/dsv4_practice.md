# DeepSeek-V4-Flash 4K SFT 实践

## 1. 背景说明

DeepSeek-V4 引入了 CSA+HCA 混合注意力、mHC 残差连接、Muon 优化器等全新架构，我们在 PaddleFormers 中实现了对 DeepSeek-V4 系列模型训练的全面支持，涵盖 PP（流水线并行）、EP（专家并行）、CP（上下文并行）、DeepEP 专家通信、Muon 优化器、BF16 混合精度、重计算、RoPE fusion、高性能 FP8 MoE 融合模块、 fused mHC kernel 以及 flex checkpoint 等训练优化能力。

本文档以 DeepSeek-V4-Flash 模型的 4K 序列长度 SFT 为例，介绍完整的训练实践流程。

## 2. 硬件与软件要求

### 2.1 硬件配置

本次实践使用 4 台 GPU 机器进行验证，每台机器 8 张 NVIDIA GPU，总计 32 张 GPU。

不同卡型的显存容量、互联带宽和算子支持存在差异，可以根据需要调整配置中的 PP size、EP size、batch size、重计算、offload 优化器等配置。

### 2.2 软件配置

| 组件 | 版本或提交 |
| --- | --- |
| PaddleFormers | `1.2.0` |
| PaddleFleet | `0.3.0` |
| paddlefleet-ops | `0.3.0` |
| PaddlePaddle GPU | `3.4.0` |
| CUDA | 13.2 |
| cuda-tile | `1.4.0` |
| Python | 3.12 |


可按如下方式安装支持版本：
pip install paddleformers==1.2.0.post0
pip install paddlefleet==0.3.0 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/

## 3. 模型准备

本文使用 DeepSeek 官方发布的 DeepSeek-V4-Flash-Base 权重。开发者可以从 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) 下载：

下载完成后，通过 dequant_fp8_to_bf16.py 脚本将权重反量化成 bf16格式。

转换后的权重目录需要包含 DSV4-Flash 的模型配置、tokenizer 文件、权重索引和权重分片。结构如下：

```text
DeepSeek-V4-Flash-Base/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-00046.safetensors
├── ...
├── model-00046-of-00046.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

## 4. 数据准备

本次实践使用 ErnieKit 格式的 SFT 数据，每行是一条 JSON 数据，包含 `src` 和 `tgt` 字段：

```json
{"src": "Give three tips for staying healthy.", "tgt": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."}
{"src": "What are the three primary colors?", "tgt": "The three primary colors are red, blue, and yellow."}
```

## 5. 启动训练

本次实践使用 4 机 32 卡启动训练，每台机器需要能访问同一份代码、模型目录和数据目录等，建议目录结构如下：

```text
DeepSeek-V4/
├── .venv/
├── PaddleFleet/
├── PaddleFormers/
├── DeepSeek-V4-Flash-Base/
├── data/
│   └── sft/
│       ├── train.jsonl
│       └── dev.jsonl
└── dsv4_sft_4K.yaml
```

启动训练需要 4 台机器同时执行：
```bash
source .venv/bin/activate

NNODES=${NNODES} \
RANK=${RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
paddleformers-cli train dsv4_sft_4K.yaml
```

如果需要训练开启 PaddleFormers 高效训练所需的融合 kernel 开关，需要显式修改 `DeepSeek-V4-Flash-Base/config.json`，添加以下字段：

```json
  "use_fused_mhc": true,
  "csa_indexer_backend": "tilelang",
  "csa_sparse_attn_backend": "cudnn",
  "use_fast_hadamard": true,
```
