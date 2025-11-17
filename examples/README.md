## 0. 环境变量

在运行前，可以通过设置环境变量 `DOWNLOAD_SOURCE` 来指定模型的下载源，默认使用 **huggingface**。

目前支持的下载源包括：
- [huggingface](https://huggingface.co)
- [modelscope](https://modelscope.cn/home)
- [aistudio](https://aistudio.baidu.com/overview)


示例：
```bash
# 使用 modelscope
export DOWNLOAD_SOURCE=modelscope

# 使用 aistudio
export DOWNLOAD_SOURCE=aistudio
```

训练前请先准备数据集，参考：

英文：
- [数据集格式说明及demo数据下载](../docs/zh/datasets_format_zh.md)
- [数据流参数说明](../docs/zh/datasets_zh.md)

英文：
- [数据集格式说明及demo数据下载](../docs/en/datasets_format.md)
- [数据流参数说明](../docs/en/datasets.md)

## 1. 预训练

### 1.1. 全参 PT

预训练需要在配置文件中指定 `stage: PT`

在线数据流
```bash
# 单卡
paddleformers-cli train ./config/pt/full.yaml
# 多卡
paddleformers-cli train ./config/pt/full_tp_pp.yaml
```

离线数据流

在配置文件中：

`input_dir`指定数据集的前缀，例如：数据集 `data-1-part0.bin` 需要设置为 `input_dir: "1.0 ./data-1-part0"`，`1.0` 为数据配比；

`split` 字段为 `train/eval` 的分配比例，如：`split: "998,2"`, 其中`train`为训练集，`eval`为评估集

`dataset_type` 指定为 `pretrain`，例如：`dataset_type: "pretrain"`

```bash
paddleformers-cli train ./config/pt/full_offline_data.yaml
```

### 1.2. LoRA PT

LoRA SFT 启动命令参考
```bash
# 单卡
paddleformers-cli train ./config/pt/lora.yaml
# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train ./config/pt/lora_tp_pp.yaml
```

## 2. 精调

### 2.1 全参 SFT

```bash
# 单卡
paddleformers-cli train ./config/sft/full.yaml
# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train ./config/sft/full_tp_pp.yaml
```

### 2.2 LoRA SFT

LoRA SFT 启动命令参考
```bash
paddleformers-cli train ./config/sft/lora.yaml
```


## 3. 对齐

### 3.1 全参 DPO

```bash
# 单卡
paddleformers-cli train ./config/dpo/full.yaml
# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 paddleformers-cli train ./config/dpo/full_tp_pp.yaml
```

### 3.2 LoRA DPO

LoRA DPO 启动命令参考
```bash
paddleformers-cli ./config/dpo/lora.yaml
```

## 4. LoRA 参数合并

使用 LoRA 方式训练模型后，为了方便推理，我们提供将 LoRA 参数合并到模型主权重中的脚本`paddleformers-cli export`。

运行示例（默认加载和保存 **HuggingFace** 权重参数）：

```bash
# model_name_or_path 为完整模型路径
# output_dir为lora训练保存的ckpt路径
# paddleformers-cli export会将合并完的权重保存到output_dir/export下面
paddleformers-cli export examples/config/run_export.yaml \
    output_dir=${lora_model_path} \
    model_name_or_path=${base_model_path}
```

### Paddle 权重使用说明

如需使用 **Paddle** 格式权重，需要在启动脚本中添加 `convert_from_hf=False` 和 `save_to_hf=False` 参数。

```bash
paddleformers-cli export examples/config/run_export.yaml \
    output_dir=${lora_model_path} \
    model_name_or_path=${base_model_path} \
    convert_from_hf=False \
    save_to_hf=False
```
