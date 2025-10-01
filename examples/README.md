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

### Paddle 权重使用说明

使用 **Paddle** 格式权重，需要在配置文件（如 `sft_full.yaml`、`sft_lora.yaml`等）中手动添加以下参数，以避免与 **HuggingFace** 格式冲突：

```yaml
model_name_or_path: your_model_name_or_path
convert_from_hf: false
save_to_hf: false
```


## 1. 精调

### 1.1 数据准备

我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：

```text
{"src": "Give three tips for staying healthy.", "tgt": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."}
...
```

为了方便测试，我们也提供了[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)demo 数据集可以直接使用：

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/examples/alpaca_demo.gz
tar -xvf alpaca_demo.gz
```

### 1.2 全参 SFT

单卡
```bash
python -u run_finetune.py ./config/sft_full.yaml
```

多卡
```bash
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py ./config/sft_full.yaml
```

### 1.3 LoRA SFT

LoRA SFT 启动命令参考
```bash
python -u run_finetune.py ./config/sft_lora.yaml
```


## 2. 对齐

### 2.1 数据准备

我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `src` : `str, List(str)`, 用户对话内容。
- `tgt` : `str, List(str)`, 系统回复内容。
- `response` : `str, List(str)`, 包含 chosen 和 rejected 回复。
- `sort` : `List(int)`, sort 值用于区分 response 中 chosen 和 rejected（sort 值小的是 rejected，sort 值大的是 chosen）。

样例数据：

```text
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?",
        "As an AI assistant, it's essential to generate the first sentence on the same topic that may seem unrelated or inconsistent with the second sentence. Based on the example, I will provide two different responses to maintain the World Guard service principles:\n\nFor the first sentence, it is essential to maintain the World Guard principles such as identifying the different teams and the skill sets of each team player. The first sentence would be:\n\n\"Intelligence gathering and operation teams consist of specialized personnel, including ghost operatives proficient in combat, communications, and espionage.\"\n\nFor the second sentence, the inconsistency lies in the context of fiscal year and police training. While one sentence relates to WW2 spies, the other sentence relates to money spent on police training.\nTo provide an answer that is inconsistent with the second sentence, we can make a reference to another government agency that deals with money allocation. Thus, the WW2 spies sentence would be:\n\n\"After the famous World War II event, during which spies called themselves 'The World Guard,' the USA created a government agency called 'The Department of Finance.' Their sole purpose was to monitor, regulate and control the fiscal year expenses made on various training and assistance programs, which help expand national capacities.\"\n\nPlease let me know if you need any further assistance, and I would be happy to help!"
        ],

    "sort": [1, 0]
}
...
```

为了方便测试，我们也提供了偏好数据集可以直接使用：

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```

### 2.2 全参 DPO

单卡
```bash
python -u ./alignment/dpo/run_dpo.py ./config/dpo_full.yaml
```

多卡
```bash
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/dpo_full.yaml
```

### 2.3 LoRA DPO

LoRA DPO 启动命令参考
```bash
python -u ./alignment/dpo/run_dpo.py ./config/dpo_lora.yaml
```


## 3. LoRA 参数合并

使用 LoRA 方式训练模型后，为了方便推理，我们提供将 LoRA 参数合并到模型主权重中的脚本`tools/mergekit.py`。

运行示例（默认加载和保存 **HuggingFace** 权重参数）：

单卡
```bash
python -u ./tools/mergekit.py \
    --lora_model_path ${lora_model_path} \
    --model_name_or_path ${base_model_path} \
    --output_path ${merged_output_path}
```

多卡
```bash
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./tools/mergekit.py \
    --lora_model_path ${lora_model_path} \
    --model_name_or_path ${base_model_path} \
    --output_path ${merged_output_path}
```

### Paddle 权重使用说明

如需使用 **Paddle** 格式权重，需要在启动脚本中添加 `--convert_from_hf False` 和 `--save_to_hf False` 参数。

单卡
```bash
python -u ./tools/mergekit.py \
    --lora_model_path ${lora_model_path} \
    --model_name_or_path ${base_model_path} \
    --output_path ${merged_output_path} \
    --convert_from_hf False \
    --save_to_hf False
```

多卡
```bash
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./tools/mergekit.py \
    --lora_model_path ${lora_model_path} \
    --model_name_or_path ${base_model_path} \
    --output_path ${merged_output_path} \
    --convert_from_hf False \
    --save_to_hf False
```
