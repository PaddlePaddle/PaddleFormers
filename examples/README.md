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


## 1. 预训练

### 1.1. 数据准备

#### 1.1.1. 在线数据流

我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `text` : `str, List(str)`, 预训练文本。

样例数据：

```text
{"text": ["一个需要连续输入值的分类问题的示例是房屋价格预测。房屋的价格通常基于诸如平方英尺、位置、卧室和浴室数量以及像后院或车库等功能这样的因素定价。为了准确预测房屋价格，这些标准必须作为连续输入值输入到分类模型中。"]}
...
```

为了方便测试，我们也提供了[demo 数据集](https://paddleformers.bj.bcebos.com/datasets/pt_data.tar.gz)可以直接使用：

```shell
wget https://paddleformers.bj.bcebos.com/datasets/pt_data.tar.gz
mkdir -p data/pt && tar -xf pt_data.tar.gz -C data/sft/
```

#### 1.1.2. 离线数据流

我们也可以选择使用离线的比特预训练数据流，更节省内存。离线数据流制作方法如下：

下载一个文本数据集，例如 https://modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset

格式需为jsonl，每行格式例如BazingaLyn/mini_pretrain_dataset/pretrain_hq_v7.jsonl：
```text
{"text": "番茄炒蛋\n材料：\n鸡蛋3个、番茄1个、油、盐、糖、水淀粉\n做法：..."}
{"text": "请描述一下如何正确规划个人理财。正确规划个人理财需要以下几个步骤..."}
{"text": "请输入一段描述有关海洋保护的情景对话。Person A: 哇，这个海滩真..."}
{"text": "鉴别两种不同类型的葡萄酒。鉴别葡萄酒的方法因其类型和品种而异，下..."}
```

运行`examples/tools/create_pretraining_data.py`，生成数据将会保存在当前目录下的`./pretrain_data.bin`和`./pretrain_data.idx`
```text
python -u examples/tools/create_pretraining_data.py \
    --model_name_or_path "/path/to/your/Qwen3-0.6B-base" \
    --data_format "JSON" \
    --input_path "/path/to/your/BazingaLyn/mini_pretrain_dataset/pretrain_hq_v7.jsonl" \
    --append_eos \
    --output_prefix "./pretrain_data"  \
    --workers 1 \
    --log_interval 10000 \
    --data_impl "mmap"
```

- 参数说明
 
| 参数名              | 类型        | 说明                 |
|--------------------|----------- |-----------------|
| `--model_name_or_path`     | string     | 模型路径  |
| `--data_format`    | string     | 支持的文件格式，之前只支持 json |
| `--input_path`     | string     | 输入的json文件的路径  |
| `--append_eos`     | store_true | 是否在document的结尾添加eos token  |
| `--output_prefix`  | str        | 输出文件的前缀    |
| `--workers`        | int        | 运行的进程数     |
| `--log_interval`   | int        | 打印日志间隔   |
| `--data_impl`      | str        | 制作的数据集类型，默认为mmap，也可以选择lazy |

### 1.2. 全参 PT

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

### 1.3. LoRA PT

LoRA SFT 启动命令参考
```bash
# 单卡
paddleformers-cli train ./config/pt/lora.yaml
# 多卡
paddleformers-cli train ./config/pt/lora_tp_pp.yaml
```

## 2. 精调

### 2.1 数据准备

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
mkdir -p data/sft && tar -xf alpaca_demo.gz -C data/sft/ --strip-components=1
```

### 2.2 全参 SFT

单卡
```bash
python -u run_finetune.py ./config/sft/full.yaml
```

多卡
```bash
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py ./config/sft/full_tp_pp.yaml
```

### 2.3 LoRA SFT

LoRA SFT 启动命令参考
```bash
python -u run_finetune.py ./config/sft/lora.yaml
```


## 3. 对齐

### 3.1 数据准备

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
mkdir -p data/dpo && tar -zxf ultrafeedback_binarized.tar.gz -C data/dpo/ --strip-components=1
```

### 3.2 全参 DPO

单卡
```bash
python -u ./alignment/dpo/run_dpo.py ./config/dpo/full.yaml
```

多卡
```bash
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/dpo/full_tp_pp.yaml
```

### 3.3 LoRA DPO

LoRA DPO 启动命令参考
```bash
python -u ./alignment/dpo/run_dpo.py ./config/dpo/lora.yaml
```

### 3.4 function-call DPO
#### 3.4.1 数据准备
我们支持的 function-call DPO 训练数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：
- `messages` : `List(dict)`, 对话历史列表。
  - 普通轮次：包含 `role` (`"user"` 或 `"assistant"`) 和 `content` (`str`) 字段。
  - 偏好/非偏好轮次（用于偏好学习）：包含以下两个关键字段，用于表示对同一用户查询的不同系统回复的偏好排序。
    - `preferred_output` : `dict`, 偏好（chosen）的系统回复，包含 `role` (`"assistant"`) 和 `content` (`str`) 等字段，根据是否调用工具可能包含工具调用信息 (`tool_calls`)。
    - `non_preferred_output` : `dict`, 非偏好（rejected）的系统回复，包含 `role` (`"assistant"`) 和 `content` (`str`) 等字段。
- `tools` : `List(dict)`, 对话中可能用到的工具（函数）的定义列表。
- `label` : `List(int)`, 用于区分 `preferred_output` 和 `non_preferred_output` 的排序标签。其中 0 对应 `non_preferred_output` (rejected)， 1 对应 `preferred_output` (chosen)。

详细的数据格式可见[function call说明](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/examples/best_practices/function_call.md)

样例数据
```text
{
    "messages": [
        {
            "role": "system",
            "content": "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n<tools>\n[{'type': 'function', 'function': {'name': 'play_music', 'description': 'Play music from a specified playlist or genre', 'parameters': {'type': 'object', 'properties': {'playlist': {'type': 'string', 'description': 'The playlist to play'}, 'genre': {'type': 'string', 'description': 'The genre of music to play'}}, 'required': []}}}, {'type': 'function', 'function': {'name': 'analyze_sentiment', 'description': 'Analyze the sentiment of a text', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': 'The text to analyze'}, 'language': {'type': 'string', 'description': 'The language of the text (optional)'}}, 'required': ['text']}}}]\n</tools>\nFor each function call return a json object with function name and arguments within <tool_call> </tool_call> XML tags with the following schema:\n<tool_call>\n{'arguments': <args-dict>, 'name': <function-name>}\n</tool_call>\n"
        },
        {
            "role": "user",
            "content": "I want to listen to some music. Can you play something for me?"
        },
        {
            "preferred_output": {
                "role": "assistant",
                "content": "Of course! Do you have a specific playlist or genre in mind?"
            },
            "non_preferred_output": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "play_music",
                            "arguments": "{\n\t\"playlist\": \"Top hits\"\n}"
                        }
                    }
                ]
            }
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "play_music",
                "description": "Play music from a specified playlist or genre",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "playlist": {
                            "type": "string",
                            "description": "The playlist to play"
                        },
                        "genre": {
                            "type": "string",
                            "description": "The genre of music to play"
                        }
                    },
                    "required": []
                }
            }
        },
    ],
    "label": [
        1,
        0
    ]
}
```
为了方便测试，我们也提供了 function-call DPO 数据集可以直接使用：
```bash
wget https://paddleformers.bj.bcebos.com/datasets/dpo_function_call_1k.tar.gz

mkdir -p data/dpo_fc && tar -zxf dpo_function_call_1k.tar.gz -C data/dpo_fc/
```

function call DPO 启动命令参考:
```bash
python -u ./alignment/dpo/run_dpo.py ./config/dpo/full_function_call.yaml
```


## 4. LoRA 参数合并

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
