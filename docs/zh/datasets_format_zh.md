# 数据流格式说明文档

## 数据流文件格式支持

当前预训练、后训练数据流支持`jsonl`、`json`格式的数据

## 1. 预训练/后预训练数据流

### 1.1. 在线数据流

#### 1.1.1. messages 格式

使用 `messages` 格式需要在 `train(/eval)_dataset_type` 处指定为 `messages`

messages 格式：每条数据都是一个字典，包含以下字段：

- `messages` : `List(Dict）`

样例数据：

```json
{"messages": {"role": "assistant", "content": "一个需要连续输入值的分类问题的示例是房屋价格预测。房屋的价格通常基于诸如平方英尺、位置、卧室和浴室数量以及像后院或车库等功能这样的因素定价。为了准确预测房屋价格，这些标准必须作为连续输入值输入到分类模型中。"}}
...
```

#### 1.1.2. erniekit 格式

使用 `erniekit` 格式需要在 `train(/eval)_dataset_type` 处指定为 `erniekit`

erniekit 格式：每条数据都是一个字典，包含以下字段：

- `text` : `str`

样例数据：

```json
{"text": "一个需要连续输入值的分类问题的示例是房屋价格预测。房屋的价格通常基于诸如平方英尺、位置、卧室和浴室数量以及像后院或车库等功能这样的因素定价。为了准确预测房屋价格，这些标准必须作为连续输入值输入到分类模型中。"}
...
```

为了方便测试，我们也提供了demo 数据集可以直接使用：

```shell
# messages格式
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/pt_online_data_messages.tar.gz
mkdir -p data/pt && tar -xf pt_online_data_messages.tar.gz -C data/pt/
# erniekit格式
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/pt_online_data_erniekit.tar.gz
mkdir -p data/pt && tar -xf pt_online_data_erniekit.tar.gz -C data/pt/
```

### 1.2. 离线数据流

我们也可以选择使用离线的比特预训练数据流，更节省内存。

为了方便测试，我们也提供了[离线预训练 demo 数据集](https://paddleformers.bj.bcebos.com/datasets/pretrain_offline_data.tar.gz)可以直接使用：

```shell
wget https://paddleformers.bj.bcebos.com/datasets/pretrain_offline_data.tar.gz
mkdir -p data/pre-training && tar -xf pretrain_offline_data.tar.gz -C data/pre-training/
```

您也可以制作自己的离线数据流，离线数据流制作方法如下：

下载一个文本数据集，例如 https://modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset

格式需为 jsonl，每行格式例如 BazingaLyn/mini_pretrain_dataset/pretrain_hq_v7.jsonl：
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
| `--data_format`    | string     | 支持的文件格式，当前只支持 JSON |
| `--input_path`     | string     | 输入的 json 文件的路径  |
| `--append_eos`     | store_true | 是否在每条数据的结尾添加 eos token  |
| `--output_prefix`  | str        | 输出文件的前缀    |
| `--workers`        | int        | 运行的进程数     |
| `--log_interval`   | int        | 打印日志间隔   |
| `--data_impl`      | str        | 制作的数据集类型，默认为 mmap，也可以选择 lazy |

## 2. SFT 数据流

### 2.1. messages 格式

使用 `messages` 格式需要在 `train(/eval)_dataset_type` 处指定为 `messages`

SFT 数据流中，每条数据都是一个字典，包含以下字段：

- `messages` : `List(Dict)`, 每个字典包含 `role`、`content`、`tool_calls(optional)` 三种 key。
    - `role` 的值可以选择 `system`, `user`, `assistant`, `tool/tool_response/observation(optional)` 或 `tool_call/tool_call/function(optional)`。
    - `content`为具体的对话内容。
    - `tool_calls(optional)` 为申请工具调用。
- `tools(optional)` : `List(Dict)`, 表示工具信息。
- `label(optional)`: Training flag (1=参与训练, 0=不参与训练)

Notes:
* 每个训练样本均为 JSON 格式，多个样本以换行符分隔

样例数据：

```json
[
    {
        "messages": [
            {"role": "system", "content": "You are a good coder."},
            {"role": "user", "content": "Given an integer array nums and an integer target value target, find two integers in the array whose sum equals the target value target, and return their array indices. You may assume that each input will have exactly one solution, and you may not use the same element twice. You can return the answer in any order. Example 1: Input: nums = [2,7,11,15], target = 9\nOutput: [0,1]\nExplanation: Because nums[0] + nums[1] == 9, return [0, 1]."},
            {"role": "assistant", "content": "<think>We are going to use a hash map (dictionary) to store the numbers we have seen so far along with their indices.\n For each number in the array, we calculate the complement (target - current number).\n If the complement exists in the hash map, that means we have found the two numbers that add up to the target.\n We then return the current index and the index of the complement from the hash map.\n Since we are guaranteed exactly one solution, we can return immediately when we find it.\n</think>\nTo solve this problem efficiently, we can use a hash map to store each number's index as we iterate through the array. For each number, we calculate its complement (target minus the current number). If the complement exists in the hash map, we immediately return the current index and the complement's index. This approach ensures we find the solution in linear time.\n\n### Approach\n1. **Initialization**: Create an empty dictionary to store numbers and their indices.\n2. **Iterate through the array**: For each number in the array:\n   - Calculate the complement (target - current number).\n   - Check if the complement exists in the dictionary.\n   - If found, return the current index and the complement's index.\n   - Otherwise, add the current number and its index to the dictionary.\n3. **Result**: Since the problem guarantees exactly one solution, we will always find it during iteration.\n\n### Solution Code\n```python\ndef twoSum(nums, target):\n    num_map = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in num_map:\n            return [num_map[complement], i]\n        num_map[num] = i\n```\n\n### Explanation\n- **Initialization**: The dictionary `num_map` will track each number encountered and its index.\n- **Iteration**: For each number `num` at index `i`:\n  - Compute `complement = target - num`.\n  - If `complement` is in `num_map`, return `[num_map[complement], i]` since these two numbers sum to the target.\n  - Otherwise, store `num` with its index in `num_map` for future checks.\n- **Efficiency**: This approach runs in O(n) time, where n is the number of elements, as each dictionary operation (insertion and lookup) is O(1) on average. The space complexity is O(n) to store the dictionary.\n"}
        ]
    }
]
```

### 2.2. erniekit 格式

使用 `erniekit` 格式需要在 `train(/eval)_dataset_type` 处指定为 `erniekit`

SFT 数据流中，每条数据都是一个字典，包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。
- `system(optional)` : 系统配置
- `label(optional)`: Training flag (1=参与训练, 0=不参与训练)
- `is_system(optional)` : 标志 src 的第一条数据是否是 system

Notes:
* `src` 和 `tgt` 为支持多轮对话的列表（List）对象
* 每个训练样本均为 JSON 格式，多个样本以换行符分隔

样例数据：
```json
{
    "system": "你是一个生活小助理",
    "src": [
        "我们如何在日常生活中减少用水？",
        "还有别的建议吗？"
    ],
    "tgt": [
        "1. 使用节水装置，如节水淋浴喷头和水龙头。",
        "2. 使用水箱或水桶收集家庭废水，例如洗碗和洗浴。 \n3. 在社区中提高节水意识。 \n4. 检查水管和灌溉系统的漏水情况，并及时修复它们。 \n5. 洗澡时间缩短，使用低流量淋浴头节约用水。 \n6. 收集雨水，用于园艺或其他非饮用目的。 \n7. 刷牙或擦手时关掉水龙头。 \n8. 减少浇水草坪的时间。 \n9. 尽可能多地重复使用灰水（来自洗衣机、浴室水槽和淋浴的水）。 \n10. 只购买能源效率高的洗碗机和洗衣机。"
    ],
    "label": [0, 1],
}
```

为了方便测试，我们也提供了 [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) demo 数据集可以直接使用：

```shell
# messages格式
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/sft_online_data_messages.tar.gz
mkdir -p data/sft && tar -xf sft_online_data_messages.tar.gz -C data/sft/
# erniekit格式
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/sft_online_data_erniekit.tar.gz
mkdir -p data/sft && tar -xf sft_online_data_erniekit.tar.gz -C data/sft/
```

用于 function call 训练的 demo 数据：

```json
[
    {
        "messages": [
            {"role": "user", "content": "I'm feeling a bit down. Can you tell me a joke to cheer me up?"},
            {"role": "assistant", "content": "<think>Okay, let me try to figure out how to approach this. The user is feeling down and asks for a joke to cheer up. I need to connect this request to the appropriate function call. Looking at the available tools, there's a function called get_random_joke which is designed exactly for this purpose. Since the user's main need is to feel better, providing a joke makes sense. The function doesn't require any parameters, so it's straightforward to call it without any additional arguments.\n</think>", "tool_calls": [{"type": "function", "function": {"name": "get_random_joke", "arguments": {}}}]},
            {"role": "tool", "content": [{"joke": "Why don't scientists trust atoms? Because they make up everything!"}]},
            {"role": "assistant", "content": "Sure, here's a joke for you: \"Why don't scientists trust atoms? Because they make up everything!\" I hope that brings a smile to your face."}
        ],
        "tools": [
            {"type": "function", "function": {"name": "get_random_joke", "description": "Get a random joke", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "generate_random_number", "description": "Generate a random number within a specified range", "parameters": {"type": "object", "properties": {"min": {"type": "number", "description": "The minimum value of the range"}, "max": {"type": "number", "description": "The maximum value of the range"}}, "required": ["min", "max"]}}}
        ]
    }
]
```

为了方便测试，我们也提供了 `messages` 格式的 `function call` 数据集可以直接使用：
```bash
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/sft_function_call_data.tar.gz
mkdir -p data/sft && tar -zxf sft_function_call_data.tar.gz -C data/sft/
```

## 3. DPO 数据流

### 3.1. messages 格式

使用 `messages` 格式需要在 `train(/eval)_dataset_type` 处指定为 `messages`

DPO 数据流中，每条数据都是一个字典，包含以下字段：
- `messages` : `List(dict)`, 对话历史列表, 包含 `role` (`"user"` 或 `"assistant"`) 和 `content` (`str`) 字段。
- `chosen_response` : `List(dict)`, 偏好（chosen）的系统回复, 包含 `role` (`"assistant"`) 和 `content` (`str`) 等字段，根据是否调用工具可能包含工具调用信息 (`tool_calls`)。
- `rejected_response` : `List(dict)`, 非偏好（rejected）的系统回复, 包含 `role` (`"assistant"`) 和 `content` (`str`) 等字段。
- `tools` : `List(dict)`, 对话中可能用到的工具（函数）的定义列表。

样例数据
```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n<tools>\n[{'type': 'function', 'function': {'name': 'play_music', 'description': 'Play music from a specified playlist or genre', 'parameters': {'type': 'object', 'properties': {'playlist': {'type': 'string', 'description': 'The playlist to play'}, 'genre': {'type': 'string', 'description': 'The genre of music to play'}}, 'required': []}}}, {'type': 'function', 'function': {'name': 'analyze_sentiment', 'description': 'Analyze the sentiment of a text', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': 'The text to analyze'}, 'language': {'type': 'string', 'description': 'The language of the text (optional)'}}, 'required': ['text']}}}]\n</tools>\nFor each function call return a json object with function name and arguments within <tool_call> </tool_call> XML tags with the following schema:\n<tool_call>\n{'arguments': <args-dict>, 'name': <function-name>}\n</tool_call>\n"
        },
        {
            "role": "user",
            "content": "I want to listen to some music. Can you play something for me?"
        }
    ],
    "chosen_response": [{
        "role": "assistant",
        "content": "Of course! Do you have a specific playlist or genre in mind?"
    }],
    "rejected_response": [{
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
    }],
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
}
```

### 3.2. erniekit 格式

使用 `erniekit` 格式需要在 `train(/eval)_dataset_type` 处指定为 `erniekit`

DPO 数据流中，每条数据都是一个字典，包含以下字段：

- `system(optional)`: 系统配置
- `src` : `str, List(str)`, 用户对话内容
- `tgt` : `str, List(str)`, 系统回复内容（比 src 少一个）
- `response` : `str, List(str)`, 包含 chosen 和 rejected 回复。
- `sort` : `List(int)`, sort 值用于区分 response 中 chosen 和 rejected（sort 值小的是 rejected，sort 值大的是 chosen）。
- `is_system(optional)` : 标志 src 的第一条数据是否是 system

Notes:
* 每个训练样本均为 JSON 格式，多个样本以换行符分隔

样例数据：

```json
{
    "system": "你是一个生活小助理",
    "src": [
        "你好。",
        "哪一个富含蛋白质，床还是墙？"
    ],
    "tgt": ["你好呀，我是你的生活小助理。"],
    "response": [
        [
            "床和墙都不是蛋白质的来源，因为它们都是无生命的物体。蛋白质通常存在于肉类、奶制品、豆类和坚果等食物中。"
        ],
        [
            "对不起，我无法回答那个问题。请提供更具体的信息，让我知道你需要什么帮助。"
        ]
    ],
    "sort": [
        1,
        0
    ]
}
...
```

为了方便测试，我们也提供了偏好数据集可以直接使用：
```bash
# messages格式
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/dpo_online_data_messages.tar.gz
mkdir -p data/dpo && tar -xf dpo_online_data_messages.tar.gz -C data/dpo/
# erniekit格式
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/dpo_online_data_erniekit.tar.gz
mkdir -p data/dpo && tar -xf dpo_online_data_erniekit.tar.gz -C data/dpo/
```

为了方便测试，我们也提供了 function call DPO 数据集可以直接使用：
```bash
# messages格式
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/dpo_function_call_data.tar.gz
mkdir -p data/dpo && tar -xf dpo_function_call_data.tar.gz -C data/dpo/
```

## 4. 多模 SFT 数据流

### 4.1. messages 格式

使用 `messages` 格式需要在 `train(/eval)_dataset_type` 处指定为 `messages`

多模 messages 格式需要在纯文 messages 格式的基础上加上`images`、`videos`、`audios`几个 key，用于传入多模态资源的`url`或者`path`，同时在`messages`中插入`<image>`、`<video>`、`<audio>`标签来表述插入多模态数据的位置：

纯文：
```json
{"messages": [{"role": "assistant", "content": "预训练的文本在这里"}]}
```
加入图片：
```json
{"messages": [{"role": "assistant", "content": "<image>是一只小狗，<image>是一只小猫"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
```
加入音频：
```json
{"messages": [{"role": "assistant", "content": "<audio>描述了今天天气真不错"}], "audios": ["/xxx/x.wav"]}
```
加入图片与视频：
```json
{"messages": [{"role": "assistant", "content": "<image>是一个大象，<video>是一只狮子在跑步"}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
```

### 4.2. erniekit 格式

使用 `erniekit` 格式需要在 `train(/eval)_dataset_type` 处指定为 `erniekit`

SFT 数据流中，每条数据都是一个字典，包含以下字段：

* `text_info`: 纯文本的列表，每个元素包含一个 `text` 和一个 `tag`
  * `text`: 来自使用者的问题或系统回复的文字内容
  * `tag`: 遮挡标签 (`no_mask`=包含在训练中, `mask`=排除)
* `image_info`: 图像组成的列表，每个元素包含一个 `image_url` 和一个 `matched_text_index`
  * `image_url`: 线上下载图像的网址或本地存取图像的路径
  * `matched_text_index`: `text_info` 中匹配文字的索引
    * 预设值: `matched_text_index=0` 表示图像与第一个文字匹配，并将其放置在第一个文字之前
* `is_system(optional)`: 系统标志 (1=系统配置 0=无系统配置)
  * 系统配置 = 如果 `is_system=1`，则为 `text_info[0]`

注意：
* 通过将 `image_info` 替换为 `video_info` 来支持视频数据
* 请确保 `mask` 和 `no_mask` 在 `text_info` 中交替出现

这是一个 SFT VL 数据集的多图像示例：

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./DoclingMatix/218/0.png"},
        {"matched_text_index": 0, "image_url": "./DoclingMatix/218/1.png"}
    ],
    "text_info": [
        {"text": "What is the purpose of the resolution discussed in the text?", "tag": "mask"},
        {"text": "The purpose of the resolution is to approve the redevelopment contract of the Philadelphia Redevelopment Authority for the redevelopment and urban renewal of a portion of the Haddington Urban Renewal Area, Unit Nos. 2 and 3, and to authorize the Redevelopment Authority to execute the redevelopment contract with Danielle M. Carson-Varns.", "tag": "no_mask"},
        {"text": "Who introduced Resolution No. 160204 to the City Council?", "tag": "mask"},
        {"text": "Councilmember Blackwell introduced Resolution No. 160204 to the City Council.", "tag": "no_mask"},
        ...
    ]
}
```

这是一个 SFT VL 数据集的单视频示例：
```json
{
    "video_info": [
        {"matched_text_index": 0, "image_url": "./NExTVideo/1027/4789497818.mp4"}
    ],
    "text_info": [
        {"text": "how does the man sit on the grass?\nA. kneel\nB. one leg in the air\nC. sitting on bicycle seat\nD. legs spread out\nE. squatting down\n Answer with the option's letter from the given choices directly.", "tag": "mask"},
        {"text": "D", "tag": "no_mask"}
    ]
}
```

这是一个 SFT VL 数据集的系统配置示例:
```json
{
    "is_system": 1,
    "text_info": [
        {"text": "Your role as ...", "tag": "mask"},
        {"text": "好的", "tag": "no_mask"},
        {"text": "What is written...", "tag": "mask"},
        {"text": "<think>So I've got...", "tag": "no_mask"},
        ...
    ]
    "image_info": [...]
}
```

为了方便测试，我们也提供了用于快速训练的 demo 数据：

```shell
# messages格式
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/sft_vl_data_messages.tar.gz
mkdir -p data/sft-vl && tar -xf sft_vl_data_messages.tar.gz -C data/sft-vl
# erniekit格式
wget https://paddleformers.bj.bcebos.com/datasets/release/v1.0/sft_vl_data_erniekit.tar.gz
mkdir -p data/sft-vl && tar -xf sft_vl_data_erniekit.tar.gz -C data/sft-vl
```
