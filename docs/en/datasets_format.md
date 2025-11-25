# Data Stream Format Documentation

## Data Stream File Format Support

Currently, pre-training and post-training data streams only support the `jsonl` format.

## 1. Pre-training Data Stream

### 1.1. Online Data Stream

In the pre-training data stream, each data entry is a dictionary containing the following fields:

- `text` : `str, List(str)`, pre-training text.

Sample data:

```text
{"text": ["An example of a classification problem that requires continuous input values is house price prediction. The price of a house is usually based on factors such as square footage, location, number of bedrooms and bathrooms, and features like a backyard or garage. To accurately predict house prices, these criteria must be entered into the classification model as continuous input values."]}
...
```

For ease of testing, we also provide a [demo dataset](https://paddleformers.bj.bcebos.com/datasets/pt_data.tar.gz) that can be used directly:

```shell
wget https://paddleformers.bj.bcebos.com/datasets/pt_data.tar.gz
mkdir -p data/pt && tar -xf pt_data.tar.gz -C data/pt/
```

### 1.2. Offline Data Stream

We can also choose to use offline bit pre-training data streams, which saves more memory.

For ease of testing, we also provide an [offline pre-training demo dataset](https://paddleformers.bj.bcebos.com/datasets/pretrain_offline_data.tar.gz) that can be used directly:

```shell
wget https://paddleformers.bj.bcebos.com/datasets/pretrain_offline_data.tar.gz
tar -xf pretrain_offline_data.tar.gz -C data/pre-training/
```

You can also create your own offline data stream. The method for creating an offline data stream is as follows:

Download a text dataset, such as https://modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset

The format must be jsonl, and the format of each line is like BazingaLyn/mini_pretrain_dataset/pretrain_hq_v7.jsonl:
```text
{"text": "Scrambled eggs with tomatoes\nIngredients:\n3 eggs, 1 tomato, oil, salt, sugar, cornstarch\nInstructions:..."}
{"text": "Please describe how to properly plan personal finance. Properly planning personal finance requires the following steps..."}
{"text": "Please enter a scene dialogue about marine conservation. Person A: Wow, this beach is really..."}
{"text": "Identify two different types of wine. The method of identifying wine varies depending on its type and variety, below..."}
```

Run `examples/tools/create_pretraining_data.py`, and the generated data will be saved in `./pretrain_data.bin` and `./pretrain_data.idx` in the current directory.
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

- Parameter Description

| Parameter Name              | Type        | Description                 |
|--------------------|----------- |-----------------|
| `--model_name_or_path`     | string     | Model path  |
| `--data_format`    | string     | Supported file format, currently only supports JSON |
| `--input_path`     | string     | Path to the input json file  |
| `--append_eos`     | store_true | Whether to add an eos token at the end of the document  |
| `--output_prefix`  | str        | Prefix of the output file    |
| `--workers`        | int        | Number of processes to run     |
| `--log_interval`   | int        | Log printing interval   |
| `--data_impl`      | str        | Type of dataset to create, default is mmap, can also choose lazy |

## 2. SFT Data Stream

### erniekit format

To use the `erniekit` format, specify `erniekit` at `train(/eval)_dataset_type`.

In the SFT data stream, each data entry is a dictionary containing the following fields:

- `src` : `str, List(str)`, the model's input instructions, prompts, and the task the model should perform.
- `tgt` : `str, List(str)`, the model's output.
- `system(optional)` : System configuration
- `label(optional)`: Training flag (1=participate in training, 0=do not participate in training)
- `is_system(optional)` : Indicates whether the first piece of data in src is system

Notes:
* `src` and `tgt` are list objects that support multi-turn conversations.
* Each training sample is in JSON format, with multiple samples separated by line breaks.

Sample data:
```json
{
    "system": "You are a life assistant",
    "src": [
        "How can we reduce water consumption in our daily lives?",
        "Any other suggestions?"
    ],
    "tgt": [
        "1. Use water-saving devices, such as water-saving shower heads and faucets.",
        "2. Use tanks or buckets to collect household wastewater, such as dishwashing and bathing. \n3. Raise awareness of water conservation in the community. \n4. Check water pipes and irrigation systems for leaks and repair them promptly. \n5. Shorten shower time and use low-flow shower heads to save water. \n6. Collect rainwater for gardening or other non-drinking purposes. \n7. Turn off the faucet when brushing your teeth or wiping your hands. \n8. Reduce the time spent watering lawns. \n9. Reuse gray water (water from washing machines, bathroom sinks, and showers) as much as possible. \n10. Only buy energy-efficient dishwashers and washing machines."
    ],
    "label": [0, 1],
}
```

For ease of testing, we also provide the [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) demo dataset that can be used directly:

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/examples/alpaca_demo.gz
mkdir -p data/sft && tar -xf alpaca_demo.gz -C data/sft/ --strip-components=1
```

### chatml format

To use the `chatml` format, specify `chatml` at `train(/eval)_dataset_type`.

In the SFT data stream, each data entry is a dictionary containing the following fields:

- `messages` : `List(Dict)`, each dictionary contains three keys: `role`, `content`, and `tool_calls(optional)`.
    - The value of `role` can be `system`, `user`, `assistant` or `tool(optional)`.
    - `content` is the specific dialogue content.
    - `tool_calls(optional)` is for requesting tool calls.
- `tools(optional)` : `List(Dict)`, represents tool information.
- `label(optional)`: Training flag (1=participate in training, 0=do not participate in training)

Notes:
* Each training sample is in JSON format, with multiple samples separated by line breaks.

Sample data:

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

- Note: The demo datasets provided in `examples/data/sft_think-train.jsonl` and `examples/data/sft_think-eval.jsonl` come from the [OpenCodeReasoning dataset](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) released by nvidia. This dataset needs to follow the Creative Commons Attribution 4.0 International License (CC BY 4.0) agreement.

Demo data for function call training:

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

For ease of testing, we also provide a `chatml` function call SFT dataset that can be used directly:
```bash
wget https://paddleformers.bj.bcebos.com/datasets/sft_function_call_demo.tar.gz

mkdir -p data/sft && tar -zxf sft_function_call_demo.tar.gz -C data/sft/
```

## 3. DPO Data Stream

### erniekit format

To use the `erniekit` format, specify `erniekit` at `train(/eval)_dataset_type`.

In the DPO data stream, each data entry is a dictionary containing the following fields:

- `system(optional)`: System configuration
- `src` : `str, List(str)`, User dialogue content
- `tgt` : `str, List(str)`, System reply content (one less than src)
- `response` : `str, List(str)`, Contains chosen and rejected replies.
- `sort` : `List(int)`, The sort value is used to distinguish between chosen and rejected in the response (the smaller sort value is rejected, and the larger sort value is chosen).
- `is_system(optional)` : Indicates whether the first piece of data in src is system

Notes:
* Each training sample is in JSON format, with multiple samples separated by line breaks.

Sample data:

```json
{
    "system": "You are a life assistant",
    "src": [
        "Hello.",
        "Which is richer in protein, a bed or a wall?"
    ],
    "tgt": ["Hello, I am your life assistant."],
    "response": [
        [
            "Neither beds nor walls are sources of protein, as they are both inanimate objects. Protein is usually found in foods such as meat, dairy products, beans, and nuts."
        ],
        [
            "Sorry, I can't answer that question. Please provide more specific information so I know what help you need."
        ]
    ],
    "sort": [
        1,
        0
    ]
}
...
```

For ease of testing, we also provide a preference dataset that can be used directly:

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
mkdir -p data/dpo && tar -zxf ultrafeedback_binarized.tar.gz -C data/dpo/ --strip-components=1
```

### chatml format

To use the `chatml` format, specify `chatml` at `train(/eval)_dataset_type`.

In the DPO data stream, each data entry is a dictionary containing the following fields:
- `messages` : `List(dict)`, a list of dialogue history.
  - Normal rounds: Contains `role` (`"user"` or `"assistant"`) and `content` (`str`) fields.
  - Preference/Non-preference rounds (for preference learning): Contains the following two key fields to represent the preference ranking of different system responses to the same user query.
    - `preferred_output` : `dict`, the preferred (chosen) system response, including fields such as `role` (`"assistant"`) and `content` (`str`), and may include tool call information (`tool_calls`) depending on whether the tool is called.
    - `non_preferred_output` : `dict`, the non-preferred (rejected) system response, including fields such as `role` (`"assistant"`) and `content` (`str`).
- `tools` : `List(dict)`, a list of definitions of tools (functions) that may be used in the dialogue.
- `label` : `List(int)`, a sorting label used to distinguish between `preferred_output` and `non_preferred_output`. Where 0 corresponds to `non_preferred_output` (rejected) and 1 corresponds to `preferred_output` (chosen).

Detailed data format can be found in [function call instructions](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/examples/best_practices/function_call.md)

Sample data
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

For ease of testing, we also provide a `chatml` function call DPO dataset that can be used directly:
```bash
wget https://paddleformers.bj.bcebos.com/datasets/dpo_function_call_1k.tar.gz

mkdir -p data/dpo_fc && tar -zxf dpo_function_call_1k.tar.gz -C data/dpo_fc/
```
