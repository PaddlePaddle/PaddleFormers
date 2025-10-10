# Function Call Support

## Data Format

### SFT Data Format

Demo data for function call training:

```json
[
    {
        "messages": [
            {
                "role": "user",
                "content": "I'm feeling a bit down. Can you tell me a joke to cheer me up?"
            },
            {
                "role": "assistant",
                "content": "<think>The user is feeling down and requested a joke. The available tool 'get_random_joke' is directly relevant to this request. Since it requires no parameters, I'll call it immediately.</think>",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_random_joke",
                            "arguments": {}
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": {
                    "joke": "Why don't scientists trust atoms? Because they make up everything!"
                }
            },
            {
                "role": "assistant",
                "content": "Here's a joke to cheer you up: \"Why don't scientists trust atoms? Because they make up everything!\" I hope that brightens your day."
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_random_joke",
                    "description": "Retrieves a random joke to lighten the mood.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_random_number",
                    "description": "Generates a random integer within a specified range.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min": {
                                "type": "number",
                                "description": "The inclusive minimum value of the range."
                            },
                            "max": {
                                "type": "number",
                                "description": "The inclusive maximum value of the range."
                            }
                        },
                        "required": ["min", "max"]
                    }
                }
            }
        ]
    }
]
```

### DPO Data Format

Demo data for function call training:

```json
[
    {
        "messages": [
            {"role": "user", "content": "This is normal turn's query."},
            {"role": "assistant", "content": "This is normal turn's answer."},
            {"role": "user", "content": "Hello, can you tell me how cold San Francisco is today?"},
            {
                "non_preferred_output": {
                    "role": "assistant", 
                    "content": "It is not particularly cold in San Francisco today."
                }, 
                "preferred_output": {
                    "role": "assistant", 
                    "content": "<think>I need to call function to get temperature info of SF.</think>", 
                    "tool_calls": [
                        {
                            "type": "function", 
                            "function": {
                                "name": "get_temperature", 
                                "arguments": {"area": "San Francisco"}
                            }
                        }
                    ]
                }
            }
        ],
        "tools": [],
        "label": [0, 1]
    }
]
```
- The assistant in normal Q&A rounds does not participate in training.
- For DPO training rounds, corresponding positions in the label field need to be indicated, where `preferred_output` represents the preferred output and `non_preferred_output` represents the rejected output.
- The `tools` definition field is the same as that in SFT.


## 训练
- 训练脚本请参考：[训练脚本](../README.md)
- 训练配置请参考：[SFT 训练配置](../config/fc/sft_full_fc.yaml)，[DPO 训练配置](../config/fc/dpo_full_fc.yaml)

## 推理

- 原始模型或者全参数训练后模型的推理或部署，请参考[Fastdeploy](https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/get_started)
