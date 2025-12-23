# 1. 注册template

## 1.1. 注册方法

在paddleformers/datasets/template/template.py文件中实现模型chat template的注册，如：
```python
register_template(
    name="ernie",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n\n<|im_start|>assistant\n"]),
    default_system="<global_setting>\nthink_mode=True\n</global_setting>",
    stop_words=["<|im_end|>"],
)
```

## 1.2. 参数说明

| 参数名              | 解释       |
|--------------------|-----------|
| `name` | template的名字，也就是训练的时候需要指定的template参数 |
| `format_user` | 对role为user的content进行format，{{content}}表示塞入实际的content，其他为拼接的token |
| `format_assistant` | 对role为assistant的content进行format |
| `format_system` | 对role为system的content进行format |
| `format_function` | 对role为function（申请工具调用）的content进行format |
| `format_observation` | format_observation |
| `format_tools` | 对tools信息进行format |
| `format_prefix` | 在system前面加的内容 |
| `default_system` | 默认的system信息，如果数据里面没有role为system的，就用这个 |
| `stop_words` | 当replace_eos为true的时候，会用stop words替换掉实际的eos token |
| `replace_eos` | 是否使用stop_words替换默认的eos token |
| `thought_words` | 数据里面的思考标志是什么，比如<think></think> |
| `efficient_eos` | eos是否有效，即是否在最后拼接eos token |
| `chat_sep` | 历史轮对话末尾加的字符串 |
| `auto_add_bos` | 如果bos没添加，会自动添加上 |
| `enable_thinking` | 否的话，会把思考信息删掉（当template_class选ReasoningTemplate时候生效） |
| `mm_plugin` | 使用什么插件来处理多模信息 |
| `grounding_plugin` | 使用什么插件来处理grounding任务的target信息 |
| `template_class` | template类，可以选Template或ReasoningTemplate，ReasoningTemplate一般是思考模型会用的，会根据enable_thinking决定是否删除思考信息 |

## 1.3. 示例

如果chat template长这样：
```text
<s><user>user prompt here
<model>model response here</s>
<user>user prompt here
<model>model response here</s>
```

相对应的register_template应该这样写：
```python
register_template(
    name="custom",
    format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_prefix=EmptyFormatter("<s>"),
    chat_sep="</s>\n",
)
```

# 2. 注册mm_plugin
多模模型需要实现自己的多模数据处理方法，包括图片处理、视频处理、音频处理、获取处理后的tokens数量来填充占位符
可以参考Qwen2VLPlugin类，类实现后在下面注册：
```python
PLUGINS = {
    "base": BasePlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
    "glm4v": GLM4VPlugin,
}
```
