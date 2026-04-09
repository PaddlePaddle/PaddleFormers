# 1. 注册 template

## 1.1. 注册方法

### 1.1.1 源代码修改（适用于 git clone 安装用户）
在 `paddleformers/datasets/template/template.py` 文件中实现模型 chat template 的注册，如：
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
### 1.1.2 运行时外挂（适用于 pip 安装用户）
可通过`--custom_register_path`参数指定一个 Python 文件来注册自定义对话模板。该文件应包含类似以下代码：
```python
from paddleformers.datasets.template.template import *

register_template(
    ...同上
)
```


## 1.2. 参数说明

| 参数名              | 解释       |
|--------------------|-----------|
| `name` | template 的名字，也就是训练的时候需要指定的 template 参数 |
| `format_user` | 对 role 为 user 的 content 进行 format，{{content}}表示塞入实际的 content，其他为拼接的 token |
| `format_assistant` | 对 role 为 assistant 的 content 进行 format |
| `format_system` | 对 role 为 system 的 content 进行 format |
| `format_function` | 对 role 为 function（申请工具调用）的 content 进行 format |
| `format_observation` | format_observation |
| `format_tools` | 对 tools 信息进行 format |
| `format_prefix` | 在格式化后最开头固定添加的内容 |
| `default_system` | 默认的 system 信息，如果数据里面没有 role 为 system 的，就用这个 |
| `stop_words` | 当 replace_eos 为 true 的时候，会用 stop words 替换掉实际的 eos token |
| `replace_eos` | 是否使用 stop_words 替换默认的 eos token |
| `thought_words` | 数据里面的思考标志是什么，比如<think></think> |
| `efficient_eos` | eos 是否有效，即是否在最后拼接 eos token |
| `chat_sep` | 历史轮对话末尾加的字符串 |
| `auto_add_bos` | 如果 bos 没添加，会自动添加上 |
| `enable_thinking` | 否的话，会把思考信息删掉（当 template_class 选 ReasoningTemplate 时候生效） |
| `mm_plugin` | 使用什么插件来处理多模信息 |
| `grounding_plugin` | 使用什么插件来处理 grounding 任务的 target 信息 |
| `template_class` | template 类，可以选 Template 或 ReasoningTemplate，ReasoningTemplate 一般是思考模型会用的，会根据 enable_thinking 决定是否删除思考信息 |

## 1.3. 示例

如果 chat template 长这样：
```text
<s><user>user prompt here
<model>model response here</s>
<user>user prompt here
<model>model response here</s>
```

相对应的 register_template 应该这样写：
```python
register_template(
    name="custom",
    format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_prefix=EmptyFormatter("<s>"),
    chat_sep="</s>\n",
)
```

# 2. 注册 mm_plugin
## 2.1. 注册方法
### 2.1.1 源代码修改（适用于 git clone 安装用户）
多模态模型需要实现自己的多模数据处理方法，包括图片处理、视频处理、音频处理、获取处理后的 tokens 数量来填充占位符
可以参考 Qwen2VLPlugin 类，类实现后在 `paddleformers/datasets/template/mm_plugin.py` 中注册：
```python
@dataclass
class MyCustomPlugin(BasePlugin):
    ...
PLUGINS = {
    "base": BasePlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
    "glm4v": GLM4VPlugin,
}
```
### 2.1.2 运行时外挂（适用于 pip 安装用户）
可通过`--custom_register_path`参数指定一个 Python 文件来注册自定义 mm_plugin。该文件应包含类似以下代码：
```python
from paddleformers.datasets.template.template import *
from paddleformers.datasets.template.mm_plugin import *

@dataclass
class MyCustomPlugin(BasePlugin):
    ...
register_mm_plugin(
    name = "myplugin",
    plugin_class = MyCustomPlugin,
)
register_template(
    name="mytemplate",
    mm_plugin=get_mm_plugin(name="myplugin"...),
    ...
)
```
