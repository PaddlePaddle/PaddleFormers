# 1. 背景说明

大语言模型依托对话交互能力，能够输出符合人类语境的智能回复。而这一能力需依托 **Chat Template **结构化标注角色与上下文，定义多轮对话数据如何被转换为模型可训练的 token 序列，从而确保交互逻辑精准。

**Chat Template ** 规定了：

* 不同角色（如 `system / user / assistant`）的拼接方式

* 特殊 token（如 BOS / EOS / role token）的插入规则

* 哪些 token 参与 loss 计算（label / mask 规则）

# 2. 使用说明

PaddleFormers 内置了常用模型的默认 Chat Template，普通用户无需额外开发，在训练配置中指定使用即可，如 qwen3：

```yaml
template_backend: custom
template: qwen3
```

值得注意的是，哪怕是同一系列的模型，在不同训练任务（如 思考模型和非思考模型）对 label 和 mask 的策略也存在差异，务必确保为对应的模型设置正确的 Chat Template，使用不匹配的 Chat Template 可能导致：

* 训练不收敛或效果明显下降

* 对话边界学习错误

* 推理阶段输出异常

在模型列表中，我们给出了不同模型训练时所需使用的 Template，在训练时请按照表格所建议的 Template 进行设置。

# 3. 自定义新的 Template

在以下情况中，您可能想要使用自己定义的 Chat Template：

* 希望自定义对话格式或特殊 token 规则

* 新增或接入框架尚未内置的新模型

* 需要调整 label / mask 行为（如特殊 EOS、packing 场景）

在这些情况下，您需要显式注册 Chat Template，以确保模型输入格式与训练目标一致。

对于未注册的情况，框架将使用对应模型的默认模板。

## 3.1. 注册方法

在 `paddleformers/datasets/template/template.py` 文件中实现模型 chat template 的注册，如：

```python
register_template(
    name="glm4_moe",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["[gMASK]<sop><|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4_moe"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4_moe"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    suffix=["<|user|>"],
    thought_words=("<think>", "</think>"),
    template_class=ReasoningTemplate,
)
```

## 3.2. 参数说明

|参数名|解释|
|-|-|
|`name`|template 的名字，也就是训练的时候需要指定的 template 参数|
|`format_user`|对 role 为 user 的 content 进行 format，{{content}}表示塞入实际的 content，其他为拼接的 token|
|`format_assistant`|对 role 为 assistant 的 content 进行 format|
|`format_system`|对 role 为 system 的 content 进行 format|
|`format_function`|对 role 为 function（申请工具调用）的 content 进行 format|
|`format_observation`|对 role 为 observation（工具返回信息）的 content 进行 format|
|`format_tools`|对 tools 信息进行 format|
|`format_prefix`|在 system 前面加的内容|
|`default_system`|默认的 system 信息，如果数据里面没有 role 为 system 的，就用这个|
|`chat_sep`|历史轮对话末尾加的字符串|
|`suffix`|默认为 eos token，在多轮对话的最后面添加|
|`efficient_eos`|suffix 是否有效，即是否在最后拼接 suffix token|
|`auto_add_bos`|设置为 true 的时候，如果 bos 没添加，会自动添加上|
|`thought_words`|数据里面的思考标志是什么，比如<think></think>|
|`enable_thinking`|否的话，会把思考信息删掉（当 template_class 选 ReasoningTemplate 时候生效）|
|`mm_plugin`|使用什么插件来处理多模信息|
|`grounding_plugin`|使用什么插件来处理 grounding 任务的 target 信息|
|`template_class`|template 类，可以选 Template 或 ReasoningTemplate，ReasoningTemplate 一般是思考模型会用的，会根据 enable_thinking 决定是否删除思考信息|

## 3.3. 多模态处理（注册 mm_plugin）

多模模型需要实现自己的多模数据处理方法，包括图片处理、视频处理、音频处理、获取处理后的 tokens 数量来填充占位符

具体实现方式可以参考 Qwen2VLPlugin 类

### 3.3.1. 多模数据下载（选做）

在基类 `MMPluginMixin` 中，PaddleFormers 已经为大家实现了最基本的数据下载函数：

* `_regularize_images`：负责图片数据下载

* `_regularize_videos`：负责视频数据下载和抽帧

* `_regularize_videos`：负责音频数据下载（暂未实现）

若大家有定制化的数据下载需求，只需重写对应的处理函数即可

### 3.3.2. 多模数据预处理（选做）

在基类 `MMPluginMixin` 中，PaddleFormers 已经为大家实现了最基本的多模数据预处理函数：`_get_mm_inputs`

在 `_get_mm_inputs` 中，PaddleFormers 会调用多模 `processor` 来处理对应模态的数据，并返回处理结果

若大家有定制化的数据预处理需求，只需重写 `_get_mm_inputs` 函数即可

### 3.3.3. 多模 token 拼接（必做）

PaddleFormers 需要重写 `process_messages` 函数来实现 template 中多模 token 的拼接逻辑。

在模型的 template 中，各个模态的数据会有各自的占位符，图片数据通常使用占位符 `<image>` ，视频数据通常使用占位符 `<video>` ，音频数据通常使用占位符 `<audio>` 。在 `process_messages` 函数中，PaddleFormers 需要将每个占位符替换为对应数量的多模 token。

以 `Qwen2VLPlugin` 中对图片 token 的处理为例：

```python
# ... 省略前序处理逻辑
for message in messages:    # 遍历messages中的每一轮对话
    content = message["content"]    # 获取每轮对话的内容
    while IMAGE_PLACEHOLDER in content: # 当对话内容中存在图片占位符时
        image_seqlen = (
            image_grid_thw[num_image_tokens].prod().item() // merge_length if self.expand_mm_tokens else 1
        )   # 根据image_grid_thw计算图片token的个数
        content = content.replace(
            IMAGE_PLACEHOLDER,
            f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
            1,
        )   # 将图片占位符替换为对应的多模special tokens
        num_image_tokens += 1
# ... 省略后续处理逻辑
```

### 3.3.4. mm_plugin 注册

类实现后在下面注册：

```python
PLUGINS = {
    "base": BasePlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
    "glm4v": GLM4VPlugin,
}
```
### 3.3.5. template 注册

mm_plugin 注册完后，还需要在 template 中注册：

```python
register_template(
    name="qwen2_vl",
    ...
    mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)
```

其中，`name`需要填入在`PLUGINS`中注册的`key`名，`image_token`、`video_token`、`audio_token`为模型各模态的 special token。

## 3.4 示例
如果模型的 chat template 为：

```jinja
<s><user>user prompt here
<model>model response here</s>
<user>user prompt here
<model>model response here</s>
```

对应的 register_template 为：

```python
register_template(
    name="custom",
    format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_prefix=EmptyFormatter("<s>"),
    chat_sep="</s>\n",
    suffix="</s>",
)
```

## 3.5 查看 Template 处理效果

在进行 sft 训练的时候，打开 FLAGS_enable_dataset_debug，即可打印 decode 之后的 input_ids 和 label，如：

![template-demo](https://github.com/user-attachments/assets/b4dd54bb-1968-47d5-b662-3404b9baefa9)
查看打印的 input 和 labels 是否符合预期来确认 template 实现是否正确

# 4. 推理使用说明

训练阶段自定义注册的 template 必须与推理阶段保持一致，确保相同输入在两阶段生成完全一致的 token。否则将导致训练与推理输入不一致，影响模型实际推理表现。

具体方法：您可以使用 huggingface 提供的 [https://huggingface.co/spaces/huggingfacejs/chat-template-playground](https://huggingface.co/spaces/huggingfacejs/chat-template-playground) 测试该模型 template 对数据渲染后的结果，与打开 FLAGS_enable_dataset_debug 后打印的训练数据进行对比，如果不一致，需要调整自定义的 template 或 plugin。
