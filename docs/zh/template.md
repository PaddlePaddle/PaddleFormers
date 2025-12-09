## 指定训练使用的template

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `template_backend` | str | 指定为`custom`表示使用自定义的template，`jinja`表示使用apply_chat_template方法进行拼接，不适合多轮对话，不推荐使用 |
| `template` | str | （只在 `template_backend` 为 `custom` 时生效）指定训练用的 template
| `split_multi_turn` | bool | 只在 `template_backend` 为 `jinja` 时生效）`True`表示将多轮数据拆成多条数据进行训练，`False`表示每次只学习最后一轮的回复 |
| `encode_one_turn` | str | 只在 `template_backend` 为 `jinja` 时生效）`True`表示将多轮对话进行拆分，分别对每一轮对话套用`apply_chat_template`，`False`表示直接对整段对话套用`apply_chat_template` |

## 自定义template
在`paddleformers/datasets/template/template.py`文件中，通过`register_template`实现自定义template

## 多模plugin接入流程

在 `paddleformers/datasets/template/mm_plugin.py` 文件中实现各种多模预处理的处理，基类是`BasePlugin`，已经实现了各种图片、视频、音频的预处理操作，如果需要自定义plugin，需要继承`BasePlugin`实现自定义的类，如`Qwen2VLPlugin`。在自定义的类中实现各种多模数据预处理的操作，并在`PLUGINS`里面注册template名字和类名的对应关系